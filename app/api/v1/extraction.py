"""Extraction API endpoints for document processing."""

import os
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile, status

from app.config import get_settings
from app.core.auth import UserContext, get_current_user
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.schemas.extraction import (
    DocumentType,
    ExtractionJobResponse,
    ExtractionListResponse,
    ExtractionReprocessRequest,
    ExtractionStatus,
    ExtractionStatusResponse,
)
from app.services.extraction.router import get_document_router
from app.services.extraction.storage import get_storage_service

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/extraction", tags=["extraction"])


# File size limit in bytes
MAX_FILE_SIZE = settings.extraction_max_file_size_mb * 1024 * 1024

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json", ".xml"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/csv",
    "application/json",
    "text/xml",
    "application/xml",
    "text/plain",  # Some CSV files come as text/plain
}


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file size and type."""
    # Check file extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '{ext}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

    # Check content type
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Content type '{file.content_type}' not supported.",
        )


@router.post("/upload", response_model=ExtractionJobResponse)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document file to extract data from")],
    client_id: Annotated[UUID, Form(description="Client ID for this document")],
    document_type: Annotated[
        DocumentType | None,
        Form(description="Optional hint for document type"),
    ] = None,
    webhook_url: Annotated[
        str | None,
        Form(description="Optional webhook URL for completion notification"),
    ] = None,
    x_extraction_provider: Annotated[
        str | None,
        Header(description="Extraction provider to use (docling, reducto)"),
    ] = None,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> ExtractionJobResponse:
    """
    Upload a document for extraction.

    Accepts PDF, CSV, JSON, or XML files up to 50MB.
    Returns a job ID immediately. Poll /status/{job_id} for results.

    **Headers:**
    - `X-Extraction-Provider`: Optional. Extraction provider to use ("docling", "reducto").
      If not specified, uses the configured default (typically "docling").

    Document types:
    - **inbody**: InBody body composition scan (PDF)
    - **lab_results**: Lab results from Quest, LabCorp, etc. (PDF/CSV)
    - **garmin**: Garmin activity export (CSV/JSON)
    - **whoop**: WHOOP recovery/strain export (CSV)
    - **apple_health**: Apple Health export (XML/JSON)
    - **nutrition**: Nutrition log (CSV)

    If document_type is not provided, it will be auto-detected.
    """
    # Validate file
    validate_file(file)

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.extraction_max_file_size_mb}MB",
        )

    # Detect document type if not provided
    doc_router = get_document_router()
    detected_type = doc_router.detect_document_type(
        content,
        file.filename or "unknown",
        hint=document_type.value if document_type else None,
    )

    # Upload to S3
    storage = get_storage_service()
    file_url = await storage.upload_file(
        file_content=content,
        client_id=str(client_id),
        filename=file.filename or "document",
        content_type=file.content_type or "application/octet-stream",
    )

    # Create extraction job record
    query = """
        INSERT INTO ai_backend.extraction_jobs (
            client_id, coach_id, file_url, file_name, file_type, file_size,
            document_type, status, webhook_url
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id, client_id, coach_id, file_name, file_type, file_size,
                  document_type, status, created_at
    """

    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    row = await db.fetchrow(
        query,
        client_id,
        user.user_id,  # coach_id from authenticated user
        file_url,
        file.filename or "document",
        ext or "unknown",
        len(content),
        detected_type,
        "pending",
        webhook_url,
    )

    # Validate and normalize provider header
    valid_providers = {"docling", "reducto"}
    provider = None
    if x_extraction_provider:
        normalized = x_extraction_provider.lower().strip()
        if normalized in valid_providers:
            provider = normalized
        else:
            logger.warning(
                "Invalid extraction provider requested, using default",
                requested=x_extraction_provider,
            )

    logger.info(
        "Extraction job created",
        job_id=str(row["id"]),
        client_id=str(client_id),
        document_type=detected_type,
        file_size=len(content),
        provider=provider,
    )

    # Queue Celery task
    from app.workers.tasks.extraction import process_extraction

    process_extraction.delay(str(row["id"]), provider=provider)

    return ExtractionJobResponse(
        id=row["id"],
        client_id=row["client_id"],
        coach_id=row["coach_id"],
        file_name=row["file_name"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        document_type=row["document_type"],
        status=ExtractionStatus(row["status"]),
        created_at=row["created_at"],
    )


@router.get("/status/{job_id}", response_model=ExtractionStatusResponse)
async def get_extraction_status(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> ExtractionStatusResponse:
    """
    Get extraction job status and results.

    Returns the current status and extracted data when available.
    """
    query = """
        SELECT id, client_id, coach_id, file_name, file_type, file_size,
               document_type, status, started_at, completed_at,
               confidence_score, validation_errors, error_message,
               extracted_data, created_at, updated_at
        FROM ai_backend.extraction_jobs
        WHERE id = $1 AND coach_id = $2
    """

    row = await db.fetchrow(query, job_id, user.user_id)

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction job not found",
        )

    return ExtractionStatusResponse(
        id=row["id"],
        client_id=row["client_id"],
        coach_id=row["coach_id"],
        file_name=row["file_name"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        document_type=row["document_type"],
        status=ExtractionStatus(row["status"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        confidence_score=float(row["confidence_score"]) if row["confidence_score"] else None,
        validation_errors=row["validation_errors"],
        error_message=row["error_message"],
        extracted_data=row["extracted_data"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("/jobs", response_model=ExtractionListResponse)
async def list_extraction_jobs(
    client_id: Annotated[UUID | None, Query(description="Filter by client ID")] = None,
    status_filter: Annotated[
        ExtractionStatus | None,
        Query(alias="status", description="Filter by status"),
    ] = None,
    document_type: Annotated[
        DocumentType | None,
        Query(description="Filter by document type"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> ExtractionListResponse:
    """
    List extraction jobs for the current coach.

    Supports filtering by client, status, and document type.
    """
    # Build query with optional filters
    conditions = ["coach_id = $1"]
    params: list = [user.user_id]
    param_idx = 2

    if client_id:
        conditions.append(f"client_id = ${param_idx}")
        params.append(client_id)
        param_idx += 1

    if status_filter:
        conditions.append(f"status = ${param_idx}")
        params.append(status_filter.value)
        param_idx += 1

    if document_type:
        conditions.append(f"document_type = ${param_idx}")
        params.append(document_type.value)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    # Get total count
    count_query = f"SELECT COUNT(*) FROM ai_backend.extraction_jobs WHERE {where_clause}"
    total = await db.fetchval(count_query, *params)

    # Get jobs
    query = f"""
        SELECT id, client_id, coach_id, file_name, file_type, file_size,
               document_type, status, created_at
        FROM ai_backend.extraction_jobs
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """
    params.extend([limit, offset])

    rows = await db.fetch(query, *params)

    jobs = [
        ExtractionJobResponse(
            id=row["id"],
            client_id=row["client_id"],
            coach_id=row["coach_id"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            document_type=row["document_type"],
            status=ExtractionStatus(row["status"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]

    return ExtractionListResponse(
        jobs=jobs,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/reprocess/{job_id}", response_model=ExtractionJobResponse)
async def reprocess_extraction(
    job_id: UUID,
    request: ExtractionReprocessRequest | None = None,
    x_extraction_provider: Annotated[
        str | None,
        Header(description="Extraction provider to use (docling, reducto)"),
    ] = None,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> ExtractionJobResponse:
    """
    Retry a failed extraction job.

    By default, only failed jobs can be reprocessed.
    Use force=true to reprocess any job.

    **Headers:**
    - `X-Extraction-Provider`: Optional. Extraction provider to use ("docling", "reducto").
      Useful for retrying with a different provider after failure.
    """
    force = request.force if request else False

    # Validate and normalize provider header
    valid_providers = {"docling", "reducto"}
    provider = None
    if x_extraction_provider:
        normalized = x_extraction_provider.lower().strip()
        if normalized in valid_providers:
            provider = normalized

    # Get job
    query = """
        SELECT id, client_id, coach_id, file_name, file_type, file_size,
               document_type, status, retry_count, max_retries, created_at
        FROM ai_backend.extraction_jobs
        WHERE id = $1 AND coach_id = $2
    """
    row = await db.fetchrow(query, job_id, user.user_id)

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction job not found",
        )

    # Check if reprocessing is allowed
    if row["status"] != "failed" and not force:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is in '{row['status']}' state. Use force=true to reprocess.",
        )

    if row["retry_count"] >= row["max_retries"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job has exceeded maximum retries ({row['max_retries']})",
        )

    # Update job status to pending
    update_query = """
        UPDATE ai_backend.extraction_jobs
        SET status = 'pending',
            error_message = NULL,
            retry_count = retry_count + 1,
            updated_at = NOW()
        WHERE id = $1
        RETURNING status
    """
    await db.execute(update_query, job_id)

    logger.info(
        "Extraction job requeued",
        job_id=str(job_id),
        retry_count=row["retry_count"] + 1,
        provider=provider,
    )

    # Queue Celery task
    from app.workers.tasks.extraction import process_extraction

    process_extraction.delay(str(job_id), provider=provider)

    return ExtractionJobResponse(
        id=row["id"],
        client_id=row["client_id"],
        coach_id=row["coach_id"],
        file_name=row["file_name"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        document_type=row["document_type"],
        status=ExtractionStatus.PENDING,
        created_at=row["created_at"],
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_extraction_job(
    job_id: UUID,
    delete_file: Annotated[bool, Query(description="Also delete the source file")] = False,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> None:
    """
    Delete an extraction job.

    Optionally deletes the source file from storage.
    Cannot delete jobs that are currently processing.
    """
    # Get job
    query = """
        SELECT id, status, file_url
        FROM ai_backend.extraction_jobs
        WHERE id = $1 AND coach_id = $2
    """
    row = await db.fetchrow(query, job_id, user.user_id)

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction job not found",
        )

    if row["status"] == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a job that is currently processing",
        )

    # Delete source file if requested
    if delete_file and row["file_url"]:
        try:
            storage = get_storage_service()
            await storage.delete_file(row["file_url"])
            logger.info("Source file deleted", job_id=str(job_id), file_url=row["file_url"])
        except Exception as e:
            logger.warning("Failed to delete source file", job_id=str(job_id), error=str(e))

    # Delete job (cascade will delete extraction_cache entries)
    delete_query = "DELETE FROM ai_backend.extraction_jobs WHERE id = $1"
    await db.execute(delete_query, job_id)

    logger.info("Extraction job deleted", job_id=str(job_id))
