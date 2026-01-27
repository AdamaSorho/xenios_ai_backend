"""Celery tasks for document extraction."""

import os
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import httpx

from app.config import get_settings
from app.core.logging import get_logger
from app.schemas.extraction import ExtractionStatus, ExtractionWebhookPayload
from app.services.extraction.base import ExtractionResult
from app.services.extraction.router import get_document_router
from app.services.extraction.storage import get_storage_service
from app.workers.celery_app import celery_app
from app.workers.tasks.base import BaseTask

logger = get_logger(__name__)
settings = get_settings()

# Track if extractors have been registered
_extractors_registered = False


def ensure_extractors_registered() -> None:
    """Ensure extractors are registered with the document router."""
    global _extractors_registered
    if not _extractors_registered:
        from app.services.extraction import register_extractors

        register_extractors()
        _extractors_registered = True
        logger.info("Extractors registered successfully")


def get_sync_db():
    """Get synchronous database connection for Celery tasks."""
    import asyncpg
    import asyncio

    async def _get_connection():
        return await asyncpg.connect(settings.database_url)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_get_connection())
    finally:
        loop.close()


def run_async(coro):
    """Run async function in sync context."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="extraction",
    max_retries=3,
    soft_time_limit=300,  # 5 minutes
    time_limit=600,  # 10 minutes hard limit
)
def process_extraction(self, job_id: str) -> dict[str, Any]:
    """
    Process a document extraction job.

    Workflow:
    1. Fetch job from database
    2. Download file from S3
    3. Detect/confirm document type
    4. Route to appropriate extractor
    5. Run extraction
    6. Validate results
    7. Store results
    8. Update job status
    9. Send webhook notification (optional)
    """
    # Ensure extractors are registered
    ensure_extractors_registered()

    logger.info("Starting extraction job", job_id=job_id)

    conn = get_sync_db()
    temp_file_path: str | None = None

    try:
        # 1. Fetch job from database
        job = run_async(
            conn.fetchrow(
                """
                SELECT id, client_id, coach_id, file_url, file_name, file_type,
                       document_type, webhook_url
                FROM ai_backend.extraction_jobs
                WHERE id = $1
                """,
                UUID(job_id),
            )
        )

        if not job:
            logger.error("Extraction job not found", job_id=job_id)
            return {"success": False, "error": "Job not found"}

        # Update status to processing
        run_async(
            conn.execute(
                """
                UPDATE ai_backend.extraction_jobs
                SET status = 'processing', started_at = NOW()
                WHERE id = $1
                """,
                UUID(job_id),
            )
        )

        # 2. Download file from S3
        storage = get_storage_service()
        temp_file_path = run_async(storage.download_to_temp_file(job["file_url"]))
        logger.info("File downloaded", job_id=job_id, temp_path=temp_file_path)

        # 3. Detect/confirm document type
        doc_router = get_document_router()
        document_type = job["document_type"]

        if not document_type:
            with open(temp_file_path, "rb") as f:
                content = f.read()
            document_type = doc_router.detect_document_type(content, job["file_name"])
            logger.info("Document type detected", job_id=job_id, document_type=document_type)

        if not document_type:
            raise ValueError("Could not detect document type")

        # 4. Route to appropriate extractor
        extractor = doc_router.get_extractor(document_type)

        if not extractor:
            raise ValueError(f"No extractor registered for document type: {document_type}")

        # 5. Run extraction
        logger.info("Running extraction", job_id=job_id, document_type=document_type)
        result: ExtractionResult = run_async(extractor.extract(temp_file_path))

        # 6. Store results
        if result.success:
            run_async(
                conn.execute(
                    """
                    UPDATE ai_backend.extraction_jobs
                    SET status = 'completed',
                        completed_at = NOW(),
                        document_type = $2,
                        extracted_data = $3,
                        confidence_score = $4,
                        validation_errors = $5,
                        error_message = NULL
                    WHERE id = $1
                    """,
                    UUID(job_id),
                    document_type,
                    result.data,
                    result.confidence,
                    result.errors if result.errors else None,
                )
            )

            # Store in extraction cache for quick lookups
            if result.data:
                run_async(_store_extraction_cache(conn, UUID(job_id), document_type, result.data))

            logger.info(
                "Extraction completed successfully",
                job_id=job_id,
                confidence=result.confidence,
            )
        else:
            run_async(
                conn.execute(
                    """
                    UPDATE ai_backend.extraction_jobs
                    SET status = 'failed',
                        completed_at = NOW(),
                        document_type = $2,
                        validation_errors = $3,
                        error_message = $4
                    WHERE id = $1
                    """,
                    UUID(job_id),
                    document_type,
                    result.errors,
                    "; ".join(result.errors) if result.errors else "Extraction failed",
                )
            )

            logger.warning(
                "Extraction failed",
                job_id=job_id,
                errors=result.errors,
            )

        # 7. Send webhook notification if configured
        webhook_url = job["webhook_url"]
        if webhook_url:
            run_async(
                _send_webhook(
                    webhook_url=webhook_url,
                    job_id=UUID(job_id),
                    client_id=job["client_id"],
                    coach_id=job["coach_id"],
                    document_type=document_type,
                    status=ExtractionStatus.COMPLETED if result.success else ExtractionStatus.FAILED,
                    confidence_score=result.confidence if result.success else None,
                    extracted_data=result.data if result.success else None,
                    error_message="; ".join(result.errors) if result.errors else None,
                )
            )

            # Update webhook sent timestamp
            run_async(
                conn.execute(
                    "UPDATE ai_backend.extraction_jobs SET webhook_sent_at = NOW() WHERE id = $1",
                    UUID(job_id),
                )
            )

        return {
            "success": result.success,
            "job_id": job_id,
            "document_type": document_type,
            "confidence": result.confidence,
            "errors": result.errors,
        }

    except Exception as e:
        logger.error(
            "Extraction job failed with exception",
            job_id=job_id,
            error=str(e),
            exc_info=True,
        )

        # Update job status to failed
        run_async(
            conn.execute(
                """
                UPDATE ai_backend.extraction_jobs
                SET status = 'failed',
                    completed_at = NOW(),
                    error_message = $2
                WHERE id = $1
                """,
                UUID(job_id),
                str(e),
            )
        )

        # Re-raise for Celery retry logic
        raise

    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file", path=temp_file_path, error=str(e))

        # Close database connection
        run_async(conn.close())


async def _store_extraction_cache(
    conn,
    job_id: UUID,
    document_type: str,
    extracted_data: dict,
) -> None:
    """Store extracted data in cache table for quick lookups."""
    from datetime import date

    # Try to extract date from data
    extraction_date = date.today()

    if "scan_date" in extracted_data and extracted_data["scan_date"]:
        extraction_date = extracted_data["scan_date"]
    elif "collection_date" in extracted_data and extracted_data["collection_date"]:
        extraction_date = extracted_data["collection_date"]
    elif "date_range_start" in extracted_data and extracted_data["date_range_start"]:
        extraction_date = extracted_data["date_range_start"]

    # Build metrics summary based on document type
    if document_type == "inbody":
        metrics = {
            "weight_kg": extracted_data.get("weight_kg"),
            "body_fat_percent": extracted_data.get("body_fat_percent"),
            "skeletal_muscle_mass_kg": extracted_data.get("skeletal_muscle_mass_kg"),
            "basal_metabolic_rate_kcal": extracted_data.get("basal_metabolic_rate_kcal"),
        }
    elif document_type == "lab_results":
        # Summarize key biomarkers
        biomarkers = extracted_data.get("biomarkers", [])
        metrics = {b["name"]: b["value"] for b in biomarkers[:20]}  # Limit to top 20
    else:
        # For wearables, store summary stats
        metrics = {
            "source": extracted_data.get("source"),
            "total_days": extracted_data.get("total_days"),
            "date_range_start": str(extracted_data.get("date_range_start")),
            "date_range_end": str(extracted_data.get("date_range_end")),
        }

    await conn.execute(
        """
        INSERT INTO ai_backend.extraction_cache (job_id, document_type, extraction_date, metrics)
        VALUES ($1, $2, $3, $4)
        """,
        job_id,
        document_type,
        extraction_date,
        metrics,
    )


async def _send_webhook(
    webhook_url: str,
    job_id: UUID,
    client_id: UUID,
    coach_id: UUID,
    document_type: str | None,
    status: ExtractionStatus,
    confidence_score: float | None,
    extracted_data: dict | None,
    error_message: str | None,
) -> None:
    """Send webhook notification on extraction completion."""
    payload = ExtractionWebhookPayload(
        event="extraction.completed",
        job_id=job_id,
        client_id=client_id,
        coach_id=coach_id,
        document_type=document_type,
        status=status,
        confidence_score=confidence_score,
        extracted_data=extracted_data,
        error_message=error_message,
        completed_at=datetime.now(timezone.utc),
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                webhook_url,
                json=payload.model_dump(mode="json"),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        logger.info(
            "Webhook notification sent",
            job_id=str(job_id),
            webhook_url=webhook_url,
            status_code=response.status_code,
        )
    except Exception as e:
        logger.warning(
            "Failed to send webhook notification",
            job_id=str(job_id),
            webhook_url=webhook_url,
            error=str(e),
        )
