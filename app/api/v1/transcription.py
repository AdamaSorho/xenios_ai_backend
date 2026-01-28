"""Transcription API endpoints for audio processing."""

import os
import secrets
import tempfile
from datetime import date
from typing import Annotated
from uuid import UUID

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.config import get_settings
from app.core.auth import UserContext, get_current_user
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.schemas.transcription import (
    ReprocessRequest,
    SessionSummaryResponse,
    SessionType,
    SpeakerLabelUpdate,
    TranscriptionJobResponse,
    TranscriptionListResponse,
    TranscriptionStatus,
    TranscriptionStatusResponse,
    TranscriptResponse,
    UtteranceResponse,
)
from app.services.transcription.audio import get_audio_service
from app.services.transcription.storage import get_transcription_storage_service

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/transcription", tags=["transcription"])


# File limits per spec
MAX_FILE_SIZE = settings.transcription_max_file_size_mb * 1024 * 1024  # 500 MB
MAX_DURATION_SECONDS = settings.transcription_max_duration_minutes * 60  # 2 hours
MIN_DURATION_SECONDS = settings.transcription_min_duration_seconds  # 10 seconds


def generate_webhook_secret() -> str:
    """Generate a secure webhook signing secret."""
    return secrets.token_urlsafe(32)


def validate_webhook_url(url: str) -> tuple[bool, str | None]:
    """
    Validate webhook URL format and security.

    Per spec: HTTPS required except for localhost.
    Does NOT make network requests to avoid SSRF risk.

    Returns:
        Tuple of (is_valid, error_message)
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    # Must have a scheme and netloc
    if not parsed.scheme or not parsed.netloc:
        return False, "Invalid URL format"

    # Allow localhost for development
    hostname = parsed.hostname or ""
    if hostname in ("localhost", "127.0.0.1", "::1"):
        return True, None

    # HTTPS required for non-localhost
    if parsed.scheme != "https":
        return False, "Webhook URL must use HTTPS"

    # Block private IP ranges to prevent SSRF
    # Note: We don't make network requests to validate reachability
    # The webhook delivery will handle unreachable endpoints with retries
    import ipaddress

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
            return False, "Webhook URL cannot point to private IP addresses"
    except ValueError:
        # Not an IP address, it's a hostname - that's fine
        pass

    return True, None


async def get_job_or_404(
    job_id: UUID,
    coach_id: str,
    db,
) -> dict:
    """
    Get job if owned by coach, else raise 404.

    Per spec: Return 404 (not 403) to prevent enumeration.
    """
    query = """
        SELECT * FROM ai_backend.transcription_jobs
        WHERE id = $1 AND coach_id = $2
    """
    row = await db.fetchrow(query, job_id, coach_id)

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription job not found",
        )

    return dict(row)


@router.post("/upload", response_model=TranscriptionJobResponse)
async def upload_audio(
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    client_id: Annotated[UUID, Form(description="Client ID for this session")],
    session_date: Annotated[
        date | None,
        Form(description="Date when the session occurred"),
    ] = None,
    session_type: Annotated[
        SessionType | None,
        Form(description="Type of session (in_person, video_call, phone_call)"),
    ] = None,
    session_title: Annotated[
        str | None,
        Form(max_length=255, description="Optional title for the session"),
    ] = None,
    conversation_id: Annotated[
        UUID | None,
        Form(description="Link to existing conversation in MVP"),
    ] = None,
    webhook_url: Annotated[
        str | None,
        Form(description="Webhook URL to notify on completion"),
    ] = None,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> TranscriptionJobResponse:
    """
    Upload audio for transcription.

    Accepts audio files up to 500MB, 2 hours max duration.
    Supported formats: MP3, M4A, WAV, MP4, AAC, OGG, WEBM.

    Returns job_id immediately. Poll /status/{job_id} for progress.
    """
    audio_service = get_audio_service()

    # Validate file format
    filename = file.filename or "audio"
    valid, error = audio_service.validate_format(filename)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error,
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.transcription_max_file_size_mb}MB",
        )

    # Save to temp file for duration validation
    ext = audio_service.get_format_from_filename(filename)
    fd, temp_path = tempfile.mkstemp(suffix=f".{ext}")
    try:
        os.write(fd, content)
        os.close(fd)

        # Validate audio duration
        valid, error = audio_service.validate_audio(temp_path)
        if not valid:
            if "too large" in error.lower():
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=error,
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error,
            )

        # Get audio info for metadata
        audio_info = audio_service.get_audio_info(temp_path)

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Validate webhook URL if provided
    webhook_secret = None
    if webhook_url:
        is_valid, error_msg = validate_webhook_url(webhook_url)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg or "Invalid webhook URL",
            )
        webhook_secret = generate_webhook_secret()

    # Upload to S3 (transcription-specific storage)
    storage = get_transcription_storage_service()
    content_type = audio_service.get_content_type(filename)
    file_url = await storage.upload_file(
        file_content=content,
        client_id=str(client_id),
        filename=filename,
        content_type=content_type,
    )

    # Create transcription job record
    query = """
        INSERT INTO ai_backend.transcription_jobs (
            client_id, coach_id, conversation_id,
            audio_url, audio_filename, audio_format, audio_size_bytes, audio_duration_seconds,
            status, session_date, session_type, session_title,
            webhook_url, webhook_secret
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        RETURNING id, client_id, coach_id, audio_filename, audio_format,
                  audio_size_bytes, audio_duration_seconds, status,
                  session_date, session_type, session_title, created_at
    """

    row = await db.fetchrow(
        query,
        client_id,
        user.user_id,  # coach_id from authenticated user
        conversation_id,
        file_url,
        filename,
        ext,
        len(content),
        audio_info.duration_seconds,
        "pending",
        session_date,
        session_type.value if session_type else None,
        session_title,
        webhook_url,
        webhook_secret,
    )

    logger.info(
        "Transcription job created",
        job_id=str(row["id"]),
        client_id=str(client_id),
        duration_seconds=audio_info.duration_seconds,
        file_size=len(content),
    )

    # Queue Celery task
    from app.workers.tasks.transcription import process_transcription

    process_transcription.delay(str(row["id"]))

    return TranscriptionJobResponse(
        id=row["id"],
        client_id=row["client_id"],
        coach_id=row["coach_id"],
        audio_filename=row["audio_filename"],
        audio_format=row["audio_format"],
        audio_size_bytes=row["audio_size_bytes"],
        audio_duration_seconds=float(row["audio_duration_seconds"]) if row["audio_duration_seconds"] else None,
        status=TranscriptionStatus(row["status"]),
        session_date=row["session_date"],
        session_type=row["session_type"],
        session_title=row["session_title"],
        webhook_secret=webhook_secret,  # One-time display
        created_at=row["created_at"],
    )


@router.get("/status/{job_id}", response_model=TranscriptionStatusResponse)
async def get_transcription_status(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> TranscriptionStatusResponse:
    """
    Get transcription job status and progress.

    Returns the current status, progress percentage, and timing information.
    """
    row = await get_job_or_404(job_id, user.user_id, db)

    return TranscriptionStatusResponse(
        id=row["id"],
        client_id=row["client_id"],
        coach_id=row["coach_id"],
        status=TranscriptionStatus(row["status"]),
        progress=row["progress"] or 0,
        audio_duration_seconds=float(row["audio_duration_seconds"]) if row["audio_duration_seconds"] else None,
        started_at=row["started_at"],
        transcription_completed_at=row["transcription_completed_at"],
        summary_completed_at=row["summary_completed_at"],
        completed_at=row["completed_at"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("/{job_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> TranscriptResponse:
    """
    Get full transcript with utterances.

    Returns the complete transcript text and individual speaker utterances.
    Available once status is 'diarizing', 'summarizing', 'completed', or 'partial'.
    """
    job = await get_job_or_404(job_id, user.user_id, db)

    # Check if transcript is available
    if job["status"] in ("pending", "uploading", "transcribing"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript not yet available. Current status: {job['status']}",
        )

    # Get transcript
    transcript_query = """
        SELECT id, job_id, full_text, word_count, duration_seconds,
               confidence_score, deepgram_request_id, model_used, created_at
        FROM ai_backend.transcripts
        WHERE job_id = $1
    """
    transcript = await db.fetchrow(transcript_query, job_id)

    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found",
        )

    # Get utterances
    utterances_query = """
        SELECT id, speaker_number, speaker_label, speaker_confidence,
               text, start_time, end_time, confidence, intent, sentiment, sequence_number
        FROM ai_backend.utterances
        WHERE transcript_id = $1
        ORDER BY sequence_number
    """
    utterance_rows = await db.fetch(utterances_query, transcript["id"])

    utterances = [
        UtteranceResponse(
            id=u["id"],
            speaker_number=u["speaker_number"],
            speaker_label=u["speaker_label"],
            speaker_confidence=float(u["speaker_confidence"]) if u["speaker_confidence"] else None,
            text=u["text"],
            start_time=float(u["start_time"]),
            end_time=float(u["end_time"]),
            confidence=float(u["confidence"]) if u["confidence"] else None,
            intent=u["intent"],
            sentiment=u["sentiment"],
            sequence_number=u["sequence_number"],
        )
        for u in utterance_rows
    ]

    return TranscriptResponse(
        id=transcript["id"],
        job_id=transcript["job_id"],
        full_text=transcript["full_text"],
        word_count=transcript["word_count"],
        duration_seconds=float(transcript["duration_seconds"]),
        confidence_score=float(transcript["confidence_score"]) if transcript["confidence_score"] else None,
        deepgram_request_id=transcript["deepgram_request_id"],
        model_used=transcript["model_used"],
        utterances=utterances,
        created_at=transcript["created_at"],
    )


@router.get("/{job_id}/summary", response_model=SessionSummaryResponse)
async def get_summary(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> SessionSummaryResponse:
    """
    Get AI-generated session summary.

    Returns executive summary, key topics, action items, and coaching moments.
    Available once status is 'completed'. Returns 404 if summarization failed.
    """
    job = await get_job_or_404(job_id, user.user_id, db)

    # Check status
    if job["status"] == "partial":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Summary not available - summarization failed. Transcript is still accessible.",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Summary not yet available. Current status: {job['status']}",
        )

    # Get summary
    query = """
        SELECT id, job_id, transcript_id,
               executive_summary, key_topics, client_concerns, coach_recommendations,
               action_items, goals_discussed, coaching_moments,
               session_type_detected, client_sentiment, engagement_score,
               llm_model, prompt_tokens, completion_tokens, created_at
        FROM ai_backend.session_summaries
        WHERE job_id = $1
    """
    summary = await db.fetchrow(query, job_id)

    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Summary not found",
        )

    return SessionSummaryResponse(
        id=summary["id"],
        job_id=summary["job_id"],
        transcript_id=summary["transcript_id"],
        executive_summary=summary["executive_summary"],
        key_topics=summary["key_topics"] or [],
        client_concerns=summary["client_concerns"] or [],
        coach_recommendations=summary["coach_recommendations"] or [],
        action_items=summary["action_items"] or [],
        goals_discussed=summary["goals_discussed"] or [],
        coaching_moments=summary["coaching_moments"] or [],
        session_type_detected=summary["session_type_detected"],
        client_sentiment=summary["client_sentiment"],
        engagement_score=float(summary["engagement_score"]) if summary["engagement_score"] else None,
        llm_model=summary["llm_model"],
        prompt_tokens=summary["prompt_tokens"],
        completion_tokens=summary["completion_tokens"],
        created_at=summary["created_at"],
    )


@router.get("/sessions", response_model=TranscriptionListResponse)
async def list_sessions(
    client_id: Annotated[UUID | None, Query(description="Filter by client ID")] = None,
    status_filter: Annotated[
        TranscriptionStatus | None,
        Query(alias="status", description="Filter by status"),
    ] = None,
    from_date: Annotated[date | None, Query(description="Filter sessions from this date")] = None,
    to_date: Annotated[date | None, Query(description="Filter sessions to this date")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> TranscriptionListResponse:
    """
    List transcription jobs for the current coach.

    Supports filtering by client, status, and date range.
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

    if from_date:
        conditions.append(f"(session_date >= ${param_idx} OR created_at >= ${param_idx})")
        params.append(from_date)
        param_idx += 1

    if to_date:
        conditions.append(f"(session_date <= ${param_idx} OR created_at <= ${param_idx})")
        params.append(to_date)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    # Get total count
    count_query = f"SELECT COUNT(*) FROM ai_backend.transcription_jobs WHERE {where_clause}"
    total = await db.fetchval(count_query, *params)

    # Get jobs
    query = f"""
        SELECT id, client_id, coach_id, audio_filename, audio_format,
               audio_size_bytes, audio_duration_seconds, status,
               session_date, session_type, session_title, created_at
        FROM ai_backend.transcription_jobs
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """
    params.extend([limit, offset])

    rows = await db.fetch(query, *params)

    jobs = [
        TranscriptionJobResponse(
            id=row["id"],
            client_id=row["client_id"],
            coach_id=row["coach_id"],
            audio_filename=row["audio_filename"],
            audio_format=row["audio_format"],
            audio_size_bytes=row["audio_size_bytes"],
            audio_duration_seconds=float(row["audio_duration_seconds"]) if row["audio_duration_seconds"] else None,
            status=TranscriptionStatus(row["status"]),
            session_date=row["session_date"],
            session_type=row["session_type"],
            session_title=row["session_title"],
            created_at=row["created_at"],
        )
        for row in rows
    ]

    return TranscriptionListResponse(
        jobs=jobs,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/{job_id}/reprocess", response_model=TranscriptionJobResponse)
async def reprocess_transcription(
    job_id: UUID,
    request: ReprocessRequest | None = None,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> TranscriptionJobResponse:
    """
    Retry failed transcription or regenerate summary.

    Options:
    - reprocess_transcript: Re-run Deepgram transcription
    - reprocess_summary: Re-run LLM summarization (default)
    """
    job = await get_job_or_404(job_id, user.user_id, db)

    reprocess_transcript = request.reprocess_transcript if request else False
    reprocess_summary = request.reprocess_summary if request else True

    # Determine new status based on what to reprocess
    if reprocess_transcript:
        # Full reprocess from transcription
        if job["status"] not in ("failed", "partial", "completed"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot reprocess transcript in '{job['status']}' state",
            )
        new_status = "pending"
    elif reprocess_summary:
        # Only reprocess summary
        if job["status"] not in ("partial", "completed"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Summary reprocess requires transcript to be available (partial or completed)",
            )
        new_status = "summarizing"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify reprocess_transcript or reprocess_summary",
        )

    # Update job status
    update_query = """
        UPDATE ai_backend.transcription_jobs
        SET status = $2,
            error_message = NULL,
            retry_count = retry_count + 1,
            updated_at = NOW()
        WHERE id = $1
        RETURNING id, client_id, coach_id, audio_filename, audio_format,
                  audio_size_bytes, audio_duration_seconds, status,
                  session_date, session_type, session_title, created_at
    """
    row = await db.fetchrow(update_query, job_id, new_status)

    logger.info(
        "Transcription job requeued",
        job_id=str(job_id),
        new_status=new_status,
        retry_count=job["retry_count"] + 1,
    )

    # Queue Celery task
    from app.workers.tasks.transcription import process_transcription

    process_transcription.delay(str(job_id))

    return TranscriptionJobResponse(
        id=row["id"],
        client_id=row["client_id"],
        coach_id=row["coach_id"],
        audio_filename=row["audio_filename"],
        audio_format=row["audio_format"],
        audio_size_bytes=row["audio_size_bytes"],
        audio_duration_seconds=float(row["audio_duration_seconds"]) if row["audio_duration_seconds"] else None,
        status=TranscriptionStatus(row["status"]),
        session_date=row["session_date"],
        session_type=row["session_type"],
        session_title=row["session_title"],
        created_at=row["created_at"],
    )


@router.patch("/{job_id}/speakers")
async def update_speaker_labels(
    job_id: UUID,
    speaker_updates: list[SpeakerLabelUpdate],
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> dict:
    """
    Manually correct speaker labels.

    Allows coaches to override automatic speaker identification.
    """
    job = await get_job_or_404(job_id, user.user_id, db)

    # Get transcript
    transcript_query = "SELECT id FROM ai_backend.transcripts WHERE job_id = $1"
    transcript = await db.fetchrow(transcript_query, job_id)

    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found",
        )

    # Update utterances for each speaker
    updated_count = 0
    for update in speaker_updates:
        result = await db.execute(
            """
            UPDATE ai_backend.utterances
            SET speaker_label = $3,
                speaker_confidence = 1.0
            WHERE transcript_id = $1 AND speaker_number = $2
            """,
            transcript["id"],
            update.speaker_number,
            update.label.value,
        )
        # Count affected rows
        if result:
            updated_count += int(result.split()[-1])

    logger.info(
        "Speaker labels updated",
        job_id=str(job_id),
        updates=len(speaker_updates),
        affected_utterances=updated_count,
    )

    return {"updated_utterances": updated_count}


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transcription(
    job_id: UUID,
    delete_audio: Annotated[bool, Query(description="Also delete the audio file")] = True,
    user: UserContext = Depends(get_current_user),
    db=Depends(get_db_session),
) -> None:
    """
    Delete transcription job and all associated data.

    By default, also deletes the audio file from storage.
    Cannot delete jobs that are currently processing.
    """
    job = await get_job_or_404(job_id, user.user_id, db)

    if job["status"] in ("transcribing", "diarizing", "summarizing"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete a job that is currently {job['status']}",
        )

    # Delete audio file if requested
    if delete_audio and job["audio_url"]:
        try:
            storage = get_transcription_storage_service()
            await storage.delete_file(job["audio_url"])
            logger.info("Audio file deleted", job_id=str(job_id), audio_url=job["audio_url"])
        except Exception as e:
            logger.warning("Failed to delete audio file", job_id=str(job_id), error=str(e))

    # Delete job (CASCADE handles transcript, utterances, summary)
    delete_query = "DELETE FROM ai_backend.transcription_jobs WHERE id = $1"
    await db.execute(delete_query, job_id)

    logger.info("Transcription job deleted", job_id=str(job_id))
