"""Celery tasks for audio transcription processing."""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import httpx

from app.config import get_settings
from app.core.logging import get_logger
from app.schemas.transcription import TranscriptionStatus, TranscriptionWebhookPayload
from app.services.transcription.storage import get_transcription_storage_service
from app.services.transcription.deepgram import (
    DeepgramError,
    DeepgramRateLimitError,
    DeepgramTimeoutError,
    get_deepgram_service,
)
from app.services.transcription.diarization import get_diarization_service
from app.services.transcription.summarization import SummarizationError, get_summarization_service
from app.workers.celery_app import celery_app
from app.workers.tasks.base import BaseTask

logger = get_logger(__name__)
settings = get_settings()


def get_sync_db():
    """Get synchronous database connection for Celery tasks."""
    import asyncpg

    async def _get_connection():
        return await asyncpg.connect(settings.database_url)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_get_connection())
    finally:
        loop.close()


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="transcription",
    max_retries=2,
    # Longer time limits to accommodate 2-hour audio + LLM summarization
    # Deepgram processes ~1 min audio in ~3-5 seconds, so 2 hours = ~10 minutes
    # Add buffer for download, diarization, and LLM = ~30 minutes max
    soft_time_limit=7200,  # 2 hours soft limit
    time_limit=7500,  # 2 hours + 5 min hard limit
    autoretry_for=(DeepgramTimeoutError, DeepgramRateLimitError),
    retry_backoff=30,  # 30s, 60s, 120s
    retry_backoff_max=300,
    retry_jitter=True,
)
def process_transcription(self, job_id: str) -> dict[str, Any]:
    """
    Process an audio transcription job.

    Workflow:
    1. Update status to 'transcribing'
    2. Download audio from S3
    3. Send to Deepgram for transcription
    4. Store transcript and utterances
    5. Update status to 'diarizing'
    6. Assign speaker labels (coach/client)
    7. Update status to 'summarizing'
    8. Generate AI summary with LLM
    9. Store summary
    10. Update status to 'completed'
    11. Send webhook notification
    """
    logger.info("Starting transcription job", job_id=job_id)

    conn = get_sync_db()
    temp_file_path: str | None = None

    try:
        # 1. Fetch job from database
        job = run_async(
            conn.fetchrow(
                """
                SELECT id, client_id, coach_id, audio_url, audio_filename,
                       status, webhook_url, webhook_secret
                FROM ai_backend.transcription_jobs
                WHERE id = $1
                """,
                UUID(job_id),
            )
        )

        if not job:
            logger.error("Transcription job not found", job_id=job_id)
            return {"success": False, "error": "Job not found"}

        # Check if we're resuming from a specific state
        current_status = job["status"]

        # Handle resuming from summarizing state (reprocess summary only)
        if current_status == "summarizing":
            return _process_summary_only(self, conn, job)

        # 2. Update status to transcribing
        run_async(_update_status(conn, UUID(job_id), "transcribing", progress=10))

        # 3. Download audio from S3 (transcription storage)
        storage = get_transcription_storage_service()
        temp_file_path = run_async(storage.download_to_temp_file(job["audio_url"]))
        logger.info("Audio downloaded", job_id=job_id, temp_path=temp_file_path)

        run_async(_update_status(conn, UUID(job_id), "transcribing", progress=20))

        # 4. Transcribe with Deepgram
        deepgram = get_deepgram_service()
        result = run_async(deepgram.transcribe_file(temp_file_path))

        logger.info(
            "Transcription completed",
            job_id=job_id,
            duration=result.duration_seconds,
            utterances=len(result.utterances),
            confidence=result.confidence,
        )

        run_async(_update_status(conn, UUID(job_id), "transcribing", progress=50))

        # 5. Store transcript
        transcript_id = run_async(
            _store_transcript(
                conn,
                UUID(job_id),
                result.full_text,
                result.word_count,
                result.duration_seconds,
                result.request_id,
                result.model,
                result.confidence,
            )
        )

        # 6. Store utterances
        utterances = [
            {
                "speaker_number": u.speaker,
                "text": u.text,
                "start_time": u.start,
                "end_time": u.end,
                "confidence": u.confidence,
            }
            for u in result.utterances
        ]
        run_async(_store_utterances(conn, transcript_id, utterances))

        # Update transcription completed timestamp
        run_async(
            conn.execute(
                """
                UPDATE ai_backend.transcription_jobs
                SET transcription_completed_at = NOW()
                WHERE id = $1
                """,
                UUID(job_id),
            )
        )

        run_async(_update_status(conn, UUID(job_id), "diarizing", progress=60))

        # 7. Assign speaker labels
        diarization = get_diarization_service()
        labeled_utterances = diarization.assign_speaker_labels(utterances)
        run_async(_update_utterance_labels(conn, transcript_id, labeled_utterances))

        run_async(_update_status(conn, UUID(job_id), "summarizing", progress=70))

        # 8. Generate summary
        try:
            summarization = get_summarization_service()
            summary_data = run_async(
                summarization.generate_summary(
                    full_text=result.full_text,
                    utterances=labeled_utterances,
                )
            )

            # 9. Store summary
            run_async(_store_summary(conn, UUID(job_id), transcript_id, summary_data))

            # Update summary completed timestamp
            run_async(
                conn.execute(
                    """
                    UPDATE ai_backend.transcription_jobs
                    SET summary_completed_at = NOW()
                    WHERE id = $1
                    """,
                    UUID(job_id),
                )
            )

            # 10. Update status to completed
            run_async(_update_status(conn, UUID(job_id), "completed", progress=100))

            # 11. Trigger session analytics computation (Spec 0005)
            from app.workers.tasks.analytics import compute_session_analytics
            compute_session_analytics.delay(job_id)
            logger.info("Triggered session analytics computation", job_id=job_id)

        except SummarizationError as e:
            # Summary failed but transcript is available
            logger.warning(
                "Summarization failed, marking as partial",
                job_id=job_id,
                error=str(e),
            )
            run_async(
                conn.execute(
                    """
                    UPDATE ai_backend.transcription_jobs
                    SET status = 'partial',
                        progress = 80,
                        error_message = $2,
                        completed_at = NOW()
                    WHERE id = $1
                    """,
                    UUID(job_id),
                    f"Summarization failed: {e}",
                )
            )

            # Trigger analytics even for partial (transcript available)
            from app.workers.tasks.analytics import compute_session_analytics
            compute_session_analytics.delay(job_id)
            logger.info("Triggered session analytics for partial job", job_id=job_id)

        # 12. Send webhook notification
        if job["webhook_url"]:
            run_async(
                _send_webhook(
                    conn=conn,
                    job_id=UUID(job_id),
                    webhook_url=job["webhook_url"],
                    webhook_secret=job["webhook_secret"],
                )
            )

        logger.info("Transcription job completed", job_id=job_id)

        return {
            "success": True,
            "job_id": job_id,
            "transcript_id": str(transcript_id),
            "duration_seconds": result.duration_seconds,
            "word_count": result.word_count,
        }

    except DeepgramError as e:
        logger.error(
            "Deepgram transcription failed",
            job_id=job_id,
            error=str(e),
        )

        run_async(
            conn.execute(
                """
                UPDATE ai_backend.transcription_jobs
                SET status = 'failed',
                    completed_at = NOW(),
                    error_message = $2
                WHERE id = $1
                """,
                UUID(job_id),
                str(e),
            )
        )

        # Send failure webhook
        if job and job["webhook_url"]:
            run_async(
                _send_webhook(
                    conn=conn,
                    job_id=UUID(job_id),
                    webhook_url=job["webhook_url"],
                    webhook_secret=job["webhook_secret"],
                    is_failure=True,
                    error_message=str(e),
                )
            )

        raise

    except Exception as e:
        logger.error(
            "Transcription job failed with exception",
            job_id=job_id,
            error=str(e),
            exc_info=True,
        )

        run_async(
            conn.execute(
                """
                UPDATE ai_backend.transcription_jobs
                SET status = 'failed',
                    completed_at = NOW(),
                    error_message = $2
                WHERE id = $1
                """,
                UUID(job_id),
                str(e),
            )
        )

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


def _process_summary_only(self, conn, job) -> dict[str, Any]:
    """Process only the summarization step (for reprocessing)."""
    job_id = str(job["id"])

    try:
        # Get existing transcript
        transcript = run_async(
            conn.fetchrow(
                "SELECT id, full_text FROM ai_backend.transcripts WHERE job_id = $1",
                job["id"],
            )
        )

        if not transcript:
            raise ValueError("Transcript not found for summary reprocessing")

        # Get utterances with labels
        utterance_rows = run_async(
            conn.fetch(
                """
                SELECT speaker_number, speaker_label, text, start_time, end_time, confidence
                FROM ai_backend.utterances
                WHERE transcript_id = $1
                ORDER BY sequence_number
                """,
                transcript["id"],
            )
        )

        utterances = [dict(u) for u in utterance_rows]

        # Delete existing summary
        run_async(
            conn.execute(
                "DELETE FROM ai_backend.session_summaries WHERE job_id = $1",
                job["id"],
            )
        )

        # Generate new summary
        summarization = get_summarization_service()
        summary_data = run_async(
            summarization.generate_summary(
                full_text=transcript["full_text"],
                utterances=utterances,
            )
        )

        # Store summary
        run_async(_store_summary(conn, job["id"], transcript["id"], summary_data))

        # Update status
        run_async(
            conn.execute(
                """
                UPDATE ai_backend.transcription_jobs
                SET status = 'completed',
                    progress = 100,
                    summary_completed_at = NOW(),
                    completed_at = NOW(),
                    error_message = NULL
                WHERE id = $1
                """,
                job["id"],
            )
        )

        # Send webhook
        if job["webhook_url"]:
            run_async(
                _send_webhook(
                    conn=conn,
                    job_id=job["id"],
                    webhook_url=job["webhook_url"],
                    webhook_secret=job["webhook_secret"],
                )
            )

        logger.info("Summary reprocessing completed", job_id=job_id)

        return {
            "success": True,
            "job_id": job_id,
            "reprocessed": "summary",
        }

    except Exception as e:
        logger.error("Summary reprocessing failed", job_id=job_id, error=str(e))

        run_async(
            conn.execute(
                """
                UPDATE ai_backend.transcription_jobs
                SET status = 'partial',
                    error_message = $2
                WHERE id = $1
                """,
                job["id"],
                f"Summary reprocessing failed: {e}",
            )
        )

        raise


async def _update_status(
    conn,
    job_id: UUID,
    status: str,
    progress: int = 0,
) -> None:
    """Update job status and progress."""
    await conn.execute(
        """
        UPDATE ai_backend.transcription_jobs
        SET status = $2,
            progress = $3,
            started_at = COALESCE(started_at, NOW())
        WHERE id = $1
        """,
        job_id,
        status,
        progress,
    )


async def _store_transcript(
    conn,
    job_id: UUID,
    full_text: str,
    word_count: int,
    duration_seconds: float,
    request_id: str,
    model: str,
    confidence: float,
) -> UUID:
    """Store transcript in database."""
    row = await conn.fetchrow(
        """
        INSERT INTO ai_backend.transcripts (
            job_id, full_text, word_count, duration_seconds,
            deepgram_request_id, model_used, confidence_score
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
        """,
        job_id,
        full_text,
        word_count,
        duration_seconds,
        request_id,
        model,
        confidence,
    )
    return row["id"]


async def _store_utterances(
    conn,
    transcript_id: UUID,
    utterances: list[dict],
) -> None:
    """Store utterances in database (batch insert)."""
    for i, u in enumerate(utterances):
        await conn.execute(
            """
            INSERT INTO ai_backend.utterances (
                transcript_id, speaker_number, text,
                start_time, end_time, confidence, sequence_number
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            transcript_id,
            u["speaker_number"],
            u["text"],
            u["start_time"],
            u["end_time"],
            u.get("confidence"),
            i,
        )


async def _update_utterance_labels(
    conn,
    transcript_id: UUID,
    labeled_utterances: list[dict],
) -> None:
    """Update utterance speaker labels after diarization."""
    for i, u in enumerate(labeled_utterances):
        await conn.execute(
            """
            UPDATE ai_backend.utterances
            SET speaker_label = $3,
                speaker_confidence = $4
            WHERE transcript_id = $1 AND sequence_number = $2
            """,
            transcript_id,
            i,
            u.get("speaker_label"),
            u.get("speaker_confidence"),
        )


async def _store_summary(
    conn,
    job_id: UUID,
    transcript_id: UUID,
    summary_data: dict,
) -> UUID:
    """Store session summary in database."""
    import json

    row = await conn.fetchrow(
        """
        INSERT INTO ai_backend.session_summaries (
            job_id, transcript_id,
            executive_summary, key_topics, client_concerns, coach_recommendations,
            action_items, goals_discussed, coaching_moments,
            session_type_detected, client_sentiment, engagement_score,
            llm_model, prompt_tokens, completion_tokens
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        RETURNING id
        """,
        job_id,
        transcript_id,
        summary_data.get("executive_summary", ""),
        json.dumps(summary_data.get("key_topics", [])),
        json.dumps(summary_data.get("client_concerns", [])),
        json.dumps(summary_data.get("coach_recommendations", [])),
        json.dumps(summary_data.get("action_items", [])),
        json.dumps(summary_data.get("goals_discussed", [])),
        json.dumps(summary_data.get("coaching_moments", [])),
        summary_data.get("session_type_detected"),
        summary_data.get("client_sentiment"),
        summary_data.get("engagement_score"),
        summary_data.get("llm_model"),
        summary_data.get("prompt_tokens"),
        summary_data.get("completion_tokens"),
    )
    return row["id"]


async def _send_webhook(
    conn,
    job_id: UUID,
    webhook_url: str,
    webhook_secret: str | None,
    is_failure: bool = False,
    error_message: str | None = None,
) -> None:
    """Send webhook notification."""
    import hmac
    import hashlib
    import json
    from uuid import uuid4

    # Get job details
    job = await conn.fetchrow(
        """
        SELECT id, client_id, coach_id, status, audio_duration_seconds
        FROM ai_backend.transcription_jobs
        WHERE id = $1
        """,
        job_id,
    )

    # Get transcript details if available
    transcript = await conn.fetchrow(
        "SELECT id, word_count FROM ai_backend.transcripts WHERE job_id = $1",
        job_id,
    )

    # Build payload
    if is_failure:
        payload = {
            "event": "transcription.failed",
            "job_id": str(job_id),
            "client_id": str(job["client_id"]),
            "coach_id": str(job["coach_id"]),
            "status": "failed",
            "error_message": error_message,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        payload = {
            "event": "transcription.completed",
            "job_id": str(job_id),
            "client_id": str(job["client_id"]),
            "coach_id": str(job["coach_id"]),
            "status": job["status"],
            "transcript_id": str(transcript["id"]) if transcript else None,
            "summary_status": "available" if job["status"] == "completed" else "failed",
            "duration_seconds": float(job["audio_duration_seconds"]) if job["audio_duration_seconds"] else None,
            "word_count": transcript["word_count"] if transcript else None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Sign payload
    headers = {
        "Content-Type": "application/json",
        "X-Webhook-ID": str(uuid4()),
        "X-Webhook-Timestamp": str(int(datetime.now(timezone.utc).timestamp())),
    }

    if webhook_secret:
        message = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            webhook_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={signature}"

    # Send webhook
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        # Update webhook sent timestamp
        await conn.execute(
            "UPDATE ai_backend.transcription_jobs SET webhook_sent_at = NOW() WHERE id = $1",
            job_id,
        )

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
