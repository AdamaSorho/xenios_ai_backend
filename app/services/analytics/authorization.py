"""Authorization utilities for analytics API."""

from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger

logger = get_logger(__name__)


async def verify_coach_client_relationship(
    coach_id: UUID,
    client_id: UUID,
    db: AsyncSession,
) -> bool:
    """
    Check if coach has active relationship with client.

    Queries the MVP's coach_clients table in public schema.

    Args:
        coach_id: UUID of the coach (from JWT)
        client_id: UUID of the client
        db: Database session

    Returns:
        True if relationship exists, False otherwise
    """
    result = await db.execute(
        text("""
            SELECT 1 FROM public.coach_clients
            WHERE coach_id = :coach_id
            AND client_id = :client_id
            AND status = 'active'
        """),
        {"coach_id": str(coach_id), "client_id": str(client_id)},
    )
    return result.scalar() is not None


async def require_coach_client_relationship(
    coach_id: UUID,
    client_id: UUID,
    db: AsyncSession,
) -> None:
    """
    Verify coach-client relationship or raise 404.

    Returns 404 (not 403) to prevent enumeration attacks.

    Args:
        coach_id: UUID of the coach (from JWT)
        client_id: UUID of the client
        db: Database session

    Raises:
        HTTPException 404 if relationship doesn't exist
    """
    has_relationship = await verify_coach_client_relationship(coach_id, client_id, db)
    if not has_relationship:
        logger.warning(
            "unauthorized_client_access_attempt",
            coach_id=str(coach_id),
            client_id=str(client_id),
        )
        raise HTTPException(status_code=404, detail="Client not found")


async def verify_job_ownership(
    job_id: UUID,
    coach_id: UUID,
    db: AsyncSession,
) -> bool:
    """
    Check if coach owns the transcription job.

    Args:
        job_id: UUID of the transcription job
        coach_id: UUID of the coach (from JWT)
        db: Database session

    Returns:
        True if coach owns the job, False otherwise
    """
    result = await db.execute(
        text("""
            SELECT 1 FROM ai_backend.transcription_jobs
            WHERE id = :job_id AND coach_id = :coach_id
        """),
        {"job_id": str(job_id), "coach_id": str(coach_id)},
    )
    return result.scalar() is not None


async def require_job_ownership(
    job_id: UUID,
    coach_id: UUID,
    db: AsyncSession,
) -> None:
    """
    Verify job ownership or raise 404.

    Args:
        job_id: UUID of the transcription job
        coach_id: UUID of the coach (from JWT)
        db: Database session

    Raises:
        HTTPException 404 if job not owned by coach
    """
    owns_job = await verify_job_ownership(job_id, coach_id, db)
    if not owns_job:
        logger.warning(
            "unauthorized_job_access_attempt",
            job_id=str(job_id),
            coach_id=str(coach_id),
        )
        raise HTTPException(status_code=404, detail="Session not found")
