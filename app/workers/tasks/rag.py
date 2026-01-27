"""Celery tasks for RAG system (Spec 0004)."""

import asyncio
from uuid import UUID

from celery import shared_task
from sqlalchemy import text

from app.core.database import get_db_session
from app.core.logging import get_logger
from app.schemas.rag import EmbeddingSourceType, InsightTrigger
from app.services.rag.embeddings import EmbeddingService
from app.services.rag.insights import DuplicateInsightError, InsightGenerationService

logger = get_logger(__name__)


@shared_task(
    bind=True,
    name="app.workers.tasks.rag.update_client_embeddings",
    queue="llm",
    max_retries=2,
    soft_time_limit=300,  # 5 minutes
    time_limit=600,  # 10 minutes hard limit
)
def update_client_embeddings(
    self,
    client_id: str,
    source_types: list[str] | None = None,
    force: bool = False,
) -> dict:
    """
    Update embeddings for a client's health data.

    Args:
        client_id: Client UUID as string
        source_types: Optional list of source types to update
        force: If True, regenerate even if content unchanged

    Returns:
        Dict with updated_count and skipped_count
    """

    async def _run():
        async with get_db_session() as db:
            service = EmbeddingService(db)

            # Convert string source types to enums
            parsed_types = None
            if source_types:
                parsed_types = [EmbeddingSourceType(t) for t in source_types]

            result = await service.update_client_embeddings(
                client_id=UUID(client_id),
                source_types=parsed_types,
                force=force,
            )

            return {"updated_count": result.updated_count, "skipped_count": result.skipped_count}

    try:
        return asyncio.run(_run())
    except Exception as e:
        logger.error(
            "Embedding update task failed",
            client_id=client_id,
            error=str(e),
        )
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2**self.request.retries))


@shared_task(
    bind=True,
    name="app.workers.tasks.rag.generate_client_insight",
    queue="llm",
    max_retries=1,
    soft_time_limit=60,  # 1 minute
    time_limit=120,  # 2 minutes hard limit
)
def generate_client_insight(
    self,
    client_id: str,
    coach_id: str,
    trigger: str,
    context: dict | None = None,
) -> dict:
    """
    Generate an insight for a client.

    Args:
        client_id: Client UUID as string
        coach_id: Coach UUID as string
        trigger: InsightTrigger value
        context: Optional trigger context

    Returns:
        Dict with status and optional insight_id
    """

    async def _run():
        async with get_db_session() as db:
            service = InsightGenerationService(db)

            try:
                result = await service.generate_insight(
                    client_id=UUID(client_id),
                    coach_id=UUID(coach_id),
                    trigger=InsightTrigger(trigger),
                    context=context,
                )
                return {"status": "generated", "insight_id": str(result.id)}

            except DuplicateInsightError:
                return {"status": "duplicate", "insight_id": None}

    try:
        return asyncio.run(_run())
    except Exception as e:
        logger.error(
            "Insight generation task failed",
            client_id=client_id,
            error=str(e),
        )
        # Single retry
        if self.request.retries < 1:
            raise self.retry(exc=e, countdown=30)
        return {"status": "failed", "error": str(e)}


@shared_task(
    name="app.workers.tasks.rag.batch_update_embeddings",
    queue="llm",
)
def batch_update_embeddings() -> dict:
    """
    Nightly batch update for all active clients' embeddings.

    This is a scheduled task that runs daily to keep embeddings fresh.
    """

    async def _run():
        async with get_db_session() as db:
            # Query all active client-coach pairs
            result = await db.execute(
                text("""
                    SELECT DISTINCT client_id FROM public.coach_clients
                    WHERE status = 'active'
                """)
            )
            clients = result.fetchall()

            queued = 0
            for row in clients:
                client_id = str(row.client_id)
                # Queue individual update task
                update_client_embeddings.delay(client_id, force=False)
                queued += 1

            logger.info("Batch embedding update queued", client_count=queued)
            return {"queued": queued}

    return asyncio.run(_run())


@shared_task(
    name="app.workers.tasks.rag.batch_generate_insights",
    queue="llm",
)
def batch_generate_insights() -> dict:
    """
    Scheduled task to evaluate and generate insights for all active clients.

    Runs periodically (e.g., daily) to check for clients who might benefit
    from proactive insights based on their recent data.
    """

    async def _run():
        async with get_db_session() as db:
            # Query all active client-coach pairs
            result = await db.execute(
                text("""
                    SELECT cc.client_id, cc.coach_id
                    FROM public.coach_clients cc
                    WHERE cc.status = 'active'
                """)
            )
            pairs = result.fetchall()

            queued = 0
            for row in pairs:
                # Queue insight generation with "scheduled" trigger
                generate_client_insight.delay(
                    str(row.client_id),
                    str(row.coach_id),
                    InsightTrigger.SCHEDULED.value,
                    None,
                )
                queued += 1

            logger.info("Batch insight generation queued", pair_count=queued)
            return {"queued": queued}

    return asyncio.run(_run())


@shared_task(
    bind=True,
    name="app.workers.tasks.rag.update_embeddings_on_data_change",
    queue="llm",
    max_retries=2,
)
def update_embeddings_on_data_change(
    self,
    client_id: str,
    source_type: str,
) -> dict:
    """
    Update embeddings for a specific source type when data changes.

    Called by MVP webhooks when high-priority data is updated:
    - Lab results (high impact, infrequent)
    - Session summaries (immediate relevance)
    - Health profile changes (affects all context)

    Args:
        client_id: Client UUID as string
        source_type: The EmbeddingSourceType that changed

    Returns:
        Dict with update result
    """

    async def _run():
        async with get_db_session() as db:
            service = EmbeddingService(db)

            result = await service.update_client_embeddings(
                client_id=UUID(client_id),
                source_types=[EmbeddingSourceType(source_type)],
                force=True,  # Force update on data change
            )

            return {"updated_count": result.updated_count, "skipped_count": result.skipped_count}

    try:
        return asyncio.run(_run())
    except Exception as e:
        logger.error(
            "Data change embedding update failed",
            client_id=client_id,
            source_type=source_type,
            error=str(e),
        )
        raise self.retry(exc=e, countdown=30 * (2**self.request.retries))
