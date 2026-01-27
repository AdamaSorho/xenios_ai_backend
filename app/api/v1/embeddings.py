"""Embeddings API endpoints for RAG system (Spec 0004)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import UserContext, get_current_user, verify_coach_client_relationship
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.schemas.rag import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingSearchRequest,
    EmbeddingSearchResponse,
    EmbeddingUpdateRequest,
    EmbeddingUpdateResult,
)
from app.services.rag.embeddings import EmbeddingService

logger = get_logger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/update", response_model=EmbeddingUpdateResult)
async def update_embeddings(
    request: EmbeddingUpdateRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
):
    """
    Update embeddings for a client's health data.

    Generates or updates vector embeddings for the specified client's data.
    Uses content hashing to skip unchanged content unless force=True.
    """
    async with get_db_session() as db:
        # Verify coach has access to this client
        await verify_coach_client_relationship(
            db,
            coach_id=user.user_id,
            client_id=str(request.client_id),
            raise_404=True,
        )

        logger.info(
            "Updating embeddings",
            coach_id=user.user_id,
            client_id=str(request.client_id),
            source_types=[t.value for t in request.source_types] if request.source_types else None,
            force=request.force,
        )

        service = EmbeddingService(db)
        result = await service.update_client_embeddings(
            client_id=request.client_id,
            source_types=request.source_types,
            force=request.force,
        )

        return result


@router.post("/search", response_model=EmbeddingSearchResponse)
async def search_embeddings(
    request: EmbeddingSearchRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
):
    """
    Search embeddings using semantic similarity.

    Returns relevant content from the client's embedded data based on
    the query text. Results are filtered by similarity threshold (0.7).
    """
    # Import here to avoid circular imports
    from app.services.rag.retrieval import RetrievalService

    async with get_db_session() as db:
        # Verify coach has access to this client
        await verify_coach_client_relationship(
            db,
            coach_id=user.user_id,
            client_id=str(request.client_id),
            raise_404=True,
        )

        logger.info(
            "Searching embeddings",
            coach_id=user.user_id,
            client_id=str(request.client_id),
            query_length=len(request.query),
        )

        service = RetrievalService(db)
        results = await service.retrieve_context(
            client_id=request.client_id,
            query=request.query,
            max_items=request.limit,
            source_types=[t.value for t in request.source_types] if request.source_types else None,
        )

        return EmbeddingSearchResponse(results=results, query=request.query)


@router.post("/batch-update", response_model=BatchEmbeddingResponse)
async def batch_update_embeddings(
    request: BatchEmbeddingRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
):
    """
    Queue batch embedding updates for multiple clients.

    Useful for onboarding or bulk data refresh. Each client is processed
    as a separate background task.
    """
    from app.workers.tasks.rag import update_client_embeddings

    queued = 0

    async with get_db_session() as db:
        for client_id in request.client_ids:
            # Verify coach has access to each client
            try:
                await verify_coach_client_relationship(
                    db,
                    coach_id=user.user_id,
                    client_id=str(client_id),
                    raise_404=True,
                )

                # Queue Celery task
                update_client_embeddings.delay(str(client_id), force=request.force)
                queued += 1

                logger.info(
                    "Queued embedding update",
                    coach_id=user.user_id,
                    client_id=str(client_id),
                )

            except HTTPException:
                # Skip clients the coach doesn't have access to
                logger.warning(
                    "Skipping unauthorized client in batch",
                    coach_id=user.user_id,
                    client_id=str(client_id),
                )
                continue

    return BatchEmbeddingResponse(queued=queued)
