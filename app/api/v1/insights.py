"""Insights API endpoints for RAG-generated insights (Spec 0004)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.core.auth import UserContext, get_current_user, verify_coach_client_relationship
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.schemas.rag import (
    GeneratedInsight,
    InsightGenerationRequest,
    InsightGenerationResponse,
    PendingInsightsResponse,
)
from app.services.rag.insights import (
    DuplicateInsightError,
    InsightGenerationService,
    RateLimitExceededError,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])


@router.post("/generate", response_model=InsightGenerationResponse)
async def generate_insight(
    request: InsightGenerationRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    http_request: Request,
):
    """
    Generate a proactive insight for a client.

    Analyzes client health data and generates an actionable insight
    that is written to the MVP insights table for coach review.

    Rate limits:
    - Max 3 insights per client per week
    - Max 1 insight per trigger type per day
    - 48-hour cooldown after insight approval/rejection

    Returns status: 'generated', 'duplicate', or raises 429 for rate limit.
    """
    async with get_db_session() as db:
        # Verify coach has access to this client
        await verify_coach_client_relationship(
            db,
            coach_id=user.user_id,
            client_id=str(request.client_id),
            raise_404=True,
        )

        # Set client_id for audit logging
        http_request.state.client_id = str(request.client_id)

        logger.info(
            "Insight generation request",
            coach_id=user.user_id,
            client_id=str(request.client_id),
            trigger=request.trigger.value,
        )

        insight_service = InsightGenerationService(db)

        try:
            result = await insight_service.generate_insight(
                client_id=request.client_id,
                coach_id=user.user_id,
                trigger=request.trigger,
                context=request.context,
            )

            return InsightGenerationResponse(
                insight_id=result.id,
                title=result.title,
                confidence_score=result.confidence_score,
                status="generated",
            )

        except DuplicateInsightError:
            logger.info(
                "Duplicate insight detected",
                client_id=str(request.client_id),
            )
            return InsightGenerationResponse(
                insight_id=None,
                title=None,
                confidence_score=None,
                status="duplicate",
            )

        except RateLimitExceededError as e:
            logger.warning(
                "Insight rate limit exceeded",
                client_id=str(request.client_id),
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(e),
                headers={"Retry-After": "3600"},  # 1 hour
            )


@router.get("/pending", response_model=PendingInsightsResponse)
async def get_pending_insights(
    user: Annotated[UserContext, Depends(get_current_user)],
    limit: int = Query(default=20, ge=1, le=100, description="Maximum insights to return"),
):
    """
    Get pending insights for the coach to review.

    Returns insights that:
    - Belong to this coach
    - Have status 'pending'
    - Have not expired
    """
    async with get_db_session() as db:
        logger.info(
            "Fetching pending insights",
            coach_id=user.user_id,
            limit=limit,
        )

        insight_service = InsightGenerationService(db)
        insights = await insight_service.get_pending_insights(
            coach_id=user.user_id,
            limit=limit,
        )

        return PendingInsightsResponse(
            insights=insights,
            total=len(insights),
        )
