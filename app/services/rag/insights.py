"""Insight generation service for proactive client insights."""

import json
import time
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.logging import get_logger
from app.models.rag import InsightGenerationLog
from app.schemas.rag import GeneratedInsight, InsightTrigger
from app.services.llm.client import LLMClient
from app.services.rag.openai_client import OpenAIEmbeddingClient
from app.services.rag.prompts import INSIGHT_GENERATION_PROMPT
from app.services.rag.retrieval import RetrievalService

logger = get_logger(__name__)


class DuplicateInsightError(Exception):
    """Raised when a similar insight already exists."""

    pass


class RateLimitExceededError(Exception):
    """Raised when insight generation rate limit is exceeded."""

    pass


class InsightGenerationService:
    """
    Service for generating proactive insights from client health data.

    Includes deduplication via title embedding similarity and rate limiting.
    """

    # Rate limit constants
    MAX_INSIGHTS_PER_WEEK = 3
    MAX_PER_TRIGGER_TYPE_PER_DAY = 1
    COOLDOWN_HOURS = 48
    DUPLICATE_SIMILARITY_THRESHOLD = 0.85

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.retrieval_service = RetrievalService(db)
        self.openai_client = OpenAIEmbeddingClient()
        self.llm_client = LLMClient()
        self.settings = get_settings()

    async def generate_insight(
        self,
        client_id: UUID,
        coach_id: UUID,
        trigger: InsightTrigger,
        context: dict | None = None,
    ) -> GeneratedInsight:
        """
        Generate an insight for a client and write to MVP insights table.

        Args:
            client_id: Client to generate insight for
            coach_id: Coach who will review the insight
            trigger: What triggered the generation
            context: Additional context (e.g., metric changes)

        Returns:
            GeneratedInsight with the created insight details

        Raises:
            DuplicateInsightError: If similar insight already exists
            RateLimitExceededError: If rate limits are exceeded
        """
        start_time = time.time()

        logger.info(
            "Generating insight",
            client_id=str(client_id),
            coach_id=str(coach_id),
            trigger=trigger.value,
        )

        # Check rate limits
        await self._check_rate_limits(client_id, trigger)

        # Gather client context
        client_context = await self._gather_client_context(client_id)

        # Identify recent changes
        changes = await self._identify_changes(client_id, trigger, context)

        # Generate insight via LLM
        prompt = INSIGHT_GENERATION_PROMPT.format(
            context=client_context,
            changes=json.dumps(changes, default=str),
        )

        result = await self.llm_client.complete(
            task="insight_generation",
            messages=[{"role": "user", "content": prompt}],
        )

        response_content = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)

        # Parse the JSON response
        insight_data = self._parse_insight_response(response_content)

        # Generate title embedding for deduplication
        title_embedding = await self.openai_client.generate_embedding(insight_data["title"])

        # Check for duplicates
        if await self._is_duplicate_insight(
            client_id, insight_data["insight_type"], title_embedding
        ):
            # Log as duplicate
            await self._log_generation(
                client_id=client_id,
                coach_id=coach_id,
                trigger=trigger,
                triggering_data=context or {},
                status="duplicate",
                insight_type=insight_data["insight_type"],
                title=insight_data["title"],
                title_embedding=title_embedding,
                tokens_used=tokens_used,
            )
            raise DuplicateInsightError("Similar insight already pending for this client")

        # Write to MVP insights table
        insight_id = await self._write_to_mvp_insights(
            client_id=client_id,
            coach_id=coach_id,
            insight_data=insight_data,
            triggering_data={
                "trigger": trigger.value,
                "changes": changes,
                "context_summary": client_context[:500] if client_context else "",
            },
        )

        generation_time_ms = int((time.time() - start_time) * 1000)

        # Log successful generation
        await self._log_generation(
            client_id=client_id,
            coach_id=coach_id,
            trigger=trigger,
            triggering_data=context or {},
            status="generated",
            insight_id=insight_id,
            insight_type=insight_data["insight_type"],
            title=insight_data["title"],
            title_embedding=title_embedding,
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
        )

        logger.info(
            "Insight generated successfully",
            client_id=str(client_id),
            insight_id=str(insight_id),
            insight_type=insight_data["insight_type"],
            generation_time_ms=generation_time_ms,
        )

        return GeneratedInsight(
            id=insight_id,
            client_id=client_id,
            coach_id=coach_id,
            title=insight_data["title"],
            client_message=insight_data["client_message"],
            rationale=insight_data["rationale"],
            suggested_actions=insight_data["suggested_actions"],
            confidence_score=insight_data["confidence_score"],
            triggering_data={
                "trigger": trigger.value,
                "changes": changes,
            },
            insight_type=insight_data["insight_type"],
            expires_at=datetime.utcnow() + timedelta(days=7),
        )

    async def _check_rate_limits(self, client_id: UUID, trigger: InsightTrigger) -> None:
        """Check all rate limit rules from spec."""
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        day_ago = now - timedelta(days=1)

        # Rule 1: Max 3 insights per client per week
        result = await self.db.execute(
            select(InsightGenerationLog).where(
                and_(
                    InsightGenerationLog.client_id == client_id,
                    InsightGenerationLog.status == "generated",
                    InsightGenerationLog.created_at >= week_ago,
                )
            )
        )
        weekly_count = len(result.scalars().all())

        if weekly_count >= self.MAX_INSIGHTS_PER_WEEK:
            logger.warning(
                "Rate limit exceeded: weekly limit",
                client_id=str(client_id),
                count=weekly_count,
            )
            raise RateLimitExceededError(
                f"Max {self.MAX_INSIGHTS_PER_WEEK} insights per client per week exceeded"
            )

        # Rule 2: Max 1 insight per trigger type per day
        result = await self.db.execute(
            select(InsightGenerationLog).where(
                and_(
                    InsightGenerationLog.client_id == client_id,
                    InsightGenerationLog.trigger == trigger.value,
                    InsightGenerationLog.status == "generated",
                    InsightGenerationLog.created_at >= day_ago,
                )
            )
        )
        daily_trigger_count = result.scalars().first()

        if daily_trigger_count:
            logger.warning(
                "Rate limit exceeded: daily trigger limit",
                client_id=str(client_id),
                trigger=trigger.value,
            )
            raise RateLimitExceededError(
                f"Max {self.MAX_PER_TRIGGER_TYPE_PER_DAY} {trigger.value} insight per day exceeded"
            )

        # Rule 3: 48-hour cooldown after approval/rejection of same type
        # Check MVP insights table
        cooldown_ago = now - timedelta(hours=self.COOLDOWN_HOURS)

        cooldown_query = text("""
            SELECT 1 FROM public.insights
            WHERE client_id = :client_id
            AND status IN ('approved', 'rejected')
            AND updated_at >= :cooldown_ago
            LIMIT 1
        """)

        cooldown_result = await self.db.execute(
            cooldown_query,
            {"client_id": str(client_id), "cooldown_ago": cooldown_ago},
        )

        if cooldown_result.scalar_one_or_none():
            logger.warning(
                "Rate limit exceeded: cooldown period",
                client_id=str(client_id),
            )
            raise RateLimitExceededError(
                f"{self.COOLDOWN_HOURS}-hour cooldown after insight approval/rejection"
            )

    async def _is_duplicate_insight(
        self,
        client_id: UUID,
        insight_type: str,
        title_embedding: list[float],
    ) -> bool:
        """
        Check for duplicate insights using type and title similarity.

        Deduplication criteria:
        - Same insight_type within 7-day window
        - Title embedding similarity > 0.85
        """
        week_ago = datetime.utcnow() - timedelta(days=7)
        embedding_str = "[" + ",".join(str(x) for x in title_embedding) + "]"

        query = text("""
            SELECT id, 1 - (title_embedding <=> :embedding::vector) as similarity
            FROM ai_backend.insight_generation_log
            WHERE client_id = :client_id
            AND insight_type = :insight_type
            AND status = 'generated'
            AND created_at >= :week_ago
            AND title_embedding IS NOT NULL
            AND 1 - (title_embedding <=> :embedding::vector) > :threshold
            LIMIT 1
        """)

        result = await self.db.execute(
            query,
            {
                "client_id": str(client_id),
                "insight_type": insight_type,
                "embedding": embedding_str,
                "week_ago": week_ago,
                "threshold": self.DUPLICATE_SIMILARITY_THRESHOLD,
            },
        )

        return result.fetchone() is not None

    async def _write_to_mvp_insights(
        self,
        client_id: UUID,
        coach_id: UUID,
        insight_data: dict,
        triggering_data: dict,
    ) -> UUID:
        """Write insight to MVP's public.insights table."""
        query = text("""
            INSERT INTO public.insights (
                coach_id, client_id, title, client_message, rationale,
                suggested_actions, confidence_score, triggering_data,
                insight_type, status, expires_at
            ) VALUES (
                :coach_id, :client_id, :title, :client_message, :rationale,
                :suggested_actions::jsonb, :confidence_score, :triggering_data::jsonb,
                :insight_type, 'pending', NOW() + INTERVAL '7 days'
            ) RETURNING id
        """)

        result = await self.db.execute(
            query,
            {
                "coach_id": str(coach_id),
                "client_id": str(client_id),
                "title": insight_data["title"],
                "client_message": insight_data["client_message"],
                "rationale": insight_data["rationale"],
                "suggested_actions": json.dumps(insight_data["suggested_actions"]),
                "confidence_score": insight_data["confidence_score"],
                "triggering_data": json.dumps(triggering_data, default=str),
                "insight_type": insight_data["insight_type"],
            },
        )

        row = result.fetchone()
        return row[0]

    def _parse_insight_response(self, content: str) -> dict:
        """Parse LLM JSON response."""
        # Extract JSON from response (may be wrapped in markdown)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            data = json.loads(content.strip())

            # Validate required fields
            required = ["title", "client_message", "rationale", "suggested_actions", "insight_type"]
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Set default confidence if not provided
            if "confidence_score" not in data:
                data["confidence_score"] = 0.7

            # Ensure confidence is within range
            data["confidence_score"] = max(0.0, min(1.0, float(data["confidence_score"])))

            return data

        except json.JSONDecodeError as e:
            logger.error("Failed to parse insight JSON", error=str(e))
            raise ValueError(f"Invalid insight JSON: {e}")

    async def _gather_client_context(self, client_id: UUID) -> str:
        """Gather relevant context for insight generation."""
        contexts = await self.retrieval_service.retrieve_context(
            client_id=client_id,
            query="recent health progress and changes goals metrics",
            max_items=5,
        )
        return "\n".join([c.content for c in contexts])

    async def _identify_changes(
        self,
        client_id: UUID,
        trigger: InsightTrigger,
        context: dict | None,
    ) -> dict:
        """Identify relevant changes based on trigger."""
        if context:
            return context

        # Default: return basic trigger info
        return {"trigger": trigger.value, "timestamp": datetime.utcnow().isoformat()}

    async def _log_generation(self, **kwargs) -> None:
        """Log insight generation attempt."""
        log_entry = InsightGenerationLog(
            client_id=kwargs["client_id"],
            coach_id=kwargs["coach_id"],
            trigger=kwargs["trigger"].value if hasattr(kwargs["trigger"], "value") else kwargs["trigger"],
            triggering_data=kwargs.get("triggering_data", {}),
            insight_id=kwargs.get("insight_id"),
            insight_type=kwargs.get("insight_type"),
            title=kwargs.get("title"),
            title_embedding=kwargs.get("title_embedding"),
            status=kwargs["status"],
            context_items_used=kwargs.get("context_items_used"),
            tokens_used=kwargs.get("tokens_used"),
            generation_time_ms=kwargs.get("generation_time_ms"),
        )
        self.db.add(log_entry)
        await self.db.flush()

    async def get_pending_insights(self, coach_id: UUID, limit: int = 20) -> list[GeneratedInsight]:
        """Get pending insights for a coach to review."""
        query = text("""
            SELECT id, client_id, coach_id, title, client_message, rationale,
                   suggested_actions, confidence_score, triggering_data,
                   insight_type, expires_at, created_at
            FROM public.insights
            WHERE coach_id = :coach_id
            AND status = 'pending'
            AND expires_at > NOW()
            ORDER BY created_at DESC
            LIMIT :limit
        """)

        result = await self.db.execute(query, {"coach_id": str(coach_id), "limit": limit})
        rows = result.fetchall()

        return [
            GeneratedInsight(
                id=row.id,
                client_id=row.client_id,
                coach_id=row.coach_id,
                title=row.title,
                client_message=row.client_message,
                rationale=row.rationale,
                suggested_actions=row.suggested_actions if isinstance(row.suggested_actions, list) else json.loads(row.suggested_actions or "[]"),
                confidence_score=row.confidence_score,
                triggering_data=row.triggering_data if isinstance(row.triggering_data, dict) else json.loads(row.triggering_data or "{}"),
                insight_type=row.insight_type,
                expires_at=row.expires_at,
                created_at=row.created_at,
            )
            for row in rows
        ]
