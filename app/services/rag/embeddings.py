"""Embedding generation and storage service for RAG system."""

import hashlib
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.logging import get_logger
from app.models.rag import Embedding
from app.schemas.rag import EmbeddingSourceType, EmbeddingUpdateResult
from app.services.rag.openai_client import OpenAIEmbeddingClient

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating and storing embeddings for client health data.

    Supports content hashing for deduplication and multiple source types.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.openai_client = OpenAIEmbeddingClient()
        self.settings = get_settings()

    async def update_client_embeddings(
        self,
        client_id: UUID,
        source_types: list[EmbeddingSourceType] | None = None,
        force: bool = False,
    ) -> EmbeddingUpdateResult:
        """
        Update embeddings for a client's data.

        Args:
            client_id: Client to update embeddings for
            source_types: Specific types to update, or None for all
            force: If True, regenerate even if content unchanged

        Returns:
            Result with updated and skipped counts
        """
        updated = 0
        skipped = 0

        types_to_process = source_types or list(EmbeddingSourceType)

        logger.info(
            "Updating client embeddings",
            client_id=str(client_id),
            source_types=[t.value for t in types_to_process],
            force=force,
        )

        # Process each source type
        for source_type in types_to_process:
            try:
                u, s = await self._update_source_type_embeddings(
                    client_id, source_type, force
                )
                updated += u
                skipped += s
            except Exception as e:
                logger.error(
                    "Failed to update embeddings for source type",
                    client_id=str(client_id),
                    source_type=source_type.value,
                    error=str(e),
                )
                # Continue with other source types

        logger.info(
            "Completed embedding update",
            client_id=str(client_id),
            updated=updated,
            skipped=skipped,
        )

        return EmbeddingUpdateResult(updated_count=updated, skipped_count=skipped)

    async def _update_source_type_embeddings(
        self,
        client_id: UUID,
        source_type: EmbeddingSourceType,
        force: bool,
    ) -> tuple[int, int]:
        """Update embeddings for a specific source type."""
        if source_type == EmbeddingSourceType.HEALTH_PROFILE:
            return await self._update_health_profile_embedding(client_id, force)
        elif source_type == EmbeddingSourceType.HEALTH_METRIC_SUMMARY:
            return await self._update_metric_summary_embeddings(client_id, force)
        elif source_type == EmbeddingSourceType.HEALTH_GOAL:
            return await self._update_health_goal_embeddings(client_id, force)
        elif source_type == EmbeddingSourceType.LAB_RESULT:
            return await self._update_lab_result_embeddings(client_id, force)
        elif source_type == EmbeddingSourceType.SESSION_SUMMARY:
            return await self._update_session_summary_embeddings(client_id, force)
        elif source_type == EmbeddingSourceType.CHECKIN_SUMMARY:
            return await self._update_checkin_summary_embeddings(client_id, force)
        elif source_type == EmbeddingSourceType.MESSAGE_THREAD:
            return await self._update_message_thread_embeddings(client_id, force)
        else:
            return 0, 0

    async def _should_update(
        self,
        client_id: UUID,
        source_type: str,
        source_id: str,
        text: str,
        force: bool,
    ) -> bool:
        """Check if embedding needs update based on content hash."""
        if force:
            return True

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        result = await self.db.execute(
            select(Embedding.content_hash).where(
                and_(
                    Embedding.client_id == client_id,
                    Embedding.source_type == source_type,
                    Embedding.source_id == source_id,
                )
            )
        )
        existing_hash = result.scalar_one_or_none()

        return existing_hash != content_hash

    async def _store_embedding(
        self,
        client_id: UUID,
        source_type: str,
        source_id: str,
        text: str,
        source_table: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Generate and store embedding for text content."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        embedding_vector = await self.openai_client.generate_embedding(text)

        # Check for existing embedding (upsert)
        result = await self.db.execute(
            select(Embedding).where(
                and_(
                    Embedding.client_id == client_id,
                    Embedding.source_type == source_type,
                    Embedding.source_id == source_id,
                )
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing
            existing.content_text = text
            existing.content_hash = content_hash
            existing.embedding = embedding_vector
            existing.metadata_ = metadata or {}
            existing.updated_at = datetime.utcnow()
        else:
            # Create new
            embedding = Embedding(
                client_id=client_id,
                source_type=source_type,
                source_id=source_id,
                source_table=source_table,
                content_text=text,
                content_hash=content_hash,
                embedding=embedding_vector,
                metadata_=metadata or {},
            )
            self.db.add(embedding)

        await self.db.flush()

    # ========================================================================
    # Source-specific embedding builders
    # ========================================================================

    async def _update_health_profile_embedding(
        self, client_id: UUID, force: bool
    ) -> tuple[int, int]:
        """Build and store health profile embedding (1 per client)."""
        # Query health profile from MVP table
        query = text("""
            SELECT
                goals, conditions, preferences, dietary_restrictions,
                activity_level, target_weight, current_weight,
                date_of_birth, gender
            FROM public.client_health_profiles
            WHERE client_id = :client_id
        """)
        result = await self.db.execute(query, {"client_id": str(client_id)})
        row = result.fetchone()

        if not row:
            return 0, 0

        # Build text summary
        text_parts = [f"Client health profile:"]

        if row.gender:
            text_parts.append(f"Gender: {row.gender}")
        if row.date_of_birth:
            age = (datetime.now().date() - row.date_of_birth).days // 365
            text_parts.append(f"Age: {age} years old")
        if row.current_weight:
            text_parts.append(f"Current weight: {row.current_weight}kg")
        if row.target_weight:
            text_parts.append(f"Target weight: {row.target_weight}kg")
        if row.goals:
            text_parts.append(f"Goals: {row.goals}")
        if row.conditions:
            text_parts.append(f"Medical conditions: {row.conditions}")
        if row.dietary_restrictions:
            text_parts.append(f"Dietary restrictions: {row.dietary_restrictions}")
        if row.activity_level:
            text_parts.append(f"Activity level: {row.activity_level}")
        if row.preferences:
            text_parts.append(f"Preferences: {row.preferences}")

        profile_text = "\n".join(text_parts)
        source_id = f"{client_id}:profile"

        if await self._should_update(
            client_id, EmbeddingSourceType.HEALTH_PROFILE.value, source_id, profile_text, force
        ):
            await self._store_embedding(
                client_id=client_id,
                source_type=EmbeddingSourceType.HEALTH_PROFILE.value,
                source_id=source_id,
                text=profile_text,
                source_table="public.client_health_profiles",
            )
            return 1, 0
        return 0, 1

    async def _update_metric_summary_embeddings(
        self, client_id: UUID, force: bool
    ) -> tuple[int, int]:
        """Build and store metric summary embeddings (1 per metric type per week)."""
        # Get metrics from last 90 days, grouped by type and week
        query = text("""
            SELECT
                metric_type,
                date_trunc('week', recorded_at) as week_start,
                MIN(value) as min_value,
                MAX(value) as max_value,
                AVG(value) as avg_value,
                COUNT(*) as data_points,
                MIN(recorded_at) as first_recorded,
                MAX(recorded_at) as last_recorded
            FROM public.health_metrics
            WHERE client_id = :client_id
            AND recorded_at >= NOW() - INTERVAL '90 days'
            GROUP BY metric_type, date_trunc('week', recorded_at)
            ORDER BY week_start DESC
        """)
        result = await self.db.execute(query, {"client_id": str(client_id)})
        rows = result.fetchall()

        updated = 0
        skipped = 0

        for row in rows:
            week_iso = row.week_start.strftime("%Y-%m-%d")
            source_id = f"{client_id}:metric:{row.metric_type}:{week_iso}"

            # Build summary text
            text_content = (
                f"{row.metric_type.replace('_', ' ').title()} progress (week of {week_iso}):\n"
                f"Range: {row.min_value:.1f} to {row.max_value:.1f}\n"
                f"Average: {row.avg_value:.1f}\n"
                f"Data points: {row.data_points}\n"
                f"Period: {row.first_recorded.strftime('%b %d')} - {row.last_recorded.strftime('%b %d')}"
            )

            if await self._should_update(
                client_id,
                EmbeddingSourceType.HEALTH_METRIC_SUMMARY.value,
                source_id,
                text_content,
                force,
            ):
                await self._store_embedding(
                    client_id=client_id,
                    source_type=EmbeddingSourceType.HEALTH_METRIC_SUMMARY.value,
                    source_id=source_id,
                    text=text_content,
                    source_table="public.health_metrics",
                    metadata={
                        "metric_type": row.metric_type,
                        "week": week_iso,
                        "date": week_iso,
                    },
                )
                updated += 1
            else:
                skipped += 1

        return updated, skipped

    async def _update_health_goal_embeddings(
        self, client_id: UUID, force: bool
    ) -> tuple[int, int]:
        """Build and store health goal embeddings (1 per active goal)."""
        query = text("""
            SELECT
                id, metric_type, target_value, current_value,
                target_date, status, created_at
            FROM public.health_goals
            WHERE client_id = :client_id
            AND status IN ('active', 'in_progress')
        """)
        result = await self.db.execute(query, {"client_id": str(client_id)})
        rows = result.fetchall()

        updated = 0
        skipped = 0

        for row in rows:
            source_id = str(row.id)
            progress = (
                ((row.current_value or 0) / row.target_value * 100)
                if row.target_value
                else 0
            )

            text_content = (
                f"Health goal: {row.metric_type.replace('_', ' ').title()}\n"
                f"Target: {row.target_value}\n"
                f"Current: {row.current_value or 'Not recorded'}\n"
                f"Progress: {progress:.0f}%\n"
                f"Target date: {row.target_date.strftime('%b %d, %Y') if row.target_date else 'Not set'}\n"
                f"Status: {row.status}"
            )

            if await self._should_update(
                client_id,
                EmbeddingSourceType.HEALTH_GOAL.value,
                source_id,
                text_content,
                force,
            ):
                await self._store_embedding(
                    client_id=client_id,
                    source_type=EmbeddingSourceType.HEALTH_GOAL.value,
                    source_id=source_id,
                    text=text_content,
                    source_table="public.health_goals",
                    metadata={
                        "metric_type": row.metric_type,
                        "status": row.status,
                        "date": row.created_at.strftime("%Y-%m-%d") if row.created_at else None,
                    },
                )
                updated += 1
            else:
                skipped += 1

        return updated, skipped

    async def _update_lab_result_embeddings(
        self, client_id: UUID, force: bool
    ) -> tuple[int, int]:
        """Build and store lab result embeddings (1 per lab panel)."""
        query = text("""
            SELECT
                id, biomarker, value, unit, reference_range,
                status, recorded_at, source
            FROM public.lab_values
            WHERE client_id = :client_id
            ORDER BY recorded_at DESC
            LIMIT 50
        """)
        result = await self.db.execute(query, {"client_id": str(client_id)})
        rows = result.fetchall()

        updated = 0
        skipped = 0

        for row in rows:
            source_id = str(row.id)

            status_text = ""
            if row.status:
                status_text = f" ({row.status.upper()})"

            text_content = (
                f"Lab result ({row.recorded_at.strftime('%b %d, %Y')}):\n"
                f"{row.biomarker}: {row.value} {row.unit or ''}{status_text}\n"
                f"Reference range: {row.reference_range or 'Not specified'}\n"
                f"Source: {row.source or 'Not specified'}"
            )

            if await self._should_update(
                client_id,
                EmbeddingSourceType.LAB_RESULT.value,
                source_id,
                text_content,
                force,
            ):
                await self._store_embedding(
                    client_id=client_id,
                    source_type=EmbeddingSourceType.LAB_RESULT.value,
                    source_id=source_id,
                    text=text_content,
                    source_table="public.lab_values",
                    metadata={
                        "biomarker": row.biomarker,
                        "status": row.status,
                        "date": row.recorded_at.strftime("%Y-%m-%d"),
                    },
                )
                updated += 1
            else:
                skipped += 1

        return updated, skipped

    async def _update_session_summary_embeddings(
        self, client_id: UUID, force: bool
    ) -> tuple[int, int]:
        """Build and store session summary embeddings (1 per session)."""
        query = text("""
            SELECT
                ss.id, ss.executive_summary, ss.key_topics,
                ss.client_concerns, ss.coach_recommendations,
                ss.action_items, ss.session_type_detected,
                tj.session_date
            FROM ai_backend.session_summaries ss
            JOIN ai_backend.transcription_jobs tj ON tj.id = ss.job_id
            WHERE tj.client_id = :client_id
            ORDER BY tj.session_date DESC
            LIMIT 20
        """)
        result = await self.db.execute(query, {"client_id": str(client_id)})
        rows = result.fetchall()

        updated = 0
        skipped = 0

        for row in rows:
            source_id = str(row.id)
            date_str = row.session_date.strftime("%b %d, %Y") if row.session_date else "Unknown date"

            # Build comprehensive summary text
            parts = [f"Session summary ({date_str}):"]

            if row.session_type_detected:
                parts.append(f"Type: {row.session_type_detected}")

            if row.executive_summary:
                parts.append(f"Summary: {row.executive_summary}")

            if row.key_topics:
                topics = row.key_topics if isinstance(row.key_topics, list) else []
                if topics:
                    parts.append(f"Key topics: {', '.join(topics)}")

            if row.client_concerns:
                concerns = row.client_concerns if isinstance(row.client_concerns, list) else []
                if concerns:
                    parts.append(f"Client concerns: {', '.join(concerns)}")

            if row.coach_recommendations:
                recs = row.coach_recommendations if isinstance(row.coach_recommendations, list) else []
                if recs:
                    parts.append(f"Coach recommendations: {', '.join(recs)}")

            if row.action_items:
                items = row.action_items if isinstance(row.action_items, list) else []
                if items:
                    parts.append(f"Action items: {', '.join(items)}")

            text_content = "\n".join(parts)

            if await self._should_update(
                client_id,
                EmbeddingSourceType.SESSION_SUMMARY.value,
                source_id,
                text_content,
                force,
            ):
                await self._store_embedding(
                    client_id=client_id,
                    source_type=EmbeddingSourceType.SESSION_SUMMARY.value,
                    source_id=source_id,
                    text=text_content,
                    source_table="ai_backend.session_summaries",
                    metadata={
                        "session_type": row.session_type_detected,
                        "date": row.session_date.strftime("%Y-%m-%d") if row.session_date else None,
                    },
                )
                updated += 1
            else:
                skipped += 1

        return updated, skipped

    async def _update_checkin_summary_embeddings(
        self, client_id: UUID, force: bool
    ) -> tuple[int, int]:
        """Build and store check-in summary embeddings (1 per check-in)."""
        query = text("""
            SELECT
                id, responses, ai_summary, created_at
            FROM public.checkins
            WHERE client_id = :client_id
            ORDER BY created_at DESC
            LIMIT 30
        """)
        result = await self.db.execute(query, {"client_id": str(client_id)})
        rows = result.fetchall()

        updated = 0
        skipped = 0

        for row in rows:
            source_id = str(row.id)
            date_str = row.created_at.strftime("%b %d, %Y") if row.created_at else "Unknown date"

            parts = [f"Check-in ({date_str}):"]

            if row.ai_summary:
                parts.append(f"Summary: {row.ai_summary}")

            # Include key responses if available
            if row.responses and isinstance(row.responses, dict):
                for key, value in list(row.responses.items())[:5]:
                    parts.append(f"{key}: {value}")

            text_content = "\n".join(parts)

            if await self._should_update(
                client_id,
                EmbeddingSourceType.CHECKIN_SUMMARY.value,
                source_id,
                text_content,
                force,
            ):
                await self._store_embedding(
                    client_id=client_id,
                    source_type=EmbeddingSourceType.CHECKIN_SUMMARY.value,
                    source_id=source_id,
                    text=text_content,
                    source_table="public.checkins",
                    metadata={
                        "date": row.created_at.strftime("%Y-%m-%d") if row.created_at else None,
                    },
                )
                updated += 1
            else:
                skipped += 1

        return updated, skipped

    async def _update_message_thread_embeddings(
        self, client_id: UUID, force: bool
    ) -> tuple[int, int]:
        """Build and store message thread embeddings (1 per day, last 14 days)."""
        # Get messages from last 14 days, grouped by day
        query = text("""
            SELECT
                DATE(created_at) as message_date,
                array_agg(content ORDER BY created_at) as messages,
                array_agg(sender_type ORDER BY created_at) as senders
            FROM public.messages
            WHERE client_id = :client_id
            AND created_at >= NOW() - INTERVAL '14 days'
            GROUP BY DATE(created_at)
            ORDER BY message_date DESC
        """)
        result = await self.db.execute(query, {"client_id": str(client_id)})
        rows = result.fetchall()

        updated = 0
        skipped = 0

        for row in rows:
            date_iso = row.message_date.strftime("%Y-%m-%d")
            source_id = f"{client_id}:messages:{date_iso}"

            # Build conversation summary
            parts = [f"Conversation ({row.message_date.strftime('%b %d, %Y')}):"]

            messages = row.messages or []
            senders = row.senders or []

            for msg, sender in zip(messages[:10], senders[:10]):  # Limit to 10 messages
                sender_label = "Coach" if sender == "coach" else "Client"
                # Truncate long messages
                msg_truncated = msg[:200] + "..." if len(msg) > 200 else msg
                parts.append(f"{sender_label}: {msg_truncated}")

            text_content = "\n".join(parts)

            if await self._should_update(
                client_id,
                EmbeddingSourceType.MESSAGE_THREAD.value,
                source_id,
                text_content,
                force,
            ):
                await self._store_embedding(
                    client_id=client_id,
                    source_type=EmbeddingSourceType.MESSAGE_THREAD.value,
                    source_id=source_id,
                    text=text_content,
                    source_table="public.messages",
                    metadata={
                        "date": date_iso,
                        "message_count": len(messages),
                    },
                )
                updated += 1
            else:
                skipped += 1

        return updated, skipped
