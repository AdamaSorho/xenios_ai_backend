"""Client analytics service for computing aggregate metrics."""

from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.analytics import ClientAnalytics
from app.services.analytics.trends import (
    calculate_session_frequency_trend,
    calculate_talk_ratio_trend,
    calculate_trend,
)

logger = get_logger(__name__)


class ClientAnalyticsService:
    """
    Compute aggregate analytics for a client over time windows.

    Aggregates:
    - Session frequency and trends
    - Talk-time averages and trends
    - Engagement and sentiment trends
    - Cue pattern summaries
    """

    def __init__(self, db: AsyncSession):
        """Initialize the client analytics service."""
        self.db = db

    async def compute_for_client(
        self,
        client_id: UUID,
        coach_id: UUID,
        window_type: str = "90d",
    ) -> ClientAnalytics:
        """
        Compute aggregate analytics for a client.

        Args:
            client_id: UUID of the client
            coach_id: UUID of the coach
            window_type: Time window (30d, 90d, all_time)

        Returns:
            ClientAnalytics model (not yet committed)
        """
        logger.info(
            "Computing client analytics",
            client_id=str(client_id),
            coach_id=str(coach_id),
            window_type=window_type,
        )

        # Calculate window dates
        window_end = date.today()
        window_start = self._calculate_window_start(window_type, window_end)

        # Fetch session analytics for window
        sessions = await self._fetch_session_analytics(
            client_id=client_id,
            coach_id=coach_id,
            from_date=window_start,
            to_date=window_end,
        )

        # Fetch all-time session count and dates for frequency calculation
        all_sessions_meta = await self._fetch_session_meta(
            client_id=client_id,
            coach_id=coach_id,
        )

        # Calculate metrics
        analytics = self._compute_metrics(
            client_id=client_id,
            coach_id=coach_id,
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
            sessions=sessions,
            all_sessions_meta=all_sessions_meta,
        )

        logger.info(
            "Client analytics computed",
            client_id=str(client_id),
            total_sessions=analytics.total_sessions,
            engagement_trend=analytics.engagement_trend,
        )

        return analytics

    async def upsert(self, analytics: ClientAnalytics) -> ClientAnalytics:
        """
        Upsert client analytics (insert or update on conflict).

        Args:
            analytics: ClientAnalytics model to upsert

        Returns:
            The upserted ClientAnalytics
        """
        # Check if exists
        result = await self.db.execute(
            text("""
                SELECT id FROM ai_backend.client_analytics
                WHERE client_id = :client_id
                AND coach_id = :coach_id
                AND window_type = :window_type
            """),
            {
                "client_id": str(analytics.client_id),
                "coach_id": str(analytics.coach_id),
                "window_type": analytics.window_type,
            },
        )
        existing = result.scalar()

        if existing:
            # Update existing record
            analytics.id = existing
            await self.db.merge(analytics)
        else:
            # Insert new record
            self.db.add(analytics)

        await self.db.flush()
        return analytics

    def _calculate_window_start(self, window_type: str, window_end: date) -> date:
        """Calculate window start date from type."""
        if window_type == "30d":
            return window_end - timedelta(days=30)
        elif window_type == "90d":
            return window_end - timedelta(days=90)
        else:  # all_time
            return date(2020, 1, 1)  # Far enough back

    async def _fetch_session_analytics(
        self,
        client_id: UUID,
        coach_id: UUID,
        from_date: date,
        to_date: date,
    ) -> list[dict[str, Any]]:
        """Fetch session analytics for a client within a window."""
        result = await self.db.execute(
            text("""
                SELECT
                    id, session_date,
                    coach_talk_percentage, client_talk_percentage,
                    engagement_score, client_sentiment_score,
                    cue_resistance_count, cue_commitment_count, cue_breakthrough_count
                FROM ai_backend.session_analytics
                WHERE client_id = :client_id
                AND coach_id = :coach_id
                AND session_date >= :from_date
                AND session_date <= :to_date
                ORDER BY session_date ASC
            """),
            {
                "client_id": str(client_id),
                "coach_id": str(coach_id),
                "from_date": from_date,
                "to_date": to_date,
            },
        )
        return [dict(row) for row in result.mappings()]

    async def _fetch_session_meta(
        self,
        client_id: UUID,
        coach_id: UUID,
    ) -> dict[str, Any]:
        """Fetch session metadata for frequency calculations."""
        result = await self.db.execute(
            text("""
                SELECT
                    COUNT(*) as total_sessions,
                    MAX(session_date) as last_session_date,
                    ARRAY_AGG(session_date ORDER BY session_date) as session_dates
                FROM ai_backend.session_analytics
                WHERE client_id = :client_id
                AND coach_id = :coach_id
            """),
            {
                "client_id": str(client_id),
                "coach_id": str(coach_id),
            },
        )
        row = result.mappings().first()
        return dict(row) if row else {
            "total_sessions": 0,
            "last_session_date": None,
            "session_dates": [],
        }

    def _compute_metrics(
        self,
        client_id: UUID,
        coach_id: UUID,
        window_type: str,
        window_start: date,
        window_end: date,
        sessions: list[dict[str, Any]],
        all_sessions_meta: dict[str, Any],
    ) -> ClientAnalytics:
        """Compute all metrics from session data."""
        total_sessions = len(sessions)
        today = date.today()

        # Session frequency
        last_session_date = all_sessions_meta.get("last_session_date")
        days_since_last = (today - last_session_date).days if last_session_date else None

        # Sessions in last 30 days
        thirty_days_ago = today - timedelta(days=30)
        sessions_last_30_days = sum(
            1 for s in sessions
            if s["session_date"] >= thirty_days_ago
        )

        # Average days between sessions
        session_dates = all_sessions_meta.get("session_dates") or []
        if len(session_dates) >= 2:
            gaps = [
                (session_dates[i] - session_dates[i - 1]).days
                for i in range(1, len(session_dates))
            ]
            avg_days_between = sum(gaps) / len(gaps)
        else:
            avg_days_between = None

        # Session frequency trend
        session_frequency_trend = calculate_session_frequency_trend(session_dates)

        # Talk-time averages and trends
        if sessions:
            coach_percentages = [
                float(s["coach_talk_percentage"])
                for s in sessions
                if s.get("coach_talk_percentage") is not None
            ]
            client_percentages = [
                float(s["client_talk_percentage"])
                for s in sessions
                if s.get("client_talk_percentage") is not None
            ]

            avg_coach_talk = sum(coach_percentages) / len(coach_percentages) if coach_percentages else None
            avg_client_talk = sum(client_percentages) / len(client_percentages) if client_percentages else None
            talk_ratio_trend = calculate_talk_ratio_trend(client_percentages)
        else:
            avg_coach_talk = None
            avg_client_talk = None
            talk_ratio_trend = "balanced"

        # Engagement averages and trends
        engagement_scores = [
            float(s["engagement_score"])
            for s in sessions
            if s.get("engagement_score") is not None
        ]
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else None
        engagement_trend = calculate_trend(engagement_scores, higher_is_better=True)

        # Sentiment averages and trends
        sentiment_scores = [
            float(s["client_sentiment_score"])
            for s in sessions
            if s.get("client_sentiment_score") is not None
        ]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
        sentiment_trend = calculate_trend(sentiment_scores, higher_is_better=True)

        # Cue totals
        total_resistance = sum(s.get("cue_resistance_count", 0) or 0 for s in sessions)
        total_commitment = sum(s.get("cue_commitment_count", 0) or 0 for s in sessions)
        total_breakthrough = sum(s.get("cue_breakthrough_count", 0) or 0 for s in sessions)

        # Resistance to commitment ratio
        if total_commitment > 0:
            resistance_ratio = total_resistance / total_commitment
        elif total_resistance > 0:
            resistance_ratio = float(total_resistance)  # All resistance, no commitment
        else:
            resistance_ratio = 0.0

        return ClientAnalytics(
            client_id=client_id,
            coach_id=coach_id,
            window_start=window_start,
            window_end=window_end,
            window_type=window_type,
            total_sessions=all_sessions_meta.get("total_sessions", 0) or 0,
            sessions_last_30_days=sessions_last_30_days,
            average_days_between_sessions=round(avg_days_between, 2) if avg_days_between else None,
            days_since_last_session=days_since_last,
            session_frequency_trend=session_frequency_trend,
            average_coach_talk_percentage=round(avg_coach_talk, 2) if avg_coach_talk else None,
            average_client_talk_percentage=round(avg_client_talk, 2) if avg_client_talk else None,
            talk_ratio_trend=talk_ratio_trend,
            average_engagement_score=round(avg_engagement, 2) if avg_engagement else None,
            engagement_trend=engagement_trend,
            engagement_scores_history=engagement_scores[-10:] if engagement_scores else [],  # Last 10
            average_sentiment_score=round(avg_sentiment, 3) if avg_sentiment else None,
            sentiment_trend=sentiment_trend,
            sentiment_scores_history=sentiment_scores[-10:] if sentiment_scores else [],  # Last 10
            total_resistance_cues=total_resistance,
            total_commitment_cues=total_commitment,
            total_breakthrough_cues=total_breakthrough,
            resistance_to_commitment_ratio=round(resistance_ratio, 3),
            computed_at=datetime.now(timezone.utc),
        )

    async def get_for_client(
        self,
        client_id: UUID,
        coach_id: UUID,
        window_type: str = "90d",
    ) -> ClientAnalytics | None:
        """Fetch existing client analytics."""
        result = await self.db.execute(
            text("""
                SELECT * FROM ai_backend.client_analytics
                WHERE client_id = :client_id
                AND coach_id = :coach_id
                AND window_type = :window_type
            """),
            {
                "client_id": str(client_id),
                "coach_id": str(coach_id),
                "window_type": window_type,
            },
        )
        row = result.mappings().first()
        if not row:
            return None

        # Convert row to model
        return ClientAnalytics(**dict(row))


def get_client_analytics_service(db: AsyncSession) -> ClientAnalyticsService:
    """Get a client analytics service instance."""
    return ClientAnalyticsService(db)
