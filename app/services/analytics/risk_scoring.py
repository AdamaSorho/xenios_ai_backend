"""Risk scoring service for client churn prediction."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.analytics import ClientAnalytics, RiskScore
from app.schemas.analytics import RiskLevel

logger = get_logger(__name__)


def normalize_factor_contribution(
    value: float,
    warning_threshold: float,
    critical_threshold: float,
    weight: float,
    inverse: bool = False,
) -> float:
    """
    Convert a metric value to a risk contribution.

    Below warning: 0 contribution
    At warning: 50% of weight
    At critical: 100% of weight
    Above critical: 100% of weight (capped)

    Linear interpolation between thresholds.

    Args:
        value: The metric value
        warning_threshold: Value where risk starts
        critical_threshold: Value where risk is maximum
        weight: Maximum contribution (points out of 100)
        inverse: If True, lower values are worse (e.g., engagement)

    Returns:
        Risk contribution (0 to weight)
    """
    if inverse:
        # Flip for metrics where low = bad
        value = -value
        warning_threshold = -warning_threshold
        critical_threshold = -critical_threshold

    if value <= warning_threshold:
        return 0.0
    elif value >= critical_threshold:
        return weight
    else:
        # Linear interpolation
        range_size = critical_threshold - warning_threshold
        if range_size == 0:
            return weight
        position = (value - warning_threshold) / range_size
        return weight * (0.5 + position * 0.5)


class RiskScoringService:
    """
    Calculate client churn risk scores.

    Factors (weights sum to 100):
    - Session frequency (25): Declining session rate
    - Engagement trend (25): Declining engagement scores
    - Sentiment trend (20): Declining sentiment
    - Resistance ratio (15): High resistance vs commitment
    - Days since session (15): Long time since last session
    """

    # Factor weights (must sum to 100)
    WEIGHTS = {
        "session_frequency": 25,
        "engagement_trend": 25,
        "sentiment_trend": 20,
        "resistance_ratio": 15,
        "days_since_session": 15,
    }

    # Thresholds for concern
    THRESHOLDS = {
        "days_since_session_warning": 14,
        "days_since_session_critical": 30,
        "engagement_decline_warning": -10,  # percentage points
        "engagement_decline_critical": -20,
        "sentiment_decline_warning": -0.2,  # on -1 to 1 scale
        "sentiment_decline_critical": -0.4,
        "resistance_ratio_warning": 2.0,  # 2x more resistance than commitment
        "resistance_ratio_critical": 4.0,
        "session_frequency_decline_warning": 0.5,  # 50% fewer sessions
        "session_frequency_decline_critical": 0.75,
    }

    MODEL_VERSION = "v1.0.0"
    VALIDITY_DAYS = 7  # Risk scores expire after 7 days

    def __init__(self, db: AsyncSession):
        """Initialize the risk scoring service."""
        self.db = db

    async def compute_risk_score(
        self,
        client_id: UUID,
        coach_id: UUID,
        client_analytics: ClientAnalytics,
    ) -> RiskScore:
        """
        Compute risk score from client analytics.

        Args:
            client_id: UUID of the client
            coach_id: UUID of the coach
            client_analytics: ClientAnalytics with aggregated metrics

        Returns:
            RiskScore model (not yet committed)
        """
        logger.info(
            "Computing risk score",
            client_id=str(client_id),
            coach_id=str(coach_id),
        )

        factors = []

        # 1. Session frequency factor
        freq_factor = self._compute_frequency_factor(client_analytics)
        factors.append(freq_factor)

        # 2. Engagement trend factor
        engagement_factor = self._compute_engagement_factor(client_analytics)
        factors.append(engagement_factor)

        # 3. Sentiment trend factor
        sentiment_factor = self._compute_sentiment_factor(client_analytics)
        factors.append(sentiment_factor)

        # 4. Resistance ratio factor
        resistance_factor = self._compute_resistance_factor(client_analytics)
        factors.append(resistance_factor)

        # 5. Days since last session factor
        recency_factor = self._compute_recency_factor(client_analytics)
        factors.append(recency_factor)

        # Calculate weighted total
        total_score = sum(f["contribution"] for f in factors)
        total_score = round(min(100, max(0, total_score)), 2)

        # Determine risk level
        risk_level = self._score_to_level(total_score)

        # Get previous score for trend
        previous = await self._get_previous_score(client_id, coach_id)
        previous_score = float(previous["risk_score"]) if previous else None
        score_change = round(total_score - previous_score, 2) if previous_score is not None else None

        if score_change is not None:
            if score_change > 5:
                trend = "worsening"
            elif score_change < -5:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = None

        # Generate recommendation
        recommendation = self._generate_recommendation(factors, risk_level)

        now = datetime.now(timezone.utc)

        risk_score = RiskScore(
            client_id=client_id,
            coach_id=coach_id,
            risk_score=total_score,
            risk_level=risk_level.value,
            churn_probability=round(total_score / 100, 3),
            factors=factors,
            previous_risk_score=previous_score,
            score_change=score_change,
            trend=trend,
            recommended_action=recommendation,
            computed_at=now,
            valid_until=now + timedelta(days=self.VALIDITY_DAYS),
            model_version=self.MODEL_VERSION,
        )

        logger.info(
            "Risk score computed",
            client_id=str(client_id),
            risk_score=total_score,
            risk_level=risk_level.value,
            trend=trend,
        )

        return risk_score

    def _compute_frequency_factor(self, analytics: ClientAnalytics) -> dict[str, Any]:
        """Score based on session frequency decline."""
        trend = analytics.session_frequency_trend

        if trend == "decreasing":
            # Calculate severity based on sessions in last 30 days
            if analytics.sessions_last_30_days == 0:
                contribution = self.WEIGHTS["session_frequency"]  # Max risk
            else:
                # Estimate decline ratio
                expected = analytics.total_sessions / 3  # Rough estimate for 30-day expectation
                if expected > 0:
                    decline_ratio = 1 - (analytics.sessions_last_30_days / expected)
                    decline_ratio = max(0, min(1, decline_ratio))
                    contribution = decline_ratio * self.WEIGHTS["session_frequency"]
                else:
                    contribution = 0
        else:
            contribution = 0

        return {
            "factor_type": "session_frequency",
            "contribution": round(contribution, 2),
            "value": analytics.sessions_last_30_days,
            "threshold": self.THRESHOLDS["session_frequency_decline_warning"],
            "description": f"Client had {analytics.sessions_last_30_days} sessions in last 30 days",
        }

    def _compute_engagement_factor(self, analytics: ClientAnalytics) -> dict[str, Any]:
        """Score based on engagement score decline."""
        history = analytics.engagement_scores_history or []

        if len(history) >= 2:
            recent = history[-1]
            older = history[0]
            change = recent - older  # Negative = declining
        else:
            change = 0

        contribution = normalize_factor_contribution(
            value=-change,  # Negative change becomes positive for risk
            warning_threshold=-self.THRESHOLDS["engagement_decline_warning"],
            critical_threshold=-self.THRESHOLDS["engagement_decline_critical"],
            weight=self.WEIGHTS["engagement_trend"],
            inverse=False,
        )

        return {
            "factor_type": "engagement_trend",
            "contribution": round(contribution, 2),
            "value": round(change, 2),
            "threshold": self.THRESHOLDS["engagement_decline_warning"],
            "description": f"Engagement changed by {change:+.1f} points over window",
        }

    def _compute_sentiment_factor(self, analytics: ClientAnalytics) -> dict[str, Any]:
        """Score based on sentiment decline."""
        history = analytics.sentiment_scores_history or []

        if len(history) >= 2:
            recent = history[-1]
            older = history[0]
            change = recent - older  # Negative = declining sentiment
        else:
            change = 0

        contribution = normalize_factor_contribution(
            value=-change,  # Negative change becomes positive for risk
            warning_threshold=-self.THRESHOLDS["sentiment_decline_warning"],
            critical_threshold=-self.THRESHOLDS["sentiment_decline_critical"],
            weight=self.WEIGHTS["sentiment_trend"],
            inverse=False,
        )

        return {
            "factor_type": "sentiment_trend",
            "contribution": round(contribution, 2),
            "value": round(change, 3),
            "threshold": self.THRESHOLDS["sentiment_decline_warning"],
            "description": f"Sentiment changed by {change:+.2f} over window",
        }

    def _compute_resistance_factor(self, analytics: ClientAnalytics) -> dict[str, Any]:
        """Score based on resistance-to-commitment ratio."""
        commitment = analytics.total_commitment_cues or 0
        resistance = analytics.total_resistance_cues or 0

        if commitment > 0:
            ratio = resistance / commitment
        elif resistance > 0:
            ratio = 10.0  # Cap for all resistance, no commitment
        else:
            ratio = 1.0  # Neutral

        # Cap ratio for calculation
        capped_ratio = min(ratio, 10.0)

        contribution = normalize_factor_contribution(
            value=capped_ratio,
            warning_threshold=self.THRESHOLDS["resistance_ratio_warning"],
            critical_threshold=self.THRESHOLDS["resistance_ratio_critical"],
            weight=self.WEIGHTS["resistance_ratio"],
            inverse=False,
        )

        return {
            "factor_type": "resistance_ratio",
            "contribution": round(contribution, 2),
            "value": round(ratio, 2),
            "threshold": self.THRESHOLDS["resistance_ratio_warning"],
            "description": f"Resistance/commitment ratio: {ratio:.1f}:1",
        }

    def _compute_recency_factor(self, analytics: ClientAnalytics) -> dict[str, Any]:
        """Score based on days since last session."""
        days = analytics.days_since_last_session or 0

        contribution = normalize_factor_contribution(
            value=days,
            warning_threshold=self.THRESHOLDS["days_since_session_warning"],
            critical_threshold=self.THRESHOLDS["days_since_session_critical"],
            weight=self.WEIGHTS["days_since_session"],
            inverse=False,
        )

        return {
            "factor_type": "days_since_session",
            "contribution": round(contribution, 2),
            "value": days,
            "threshold": self.THRESHOLDS["days_since_session_warning"],
            "description": f"Last session was {days} days ago",
        }

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score <= 25:
            return RiskLevel.LOW
        elif score <= 50:
            return RiskLevel.MEDIUM
        elif score <= 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_recommendation(
        self,
        factors: list[dict[str, Any]],
        level: RiskLevel,
    ) -> str:
        """Generate actionable recommendation based on risk factors."""
        top_factors = sorted(factors, key=lambda f: f["contribution"], reverse=True)[:2]

        recommendations = {
            ("session_frequency", RiskLevel.HIGH): "Schedule a check-in call to re-engage",
            ("session_frequency", RiskLevel.CRITICAL): "Urgent: Reach out immediately to prevent churn",
            ("engagement_trend", RiskLevel.HIGH): "Try new engagement strategies in next session",
            ("engagement_trend", RiskLevel.CRITICAL): "Engagement declining rapidly - immediate intervention needed",
            ("sentiment_trend", RiskLevel.HIGH): "Address underlying concerns in next conversation",
            ("sentiment_trend", RiskLevel.CRITICAL): "Client sentiment declining - prioritize emotional support",
            ("resistance_ratio", RiskLevel.HIGH): "Review recent sessions for resistance patterns",
            ("resistance_ratio", RiskLevel.CRITICAL): "High resistance detected - explore barriers with client",
            ("days_since_session", RiskLevel.HIGH): "Client hasn't been seen recently - schedule follow-up",
            ("days_since_session", RiskLevel.CRITICAL): "Client hasn't been seen in 30+ days - follow up urgently",
        }

        for factor in top_factors:
            key = (factor["factor_type"], level)
            if key in recommendations:
                return recommendations[key]

        # Default recommendations by level
        if level == RiskLevel.LOW:
            return "Continue current coaching approach"
        elif level == RiskLevel.MEDIUM:
            return "Monitor closely and consider proactive check-in"
        elif level == RiskLevel.HIGH:
            return "Take action to re-engage client"
        else:
            return "Immediate intervention required to prevent churn"

    async def _get_previous_score(
        self,
        client_id: UUID,
        coach_id: UUID,
    ) -> dict[str, Any] | None:
        """Get the most recent previous risk score."""
        result = await self.db.execute(
            text("""
                SELECT risk_score, risk_level, computed_at
                FROM ai_backend.risk_scores
                WHERE client_id = :client_id
                AND coach_id = :coach_id
                ORDER BY computed_at DESC
                LIMIT 1
            """),
            {
                "client_id": str(client_id),
                "coach_id": str(coach_id),
            },
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def get_latest(
        self,
        client_id: UUID,
        coach_id: UUID,
    ) -> RiskScore | None:
        """Get the latest valid risk score for a client."""
        result = await self.db.execute(
            text("""
                SELECT *
                FROM ai_backend.risk_scores
                WHERE client_id = :client_id
                AND coach_id = :coach_id
                AND valid_until > NOW()
                ORDER BY computed_at DESC
                LIMIT 1
            """),
            {
                "client_id": str(client_id),
                "coach_id": str(coach_id),
            },
        )
        row = result.mappings().first()
        if not row:
            return None
        return RiskScore(**dict(row))

    async def get_history(
        self,
        client_id: UUID,
        coach_id: UUID,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get risk score history for a client."""
        result = await self.db.execute(
            text("""
                SELECT computed_at, risk_score, risk_level,
                       (factors->0->>'factor_type') as top_factor
                FROM ai_backend.risk_scores
                WHERE client_id = :client_id
                AND coach_id = :coach_id
                ORDER BY computed_at DESC
                LIMIT :limit
            """),
            {
                "client_id": str(client_id),
                "coach_id": str(coach_id),
                "limit": limit,
            },
        )
        return [dict(row) for row in result.mappings()]


def get_risk_scoring_service(db: AsyncSession) -> RiskScoringService:
    """Get a risk scoring service instance."""
    return RiskScoringService(db)
