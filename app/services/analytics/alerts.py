"""Risk alert service for generating and managing alerts."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.analytics import RiskAlert, RiskScore
from app.schemas.analytics import AlertSeverity, AlertStatus, AlertType, RiskLevel

logger = get_logger(__name__)


class AlertsService:
    """
    Generate and manage risk alerts for coaches.

    Generates alerts for:
    - New high/critical risk clients
    - Significant risk increases (>20 points)
    - Clients with no session in 30+ days
    """

    # Thresholds
    RISK_INCREASE_THRESHOLD = 20  # Points increase to trigger alert
    NO_SESSION_DAYS_THRESHOLD = 30  # Days without session

    def __init__(self, db: AsyncSession):
        """Initialize the alerts service."""
        self.db = db

    async def generate_alerts(
        self,
        client_id: UUID,
        coach_id: UUID,
        risk_score: RiskScore,
        previous_score: RiskScore | None = None,
    ) -> list[RiskAlert]:
        """
        Generate alerts based on risk score changes.

        Args:
            client_id: UUID of the client
            coach_id: UUID of the coach
            risk_score: Current risk score
            previous_score: Previous risk score (if any)

        Returns:
            List of generated RiskAlert models
        """
        alerts = []

        # Check for existing pending alerts to avoid duplicates
        existing = await self._get_pending_alerts(client_id, coach_id)
        existing_types = {a["alert_type"] for a in existing}

        # 1. Check for new high/critical risk
        if risk_score.risk_level in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value):
            # Only alert if new (wasn't high/critical before)
            was_high_before = previous_score and previous_score.risk_level in (
                RiskLevel.HIGH.value, RiskLevel.CRITICAL.value
            )
            if not was_high_before and AlertType.NEW_HIGH_RISK.value not in existing_types:
                alert = self._create_high_risk_alert(
                    client_id, coach_id, risk_score
                )
                alerts.append(alert)
                logger.info(
                    "Generated new_high_risk alert",
                    client_id=str(client_id),
                    risk_level=risk_score.risk_level,
                )

        # 2. Check for significant risk increase
        if previous_score and risk_score.score_change:
            increase = float(risk_score.score_change)
            if increase >= self.RISK_INCREASE_THRESHOLD:
                if AlertType.RISK_INCREASED.value not in existing_types:
                    alert = self._create_risk_increased_alert(
                        client_id, coach_id, risk_score, increase
                    )
                    alerts.append(alert)
                    logger.info(
                        "Generated risk_increased alert",
                        client_id=str(client_id),
                        increase=increase,
                    )

        # 3. Check for no session in 30+ days
        factors = risk_score.factors or []
        days_factor = next(
            (f for f in factors if f.get("factor_type") == "days_since_session"),
            None
        )
        if days_factor:
            days_since = days_factor.get("value", 0)
            if days_since >= self.NO_SESSION_DAYS_THRESHOLD:
                if AlertType.NO_SESSION_30D.value not in existing_types:
                    alert = self._create_no_session_alert(
                        client_id, coach_id, risk_score, int(days_since)
                    )
                    alerts.append(alert)
                    logger.info(
                        "Generated no_session_30d alert",
                        client_id=str(client_id),
                        days_since=days_since,
                    )

        return alerts

    def _create_high_risk_alert(
        self,
        client_id: UUID,
        coach_id: UUID,
        risk_score: RiskScore,
    ) -> RiskAlert:
        """Create alert for new high/critical risk."""
        is_critical = risk_score.risk_level == RiskLevel.CRITICAL.value

        return RiskAlert(
            client_id=client_id,
            coach_id=coach_id,
            risk_score_id=risk_score.id,
            alert_type=AlertType.NEW_HIGH_RISK.value,
            severity=AlertSeverity.URGENT.value if is_critical else AlertSeverity.WARNING.value,
            title=f"Client at {'critical' if is_critical else 'high'} risk of disengagement",
            message=self._format_risk_message(risk_score),
            status=AlertStatus.PENDING.value,
            created_at=datetime.now(timezone.utc),
        )

    def _create_risk_increased_alert(
        self,
        client_id: UUID,
        coach_id: UUID,
        risk_score: RiskScore,
        increase: float,
    ) -> RiskAlert:
        """Create alert for significant risk increase."""
        return RiskAlert(
            client_id=client_id,
            coach_id=coach_id,
            risk_score_id=risk_score.id,
            alert_type=AlertType.RISK_INCREASED.value,
            severity=AlertSeverity.WARNING.value,
            title=f"Risk score increased by {increase:.0f} points",
            message=f"Client's churn risk has increased significantly. "
                    f"Current risk score: {float(risk_score.risk_score):.0f}/100. "
                    f"{risk_score.recommended_action}",
            status=AlertStatus.PENDING.value,
            created_at=datetime.now(timezone.utc),
        )

    def _create_no_session_alert(
        self,
        client_id: UUID,
        coach_id: UUID,
        risk_score: RiskScore,
        days_since: int,
    ) -> RiskAlert:
        """Create alert for no session in 30+ days."""
        is_urgent = days_since >= 45

        return RiskAlert(
            client_id=client_id,
            coach_id=coach_id,
            risk_score_id=risk_score.id,
            alert_type=AlertType.NO_SESSION_30D.value,
            severity=AlertSeverity.URGENT.value if is_urgent else AlertSeverity.WARNING.value,
            title=f"No session in {days_since} days",
            message=f"This client hasn't had a coaching session in {days_since} days. "
                    f"Consider reaching out to re-engage and schedule a follow-up.",
            status=AlertStatus.PENDING.value,
            created_at=datetime.now(timezone.utc),
        )

    def _format_risk_message(self, risk_score: RiskScore) -> str:
        """Format detailed message for risk alert."""
        factors = risk_score.factors or []
        top_factors = sorted(factors, key=lambda f: f.get("contribution", 0), reverse=True)[:2]

        lines = [
            f"Risk score: {float(risk_score.risk_score):.0f}/100 ({risk_score.risk_level})",
            "",
            "Top contributing factors:",
        ]

        for f in top_factors:
            lines.append(f"- {f.get('description', f.get('factor_type'))}")

        lines.extend([
            "",
            f"Recommended action: {risk_score.recommended_action}",
        ])

        return "\n".join(lines)

    async def _get_pending_alerts(
        self,
        client_id: UUID,
        coach_id: UUID,
    ) -> list[dict[str, Any]]:
        """Get pending alerts for a client."""
        result = await self.db.execute(
            text("""
                SELECT id, alert_type, created_at
                FROM ai_backend.risk_alerts
                WHERE client_id = :client_id
                AND coach_id = :coach_id
                AND status = 'pending'
            """),
            {
                "client_id": str(client_id),
                "coach_id": str(coach_id),
            },
        )
        return [dict(row) for row in result.mappings()]

    async def acknowledge_alert(
        self,
        alert_id: UUID,
        coach_id: UUID,
        notes: str | None = None,
    ) -> RiskAlert | None:
        """
        Acknowledge a risk alert.

        Args:
            alert_id: UUID of the alert
            coach_id: UUID of the coach (for authorization)
            notes: Optional notes from the coach

        Returns:
            Updated RiskAlert or None if not found/unauthorized
        """
        # Fetch and verify ownership
        result = await self.db.execute(
            text("""
                SELECT * FROM ai_backend.risk_alerts
                WHERE id = :alert_id AND coach_id = :coach_id
            """),
            {"alert_id": str(alert_id), "coach_id": str(coach_id)},
        )
        row = result.mappings().first()

        if not row:
            return None

        # Update status
        await self.db.execute(
            text("""
                UPDATE ai_backend.risk_alerts
                SET status = 'acknowledged',
                    acknowledged_at = NOW(),
                    acknowledged_notes = :notes
                WHERE id = :alert_id
            """),
            {"alert_id": str(alert_id), "notes": notes},
        )

        # Fetch updated record
        result = await self.db.execute(
            text("SELECT * FROM ai_backend.risk_alerts WHERE id = :alert_id"),
            {"alert_id": str(alert_id)},
        )
        row = result.mappings().first()
        return RiskAlert(**dict(row)) if row else None

    async def get_alerts_for_coach(
        self,
        coach_id: UUID,
        status: str | None = "pending",
        severity: str | None = None,
        limit: int = 50,
    ) -> tuple[list[RiskAlert], int]:
        """
        Get alerts for a coach with filtering.

        Args:
            coach_id: UUID of the coach
            status: Filter by status (pending, acknowledged, all)
            severity: Filter by severity (warning, urgent, all)
            limit: Maximum number of alerts to return

        Returns:
            Tuple of (list of RiskAlert, total count)
        """
        # Build query conditions
        conditions = ["coach_id = :coach_id"]
        params: dict[str, Any] = {"coach_id": str(coach_id), "limit": limit}

        if status and status != "all":
            conditions.append("status = :status")
            params["status"] = status

        if severity and severity != "all":
            conditions.append("severity = :severity")
            params["severity"] = severity

        where_clause = " AND ".join(conditions)

        # Get total count
        count_result = await self.db.execute(
            text(f"SELECT COUNT(*) FROM ai_backend.risk_alerts WHERE {where_clause}"),
            params,
        )
        total = count_result.scalar() or 0

        # Get alerts
        result = await self.db.execute(
            text(f"""
                SELECT * FROM ai_backend.risk_alerts
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            params,
        )

        alerts = [RiskAlert(**dict(row)) for row in result.mappings()]
        return alerts, total

    async def get_alerts_for_client(
        self,
        client_id: UUID,
        coach_id: UUID,
    ) -> list[RiskAlert]:
        """Get active alerts for a specific client."""
        result = await self.db.execute(
            text("""
                SELECT * FROM ai_backend.risk_alerts
                WHERE client_id = :client_id
                AND coach_id = :coach_id
                AND status != 'resolved'
                ORDER BY created_at DESC
            """),
            {
                "client_id": str(client_id),
                "coach_id": str(coach_id),
            },
        )
        return [RiskAlert(**dict(row)) for row in result.mappings()]


def get_alerts_service(db: AsyncSession) -> AlertsService:
    """Get an alerts service instance."""
    return AlertsService(db)
