"""Analytics API endpoints for coaching analytics and risk detection."""

from datetime import date, datetime, timezone
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import UserContext, get_current_user
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.schemas.analytics import (
    AcknowledgeAlertRequest,
    AlertListResponse,
    ClientAnalyticsResponse,
    ClientAnalyticsSummaryResponse,
    CoachSummaryResponse,
    ComputeResponse,
    DataPoint,
    LanguageCueResponse,
    RiskAlertResponse,
    RiskDistribution,
    RiskLevel,
    RiskScoreDetailResponse,
    RiskScoreHistory,
    RiskScoreResponse,
    SessionAnalyticsDetailResponse,
    SessionAnalyticsResponse,
    SessionAnalyticsSummary,
    SessionComparison,
    SessionListResponse,
    TrendData,
    TrendDirection,
    TrendsResponse,
)
from app.services.analytics.alerts import get_alerts_service
from app.services.analytics.authorization import (
    require_coach_client_relationship,
    require_job_ownership,
)
from app.services.analytics.client_analytics import get_client_analytics_service
from app.services.analytics.risk_scoring import get_risk_scoring_service
from app.services.analytics.session_analytics import get_session_analytics_service
from app.services.analytics.trends import calculate_trend

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# Dependency for database session
async def get_db():
    """Get database session."""
    async with get_db_session() as session:
        yield session


# ============================================================================
# Session Analytics Endpoints
# ============================================================================


@router.get("/sessions/{job_id}", response_model=SessionAnalyticsDetailResponse)
async def get_session_analytics(
    job_id: UUID,
    current_user: Annotated[UserContext, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Get analytics for a specific session.

    Includes session comparison with previous session if available.
    """
    coach_id = UUID(current_user.user_id)

    # Verify ownership
    await require_job_ownership(job_id, coach_id, db)

    # Fetch session analytics
    result = await db.execute(
        text("SELECT * FROM ai_backend.session_analytics WHERE job_id = :job_id"),
        {"job_id": str(job_id)},
    )
    row = result.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Analytics not found for this session")

    analytics_dict = dict(row)

    # Fetch language cues
    cues_result = await db.execute(
        text("""
            SELECT * FROM ai_backend.language_cues
            WHERE session_analytics_id = :analytics_id
            ORDER BY timestamp_seconds
        """),
        {"analytics_id": str(analytics_dict["id"])},
    )
    cues = [dict(r) for r in cues_result.mappings()]

    # Fetch previous session for comparison
    comparison = await _get_session_comparison(
        db=db,
        client_id=analytics_dict["client_id"],
        coach_id=coach_id,
        current_session_date=analytics_dict["session_date"],
        current_analytics=analytics_dict,
    )

    return SessionAnalyticsDetailResponse(
        session_analytics=SessionAnalyticsResponse(**analytics_dict),
        cues=[LanguageCueResponse(**c) for c in cues],
        comparison=comparison,
    )


async def _get_session_comparison(
    db: AsyncSession,
    client_id: UUID,
    coach_id: UUID,
    current_session_date: date,
    current_analytics: dict[str, Any],
) -> SessionComparison | None:
    """Get comparison with previous session."""
    result = await db.execute(
        text("""
            SELECT session_date, engagement_score, client_sentiment_score, client_talk_percentage
            FROM ai_backend.session_analytics
            WHERE client_id = :client_id
            AND coach_id = :coach_id
            AND session_date < :current_date
            ORDER BY session_date DESC
            LIMIT 1
        """),
        {
            "client_id": str(client_id),
            "coach_id": str(coach_id),
            "current_date": current_session_date,
        },
    )
    prev = result.mappings().first()

    if not prev:
        return None

    prev_dict = dict(prev)

    # Calculate changes
    engagement_change = (
        float(current_analytics.get("engagement_score") or 0) -
        float(prev_dict.get("engagement_score") or 0)
    )
    sentiment_change = (
        float(current_analytics.get("client_sentiment_score") or 0) -
        float(prev_dict.get("client_sentiment_score") or 0)
    )
    talk_ratio_change = (
        float(current_analytics.get("client_talk_percentage") or 0) -
        float(prev_dict.get("client_talk_percentage") or 0)
    )

    # Generate notable changes
    notable = []
    if abs(engagement_change) > 5:
        direction = "improved" if engagement_change > 0 else "decreased"
        notable.append(f"Engagement {direction} by {abs(engagement_change):.1f} points")
    if abs(sentiment_change) > 0.1:
        direction = "improved" if sentiment_change > 0 else "decreased"
        notable.append(f"Sentiment {direction}")
    if abs(talk_ratio_change) > 5:
        direction = "more" if talk_ratio_change > 0 else "less"
        notable.append(f"Client spoke {direction} ({abs(talk_ratio_change):.1f}% change)")

    return SessionComparison(
        previous_session_date=prev_dict["session_date"],
        engagement_change=round(engagement_change, 2),
        sentiment_change=round(sentiment_change, 3),
        talk_ratio_change=round(talk_ratio_change, 2),
        notable_changes=notable,
    )


# ============================================================================
# Client Analytics Endpoints
# ============================================================================


@router.get("/clients/{client_id}/summary", response_model=ClientAnalyticsSummaryResponse)
async def get_client_summary(
    client_id: UUID,
    window: str = Query("90d", pattern="^(30d|90d|all_time)$"),
    current_user: Annotated[UserContext, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Get aggregate analytics for a client."""
    coach_id = UUID(current_user.user_id)

    # Verify relationship
    await require_coach_client_relationship(coach_id, client_id, db)

    # Fetch client analytics
    result = await db.execute(
        text("""
            SELECT * FROM ai_backend.client_analytics
            WHERE client_id = :client_id
            AND coach_id = :coach_id
            AND window_type = :window_type
        """),
        {
            "client_id": str(client_id),
            "coach_id": str(coach_id),
            "window_type": window,
        },
    )
    row = result.mappings().first()

    if not row:
        # Return empty analytics if not computed yet
        return _empty_client_summary(client_id, coach_id, window)

    analytics_dict = dict(row)

    # Get session count and latest date
    session_result = await db.execute(
        text("""
            SELECT COUNT(*) as count, MAX(session_date) as latest
            FROM ai_backend.session_analytics
            WHERE client_id = :client_id AND coach_id = :coach_id
        """),
        {"client_id": str(client_id), "coach_id": str(coach_id)},
    )
    session_row = session_result.mappings().first()
    session_count = session_row["count"] if session_row else 0
    latest_session = session_row["latest"] if session_row else None

    # Get latest risk score
    risk_result = await db.execute(
        text("""
            SELECT risk_level, valid_until, computed_at
            FROM ai_backend.risk_scores
            WHERE client_id = :client_id AND coach_id = :coach_id
            ORDER BY computed_at DESC
            LIMIT 1
        """),
        {"client_id": str(client_id), "coach_id": str(coach_id)},
    )
    risk_row = risk_result.mappings().first()

    risk_level = None
    risk_score_stale = False
    if risk_row:
        valid_until = risk_row["valid_until"]
        if valid_until and valid_until > datetime.now(timezone.utc):
            risk_level = risk_row["risk_level"]
        else:
            risk_score_stale = True

    # Compute quality warnings
    quality_warnings = []
    if session_count < 2:
        quality_warnings.append("insufficient_sessions")
    if risk_score_stale:
        quality_warnings.append("stale_risk_score")

    return ClientAnalyticsSummaryResponse(
        client_analytics=ClientAnalyticsResponse(**analytics_dict),
        session_count=session_count,
        latest_session_date=latest_session,
        risk_level=RiskLevel(risk_level) if risk_level else None,
        risk_score_stale=risk_score_stale,
        quality_warnings=quality_warnings,
    )


def _empty_client_summary(
    client_id: UUID,
    coach_id: UUID,
    window: str,
) -> ClientAnalyticsSummaryResponse:
    """Return empty summary when no analytics exist."""
    today = date.today()
    from datetime import timedelta

    if window == "30d":
        window_start = today - timedelta(days=30)
    elif window == "90d":
        window_start = today - timedelta(days=90)
    else:
        window_start = date(2020, 1, 1)

    return ClientAnalyticsSummaryResponse(
        client_analytics=ClientAnalyticsResponse(
            id=UUID("00000000-0000-0000-0000-000000000000"),
            client_id=client_id,
            coach_id=coach_id,
            window_start=window_start,
            window_end=today,
            window_type=window,
            total_sessions=0,
            sessions_last_30_days=0,
            average_days_between_sessions=None,
            days_since_last_session=None,
            session_frequency_trend=None,
            average_coach_talk_percentage=None,
            average_client_talk_percentage=None,
            talk_ratio_trend=None,
            average_engagement_score=None,
            engagement_trend=None,
            engagement_scores_history=[],
            average_sentiment_score=None,
            sentiment_trend=None,
            sentiment_scores_history=[],
            total_resistance_cues=0,
            total_commitment_cues=0,
            total_breakthrough_cues=0,
            resistance_to_commitment_ratio=None,
            computed_at=datetime.now(timezone.utc),
        ),
        session_count=0,
        latest_session_date=None,
        risk_level=None,
        risk_score_stale=False,
        quality_warnings=["insufficient_sessions"],
    )


@router.get("/clients/{client_id}/sessions", response_model=SessionListResponse)
async def get_client_sessions(
    client_id: UUID,
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    from_date: date | None = None,
    to_date: date | None = None,
    current_user: Annotated[UserContext, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Get session analytics history for a client."""
    coach_id = UUID(current_user.user_id)

    # Verify relationship
    await require_coach_client_relationship(coach_id, client_id, db)

    # Build query
    conditions = ["client_id = :client_id", "coach_id = :coach_id"]
    params: dict[str, Any] = {
        "client_id": str(client_id),
        "coach_id": str(coach_id),
        "limit": limit,
        "offset": offset,
    }

    if from_date:
        conditions.append("session_date >= :from_date")
        params["from_date"] = from_date
    if to_date:
        conditions.append("session_date <= :to_date")
        params["to_date"] = to_date

    where_clause = " AND ".join(conditions)

    # Get total count
    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM ai_backend.session_analytics WHERE {where_clause}"),
        params,
    )
    total = count_result.scalar() or 0

    # Get sessions
    result = await db.execute(
        text(f"""
            SELECT job_id, session_date, total_duration_seconds,
                   coach_talk_percentage, client_talk_percentage,
                   engagement_score, client_sentiment_score,
                   cue_resistance_count + cue_commitment_count + cue_breakthrough_count +
                   cue_concern_count + cue_deflection_count + cue_enthusiasm_count +
                   cue_doubt_count + cue_agreement_count + cue_goal_setting_count as cue_count,
                   quality_warning
            FROM ai_backend.session_analytics
            WHERE {where_clause}
            ORDER BY session_date DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )

    sessions = []
    for row in result.mappings():
        sessions.append(SessionAnalyticsSummary(
            job_id=row["job_id"],
            session_date=row["session_date"],
            duration_minutes=float(row["total_duration_seconds"] or 0) / 60,
            coach_talk_percentage=float(row["coach_talk_percentage"] or 0),
            client_talk_percentage=float(row["client_talk_percentage"] or 0),
            engagement_score=float(row["engagement_score"] or 0),
            client_sentiment_score=float(row["client_sentiment_score"] or 0),
            cue_count=row["cue_count"] or 0,
            has_warnings=row["quality_warning"] or False,
        ))

    return SessionListResponse(sessions=sessions, total=total)


@router.get("/clients/{client_id}/risk", response_model=RiskScoreDetailResponse)
async def get_client_risk(
    client_id: UUID,
    current_user: Annotated[UserContext, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Get current risk score and history."""
    coach_id = UUID(current_user.user_id)

    # Verify relationship
    await require_coach_client_relationship(coach_id, client_id, db)

    # Get latest risk score
    result = await db.execute(
        text("""
            SELECT * FROM ai_backend.risk_scores
            WHERE client_id = :client_id AND coach_id = :coach_id
            ORDER BY computed_at DESC
            LIMIT 1
        """),
        {"client_id": str(client_id), "coach_id": str(coach_id)},
    )
    row = result.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Risk score not found")

    risk_score = RiskScoreResponse(**dict(row))

    # Get history
    history_result = await db.execute(
        text("""
            SELECT computed_at, risk_score, risk_level,
                   (factors->0->>'factor_type') as top_factor
            FROM ai_backend.risk_scores
            WHERE client_id = :client_id AND coach_id = :coach_id
            ORDER BY computed_at DESC
            LIMIT 5
        """),
        {"client_id": str(client_id), "coach_id": str(coach_id)},
    )
    history = [
        RiskScoreHistory(
            computed_at=r["computed_at"],
            risk_score=float(r["risk_score"]),
            risk_level=RiskLevel(r["risk_level"]),
            top_factor=r["top_factor"] or "unknown",
        )
        for r in history_result.mappings()
    ]

    # Get active alerts
    alerts_service = get_alerts_service(db)
    alerts = await alerts_service.get_alerts_for_client(client_id, coach_id)

    return RiskScoreDetailResponse(
        risk_score=risk_score,
        history=history,
        alerts=[RiskAlertResponse(**a.to_dict()) for a in alerts],
    )


@router.get("/clients/{client_id}/trends", response_model=TrendsResponse)
async def get_client_trends(
    client_id: UUID,
    metrics: list[str] = Query(["engagement", "sentiment", "talk_ratio"]),
    window: str = Query("90d", pattern="^(30d|90d|all_time)$"),
    current_user: Annotated[UserContext, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Get trend data for specific metrics."""
    coach_id = UUID(current_user.user_id)

    # Verify relationship
    await require_coach_client_relationship(coach_id, client_id, db)

    # Calculate window
    today = date.today()
    from datetime import timedelta

    if window == "30d":
        from_date = today - timedelta(days=30)
    elif window == "90d":
        from_date = today - timedelta(days=90)
    else:
        from_date = date(2020, 1, 1)

    # Fetch session data
    result = await db.execute(
        text("""
            SELECT id, session_date, engagement_score, client_sentiment_score, client_talk_percentage
            FROM ai_backend.session_analytics
            WHERE client_id = :client_id AND coach_id = :coach_id
            AND session_date >= :from_date
            ORDER BY session_date ASC
        """),
        {
            "client_id": str(client_id),
            "coach_id": str(coach_id),
            "from_date": from_date,
        },
    )
    sessions = [dict(r) for r in result.mappings()]

    trends = {}
    valid_metrics = {"engagement", "sentiment", "talk_ratio"}

    for metric in metrics:
        if metric not in valid_metrics:
            continue

        if metric == "engagement":
            values = [float(s["engagement_score"] or 0) for s in sessions]
            data_points = [
                DataPoint(date=s["session_date"], value=float(s["engagement_score"] or 0), session_id=s["id"])
                for s in sessions
            ]
        elif metric == "sentiment":
            values = [float(s["client_sentiment_score"] or 0) for s in sessions]
            data_points = [
                DataPoint(date=s["session_date"], value=float(s["client_sentiment_score"] or 0), session_id=s["id"])
                for s in sessions
            ]
        else:  # talk_ratio
            values = [float(s["client_talk_percentage"] or 0) for s in sessions]
            data_points = [
                DataPoint(date=s["session_date"], value=float(s["client_talk_percentage"] or 0), session_id=s["id"])
                for s in sessions
            ]

        if len(values) >= 2:
            current_value = values[-1]
            previous_value = values[0]
            change = current_value - previous_value
            change_pct = (change / abs(previous_value) * 100) if previous_value != 0 else 0
            trend = calculate_trend(values, higher_is_better=(metric != "talk_ratio"))
        else:
            current_value = values[-1] if values else 0
            previous_value = current_value
            change = 0
            change_pct = 0
            trend = "stable"

        trends[metric] = TrendData(
            metric_name=metric,
            current_value=round(current_value, 3),
            previous_value=round(previous_value, 3),
            change=round(change, 3),
            change_percentage=round(change_pct, 1),
            trend=TrendDirection(trend),
            data_points=data_points,
        )

    return TrendsResponse(trends=trends)


# ============================================================================
# Risk Alerts Endpoints
# ============================================================================


@router.get("/risk/alerts", response_model=AlertListResponse)
async def get_risk_alerts(
    status: str = Query("pending", pattern="^(pending|acknowledged|all)$"),
    severity: str = Query("all", pattern="^(warning|urgent|all)$"),
    limit: int = Query(50, le=100),
    current_user: Annotated[UserContext, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Get risk alerts for coach."""
    coach_id = UUID(current_user.user_id)

    alerts_service = get_alerts_service(db)
    alerts, total = await alerts_service.get_alerts_for_coach(
        coach_id=coach_id,
        status=status if status != "all" else None,
        severity=severity if severity != "all" else None,
        limit=limit,
    )

    return AlertListResponse(
        alerts=[RiskAlertResponse(**a.to_dict()) for a in alerts],
        total=total,
    )


@router.post("/risk/alerts/{alert_id}/acknowledge", response_model=RiskAlertResponse)
async def acknowledge_alert(
    alert_id: UUID,
    request: AcknowledgeAlertRequest,
    current_user: Annotated[UserContext, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Acknowledge a risk alert."""
    coach_id = UUID(current_user.user_id)

    alerts_service = get_alerts_service(db)
    alert = await alerts_service.acknowledge_alert(
        alert_id=alert_id,
        coach_id=coach_id,
        notes=request.notes,
    )

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return RiskAlertResponse(**alert.to_dict())


# ============================================================================
# Coach Summary & Compute Endpoints
# ============================================================================


@router.post("/compute/{client_id}", status_code=202, response_model=ComputeResponse)
async def trigger_compute(
    client_id: UUID,
    current_user: Annotated[UserContext, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Trigger manual recomputation of client analytics."""
    coach_id = UUID(current_user.user_id)

    # Verify relationship
    await require_coach_client_relationship(coach_id, client_id, db)

    # Import Celery task (deferred to avoid circular import)
    from app.workers.tasks.analytics import compute_client_analytics_for_client

    # Trigger async task
    task = compute_client_analytics_for_client.delay(str(client_id), str(coach_id))

    return ComputeResponse(
        task_id=task.id,
        message="Analytics recomputation queued",
    )


@router.get("/coach/summary", response_model=CoachSummaryResponse)
async def get_coach_summary(
    current_user: Annotated[UserContext, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Get aggregate analytics for coach's clients."""
    coach_id = UUID(current_user.user_id)

    # Get client count
    client_result = await db.execute(
        text("""
            SELECT COUNT(DISTINCT client_id) as total
            FROM ai_backend.session_analytics
            WHERE coach_id = :coach_id
        """),
        {"coach_id": str(coach_id)},
    )
    total_clients = client_result.scalar() or 0

    # Get risk distribution
    risk_result = await db.execute(
        text("""
            SELECT risk_level, COUNT(*) as count
            FROM (
                SELECT DISTINCT ON (client_id) client_id, risk_level
                FROM ai_backend.risk_scores
                WHERE coach_id = :coach_id
                AND valid_until > NOW()
                ORDER BY client_id, computed_at DESC
            ) latest_scores
            GROUP BY risk_level
        """),
        {"coach_id": str(coach_id)},
    )
    risk_counts = {r["risk_level"]: r["count"] for r in risk_result.mappings()}

    risk_distribution = RiskDistribution(
        low=risk_counts.get("low", 0),
        medium=risk_counts.get("medium", 0),
        high=risk_counts.get("high", 0),
        critical=risk_counts.get("critical", 0),
    )

    clients_at_risk = (
        risk_distribution.medium +
        risk_distribution.high +
        risk_distribution.critical
    )

    # Get average engagement
    engagement_result = await db.execute(
        text("""
            SELECT AVG(average_engagement_score) as avg
            FROM ai_backend.client_analytics
            WHERE coach_id = :coach_id
            AND window_type = '90d'
        """),
        {"coach_id": str(coach_id)},
    )
    avg_engagement = engagement_result.scalar()

    # Get sessions this month
    from datetime import timedelta

    first_of_month = date.today().replace(day=1)
    sessions_result = await db.execute(
        text("""
            SELECT COUNT(*) as count
            FROM ai_backend.session_analytics
            WHERE coach_id = :coach_id
            AND session_date >= :first_of_month
        """),
        {"coach_id": str(coach_id), "first_of_month": first_of_month},
    )
    sessions_this_month = sessions_result.scalar() or 0

    return CoachSummaryResponse(
        total_clients=total_clients,
        clients_at_risk=clients_at_risk,
        average_engagement=float(avg_engagement) if avg_engagement else None,
        sessions_this_month=sessions_this_month,
        risk_distribution=risk_distribution,
    )
