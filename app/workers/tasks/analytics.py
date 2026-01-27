"""Celery tasks for coaching analytics computation."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from app.config import get_settings
from app.core.logging import get_logger
from app.workers.celery_app import celery_app
from app.workers.tasks.base import BaseTask

logger = get_logger(__name__)
settings = get_settings()


def get_sync_db():
    """Get synchronous database connection for Celery tasks."""
    import asyncpg

    async def _get_connection():
        return await asyncpg.connect(settings.database_url)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_get_connection())
    finally:
        loop.close()


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def get_async_db_session():
    """Get async SQLAlchemy session for service calls."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine(
        settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
        echo=False,
    )
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return async_session()


@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="analytics",
    max_retries=3,
    soft_time_limit=300,  # 5 minutes
    time_limit=360,  # 6 minutes hard limit
    autoretry_for=(Exception,),
    retry_backoff=30,
    retry_backoff_max=180,
    retry_jitter=True,
)
def compute_session_analytics(self, job_id: str) -> dict[str, Any]:
    """
    Compute analytics for a completed transcription.

    Triggered when transcription job status -> completed.

    Flow:
    1. Compute basic session metrics (talk-time, engagement, etc.)
    2. Run cue detection (LLM-based)
    3. Update session analytics with cue counts
    4. Save results

    Args:
        job_id: UUID string of the transcription job

    Returns:
        Dict with success status and analytics details
    """
    logger.info("Starting session analytics computation", job_id=job_id)

    return run_async(_compute_session_analytics_async(self, job_id))


async def _compute_session_analytics_async(task, job_id: str) -> dict[str, Any]:
    """Async implementation of session analytics computation."""
    from app.services.analytics.cue_detection import CueDetectionError, get_cue_detection_service
    from app.services.analytics.session_analytics import SessionAnalyticsService

    db = await get_async_db_session()

    try:
        session_service = SessionAnalyticsService(db)
        cue_service = get_cue_detection_service()

        # 1. Compute basic metrics (no LLM, fast)
        session_analytics, utterances = await session_service.compute_for_job(
            UUID(job_id)
        )

        # 2. Attempt cue detection (async LLM calls, may fail)
        try:
            cues, cue_counts = await cue_service.detect_cues(
                utterances, session_analytics.id
            )

            # Save cues
            for cue in cues:
                db.add(cue)

            # Update analytics with cue counts
            await session_service.update_cue_counts(session_analytics, cue_counts)

            logger.info(
                "Cue detection completed",
                job_id=job_id,
                cue_count=len(cues),
            )

        except CueDetectionError as e:
            logger.warning(
                "cue_detection_failed",
                job_id=job_id,
                error=str(e),
            )
            # Continue without cues - analytics still valid
            if session_analytics.quality_warnings is None:
                session_analytics.quality_warnings = []
            session_analytics.quality_warnings.append("cue_detection_failed")
            session_analytics.quality_warning = True

        # 3. Commit all changes
        await db.commit()

        logger.info(
            "Session analytics computed",
            job_id=job_id,
            analytics_id=str(session_analytics.id),
            engagement_score=float(session_analytics.engagement_score)
            if session_analytics.engagement_score
            else None,
        )

        return {
            "success": True,
            "job_id": job_id,
            "analytics_id": str(session_analytics.id),
            "engagement_score": float(session_analytics.engagement_score)
            if session_analytics.engagement_score
            else None,
            "quality_warning": session_analytics.quality_warning,
        }

    except Exception as e:
        await db.rollback()
        logger.error(
            "Session analytics computation failed",
            job_id=job_id,
            error=str(e),
        )
        raise

    finally:
        await db.close()


@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="analytics",
    max_retries=2,
    soft_time_limit=3600,  # 1 hour
    time_limit=3900,  # 65 minutes hard limit
)
def compute_client_analytics_batch(self) -> dict[str, Any]:
    """
    Daily batch job to compute aggregate client analytics.

    Scheduled via Celery Beat at 2:00 AM.

    For each active client-coach relationship:
    1. Compute 30d window analytics
    2. Compute 90d window analytics
    """
    logger.info("Starting client analytics batch computation")

    return run_async(_compute_client_analytics_batch_async(self))


async def _compute_client_analytics_batch_async(task) -> dict[str, Any]:
    """Async implementation of client analytics batch computation."""
    from app.services.analytics.client_analytics import ClientAnalyticsService

    db = await get_async_db_session()

    processed = 0
    errors = []

    try:
        # Get all active client-coach relationships from session_analytics
        # (clients with at least one session)
        result = await db.execute(
            """
            SELECT DISTINCT client_id, coach_id
            FROM ai_backend.session_analytics
            WHERE session_date >= NOW() - INTERVAL '365 days'
            """
        )
        relationships = [dict(row) for row in result.mappings()]

        logger.info(
            "Found client-coach relationships",
            count=len(relationships),
        )

        service = ClientAnalyticsService(db)

        for rel in relationships:
            client_id = rel["client_id"]
            coach_id = rel["coach_id"]

            try:
                # Compute 30d window
                analytics_30d = await service.compute_for_client(
                    client_id=client_id,
                    coach_id=coach_id,
                    window_type="30d",
                )
                await service.upsert(analytics_30d)

                # Compute 90d window
                analytics_90d = await service.compute_for_client(
                    client_id=client_id,
                    coach_id=coach_id,
                    window_type="90d",
                )
                await service.upsert(analytics_90d)

                processed += 1

            except Exception as e:
                logger.warning(
                    "Client analytics computation failed",
                    client_id=str(client_id),
                    coach_id=str(coach_id),
                    error=str(e),
                )
                errors.append(
                    {
                        "client_id": str(client_id),
                        "coach_id": str(coach_id),
                        "error": str(e),
                    }
                )

        await db.commit()

        logger.info(
            "Client analytics batch completed",
            processed=processed,
            errors=len(errors),
        )

        return {
            "success": True,
            "processed": processed,
            "errors": errors[:10],  # Only first 10 errors
            "total_errors": len(errors),
        }

    except Exception as e:
        await db.rollback()
        logger.error(
            "Client analytics batch failed",
            error=str(e),
        )
        raise

    finally:
        await db.close()


@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="analytics",
    max_retries=2,
    soft_time_limit=3600,  # 1 hour
    time_limit=3900,  # 65 minutes hard limit
)
def compute_risk_scores_batch(self) -> dict[str, Any]:
    """
    Daily batch job to compute risk scores.

    Scheduled via Celery Beat at 3:00 AM.

    For each active client:
    1. Get latest ClientAnalytics
    2. Compute risk score
    3. Generate alerts if needed
    """
    logger.info("Starting risk scores batch computation")

    return run_async(_compute_risk_scores_batch_async(self))


async def _compute_risk_scores_batch_async(task) -> dict[str, Any]:
    """Async implementation of risk scores batch computation."""
    from sqlalchemy import text

    from app.services.analytics.alerts import AlertsService
    from app.services.analytics.client_analytics import ClientAnalyticsService
    from app.services.analytics.risk_scoring import RiskScoringService

    db = await get_async_db_session()

    processed = 0
    alerts_generated = 0
    errors = []

    try:
        # Get all clients with recent analytics
        result = await db.execute(
            text("""
                SELECT DISTINCT client_id, coach_id
                FROM ai_backend.client_analytics
                WHERE computed_at >= NOW() - INTERVAL '7 days'
            """)
        )
        clients = [dict(row) for row in result.mappings()]

        logger.info(
            "Found clients with recent analytics",
            count=len(clients),
        )

        analytics_service = ClientAnalyticsService(db)
        risk_service = RiskScoringService(db)
        alerts_service = AlertsService(db)

        for client in clients:
            client_id = client["client_id"]
            coach_id = client["coach_id"]

            try:
                # Get 90d analytics (primary window for risk)
                analytics = await analytics_service.get_for_client(
                    client_id=client_id,
                    coach_id=coach_id,
                    window_type="90d",
                )

                if not analytics:
                    continue

                # Get previous score for comparison
                previous_score = await risk_service.get_latest(client_id, coach_id)

                # Compute new risk score
                risk_score = await risk_service.compute_risk_score(
                    client_id=client_id,
                    coach_id=coach_id,
                    client_analytics=analytics,
                )
                db.add(risk_score)
                await db.flush()  # Get ID for alerts

                # Generate alerts if needed
                alerts = await alerts_service.generate_alerts(
                    client_id=client_id,
                    coach_id=coach_id,
                    risk_score=risk_score,
                    previous_score=previous_score,
                )

                for alert in alerts:
                    db.add(alert)
                    alerts_generated += 1

                processed += 1

            except Exception as e:
                logger.warning(
                    "Risk score computation failed",
                    client_id=str(client_id),
                    coach_id=str(coach_id),
                    error=str(e),
                )
                errors.append(
                    {
                        "client_id": str(client_id),
                        "coach_id": str(coach_id),
                        "error": str(e),
                    }
                )

        await db.commit()

        logger.info(
            "Risk scores batch completed",
            processed=processed,
            alerts_generated=alerts_generated,
            errors=len(errors),
        )

        return {
            "success": True,
            "processed": processed,
            "alerts_generated": alerts_generated,
            "errors": errors[:10],
            "total_errors": len(errors),
        }

    except Exception as e:
        await db.rollback()
        logger.error(
            "Risk scores batch failed",
            error=str(e),
        )
        raise

    finally:
        await db.close()


@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="analytics",
    max_retries=2,
    soft_time_limit=1800,  # 30 minutes
    time_limit=2100,
)
def generate_risk_alerts(self) -> dict[str, Any]:
    """
    Generate alerts for high-risk clients.

    Scheduled via Celery Beat at 3:30 AM (after risk scores).

    Checks for:
    - New high/critical risk level
    - Significant risk increase (>20 points)
    - No session in 30+ days
    """
    logger.info("Starting risk alerts generation")

    return run_async(_generate_risk_alerts_async(self))


async def _generate_risk_alerts_async(task) -> dict[str, Any]:
    """Async implementation of risk alerts generation."""
    from sqlalchemy import text

    from app.services.analytics.alerts import AlertsService
    from app.services.analytics.risk_scoring import RiskScoringService

    db = await get_async_db_session()

    alerts_generated = 0
    errors = []

    try:
        # Get all risk scores computed today
        today = datetime.now(timezone.utc).date()
        result = await db.execute(
            text("""
                SELECT rs.id, rs.client_id, rs.coach_id, rs.risk_score,
                       rs.risk_level, rs.score_change
                FROM ai_backend.risk_scores rs
                WHERE DATE(rs.computed_at) = :today
            """),
            {"today": today},
        )
        scores = [dict(row) for row in result.mappings()]

        logger.info(
            "Found risk scores computed today",
            count=len(scores),
        )

        risk_service = RiskScoringService(db)
        alerts_service = AlertsService(db)

        for score_data in scores:
            try:
                # Fetch full risk score model
                from app.models.analytics import RiskScore

                result = await db.execute(
                    text("SELECT * FROM ai_backend.risk_scores WHERE id = :id"),
                    {"id": str(score_data["id"])},
                )
                row = result.mappings().first()
                if not row:
                    continue

                risk_score = RiskScore(**dict(row))

                # Get previous score for comparison
                previous_score = await risk_service._get_previous_score(
                    client_id=risk_score.client_id,
                    coach_id=risk_score.coach_id,
                )

                # Convert previous score data to RiskScore model if exists
                previous_risk_model = None
                if previous_score:
                    prev_result = await db.execute(
                        text("""
                            SELECT * FROM ai_backend.risk_scores
                            WHERE client_id = :client_id
                            AND coach_id = :coach_id
                            AND id != :current_id
                            ORDER BY computed_at DESC
                            LIMIT 1
                        """),
                        {
                            "client_id": str(risk_score.client_id),
                            "coach_id": str(risk_score.coach_id),
                            "current_id": str(risk_score.id),
                        },
                    )
                    prev_row = prev_result.mappings().first()
                    if prev_row:
                        previous_risk_model = RiskScore(**dict(prev_row))

                # Generate alerts
                alerts = await alerts_service.generate_alerts(
                    client_id=risk_score.client_id,
                    coach_id=risk_score.coach_id,
                    risk_score=risk_score,
                    previous_score=previous_risk_model,
                )

                for alert in alerts:
                    db.add(alert)
                    alerts_generated += 1

            except Exception as e:
                logger.warning(
                    "Alert generation failed for score",
                    score_id=str(score_data["id"]),
                    error=str(e),
                )
                errors.append(
                    {
                        "score_id": str(score_data["id"]),
                        "error": str(e),
                    }
                )

        await db.commit()

        logger.info(
            "Risk alerts generation completed",
            alerts_generated=alerts_generated,
            errors=len(errors),
        )

        return {
            "success": True,
            "alerts_generated": alerts_generated,
            "errors": errors[:10],
            "total_errors": len(errors),
        }

    except Exception as e:
        await db.rollback()
        logger.error(
            "Risk alerts generation failed",
            error=str(e),
        )
        raise

    finally:
        await db.close()


@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="analytics",
    max_retries=1,
    soft_time_limit=300,  # 5 minutes
    time_limit=360,
)
def cleanup_expired_risk_scores(self) -> dict[str, Any]:
    """
    Delete risk scores past valid_until.

    Runs daily at 4 AM.
    """
    logger.info("Starting expired risk scores cleanup")

    conn = get_sync_db()

    try:
        # Delete expired risk scores
        result = run_async(
            conn.execute(
                """
                DELETE FROM ai_backend.risk_scores
                WHERE valid_until < NOW()
                """
            )
        )

        deleted = result.split()[-1] if isinstance(result, str) else 0

        logger.info(
            "Expired risk scores cleanup completed",
            deleted=deleted,
        )

        return {
            "success": True,
            "deleted": deleted,
        }

    except Exception as e:
        logger.error(
            "Expired risk scores cleanup failed",
            error=str(e),
        )
        raise

    finally:
        run_async(conn.close())


@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="analytics",
    max_retries=1,
    soft_time_limit=7200,  # 2 hours
    time_limit=7500,
)
def archive_old_analytics(self) -> dict[str, Any]:
    """
    Archive/delete analytics older than retention period.

    Runs monthly (1st of each month at 5 AM).

    Retention per spec:
    - Session Analytics: 2 years (cascade deletes cues)
    - Client Analytics: 1 year
    - Risk Scores: 90 days (handled by cleanup_expired_risk_scores)
    - Risk Alerts: 1 year (archive resolved alerts)
    """
    logger.info("Starting old analytics archive/cleanup")

    conn = get_sync_db()

    try:
        now = datetime.now(timezone.utc)

        # Session Analytics: delete older than 2 years
        two_years_ago = now - timedelta(days=730)
        session_result = run_async(
            conn.execute(
                """
                DELETE FROM ai_backend.session_analytics
                WHERE session_date < $1
                """,
                two_years_ago.date(),
            )
        )

        # Client Analytics: delete older than 1 year
        one_year_ago = now - timedelta(days=365)
        client_result = run_async(
            conn.execute(
                """
                DELETE FROM ai_backend.client_analytics
                WHERE window_end < $1
                """,
                one_year_ago.date(),
            )
        )

        # Risk Alerts: delete resolved alerts older than 1 year
        alerts_result = run_async(
            conn.execute(
                """
                DELETE FROM ai_backend.risk_alerts
                WHERE status = 'acknowledged'
                AND acknowledged_at < $1
                """,
                one_year_ago,
            )
        )

        logger.info(
            "Old analytics archive/cleanup completed",
            session_analytics_deleted=session_result,
            client_analytics_deleted=client_result,
            alerts_deleted=alerts_result,
        )

        return {
            "success": True,
            "session_analytics_deleted": str(session_result),
            "client_analytics_deleted": str(client_result),
            "alerts_deleted": str(alerts_result),
        }

    except Exception as e:
        logger.error(
            "Old analytics archive/cleanup failed",
            error=str(e),
        )
        raise

    finally:
        run_async(conn.close())
