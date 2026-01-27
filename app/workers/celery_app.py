"""Celery application configuration."""

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery("xenios_ai")

# Configure Celery
celery_app.conf.update(
    # Broker and backend
    broker_url=settings.redis_url,
    result_backend=settings.redis_url,
    # Task routing - route tasks to specific queues based on task name
    task_routes={
        "app.workers.tasks.transcription.*": {"queue": "transcription"},
        "app.workers.tasks.extraction.*": {"queue": "extraction"},
        "app.workers.tasks.llm.*": {"queue": "llm"},
        "app.workers.tasks.analytics.*": {"queue": "analytics"},
    },
    # Default queue for unrouted tasks
    task_default_queue="default",
    # Retry policy - acknowledge tasks after completion (late ack)
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Serialization - use JSON for safety and interoperability
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Time limits
    task_soft_time_limit=300,  # 5 minutes soft limit (raises SoftTimeLimitExceeded)
    task_time_limit=600,  # 10 minutes hard limit (kills the task)
    # Result expiration
    result_expires=3600,  # 1 hour
    # Worker settings
    worker_prefetch_multiplier=1,  # Don't prefetch too many tasks
    worker_concurrency=4,  # Default concurrency
    # Beat scheduler (for periodic tasks)
    beat_scheduler="celery.beat:PersistentScheduler",
    # Timezone
    timezone="UTC",
    enable_utc=True,
)

# Auto-discover tasks in the app.workers.tasks package
celery_app.autodiscover_tasks(["app.workers.tasks"])

# Celery Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # Analytics batch jobs (Spec 0005)
    "compute-client-analytics-daily": {
        "task": "app.workers.tasks.analytics.compute_client_analytics_batch",
        "schedule": crontab(hour=2, minute=0),  # 2:00 AM UTC daily
    },
    "compute-risk-scores-daily": {
        "task": "app.workers.tasks.analytics.compute_risk_scores_batch",
        "schedule": crontab(hour=3, minute=0),  # 3:00 AM UTC daily
    },
    "generate-risk-alerts-daily": {
        "task": "app.workers.tasks.analytics.generate_risk_alerts",
        "schedule": crontab(hour=3, minute=30),  # 3:30 AM UTC daily
    },
    "cleanup-expired-risk-scores": {
        "task": "app.workers.tasks.analytics.cleanup_expired_risk_scores",
        "schedule": crontab(hour=4, minute=0),  # 4:00 AM UTC daily
    },
    "archive-old-analytics-monthly": {
        "task": "app.workers.tasks.analytics.archive_old_analytics",
        "schedule": crontab(day_of_month=1, hour=5, minute=0),  # 1st of month at 5 AM
    },
}
