"""Base task class with common functionality."""

import uuid
from typing import Any

from celery import Task

from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseTask(Task):
    """
    Base task class with built-in logging, error handling, and retry configuration.

    All custom tasks should inherit from this class.
    """

    # Default retry configuration
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_backoff_max = 600  # Max 10 minutes between retries
    retry_jitter = True
    max_retries = 3

    # Don't retry on these exceptions
    dont_autoretry_for = (ValueError, TypeError)

    def before_start(
        self,
        task_id: str,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> None:
        """Called before task execution starts."""
        # Generate correlation ID if not provided
        correlation_id = kwargs.pop("correlation_id", None) or str(uuid.uuid4())

        logger.info(
            "Task starting",
            task_id=task_id,
            task_name=self.name,
            correlation_id=correlation_id,
            retry_count=self.request.retries,
        )

    def after_return(
        self,
        status: str,
        retval: Any,
        task_id: str,
        args: tuple,
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """Called after task returns (success or failure)."""
        logger.info(
            "Task completed",
            task_id=task_id,
            task_name=self.name,
            status=status,
        )

    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> None:
        """Called on successful task completion."""
        logger.debug(
            "Task succeeded",
            task_id=task_id,
            task_name=self.name,
        )

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """Called on task failure."""
        logger.error(
            "Task failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            exc_info=einfo,
        )

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """Called when task is being retried."""
        logger.warning(
            "Task retrying",
            task_id=task_id,
            task_name=self.name,
            retry_count=self.request.retries,
            error=str(exc),
        )
