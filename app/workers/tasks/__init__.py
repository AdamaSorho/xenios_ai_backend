"""Task modules for Celery workers."""

from app.workers.tasks.base import BaseTask

__all__ = ["BaseTask"]
