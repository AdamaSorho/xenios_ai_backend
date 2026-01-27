"""Task modules for Celery workers."""

from app.workers.tasks.base import BaseTask
from app.workers.tasks.extraction import process_extraction

__all__ = ["BaseTask", "process_extraction"]
