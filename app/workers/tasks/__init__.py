"""Task modules for Celery workers."""

from app.workers.tasks.base import BaseTask
from app.workers.tasks.extraction import process_extraction
from app.workers.tasks.rag import (
    batch_generate_insights,
    batch_update_embeddings,
    generate_client_insight,
    update_client_embeddings,
    update_embeddings_on_data_change,
)

__all__ = [
    "BaseTask",
    "process_extraction",
    # RAG tasks (Spec 0004)
    "update_client_embeddings",
    "generate_client_insight",
    "batch_update_embeddings",
    "batch_generate_insights",
    "update_embeddings_on_data_change",
]
