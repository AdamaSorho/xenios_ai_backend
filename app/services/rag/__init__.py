"""RAG (Retrieval-Augmented Generation) services for Spec 0004."""

from app.services.rag.embeddings import EmbeddingService
from app.services.rag.openai_client import OpenAIEmbeddingClient

__all__ = ["OpenAIEmbeddingClient", "EmbeddingService"]
