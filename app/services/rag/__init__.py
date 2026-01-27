"""RAG (Retrieval-Augmented Generation) services for Spec 0004."""

from app.services.rag.chat import ChatService
from app.services.rag.embeddings import EmbeddingService
from app.services.rag.insights import InsightGenerationService
from app.services.rag.openai_client import OpenAIEmbeddingClient
from app.services.rag.retrieval import RetrievalService

__all__ = [
    "OpenAIEmbeddingClient",
    "EmbeddingService",
    "RetrievalService",
    "ChatService",
    "InsightGenerationService",
]
