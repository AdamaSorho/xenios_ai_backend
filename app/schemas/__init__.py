"""Pydantic schemas for API request/response validation."""

from app.schemas.common import ServiceStatus
from app.schemas.extraction import (
    DocumentType,
    ExtractionJobResponse,
    ExtractionListResponse,
    ExtractionStatus,
    ExtractionStatusResponse,
    ExtractionUploadRequest,
)
from app.schemas.health import HealthResponse, ReadinessResponse
from app.schemas.rag import (
    ChatRequest,
    ChatResponse,
    EmbeddingSearchRequest,
    EmbeddingSearchResponse,
    EmbeddingSourceType,
    EmbeddingUpdateRequest,
    EmbeddingUpdateResult,
    GeneratedInsight,
    InsightGenerationRequest,
    InsightGenerationResponse,
    InsightTrigger,
    SearchResult,
    SourceCitation,
)

__all__ = [
    # Common
    "ServiceStatus",
    # Extraction
    "DocumentType",
    "ExtractionJobResponse",
    "ExtractionListResponse",
    "ExtractionStatus",
    "ExtractionStatusResponse",
    "ExtractionUploadRequest",
    # Health
    "HealthResponse",
    "ReadinessResponse",
    # RAG (Spec 0004)
    "ChatRequest",
    "ChatResponse",
    "EmbeddingSearchRequest",
    "EmbeddingSearchResponse",
    "EmbeddingSourceType",
    "EmbeddingUpdateRequest",
    "EmbeddingUpdateResult",
    "GeneratedInsight",
    "InsightGenerationRequest",
    "InsightGenerationResponse",
    "InsightTrigger",
    "SearchResult",
    "SourceCitation",
]
