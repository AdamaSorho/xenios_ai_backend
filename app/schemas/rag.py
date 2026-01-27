"""Schemas for RAG API endpoints (Spec 0004)."""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class EmbeddingSourceType(str, Enum):
    """Types of data that can be embedded for RAG."""

    HEALTH_PROFILE = "health_profile"
    HEALTH_METRIC_SUMMARY = "health_metric_summary"
    HEALTH_GOAL = "health_goal"
    LAB_RESULT = "lab_result"
    SESSION_SUMMARY = "session_summary"
    CHECKIN_SUMMARY = "checkin_summary"
    MESSAGE_THREAD = "message_thread"


class InsightTrigger(str, Enum):
    """Triggers for insight generation."""

    SCHEDULED = "scheduled"
    METRIC_CHANGE = "metric_change"
    GOAL_PROGRESS = "goal_progress"
    CHECKIN_SUBMITTED = "checkin_submitted"
    SESSION_COMPLETED = "session_completed"


# ============================================================================
# Embedding Schemas
# ============================================================================


class EmbeddingUpdateRequest(BaseModel):
    """Request to update embeddings for a client."""

    client_id: UUID = Field(..., description="Client ID to update embeddings for")
    source_types: list[EmbeddingSourceType] | None = Field(
        default=None,
        description="Specific source types to update. If None, updates all types.",
    )
    force: bool = Field(
        default=False,
        description="Force update even if content hash unchanged",
    )


class EmbeddingUpdateResult(BaseModel):
    """Result of embedding update operation."""

    updated_count: int = Field(..., description="Number of embeddings created/updated")
    skipped_count: int = Field(..., description="Number of embeddings skipped (unchanged)")


class EmbeddingSearchRequest(BaseModel):
    """Request to search embeddings."""

    client_id: UUID = Field(..., description="Client ID to search embeddings for")
    query: str = Field(..., description="Query text to find similar embeddings", min_length=1)
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results to return")
    source_types: list[EmbeddingSourceType] | None = Field(
        default=None,
        description="Filter by source types",
    )


class SearchResult(BaseModel):
    """A single embedding search result."""

    source_type: str = Field(..., description="Type of source (e.g., health_profile)")
    source_id: str = Field(..., description="Deterministic source identifier")
    content: str = Field(..., description="The embedded content text")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbeddingSearchResponse(BaseModel):
    """Response from embedding search."""

    results: list[SearchResult] = Field(..., description="Search results ordered by relevance")
    query: str = Field(..., description="Original query")


class BatchEmbeddingRequest(BaseModel):
    """Request to batch update embeddings for multiple clients."""

    client_ids: list[UUID] = Field(..., min_length=1, description="Client IDs to update")
    force: bool = Field(default=True, description="Force update all embeddings")


class BatchEmbeddingResponse(BaseModel):
    """Response from batch embedding update."""

    queued: int = Field(..., description="Number of clients queued for update")


# ============================================================================
# Chat Schemas
# ============================================================================


class ChatRequest(BaseModel):
    """Request for grounded chat completion."""

    client_id: UUID = Field(..., description="Client ID to chat about")
    message: str = Field(..., description="User message to respond to", min_length=1)
    conversation_id: UUID | None = Field(
        default=None,
        description="Conversation ID for context continuity. Generated if not provided.",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source citations in response",
    )
    max_context_items: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum context items to include",
    )


class SourceCitation(BaseModel):
    """Citation for a source used in the response."""

    source_type: str = Field(..., description="Type of source")
    source_id: str = Field(..., description="Source identifier")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to query")
    snippet: str = Field(..., description="Relevant excerpt from the source")
    date: date | None = Field(default=None, description="Date associated with the source")


class ChatResponse(BaseModel):
    """Response from grounded chat completion."""

    response: str = Field(..., description="Generated response text")
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Sources cited in the response",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score based on context relevance",
    )
    has_context: bool = Field(
        ...,
        description="Whether relevant context was found for grounding",
    )
    conversation_id: UUID = Field(..., description="Conversation ID for continuity")
    tokens_used: int = Field(..., description="Total tokens used in generation")


class StreamChunk(BaseModel):
    """A chunk in the streaming response."""

    type: str = Field(..., description="Chunk type: 'chunk', 'done', or 'error'")
    content: str | None = Field(default=None, description="Text content for chunk type")
    sources: list[SourceCitation] | None = Field(default=None, description="Sources for done type")
    confidence: float | None = Field(default=None, description="Confidence for done type")
    has_context: bool | None = Field(default=None, description="Context flag for done type")
    conversation_id: UUID | None = Field(default=None, description="Conversation ID for done type")
    tokens_used: int | None = Field(default=None, description="Tokens used for done type")
    code: str | None = Field(default=None, description="Error code for error type")
    message: str | None = Field(default=None, description="Error message for error type")


# ============================================================================
# Insight Schemas
# ============================================================================


class InsightGenerationRequest(BaseModel):
    """Request to generate an insight for a client."""

    client_id: UUID = Field(..., description="Client ID to generate insight for")
    trigger: InsightTrigger = Field(..., description="What triggered the insight generation")
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context about the trigger (e.g., metric changes)",
    )


class InsightGenerationResponse(BaseModel):
    """Response from insight generation."""

    insight_id: UUID | None = Field(
        default=None,
        description="ID of generated insight in MVP insights table",
    )
    title: str | None = Field(default=None, description="Insight title")
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in the insight",
    )
    status: str = Field(..., description="Status: 'generated', 'duplicate', or 'rate_limited'")


class GeneratedInsight(BaseModel):
    """A generated insight for coach review."""

    id: UUID = Field(..., description="Insight ID")
    client_id: UUID = Field(..., description="Client ID")
    coach_id: UUID = Field(..., description="Coach ID")
    title: str = Field(..., description="Insight title")
    client_message: str = Field(..., description="Message to show the client")
    rationale: str = Field(..., description="Why this insight matters (for coach)")
    suggested_actions: list[str] = Field(..., description="Recommended actions")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    triggering_data: dict[str, Any] = Field(..., description="Data that triggered the insight")
    insight_type: str = Field(..., description="Type: nutrition, training, recovery, etc.")
    expires_at: datetime = Field(..., description="When the insight expires")
    created_at: datetime | None = Field(default=None, description="When created")


class PendingInsightsResponse(BaseModel):
    """Response listing pending insights for a coach."""

    insights: list[GeneratedInsight] = Field(..., description="Pending insights")
    total: int = Field(..., description="Total pending insights count")
