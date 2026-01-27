"""Schemas for transcription API."""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TranscriptionStatus(str, Enum):
    """Status of a transcription job."""

    PENDING = "pending"
    UPLOADING = "uploading"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    PARTIAL = "partial"  # Transcript available but summary failed
    FAILED = "failed"


class SessionType(str, Enum):
    """Type of coaching session."""

    IN_PERSON = "in_person"
    VIDEO_CALL = "video_call"
    PHONE_CALL = "phone_call"


class SpeakerLabel(str, Enum):
    """Speaker label types."""

    COACH = "coach"
    CLIENT = "client"
    SPEAKER = "speaker"  # For monologue
    UNKNOWN = "unknown"


class IntentType(str, Enum):
    """Intent classification for utterances."""

    QUESTION_OPEN = "question_open"
    QUESTION_CLOSED = "question_closed"
    REFLECTION = "reflection"
    ADVICE = "advice"
    ENCOURAGEMENT = "encouragement"
    CHALLENGE = "challenge"
    INSTRUCTION = "instruction"
    ACKNOWLEDGMENT = "acknowledgment"
    CONCERN = "concern"
    COMMITMENT = "commitment"
    RESISTANCE = "resistance"
    UPDATE = "update"


class CoachingMomentType(str, Enum):
    """Types of significant coaching moments."""

    BREAKTHROUGH = "breakthrough"
    CONCERN = "concern"
    GOAL_SET = "goal_set"
    COMMITMENT = "commitment"
    RESISTANCE = "resistance"


class SessionTypeDetected(str, Enum):
    """Detected session focus area."""

    NUTRITION = "nutrition"
    TRAINING = "training"
    MINDSET = "mindset"
    ACCOUNTABILITY = "accountability"
    GENERAL = "general"


class SentimentType(str, Enum):
    """Client sentiment classification."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


# ============================================================================
# Request Schemas
# ============================================================================


class TranscriptionUploadRequest(BaseModel):
    """Request schema for audio upload (metadata only, file is multipart)."""

    client_id: UUID
    session_date: date | None = Field(
        None,
        description="Date when the session occurred",
    )
    session_type: SessionType | None = Field(
        None,
        description="Type of session (in_person, video_call, phone_call)",
    )
    session_title: str | None = Field(
        None,
        max_length=255,
        description="Optional title for the session",
    )
    conversation_id: UUID | None = Field(
        None,
        description="Link to existing conversation in MVP",
    )
    webhook_url: str | None = Field(
        None,
        description="Webhook URL to notify on completion",
    )


class SpeakerLabelUpdate(BaseModel):
    """Update speaker label for an utterance."""

    speaker_number: int
    label: SpeakerLabel


class ReprocessRequest(BaseModel):
    """Request to reprocess a transcription job."""

    reprocess_transcript: bool = Field(
        False,
        description="Re-run transcription",
    )
    reprocess_summary: bool = Field(
        True,
        description="Re-run summarization",
    )


# ============================================================================
# Response Schemas - Core Objects
# ============================================================================


class WordResponse(BaseModel):
    """A single word with timing information."""

    word: str
    start: float
    end: float
    confidence: float
    speaker: int | None = None


class UtteranceResponse(BaseModel):
    """A single speaker utterance."""

    id: UUID
    speaker_number: int
    speaker_label: str | None
    speaker_confidence: float | None
    text: str
    start_time: float
    end_time: float
    confidence: float | None
    intent: str | None
    sentiment: str | None
    sequence_number: int

    class Config:
        from_attributes = True


class ActionItemResponse(BaseModel):
    """An action item from the session."""

    description: str
    owner: str  # "coach" or "client"
    priority: str  # "high", "medium", "low"
    due_date: str | None = None


class CoachingMomentResponse(BaseModel):
    """A significant coaching moment."""

    type: str
    timestamp_seconds: float
    description: str
    significance: str
    utterance_id: UUID | None = None


# ============================================================================
# Response Schemas - API Responses
# ============================================================================


class TranscriptionJobResponse(BaseModel):
    """Response schema for a transcription job (minimal)."""

    id: UUID
    client_id: UUID
    coach_id: UUID
    audio_filename: str
    audio_format: str
    audio_size_bytes: int
    audio_duration_seconds: float | None
    status: TranscriptionStatus
    session_date: date | None
    session_type: str | None
    session_title: str | None
    webhook_secret: str | None = Field(
        None,
        description="One-time display of webhook secret (only on creation)",
    )
    created_at: datetime

    class Config:
        from_attributes = True


class TranscriptionStatusResponse(BaseModel):
    """Response schema for transcription job status with progress."""

    id: UUID
    client_id: UUID
    coach_id: UUID
    status: TranscriptionStatus
    progress: int = Field(ge=0, le=100, description="Processing progress percentage")
    audio_duration_seconds: float | None
    started_at: datetime | None
    transcription_completed_at: datetime | None
    summary_completed_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TranscriptResponse(BaseModel):
    """Response schema for full transcript."""

    id: UUID
    job_id: UUID
    full_text: str
    word_count: int
    duration_seconds: float
    confidence_score: float | None
    deepgram_request_id: str | None
    model_used: str | None
    utterances: list[UtteranceResponse]
    created_at: datetime

    class Config:
        from_attributes = True


class SessionSummaryResponse(BaseModel):
    """Response schema for AI-generated session summary."""

    id: UUID
    job_id: UUID
    transcript_id: UUID

    # AI-generated content
    executive_summary: str
    key_topics: list[str]
    client_concerns: list[str]
    coach_recommendations: list[str]
    action_items: list[ActionItemResponse]
    goals_discussed: list[str]
    coaching_moments: list[CoachingMomentResponse]

    # Analysis
    session_type_detected: str | None
    client_sentiment: str | None
    engagement_score: float | None = Field(ge=0.0, le=1.0)

    # Metadata
    llm_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    created_at: datetime

    class Config:
        from_attributes = True


class TranscriptionListResponse(BaseModel):
    """Response schema for listing transcription jobs."""

    jobs: list[TranscriptionJobResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# Webhook Schemas
# ============================================================================


class TranscriptionWebhookPayload(BaseModel):
    """Payload sent to webhook URL on transcription completion."""

    event: str  # "transcription.completed" or "transcription.failed"
    job_id: UUID
    client_id: UUID
    coach_id: UUID
    status: TranscriptionStatus
    transcript_id: UUID | None = None
    summary_status: str | None = None  # "available" or "failed"
    duration_seconds: float | None = None
    word_count: int | None = None
    error_message: str | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None


# ============================================================================
# Internal Schemas
# ============================================================================


class TranscriptionResultInternal(BaseModel):
    """Internal schema for transcription processing results."""

    full_text: str
    word_count: int
    duration_seconds: float
    request_id: str
    model: str
    confidence: float
    utterances: list[dict[str, Any]]
    words: list[dict[str, Any]]


class SummaryResultInternal(BaseModel):
    """Internal schema for summarization results."""

    executive_summary: str
    key_topics: list[str]
    client_concerns: list[str]
    coach_recommendations: list[str]
    action_items: list[dict[str, Any]]
    goals_discussed: list[str]
    coaching_moments: list[dict[str, Any]]
    session_type_detected: str
    client_sentiment: str
    engagement_score: float
