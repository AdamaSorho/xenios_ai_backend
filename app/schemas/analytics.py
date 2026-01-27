"""Pydantic schemas for coaching analytics API."""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class CueType(str, Enum):
    """Types of language cues detected in sessions."""

    RESISTANCE = "resistance"
    COMMITMENT = "commitment"
    BREAKTHROUGH = "breakthrough"
    CONCERN = "concern"
    DEFLECTION = "deflection"
    ENTHUSIASM = "enthusiasm"
    DOUBT = "doubt"
    AGREEMENT = "agreement"
    GOAL_SETTING = "goal_setting"


class RiskLevel(str, Enum):
    """Risk level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Status of a risk alert."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertSeverity(str, Enum):
    """Severity of a risk alert."""

    WARNING = "warning"
    URGENT = "urgent"


class AlertType(str, Enum):
    """Types of risk alerts."""

    NEW_HIGH_RISK = "new_high_risk"
    RISK_INCREASED = "risk_increased"
    NO_SESSION_30D = "no_session_30d"


class TrendDirection(str, Enum):
    """Direction of a trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class WindowType(str, Enum):
    """Time window types for analytics."""

    DAYS_30 = "30d"
    DAYS_90 = "90d"
    ALL_TIME = "all_time"


class QualityWarning(str, Enum):
    """Types of quality warnings."""

    LOW_CONFIDENCE_DIARIZATION = "low_confidence_diarization"
    INSUFFICIENT_SESSIONS = "insufficient_sessions"
    MISSING_SENTIMENT = "missing_sentiment"
    SHORT_SESSION = "short_session"
    STALE_RISK_SCORE = "stale_risk_score"
    CUE_DETECTION_FAILED = "cue_detection_failed"


# ============================================================================
# Base Response Schemas
# ============================================================================


class LanguageCueResponse(BaseModel):
    """Individual language cue with context."""

    id: UUID
    utterance_id: UUID
    cue_type: CueType
    confidence: float = Field(ge=0.0, le=1.0)
    text_excerpt: str = Field(max_length=200)
    timestamp_seconds: float
    preceding_context: str | None = None
    interpretation: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class RiskFactorResponse(BaseModel):
    """Individual risk factor with contribution details."""

    factor_type: str
    contribution: float = Field(ge=0.0, le=100.0)
    value: float
    threshold: float
    description: str


class DataPoint(BaseModel):
    """Single data point in a trend."""

    date: date
    value: float
    session_id: UUID | None = None


class TrendData(BaseModel):
    """Trend data for a single metric."""

    metric_name: str
    current_value: float
    previous_value: float
    change: float
    change_percentage: float
    trend: TrendDirection
    data_points: list[DataPoint]


class SessionComparison(BaseModel):
    """Comparison between current and previous session."""

    previous_session_date: date
    engagement_change: float  # +/- percentage points
    sentiment_change: float  # +/- on -1 to 1 scale
    talk_ratio_change: float  # +/- percentage points (client)
    notable_changes: list[str]  # Human-readable descriptions


class RiskScoreHistory(BaseModel):
    """Historical risk score entry for trend display."""

    computed_at: datetime
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    top_factor: str


# ============================================================================
# Session Analytics Schemas
# ============================================================================


class SessionAnalyticsResponse(BaseModel):
    """Full session analytics data."""

    id: UUID
    job_id: UUID
    client_id: UUID
    coach_id: UUID
    session_date: date

    # Talk-time metrics
    total_duration_seconds: float
    coach_talk_time_seconds: float
    client_talk_time_seconds: float
    silence_time_seconds: float
    coach_talk_percentage: float = Field(ge=0.0, le=100.0)
    client_talk_percentage: float = Field(ge=0.0, le=100.0)

    # Turn-taking
    total_turns: int
    coach_turns: int
    client_turns: int
    average_turn_duration_coach: float | None
    average_turn_duration_client: float | None
    interruption_count: int

    # Coaching style
    coach_question_count: int
    coach_statement_count: int
    question_to_statement_ratio: float | None
    open_question_count: int
    closed_question_count: int

    # Language cue counts
    cue_resistance_count: int
    cue_commitment_count: int
    cue_breakthrough_count: int
    cue_concern_count: int
    cue_deflection_count: int
    cue_enthusiasm_count: int
    cue_doubt_count: int
    cue_agreement_count: int
    cue_goal_setting_count: int

    # Sentiment
    client_sentiment_score: float | None = Field(ge=-1.0, le=1.0, default=None)
    coach_sentiment_score: float | None = Field(ge=-1.0, le=1.0, default=None)
    sentiment_variance: float | None

    # Engagement
    engagement_score: float | None = Field(ge=0.0, le=100.0, default=None)
    response_elaboration_score: float | None = Field(ge=0.0, le=100.0, default=None)

    # Quality flags
    quality_warning: bool
    quality_warnings: list[str]

    # Metadata
    computed_at: datetime
    model_version: str

    class Config:
        from_attributes = True


class SessionAnalyticsSummary(BaseModel):
    """Compact view for session list endpoints."""

    job_id: UUID
    session_date: date
    duration_minutes: float
    coach_talk_percentage: float = Field(ge=0.0, le=100.0)
    client_talk_percentage: float = Field(ge=0.0, le=100.0)
    engagement_score: float = Field(ge=0.0, le=100.0)
    client_sentiment_score: float = Field(ge=-1.0, le=1.0)
    cue_count: int  # Total cues detected
    has_warnings: bool  # True if quality_warning is True

    class Config:
        from_attributes = True


class SessionAnalyticsDetailResponse(BaseModel):
    """GET /sessions/{job_id} response with full details."""

    session_analytics: SessionAnalyticsResponse
    cues: list[LanguageCueResponse]
    comparison: SessionComparison | None = None


# ============================================================================
# Client Analytics Schemas
# ============================================================================


class ClientAnalyticsResponse(BaseModel):
    """Aggregate client analytics for a time window."""

    id: UUID
    client_id: UUID
    coach_id: UUID

    # Time window
    window_start: date
    window_end: date
    window_type: WindowType

    # Session frequency
    total_sessions: int
    sessions_last_30_days: int
    average_days_between_sessions: float | None
    days_since_last_session: int | None
    session_frequency_trend: TrendDirection | None

    # Talk-time trends
    average_coach_talk_percentage: float | None
    average_client_talk_percentage: float | None
    talk_ratio_trend: str | None

    # Engagement trends
    average_engagement_score: float | None = Field(ge=0.0, le=100.0, default=None)
    engagement_trend: TrendDirection | None
    engagement_scores_history: list[float]

    # Sentiment trends
    average_sentiment_score: float | None = Field(ge=-1.0, le=1.0, default=None)
    sentiment_trend: TrendDirection | None
    sentiment_scores_history: list[float]

    # Cue patterns
    total_resistance_cues: int
    total_commitment_cues: int
    total_breakthrough_cues: int
    resistance_to_commitment_ratio: float | None

    computed_at: datetime

    class Config:
        from_attributes = True


class ClientAnalyticsSummaryResponse(BaseModel):
    """GET /clients/{id}/summary response."""

    client_analytics: ClientAnalyticsResponse
    session_count: int
    latest_session_date: date | None
    risk_level: RiskLevel | None  # Null if no valid risk score
    risk_score_stale: bool  # True if risk score > 7 days old
    quality_warnings: list[str]


# ============================================================================
# Risk Score Schemas
# ============================================================================


class RiskScoreResponse(BaseModel):
    """Current risk score with factors."""

    id: UUID
    client_id: UUID
    coach_id: UUID

    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    churn_probability: float = Field(ge=0.0, le=1.0)

    factors: list[RiskFactorResponse]

    previous_risk_score: float | None = Field(ge=0.0, le=100.0, default=None)
    score_change: float | None
    trend: TrendDirection | None

    recommended_action: str | None

    computed_at: datetime
    valid_until: datetime
    model_version: str

    class Config:
        from_attributes = True


class RiskScoreDetailResponse(BaseModel):
    """GET /clients/{id}/risk response."""

    risk_score: RiskScoreResponse
    history: list[RiskScoreHistory]  # Last 5 scores
    alerts: list["RiskAlertResponse"]  # Active alerts for this client


# ============================================================================
# Risk Alert Schemas
# ============================================================================


class RiskAlertResponse(BaseModel):
    """Risk alert with status."""

    id: UUID
    client_id: UUID
    coach_id: UUID
    risk_score_id: UUID | None

    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str

    status: AlertStatus
    acknowledged_at: datetime | None
    acknowledged_notes: str | None
    resolved_at: datetime | None

    created_at: datetime

    class Config:
        from_attributes = True


class AcknowledgeAlertRequest(BaseModel):
    """Request body for acknowledging an alert."""

    notes: str | None = Field(None, max_length=1000)


class AlertListResponse(BaseModel):
    """GET /risk/alerts response."""

    alerts: list[RiskAlertResponse]
    total: int


# ============================================================================
# Trend Schemas
# ============================================================================


class TrendsResponse(BaseModel):
    """GET /clients/{id}/trends response."""

    trends: dict[str, TrendData]


# ============================================================================
# Session List Schemas
# ============================================================================


class SessionListResponse(BaseModel):
    """GET /clients/{id}/sessions response."""

    sessions: list[SessionAnalyticsSummary]
    total: int


# ============================================================================
# Coach Summary Schemas
# ============================================================================


class RiskDistribution(BaseModel):
    """Distribution of clients by risk level."""

    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


class CoachSummaryResponse(BaseModel):
    """GET /coach/summary response."""

    total_clients: int
    clients_at_risk: int  # Medium + High + Critical
    average_engagement: float | None = Field(ge=0.0, le=100.0, default=None)
    sessions_this_month: int
    risk_distribution: RiskDistribution


# ============================================================================
# Compute Request/Response
# ============================================================================


class ComputeResponse(BaseModel):
    """Response for triggering recomputation."""

    task_id: str
    message: str


# Update forward refs for self-referential types
RiskScoreDetailResponse.model_rebuild()
