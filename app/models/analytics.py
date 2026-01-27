"""SQLAlchemy ORM models for coaching analytics."""

import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.core.database import Base


class SessionAnalytics(Base):
    """Per-session analytics computed from transcripts."""

    __tablename__ = "session_analytics"
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_session_analytics_job_id"),
        {"schema": "ai_backend"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai_backend.transcription_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    client_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    coach_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    session_date = Column(Date, nullable=False, index=True)

    # Talk-time metrics
    total_duration_seconds = Column(Numeric(10, 2), nullable=False)
    coach_talk_time_seconds = Column(Numeric(10, 2), nullable=False)
    client_talk_time_seconds = Column(Numeric(10, 2), nullable=False)
    silence_time_seconds = Column(Numeric(10, 2), nullable=False)
    coach_talk_percentage = Column(Numeric(5, 2), nullable=False)
    client_talk_percentage = Column(Numeric(5, 2), nullable=False)

    # Turn-taking
    total_turns = Column(Integer, nullable=False)
    coach_turns = Column(Integer, nullable=False)
    client_turns = Column(Integer, nullable=False)
    average_turn_duration_coach = Column(Numeric(8, 2))
    average_turn_duration_client = Column(Numeric(8, 2))
    interruption_count = Column(Integer, default=0)

    # Coaching style
    coach_question_count = Column(Integer, default=0)
    coach_statement_count = Column(Integer, default=0)
    question_to_statement_ratio = Column(Numeric(5, 3))
    open_question_count = Column(Integer, default=0)
    closed_question_count = Column(Integer, default=0)

    # Language cue counts
    cue_resistance_count = Column(Integer, default=0)
    cue_commitment_count = Column(Integer, default=0)
    cue_breakthrough_count = Column(Integer, default=0)
    cue_concern_count = Column(Integer, default=0)
    cue_deflection_count = Column(Integer, default=0)
    cue_enthusiasm_count = Column(Integer, default=0)
    cue_doubt_count = Column(Integer, default=0)
    cue_agreement_count = Column(Integer, default=0)
    cue_goal_setting_count = Column(Integer, default=0)

    # Sentiment
    client_sentiment_score = Column(Numeric(4, 3))  # -1 to 1
    coach_sentiment_score = Column(Numeric(4, 3))
    sentiment_variance = Column(Numeric(4, 3))

    # Engagement
    engagement_score = Column(Numeric(5, 2))  # 0-100
    response_elaboration_score = Column(Numeric(5, 2))

    # Quality flags
    quality_warning = Column(Boolean, default=False)
    quality_warnings = Column(JSONB, default=list)

    # Metadata
    computed_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    model_version = Column(String(50), nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "client_id": str(self.client_id),
            "coach_id": str(self.coach_id),
            "session_date": self.session_date.isoformat() if self.session_date else None,
            "total_duration_seconds": float(self.total_duration_seconds),
            "coach_talk_time_seconds": float(self.coach_talk_time_seconds),
            "client_talk_time_seconds": float(self.client_talk_time_seconds),
            "silence_time_seconds": float(self.silence_time_seconds),
            "coach_talk_percentage": float(self.coach_talk_percentage),
            "client_talk_percentage": float(self.client_talk_percentage),
            "total_turns": self.total_turns,
            "coach_turns": self.coach_turns,
            "client_turns": self.client_turns,
            "average_turn_duration_coach": float(self.average_turn_duration_coach) if self.average_turn_duration_coach else None,
            "average_turn_duration_client": float(self.average_turn_duration_client) if self.average_turn_duration_client else None,
            "interruption_count": self.interruption_count,
            "coach_question_count": self.coach_question_count,
            "coach_statement_count": self.coach_statement_count,
            "question_to_statement_ratio": float(self.question_to_statement_ratio) if self.question_to_statement_ratio else None,
            "open_question_count": self.open_question_count,
            "closed_question_count": self.closed_question_count,
            "cue_resistance_count": self.cue_resistance_count,
            "cue_commitment_count": self.cue_commitment_count,
            "cue_breakthrough_count": self.cue_breakthrough_count,
            "cue_concern_count": self.cue_concern_count,
            "cue_deflection_count": self.cue_deflection_count,
            "cue_enthusiasm_count": self.cue_enthusiasm_count,
            "cue_doubt_count": self.cue_doubt_count,
            "cue_agreement_count": self.cue_agreement_count,
            "cue_goal_setting_count": self.cue_goal_setting_count,
            "client_sentiment_score": float(self.client_sentiment_score) if self.client_sentiment_score else None,
            "coach_sentiment_score": float(self.coach_sentiment_score) if self.coach_sentiment_score else None,
            "sentiment_variance": float(self.sentiment_variance) if self.sentiment_variance else None,
            "engagement_score": float(self.engagement_score) if self.engagement_score else None,
            "response_elaboration_score": float(self.response_elaboration_score) if self.response_elaboration_score else None,
            "quality_warning": self.quality_warning,
            "quality_warnings": self.quality_warnings or [],
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
            "model_version": self.model_version,
        }


class LanguageCue(Base):
    """Language cues detected in client utterances."""

    __tablename__ = "language_cues"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_analytics_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai_backend.session_analytics.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    utterance_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai_backend.utterances.id", ondelete="CASCADE"),
        nullable=False,
    )

    cue_type = Column(String(30), nullable=False, index=True)
    confidence = Column(Numeric(4, 3), nullable=False)
    text_excerpt = Column(Text, nullable=False)
    timestamp_seconds = Column(Numeric(10, 3), nullable=False)
    preceding_context = Column(Text)
    interpretation = Column(Text)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "session_analytics_id": str(self.session_analytics_id),
            "utterance_id": str(self.utterance_id),
            "cue_type": self.cue_type,
            "confidence": float(self.confidence),
            "text_excerpt": self.text_excerpt,
            "timestamp_seconds": float(self.timestamp_seconds),
            "preceding_context": self.preceding_context,
            "interpretation": self.interpretation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ClientAnalytics(Base):
    """Aggregate analytics per client over time windows."""

    __tablename__ = "client_analytics"
    __table_args__ = (
        UniqueConstraint(
            "client_id", "coach_id", "window_type",
            name="uq_client_analytics_client_coach_window",
        ),
        {"schema": "ai_backend"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    coach_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Time window
    window_start = Column(Date, nullable=False)
    window_end = Column(Date, nullable=False)
    window_type = Column(String(20), nullable=False)  # 30d, 90d, all_time

    # Session frequency
    total_sessions = Column(Integer, nullable=False)
    sessions_last_30_days = Column(Integer, nullable=False)
    average_days_between_sessions = Column(Numeric(6, 2))
    days_since_last_session = Column(Integer)
    session_frequency_trend = Column(String(20))

    # Talk-time trends
    average_coach_talk_percentage = Column(Numeric(5, 2))
    average_client_talk_percentage = Column(Numeric(5, 2))
    talk_ratio_trend = Column(String(30))

    # Engagement trends
    average_engagement_score = Column(Numeric(5, 2))
    engagement_trend = Column(String(20))
    engagement_scores_history = Column(JSONB)  # Array of floats

    # Sentiment trends
    average_sentiment_score = Column(Numeric(4, 3))
    sentiment_trend = Column(String(20))
    sentiment_scores_history = Column(JSONB)

    # Cue patterns
    total_resistance_cues = Column(Integer, default=0)
    total_commitment_cues = Column(Integer, default=0)
    total_breakthrough_cues = Column(Integer, default=0)
    resistance_to_commitment_ratio = Column(Numeric(5, 3))

    computed_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "client_id": str(self.client_id),
            "coach_id": str(self.coach_id),
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "window_type": self.window_type,
            "total_sessions": self.total_sessions,
            "sessions_last_30_days": self.sessions_last_30_days,
            "average_days_between_sessions": float(self.average_days_between_sessions) if self.average_days_between_sessions else None,
            "days_since_last_session": self.days_since_last_session,
            "session_frequency_trend": self.session_frequency_trend,
            "average_coach_talk_percentage": float(self.average_coach_talk_percentage) if self.average_coach_talk_percentage else None,
            "average_client_talk_percentage": float(self.average_client_talk_percentage) if self.average_client_talk_percentage else None,
            "talk_ratio_trend": self.talk_ratio_trend,
            "average_engagement_score": float(self.average_engagement_score) if self.average_engagement_score else None,
            "engagement_trend": self.engagement_trend,
            "engagement_scores_history": self.engagement_scores_history or [],
            "average_sentiment_score": float(self.average_sentiment_score) if self.average_sentiment_score else None,
            "sentiment_trend": self.sentiment_trend,
            "sentiment_scores_history": self.sentiment_scores_history or [],
            "total_resistance_cues": self.total_resistance_cues,
            "total_commitment_cues": self.total_commitment_cues,
            "total_breakthrough_cues": self.total_breakthrough_cues,
            "resistance_to_commitment_ratio": float(self.resistance_to_commitment_ratio) if self.resistance_to_commitment_ratio else None,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
        }


class RiskScore(Base):
    """Churn risk scores for clients."""

    __tablename__ = "risk_scores"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    coach_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    risk_score = Column(Numeric(5, 2), nullable=False)  # 0-100
    risk_level = Column(String(20), nullable=False, index=True)
    churn_probability = Column(Numeric(4, 3), nullable=False)  # 0-1

    factors = Column(JSONB, nullable=False)  # Array of RiskFactor

    previous_risk_score = Column(Numeric(5, 2))
    score_change = Column(Numeric(5, 2))
    trend = Column(String(20))

    recommended_action = Column(Text)

    computed_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    valid_until = Column(DateTime(timezone=True), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "client_id": str(self.client_id),
            "coach_id": str(self.coach_id),
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level,
            "churn_probability": float(self.churn_probability),
            "factors": self.factors or [],
            "previous_risk_score": float(self.previous_risk_score) if self.previous_risk_score else None,
            "score_change": float(self.score_change) if self.score_change else None,
            "trend": self.trend,
            "recommended_action": self.recommended_action,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "model_version": self.model_version,
        }


class RiskAlert(Base):
    """Risk alerts generated for coaches."""

    __tablename__ = "risk_alerts"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    coach_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    risk_score_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai_backend.risk_scores.id", ondelete="SET NULL"),
    )

    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)

    status = Column(String(20), nullable=False, default="pending", index=True)
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_notes = Column(Text)
    resolved_at = Column(DateTime(timezone=True))

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "client_id": str(self.client_id),
            "coach_id": str(self.coach_id),
            "risk_score_id": str(self.risk_score_id) if self.risk_score_id else None,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "status": self.status,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_notes": self.acknowledged_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
