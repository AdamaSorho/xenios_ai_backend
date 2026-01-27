# Plan 0005: Coaching Analytics & Risk Detection

**Spec**: [codev/specs/0005-coaching-analytics-risk-detection.md](../specs/0005-coaching-analytics-risk-detection.md)
**Status**: Ready for implementation
**Estimated Phases**: 7

---

## Implementation Strategy

Build the analytics system incrementally, starting with database schema, then session-level analytics, language cue detection, client aggregates, risk scoring, and finally API endpoints with Celery tasks.

**Key Principles:**
- Database schema first (foundation for all analytics)
- Session analytics computed on transcription completion (event-driven)
- Client analytics and risk scores computed via daily batch jobs
- LLM calls only for cue detection (coaching style metrics use heuristics)
- Reuse LLM client patterns from Spec 0001
- Test with mock transcription data at each phase
- Authorization checks on all endpoints

**Dependencies:**
- Spec 0001: LLM client, Celery, Redis
- Spec 0003: Transcription tables (transcription_jobs, transcripts, utterances, session_summaries)

---

## Phase 1: Database Schema

**Goal**: Create all analytics tables and indexes.

### 1.1 Create migration file

**File**: `scripts/migrations/0005_analytics_tables.sql`

```sql
-- Session-level analytics
CREATE TABLE ai_backend.session_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ai_backend.transcription_jobs(id) ON DELETE CASCADE,
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,
    session_date DATE NOT NULL,

    -- Talk-time metrics
    total_duration_seconds DECIMAL(10,2) NOT NULL,
    coach_talk_time_seconds DECIMAL(10,2) NOT NULL,
    client_talk_time_seconds DECIMAL(10,2) NOT NULL,
    silence_time_seconds DECIMAL(10,2) NOT NULL,
    coach_talk_percentage DECIMAL(5,2) NOT NULL,
    client_talk_percentage DECIMAL(5,2) NOT NULL,

    -- Turn-taking
    total_turns INTEGER NOT NULL,
    coach_turns INTEGER NOT NULL,
    client_turns INTEGER NOT NULL,
    average_turn_duration_coach DECIMAL(8,2),
    average_turn_duration_client DECIMAL(8,2),
    interruption_count INTEGER DEFAULT 0,

    -- Coaching style
    coach_question_count INTEGER DEFAULT 0,
    coach_statement_count INTEGER DEFAULT 0,
    question_to_statement_ratio DECIMAL(5,3),
    open_question_count INTEGER DEFAULT 0,
    closed_question_count INTEGER DEFAULT 0,

    -- Language cue counts (matches CueType enum)
    cue_resistance_count INTEGER DEFAULT 0,
    cue_commitment_count INTEGER DEFAULT 0,
    cue_breakthrough_count INTEGER DEFAULT 0,
    cue_concern_count INTEGER DEFAULT 0,
    cue_deflection_count INTEGER DEFAULT 0,
    cue_enthusiasm_count INTEGER DEFAULT 0,
    cue_doubt_count INTEGER DEFAULT 0,
    cue_agreement_count INTEGER DEFAULT 0,
    cue_goal_setting_count INTEGER DEFAULT 0,

    -- Sentiment
    client_sentiment_score DECIMAL(4,3),
    coach_sentiment_score DECIMAL(4,3),
    sentiment_variance DECIMAL(4,3),

    -- Engagement
    engagement_score DECIMAL(5,2),
    response_elaboration_score DECIMAL(5,2),

    -- Quality flags
    quality_warning BOOLEAN DEFAULT FALSE,
    quality_warnings JSONB DEFAULT '[]',

    -- Metadata
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    model_version VARCHAR(50) NOT NULL,

    UNIQUE(job_id)
);

CREATE INDEX idx_session_analytics_client ON ai_backend.session_analytics(client_id);
CREATE INDEX idx_session_analytics_coach ON ai_backend.session_analytics(coach_id);
CREATE INDEX idx_session_analytics_date ON ai_backend.session_analytics(session_date DESC);

-- Language cues detected
CREATE TABLE ai_backend.language_cues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_analytics_id UUID NOT NULL REFERENCES ai_backend.session_analytics(id) ON DELETE CASCADE,
    utterance_id UUID NOT NULL REFERENCES ai_backend.utterances(id) ON DELETE CASCADE,

    cue_type VARCHAR(30) NOT NULL,
    confidence DECIMAL(4,3) NOT NULL,
    text_excerpt TEXT NOT NULL,
    timestamp_seconds DECIMAL(10,3) NOT NULL,
    preceding_context TEXT,
    interpretation TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_language_cues_session ON ai_backend.language_cues(session_analytics_id);
CREATE INDEX idx_language_cues_type ON ai_backend.language_cues(cue_type);

-- Client aggregate analytics
CREATE TABLE ai_backend.client_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    -- Time window
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    window_type VARCHAR(20) NOT NULL,

    -- Session frequency
    total_sessions INTEGER NOT NULL,
    sessions_last_30_days INTEGER NOT NULL,
    average_days_between_sessions DECIMAL(6,2),
    days_since_last_session INTEGER,
    session_frequency_trend VARCHAR(20),

    -- Talk-time trends
    average_coach_talk_percentage DECIMAL(5,2),
    average_client_talk_percentage DECIMAL(5,2),
    talk_ratio_trend VARCHAR(30),

    -- Engagement trends
    average_engagement_score DECIMAL(5,2),
    engagement_trend VARCHAR(20),
    engagement_scores_history JSONB,

    -- Sentiment trends
    average_sentiment_score DECIMAL(4,3),
    sentiment_trend VARCHAR(20),
    sentiment_scores_history JSONB,

    -- Cue patterns
    total_resistance_cues INTEGER DEFAULT 0,
    total_commitment_cues INTEGER DEFAULT 0,
    total_breakthrough_cues INTEGER DEFAULT 0,
    resistance_to_commitment_ratio DECIMAL(5,3),

    computed_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(client_id, coach_id, window_type)
);

CREATE INDEX idx_client_analytics_client ON ai_backend.client_analytics(client_id);
CREATE INDEX idx_client_analytics_coach ON ai_backend.client_analytics(coach_id);

-- Risk scores
CREATE TABLE ai_backend.risk_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    risk_score DECIMAL(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    churn_probability DECIMAL(4,3) NOT NULL,

    factors JSONB NOT NULL,

    previous_risk_score DECIMAL(5,2),
    score_change DECIMAL(5,2),
    trend VARCHAR(20),

    recommended_action TEXT,

    computed_at TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50) NOT NULL
);

CREATE INDEX idx_risk_scores_client ON ai_backend.risk_scores(client_id);
CREATE INDEX idx_risk_scores_coach ON ai_backend.risk_scores(coach_id);
CREATE INDEX idx_risk_scores_level ON ai_backend.risk_scores(risk_level);
CREATE INDEX idx_risk_scores_computed ON ai_backend.risk_scores(computed_at DESC);

-- Risk alerts
CREATE TABLE ai_backend.risk_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,
    risk_score_id UUID REFERENCES ai_backend.risk_scores(id) ON DELETE SET NULL,

    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,

    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    acknowledged_at TIMESTAMPTZ,
    acknowledged_notes TEXT,
    resolved_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_risk_alerts_coach ON ai_backend.risk_alerts(coach_id);
CREATE INDEX idx_risk_alerts_status ON ai_backend.risk_alerts(status);
CREATE INDEX idx_risk_alerts_severity ON ai_backend.risk_alerts(severity);
CREATE INDEX idx_risk_alerts_created ON ai_backend.risk_alerts(created_at DESC);
```

### 1.2 Create SQLAlchemy models

**File**: `app/models/analytics.py`

```python
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, Date, Numeric, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base
import uuid

class SessionAnalytics(Base):
    __tablename__ = "session_analytics"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("ai_backend.transcription_jobs.id"), nullable=False, unique=True)
    client_id = Column(UUID(as_uuid=True), nullable=False)
    coach_id = Column(UUID(as_uuid=True), nullable=False)
    session_date = Column(Date, nullable=False)

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
    client_sentiment_score = Column(Numeric(4, 3))
    coach_sentiment_score = Column(Numeric(4, 3))
    sentiment_variance = Column(Numeric(4, 3))

    # Engagement
    engagement_score = Column(Numeric(5, 2))
    response_elaboration_score = Column(Numeric(5, 2))

    # Quality flags
    quality_warning = Column(Boolean, default=False)
    quality_warnings = Column(JSONB, default=list)

    # Metadata
    computed_at = Column(DateTime(timezone=True), server_default="NOW()")
    model_version = Column(String(50), nullable=False)


class LanguageCue(Base):
    __tablename__ = "language_cues"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_analytics_id = Column(UUID(as_uuid=True), ForeignKey("ai_backend.session_analytics.id"), nullable=False)
    utterance_id = Column(UUID(as_uuid=True), ForeignKey("ai_backend.utterances.id"), nullable=False)

    cue_type = Column(String(30), nullable=False)
    confidence = Column(Numeric(4, 3), nullable=False)
    text_excerpt = Column(Text, nullable=False)
    timestamp_seconds = Column(Numeric(10, 3), nullable=False)
    preceding_context = Column(Text)
    interpretation = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default="NOW()")


class ClientAnalytics(Base):
    __tablename__ = "client_analytics"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False)
    coach_id = Column(UUID(as_uuid=True), nullable=False)

    window_start = Column(Date, nullable=False)
    window_end = Column(Date, nullable=False)
    window_type = Column(String(20), nullable=False)

    total_sessions = Column(Integer, nullable=False)
    sessions_last_30_days = Column(Integer, nullable=False)
    average_days_between_sessions = Column(Numeric(6, 2))
    days_since_last_session = Column(Integer)
    session_frequency_trend = Column(String(20))

    average_coach_talk_percentage = Column(Numeric(5, 2))
    average_client_talk_percentage = Column(Numeric(5, 2))
    talk_ratio_trend = Column(String(30))

    average_engagement_score = Column(Numeric(5, 2))
    engagement_trend = Column(String(20))
    engagement_scores_history = Column(JSONB)

    average_sentiment_score = Column(Numeric(4, 3))
    sentiment_trend = Column(String(20))
    sentiment_scores_history = Column(JSONB)

    total_resistance_cues = Column(Integer, default=0)
    total_commitment_cues = Column(Integer, default=0)
    total_breakthrough_cues = Column(Integer, default=0)
    resistance_to_commitment_ratio = Column(Numeric(5, 3))

    computed_at = Column(DateTime(timezone=True), server_default="NOW()")


class RiskScore(Base):
    __tablename__ = "risk_scores"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False)
    coach_id = Column(UUID(as_uuid=True), nullable=False)

    risk_score = Column(Numeric(5, 2), nullable=False)
    risk_level = Column(String(20), nullable=False)
    churn_probability = Column(Numeric(4, 3), nullable=False)

    factors = Column(JSONB, nullable=False)

    previous_risk_score = Column(Numeric(5, 2))
    score_change = Column(Numeric(5, 2))
    trend = Column(String(20))

    recommended_action = Column(Text)

    computed_at = Column(DateTime(timezone=True), server_default="NOW()")
    valid_until = Column(DateTime(timezone=True), nullable=False)
    model_version = Column(String(50), nullable=False)


class RiskAlert(Base):
    __tablename__ = "risk_alerts"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False)
    coach_id = Column(UUID(as_uuid=True), nullable=False)
    risk_score_id = Column(UUID(as_uuid=True), ForeignKey("ai_backend.risk_scores.id"))

    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)

    status = Column(String(20), nullable=False, default="pending")
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_notes = Column(Text)
    resolved_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), server_default="NOW()")
```

### 1.3 Create Pydantic schemas

**File**: `app/schemas/analytics.py`

Define request/response schemas matching the spec:
- `SessionAnalyticsResponse` - Full session analytics with cues
- `SessionAnalyticsSummary` - Compact view for list endpoints
- `LanguageCueResponse` - Individual cue with context
- `ClientAnalyticsResponse` - Aggregate client analytics
- `RiskScoreResponse` - Risk score with factors
- `RiskFactorResponse` - Individual risk factor detail
- `RiskAlertResponse` - Alert with status
- `TrendData`, `DataPoint` - Trend visualization data
- `SessionComparison` - Compare current vs previous session
- `RiskScoreHistory` - Historical risk score entry for trend display
- `ClientAnalyticsSummaryResponse` - Summary endpoint response with quality_warnings
- `AcknowledgeAlertRequest` - Request body for alert acknowledgment

**Key schema implementations:**

```python
from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime
from typing import Optional

class SessionAnalyticsSummary(BaseModel):
    """Compact view for session list endpoints (GET /clients/{id}/sessions)."""
    job_id: UUID
    session_date: date
    duration_minutes: float
    coach_talk_percentage: float
    client_talk_percentage: float
    engagement_score: float
    client_sentiment_score: float
    cue_count: int  # Total cues detected
    has_warnings: bool  # True if quality_warning is True on session
    # quality_warnings field omitted in summary; use full endpoint for details

class RiskScoreHistory(BaseModel):
    """Historical risk score for trend display."""
    computed_at: datetime
    risk_score: float
    risk_level: str
    top_factor: str  # Primary contributing factor

class RiskScoreDetailResponse(BaseModel):
    """GET /clients/{id}/risk response."""
    risk_score: RiskScoreResponse
    history: list[RiskScoreHistory]  # Last 5 scores
    alerts: list[RiskAlertResponse]  # Active alerts for this client

class ClientAnalyticsSummaryResponse(BaseModel):
    """GET /clients/{id}/summary response."""
    client_analytics: ClientAnalyticsResponse
    session_count: int
    latest_session_date: date | None
    risk_level: str | None  # Null if no valid risk score
    risk_score_stale: bool  # True if risk score > 7 days old
    quality_warnings: list[str]  # e.g., ["low_confidence_diarization"]

class SessionListResponse(BaseModel):
    """GET /clients/{id}/sessions response."""
    sessions: list[SessionAnalyticsSummary]
    total: int
```

**Query for risk score history:**
```sql
SELECT computed_at, risk_score, risk_level,
       (factors->0->>'factor_type') as top_factor
FROM ai_backend.risk_scores
WHERE client_id = :client_id
ORDER BY computed_at DESC
LIMIT 5
```

### 1.4 Update model exports

**File**: `app/models/__init__.py`

```python
from app.models.analytics import (
    SessionAnalytics,
    LanguageCue,
    ClientAnalytics,
    RiskScore,
    RiskAlert,
)
```

### Phase 1 Tests

- Verify migration runs without errors
- Verify models can be instantiated
- Test unique constraints

---

## Phase 2: Session Analytics Service

**Goal**: Compute talk-time and coaching style metrics from utterances.

### 2.1 Create analytics service directory

```
app/services/analytics/
├── __init__.py
├── session_analytics.py
├── talk_time.py
├── coaching_style.py
├── engagement.py
└── sentiment.py
```

### 2.2 Implement talk-time analyzer

**File**: `app/services/analytics/talk_time.py`

```python
from dataclasses import dataclass
from app.models.transcription import Utterance

@dataclass
class TalkTimeMetrics:
    total_duration_seconds: float
    coach_talk_time_seconds: float
    client_talk_time_seconds: float
    silence_time_seconds: float
    coach_talk_percentage: float
    client_talk_percentage: float
    total_turns: int
    coach_turns: int
    client_turns: int
    average_turn_duration_coach: float
    average_turn_duration_client: float
    interruption_count: int


class TalkTimeAnalyzer:
    def compute(self, utterances: list[Utterance]) -> TalkTimeMetrics:
        # Implementation per spec formulas
        pass

    def _count_interruptions(self, utterances: list[Utterance]) -> int:
        # Detect overlapping speech
        pass
```

### 2.3 Implement coaching style analyzer

**File**: `app/services/analytics/coaching_style.py`

```python
from dataclasses import dataclass

QUESTION_STARTERS = {"what", "how", "why", "where", "when", "who", "which"}
OPEN_STARTERS = {"what", "how", "why", "tell"}
CLOSED_STARTERS = {"do", "did", "is", "are", "can", "could", "will", "would", "have", "has"}

@dataclass
class CoachingStyleMetrics:
    coach_question_count: int
    coach_statement_count: int
    question_to_statement_ratio: float
    open_question_count: int
    closed_question_count: int


class CoachingStyleAnalyzer:
    def compute(self, coach_utterances: list) -> CoachingStyleMetrics:
        # Pattern-based detection per spec
        pass
```

### 2.4 Implement engagement score calculator

**File**: `app/services/analytics/engagement.py`

```python
def calculate_engagement_score(
    client_talk_percentage: float,
    duration_minutes: float,
    client_turns: int,
    total_client_words: int,
    sentiment_score: float,
    commitment_cue_count: int,
    resistance_cue_count: int,
) -> float:
    # Implementation per spec formula
    pass


def calculate_response_elaboration_score(client_utterances: list) -> float:
    # avg_words / 30 * 100, capped at 100
    pass
```

### 2.5 Implement sentiment analyzer

**File**: `app/services/analytics/sentiment.py`

```python
def calculate_sentiment_variance(utterance_sentiments: list[float]) -> float:
    # Standard deviation of sentiment scores
    pass


def aggregate_sentiment(utterances: list, speaker_label: str) -> float:
    # Average sentiment for speaker
    pass
```

### 2.6 Implement session analytics service

**File**: `app/services/analytics/session_analytics.py`

```python
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analytics.talk_time import TalkTimeAnalyzer
from app.services.analytics.coaching_style import CoachingStyleAnalyzer
from app.services.analytics.engagement import calculate_engagement_score
from app.services.analytics.sentiment import calculate_sentiment_variance


class SessionAnalyticsService:
    MODEL_VERSION = "v1.0.0"

    def __init__(self, db: AsyncSession):
        self.db = db
        self.talk_time_analyzer = TalkTimeAnalyzer()
        self.coaching_style_analyzer = CoachingStyleAnalyzer()

    async def compute_for_job(self, job_id: UUID) -> SessionAnalytics:
        """
        Compute all session analytics for a completed transcription job.
        Called after transcription completes.
        """
        # 1. Fetch job, transcript, utterances
        # 2. Compute talk-time metrics
        # 3. Compute coaching style metrics
        # 4. Compute sentiment metrics
        # 5. Create SessionAnalytics record (without cues - Phase 3)
        # 6. Return for cue detection
        pass
```

### Phase 2 Tests

**File**: `tests/services/analytics/test_talk_time.py`
**File**: `tests/services/analytics/test_coaching_style.py`
**File**: `tests/services/analytics/test_engagement.py`

- Test with sample utterance data
- Verify percentage calculations
- Test edge cases (single speaker, no utterances)

---

## Phase 3: Language Cue Detection

**Goal**: Detect resistance, commitment, breakthrough, and other cues using LLM.

### 3.1 Create cue detection service

**File**: `app/services/analytics/cue_detection.py`

```python
import asyncio
import json
import re
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

from app.services.llm_client import LLMClient
from app.core.logging import get_logger

logger = get_logger(__name__)


class CueType(str, Enum):
    RESISTANCE = "resistance"
    COMMITMENT = "commitment"
    BREAKTHROUGH = "breakthrough"
    CONCERN = "concern"
    DEFLECTION = "deflection"
    ENTHUSIASM = "enthusiasm"
    DOUBT = "doubt"
    AGREEMENT = "agreement"
    GOAL_SETTING = "goal_setting"


@dataclass
class DetectedCue:
    cue_type: str
    confidence: float
    interpretation: str


class CueDetectionError(Exception):
    pass


class CueDetectionService:
    MODEL = "gpt-4o-mini"  # Cost-effective for classification
    MAX_RETRIES = 2
    TIMEOUT_SECONDS = 30
    RATE_LIMIT_DELAY = 1.0
    MIN_CONFIDENCE = 0.7

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def detect_cues(
        self,
        utterances: list,
        session_analytics_id: UUID,
    ) -> list:
        """Detect cues in client utterances."""
        # Implementation per spec
        pass

    async def _analyze_utterance_with_retry(self, utterance, context: str):
        # Retry logic with timeout
        pass

    def _parse_llm_response(self, response: str) -> list[DetectedCue]:
        # JSON parsing with validation
        pass

    def _redact_pii(self, text: str) -> str:
        # Redact phone, email, SSN
        pass

    def _get_context(self, utterances: list, current: any) -> str:
        # Get preceding utterances for context
        pass
```

### 3.2 Create prompt template

**File**: `app/services/analytics/prompts.py`

```python
CUE_DETECTION_PROMPT = """Analyze this coaching session utterance for language cues.

UTTERANCE: {text}
SPEAKER: {speaker_label}
CONTEXT (previous 2 utterances): {context}

Identify any of these cue types present:
- resistance: Client pushing back, expressing doubt, or avoiding
- commitment: Client making promises or expressing determination
- breakthrough: Moments of insight or realization
- concern: Client expressing worry or anxiety
- deflection: Changing subject, giving vague answers
- enthusiasm: Expressing excitement or positive energy
- doubt: Hesitation or uncertainty
- goal_setting: Setting concrete, measurable objectives

For each cue found, provide:
1. cue_type: The type from the list above
2. confidence: 0.0-1.0
3. interpretation: Brief explanation

Respond in JSON format:
{
  "cues": [
    {"cue_type": "...", "confidence": 0.X, "interpretation": "..."}
  ]
}
"""
```

### 3.3 Update session analytics service

Update `session_analytics.py` to call cue detection after basic metrics are computed.

### Phase 3 Tests

**File**: `tests/services/analytics/test_cue_detection.py`

- Test prompt formatting
- Test JSON parsing (valid, malformed, markdown-wrapped)
- Test PII redaction
- Test retry logic
- Mock LLM responses

---

## Phase 4: Client Analytics Service

**Goal**: Aggregate session analytics into client-level trends.

### 4.1 Create client analytics service

**File**: `app/services/analytics/client_analytics.py`

```python
from datetime import date, timedelta
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analytics.trends import calculate_trend


class ClientAnalyticsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def compute_for_client(
        self,
        client_id: UUID,
        coach_id: UUID,
        window_type: str = "90d",
    ) -> ClientAnalytics:
        """
        Compute aggregate analytics for a client.
        Called by daily batch job.
        """
        # 1. Determine window dates
        # 2. Fetch session analytics in window
        # 3. Calculate averages
        # 4. Calculate trends
        # 5. Upsert ClientAnalytics record
        pass

    def _calculate_window_dates(self, window_type: str) -> tuple[date, date]:
        pass

    def _calculate_session_frequency(self, sessions: list) -> dict:
        pass
```

### 4.2 Create trend calculator

**File**: `app/services/analytics/trends.py`

```python
def calculate_trend(values: list[float], threshold: float = 0.1) -> str:
    """
    Determine trend direction from a series of values.
    Returns: "improving", "stable", or "declining"
    """
    # Implementation per spec
    pass
```

### Phase 4 Tests

**File**: `tests/services/analytics/test_client_analytics.py`
**File**: `tests/services/analytics/test_trends.py`

- Test trend calculation with various patterns
- Test window calculations
- Test aggregation logic

---

## Phase 5: Risk Scoring Service

**Goal**: Calculate churn risk scores with multi-factor analysis.

### 5.1 Create risk scoring service

**File**: `app/services/analytics/risk_scoring.py`

```python
from datetime import datetime, timedelta
from uuid import UUID
from enum import Enum

from app.models.analytics import ClientAnalytics, RiskScore


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def normalize_factor_contribution(
    value: float,
    warning_threshold: float,
    critical_threshold: float,
    weight: float,
    inverse: bool = False,
) -> float:
    """Convert metric value to risk contribution."""
    # Implementation per spec
    pass


class RiskScoringService:
    WEIGHTS = {
        "session_frequency": 25,
        "engagement_trend": 25,
        "sentiment_trend": 20,
        "resistance_ratio": 15,
        "days_since_session": 15,
    }

    THRESHOLDS = {
        "days_since_session_warning": 14,
        "days_since_session_critical": 30,
        "engagement_decline_warning": -10,
        "engagement_decline_critical": -20,
        "sentiment_decline_warning": -0.2,
        "sentiment_decline_critical": -0.4,
        "resistance_ratio_warning": 2.0,
        "resistance_ratio_critical": 4.0,
        "session_frequency_decline_warning": 0.5,
        "session_frequency_decline_critical": 0.75,
    }

    MODEL_VERSION = "v1.0.0"

    def __init__(self, db):
        self.db = db

    async def compute_risk_score(
        self,
        client_id: UUID,
        client_analytics: ClientAnalytics,
    ) -> RiskScore:
        """Compute risk score from client analytics."""
        # 1. Compute each factor
        # 2. Sum contributions
        # 3. Determine level
        # 4. Generate recommendation
        # 5. Get previous score for trend
        # 6. Create RiskScore record
        pass

    def _compute_frequency_factor(self, analytics):
        pass

    def _compute_engagement_factor(self, analytics):
        pass

    def _compute_sentiment_factor(self, analytics):
        pass

    def _compute_resistance_factor(self, analytics):
        pass

    def _compute_recency_factor(self, analytics):
        pass

    def _score_to_level(self, score: float) -> RiskLevel:
        pass

    def _generate_recommendation(self, factors, level) -> str:
        pass
```

### 5.2 Create alerts service

**File**: `app/services/analytics/alerts.py`

```python
from uuid import UUID

class AlertsService:
    def __init__(self, db):
        self.db = db

    async def generate_alerts(
        self,
        client_id: UUID,
        coach_id: UUID,
        risk_score: RiskScore,
        previous_score: RiskScore | None,
    ) -> list[RiskAlert]:
        """Generate alerts based on risk score changes."""
        # Check for:
        # - New high/critical risk
        # - Significant risk increase
        # - No session in 30+ days
        pass

    async def acknowledge_alert(
        self,
        alert_id: UUID,
        coach_id: UUID,
        notes: str | None,
    ) -> RiskAlert:
        pass
```

### Phase 5 Tests

**File**: `tests/services/analytics/test_risk_scoring.py`
**File**: `tests/services/analytics/test_alerts.py`

- Test factor calculations
- Test normalization function
- Test risk level thresholds
- Test alert generation logic

---

## Phase 6: API Endpoints

**Goal**: Expose analytics via REST API with proper authentication and authorization.

### 6.0 Authentication & Authorization Strategy

**Authentication** (inherited from Spec 0001):
- `X-API-Key` header: Backend-to-backend auth (verified by API key middleware)
- `Authorization: Bearer <JWT>`: User auth from Supabase (verified by `get_current_user` dependency)

Both are required on all analytics endpoints.

**Authorization** (per spec):
- Coach can only access their own clients
- **Return 404 (not 403) for unauthorized access** to prevent enumeration
- Client access verified via `coach_clients` relationship table in MVP database

**Implementation pattern**:

```python
from fastapi import HTTPException

async def get_authorized_client(
    client_id: UUID,
    coach_id: UUID,
    db: AsyncSession,
) -> None:
    """Verify coach-client relationship. Raises 404 if unauthorized."""
    has_relationship = await verify_coach_client_relationship(coach_id, client_id, db)
    if not has_relationship:
        # Return 404 (not 403) per spec - prevents enumeration
        raise HTTPException(status_code=404, detail="Client not found")
```

### 6.1 Create analytics router

**File**: `app/api/v1/analytics.py`

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from uuid import UUID

from app.core.auth import get_current_user, require_api_key
from app.schemas.analytics import *
from app.services.analytics.session_analytics import SessionAnalyticsService
from app.services.analytics.client_analytics import ClientAnalyticsService
from app.services.analytics.risk_scoring import RiskScoringService
from app.services.analytics.alerts import AlertsService
from app.services.analytics.authorization import get_authorized_client

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    dependencies=[Depends(require_api_key)],  # X-API-Key required on all endpoints
)


@router.get("/sessions/{job_id}", response_model=SessionAnalyticsDetailResponse)
async def get_session_analytics(
    job_id: UUID,
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get analytics for a specific session."""
    pass


@router.get("/clients/{client_id}/summary", response_model=ClientAnalyticsSummaryResponse)
async def get_client_summary(
    client_id: UUID,
    window: str = Query("90d", regex="^(30d|90d|all_time)$"),
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get aggregate analytics for a client."""
    pass


@router.get("/clients/{client_id}/sessions", response_model=SessionListResponse)
async def get_client_sessions(
    client_id: UUID,
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    from_date: date | None = None,
    to_date: date | None = None,
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get session analytics history for a client."""
    pass


@router.get("/clients/{client_id}/risk", response_model=RiskScoreDetailResponse)
async def get_client_risk(
    client_id: UUID,
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get current risk score and history."""
    pass


@router.get("/clients/{client_id}/trends", response_model=TrendsResponse)
async def get_client_trends(
    client_id: UUID,
    metrics: list[str] = Query(["engagement", "sentiment", "talk_ratio"]),
    window: str = Query("90d"),
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get trend data for specific metrics."""
    pass


@router.get("/risk/alerts", response_model=AlertListResponse)
async def get_risk_alerts(
    status: str = Query("pending", regex="^(pending|acknowledged|all)$"),
    severity: str = Query("all", regex="^(warning|urgent|all)$"),
    limit: int = Query(50, le=100),
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get risk alerts for coach."""
    pass


@router.post("/risk/alerts/{alert_id}/acknowledge", response_model=RiskAlertResponse)
async def acknowledge_alert(
    alert_id: UUID,
    request: AcknowledgeAlertRequest,
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Acknowledge a risk alert."""
    pass


@router.post("/compute/{client_id}", status_code=202)
async def trigger_compute(
    client_id: UUID,
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Trigger manual recomputation of client analytics."""
    pass


@router.get("/coach/summary", response_model=CoachSummaryResponse)
async def get_coach_summary(
    current_user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get aggregate analytics for coach's clients."""
    pass
```

### 6.2 Add authorization helper

**File**: `app/services/analytics/authorization.py`

```python
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

async def verify_coach_client_relationship(
    coach_id: UUID,
    client_id: UUID,
    db: AsyncSession,
) -> bool:
    """Check if coach has active relationship with client."""
    result = await db.execute(
        text("""
            SELECT 1 FROM public.coach_clients
            WHERE coach_id = :coach_id
            AND client_id = :client_id
            AND status = 'active'
        """),
        {"coach_id": str(coach_id), "client_id": str(client_id)}
    )
    return result.scalar() is not None
```

### 6.3 Register router

**File**: `app/api/v1/router.py`

```python
from app.api.v1.analytics import router as analytics_router

api_router.include_router(analytics_router)
```

### Phase 6 Tests

**File**: `tests/api/v1/test_analytics.py`

- Test all endpoints
- Test authorization (only own clients)
- Test query parameters
- Test 404 for unauthorized access

---

## Phase 6.5: Security, Privacy & Quality Warnings

**Goal**: Add audit logging, rate limiting, PII protection, and quality warning logic.

### 6.5.1 Audit Logging

Analytics access must be logged for compliance. Reuse existing AuditLogMiddleware from Spec 0004.

**Integration**: The `AuditLogMiddleware` from Spec 0004 already logs:
- coach_id (from request.state.user_id)
- client_id (from URL path)
- endpoint, method, timestamp

**Additional logging** in analytics service layer:

```python
# app/services/analytics/session_analytics.py

from app.core.logging import get_logger

logger = get_logger(__name__)

async def compute_for_job(self, job_id: UUID) -> SessionAnalytics:
    # ... computation logic ...

    # Audit log (structured, no PII)
    logger.info(
        "session_analytics_computed",
        job_id=str(job_id),
        client_id=str(client_id),
        coach_id=str(coach_id),
        engagement_score=float(analytics.engagement_score),
        cue_count=total_cues,
    )
```

### 6.5.2 Rate Limiting

Reuse `RateLimitMiddleware` from Spec 0004. Analytics endpoints use default rate limits.

If analytics-specific limits needed, add to `app/middleware/rate_limit.py`:

```python
RATE_LIMITS = {
    # ... existing limits ...
    "analytics_read": {"requests": 200, "window_seconds": 3600},  # 200/hour
    "analytics_compute": {"requests": 10, "window_seconds": 3600},  # 10/hour (manual triggers)
}
```

### 6.5.3 PII Protection in Logs

**Rule**: Never log raw transcript content, utterance text, or cue excerpts.

**Safe to log**: IDs, scores, counts, timestamps, cue types, risk levels.

**Implementation** in structured logging:

```python
# In cue_detection.py - DON'T do this:
# logger.info("Detected cue", text=utterance.text)  # BAD - contains PII

# DO this:
logger.info(
    "cue_detected",
    utterance_id=str(utterance.id),
    cue_type=cue.cue_type,
    confidence=cue.confidence,
    # No text content!
)
```

### 6.5.4 Quality Warnings Logic

Quality warnings are set based on data quality issues during computation.

**File**: `app/services/analytics/quality.py`

```python
from enum import Enum

class QualityWarning(str, Enum):
    LOW_CONFIDENCE_DIARIZATION = "low_confidence_diarization"
    INSUFFICIENT_SESSIONS = "insufficient_sessions"
    MISSING_SENTIMENT = "missing_sentiment"
    SHORT_SESSION = "short_session"
    STALE_RISK_SCORE = "stale_risk_score"


def compute_quality_warnings(
    utterances: list,
    session_duration_seconds: float,
    diarization_confidence: float,
) -> list[str]:
    """Compute quality warnings for a session."""
    warnings = []

    # Low confidence diarization (threshold: 0.7)
    if diarization_confidence < 0.7:
        warnings.append(QualityWarning.LOW_CONFIDENCE_DIARIZATION)

    # Short session (< 2 minutes)
    if session_duration_seconds < 120:
        warnings.append(QualityWarning.SHORT_SESSION)

    # Missing sentiment (no sentiment scores on utterances)
    client_utterances = [u for u in utterances if u.speaker_label == "client"]
    sentiments = [u.sentiment for u in client_utterances if u.sentiment is not None]
    if len(sentiments) < len(client_utterances) * 0.5:  # < 50% have sentiment
        warnings.append(QualityWarning.MISSING_SENTIMENT)

    return warnings


def compute_summary_quality_warnings(
    client_analytics,
    risk_score,
    session_count: int,
) -> list[str]:
    """Compute quality warnings for client summary response."""
    warnings = []

    # Insufficient sessions for trends (< 2)
    if session_count < 2:
        warnings.append(QualityWarning.INSUFFICIENT_SESSIONS)

    # Stale risk score (> 7 days old)
    if risk_score and risk_score.valid_until < datetime.now(timezone.utc):
        warnings.append(QualityWarning.STALE_RISK_SCORE)

    return warnings
```

### 6.5.5 Error Handling for LLM Cue Detection Failure

**Requirement**: If cue detection fails, skip cues but still compute/save other analytics.

**Implementation** in Celery task:

**Note**: Cue detection is async (LLM calls). We use `asyncio.run()` to run async code in sync Celery tasks.

```python
# app/workers/tasks/analytics.py
import asyncio
from app.core.logging import get_logger

logger = get_logger(__name__)

@celery_app.task(bind=True, max_retries=3)
def compute_session_analytics(self, job_id: str):
    """Compute analytics with graceful cue detection failure handling."""
    # Run async code in sync Celery task
    asyncio.run(_compute_session_analytics_async(self, job_id))


async def _compute_session_analytics_async(task, job_id: str):
    """Async implementation of session analytics computation."""
    async with get_async_db_session() as db:
        try:
            session_service = SessionAnalyticsService(db)
            cue_service = CueDetectionService(get_llm_client())

            # 1. Compute basic metrics (no LLM, fast)
            session_analytics, utterances = await session_service.compute_basic_metrics(job_id)

            # 2. Attempt cue detection (async LLM calls, may fail)
            try:
                cues = await cue_service.detect_cues(utterances, session_analytics.id)
                session_analytics = update_cue_counts(session_analytics, cues)
                # Save cues
                for cue in cues:
                    db.add(cue)
            except CueDetectionError as e:
                logger.warning(
                    "cue_detection_failed",
                    job_id=job_id,
                    error=str(e),
                )
                # Continue without cues - analytics still valid
                session_analytics.quality_warnings = session_analytics.quality_warnings or []
                session_analytics.quality_warnings.append("cue_detection_failed")
                session_analytics.quality_warning = True

            # 3. Save analytics (with or without cues)
            db.add(session_analytics)
            await db.commit()

        except Exception as e:
            await db.rollback()
            task.retry(exc=e, countdown=60)
```

### Phase 6.5 Tests

**File**: `tests/services/analytics/test_quality.py`

- Test quality warning generation
- Test each warning condition

**File**: `tests/api/v1/test_analytics_security.py`

- Test rate limiting on analytics endpoints
- Test audit log entries created
- Verify no PII in log output

---

## Phase 7: Celery Tasks & Integration Tests

**Goal**: Create background tasks and end-to-end tests.

### 7.1 Create analytics Celery tasks

**File**: `app/workers/tasks/analytics.py`

```python
from celery import shared_task
from app.workers.celery_app import celery_app

@celery_app.task(bind=True, max_retries=3)
def compute_session_analytics(self, job_id: str):
    """
    Compute analytics for a completed transcription.
    Triggered when transcription job status -> completed.
    """
    # 1. Get DB session
    # 2. Call SessionAnalyticsService.compute_for_job()
    # 3. Call CueDetectionService.detect_cues()
    # 4. Update cue counts on SessionAnalytics
    pass


@celery_app.task
def compute_client_analytics_batch():
    """
    Daily batch job to compute aggregate client analytics.
    Scheduled via Celery Beat at 2:00 AM.
    """
    # For each active client-coach pair:
    # 1. Compute 30d window
    # 2. Compute 90d window
    pass


@celery_app.task
def compute_risk_scores_batch():
    """
    Daily batch job to compute risk scores.
    Scheduled via Celery Beat at 3:00 AM.
    """
    # For each active client:
    # 1. Get latest ClientAnalytics
    # 2. Compute risk score
    # 3. Generate alerts if needed
    pass


@celery_app.task
def generate_risk_alerts():
    """
    Generate alerts for high-risk clients.
    Scheduled via Celery Beat at 3:30 AM (after risk scores).

    Checks for:
    - New high/critical risk level
    - Significant risk increase (>20 points)
    - No session in 30+ days
    """
    # 1. Get all recent risk scores (computed today)
    # 2. For each score, check alert conditions
    # 3. Create RiskAlert records as needed
    # 4. Avoid duplicate alerts (check existing pending alerts)
    pass


@celery_app.task
def cleanup_expired_risk_scores():
    """Delete risk scores past valid_until. Runs daily at 4 AM."""
    # DELETE FROM ai_backend.risk_scores WHERE valid_until < NOW()
    pass


@celery_app.task
def archive_old_analytics():
    """
    Archive/delete analytics older than retention period. Runs monthly.

    Retention per spec:
    - Session Analytics: 2 years (cascade deletes cues)
    - Client Analytics: 1 year
    - Risk Scores: 90 days (handled by cleanup_expired_risk_scores)
    - Risk Alerts: 1 year (archive resolved alerts)
    """
    # 1. Delete session_analytics older than 2 years
    # 2. Delete client_analytics windows older than 1 year
    # 3. Archive/delete risk_alerts older than 1 year
    pass
```

### 7.2 Add Celery Beat schedule

**File**: `app/workers/celery_app.py`

```python
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    # ... existing schedules
    "compute-client-analytics-daily": {
        "task": "app.workers.tasks.analytics.compute_client_analytics_batch",
        "schedule": crontab(hour=2, minute=0),
    },
    "compute-risk-scores-daily": {
        "task": "app.workers.tasks.analytics.compute_risk_scores_batch",
        "schedule": crontab(hour=3, minute=0),
    },
    "generate-risk-alerts-daily": {
        "task": "app.workers.tasks.analytics.generate_risk_alerts",
        "schedule": crontab(hour=3, minute=30),
    },
    "cleanup-expired-risk-scores": {
        "task": "app.workers.tasks.analytics.cleanup_expired_risk_scores",
        "schedule": crontab(hour=4, minute=0),
    },
    "archive-old-analytics-monthly": {
        "task": "app.workers.tasks.analytics.archive_old_analytics",
        "schedule": crontab(day_of_month=1, hour=5, minute=0),  # 1st of each month
    },
}
```

### 7.3 Hook into transcription completion

Update transcription task to trigger session analytics:

**File**: `app/workers/tasks/transcription.py`

```python
# After transcription completes:
from app.workers.tasks.analytics import compute_session_analytics

# In process_transcription task, after status = completed:
compute_session_analytics.delay(str(job_id))
```

### 7.4 Create integration tests

**File**: `tests/integration/test_analytics_pipeline.py`

- Test full flow: transcription complete → session analytics → cue detection
- Test batch jobs
- Test risk alert generation

### Phase 7 Tests

**File**: `tests/workers/tasks/test_analytics_tasks.py`

- Test task execution
- Test retry logic
- Test batch processing

---

## Acceptance Criteria Checklist

### AC1: Session Analytics
- [ ] Analytics computed automatically on transcription completion
- [ ] Talk-time percentages calculated accurately
- [ ] Turn counts and durations recorded
- [ ] Coach question/statement ratio computed
- [ ] Engagement score generated (0-100)

### AC2: Language Cue Detection
- [ ] Resistance cues detected with >70% confidence
- [ ] Commitment cues detected
- [ ] Breakthrough moments identified
- [ ] Cues linked to specific utterances
- [ ] Text excerpt stored (PII-redacted, max 200 chars)
- [ ] Preceding context stored when available (PII-redacted, optional)

### AC3: Client Analytics
- [ ] Daily batch computation runs successfully
- [ ] 30-day and 90-day windows supported
- [ ] Trends calculated (improving/stable/declining)
- [ ] Session frequency tracked
- [ ] Historical data maintained

### AC4: Risk Scoring
- [ ] Risk score computed (0-100 scale)
- [ ] Risk levels assigned (low/medium/high/critical)
- [ ] Contributing factors listed
- [ ] Recommended actions generated
- [ ] Previous score tracked for trend

### AC5: Risk Alerts
- [ ] Alerts generated for high-risk clients
- [ ] Alerts generated for risk increases
- [ ] Alerts generated for 30+ days without session
- [ ] Acknowledgment workflow works
- [ ] Coach only sees their own alerts

### AC6: API Completeness
- [ ] All endpoints return appropriate data
- [ ] Pagination on list endpoints
- [ ] Proper error responses
- [ ] Authorization enforced

### AC7: Performance
- [ ] Session analytics computed in <30 seconds
- [ ] Batch jobs complete within 1 hour (1000 clients)
- [ ] API responses in <500ms
- [ ] No impact on transcription pipeline

### AC8: Security
- [ ] Coach can only access their own clients
- [ ] 404 returned for unauthorized access
- [ ] No PII in logs
- [ ] Rate limiting applied

---

## File Summary

### New Files
```
scripts/migrations/0005_analytics_tables.sql
app/models/analytics.py
app/schemas/analytics.py
app/services/analytics/__init__.py
app/services/analytics/session_analytics.py
app/services/analytics/talk_time.py
app/services/analytics/coaching_style.py
app/services/analytics/engagement.py
app/services/analytics/sentiment.py
app/services/analytics/cue_detection.py
app/services/analytics/prompts.py
app/services/analytics/client_analytics.py
app/services/analytics/trends.py
app/services/analytics/risk_scoring.py
app/services/analytics/alerts.py
app/services/analytics/authorization.py
app/api/v1/analytics.py
app/workers/tasks/analytics.py
tests/services/analytics/test_talk_time.py
tests/services/analytics/test_coaching_style.py
tests/services/analytics/test_engagement.py
tests/services/analytics/test_cue_detection.py
tests/services/analytics/test_client_analytics.py
tests/services/analytics/test_trends.py
tests/services/analytics/test_risk_scoring.py
tests/services/analytics/test_alerts.py
tests/api/v1/test_analytics.py
tests/workers/tasks/test_analytics_tasks.py
tests/integration/test_analytics_pipeline.py
```

### Modified Files
```
app/models/__init__.py
app/schemas/__init__.py
app/api/v1/router.py
app/workers/celery_app.py
app/workers/tasks/transcription.py
pyproject.toml (if any new deps needed)
```

---

**Plan Status**: Ready for review
**Author**: Architect
**Created**: 2026-01-27
