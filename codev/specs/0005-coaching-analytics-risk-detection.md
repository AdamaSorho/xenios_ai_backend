# Spec 0005: Coaching Analytics & Risk Detection

## Overview

**What**: Build an analytics system that processes coaching session data to surface talk-time metrics, language cue patterns, sentiment trends, and predictive client churn risk scores.

**Why**: Coaches lack visibility into long-term patterns in their client relationships:
- No aggregate view of session dynamics (who talks more, engagement levels)
- Subtle language cues indicating client disengagement go unnoticed
- Sentiment shifts across sessions are hard to track manually
- Client churn happens without early warning signals

Analytics enables:
- Talk-time dashboards showing coach vs client speaking ratios
- Language pattern detection for resistance, commitment, breakthroughs
- Sentiment trajectory visualization across sessions
- Proactive churn risk alerts before clients disengage

**Who**:
- Coaches reviewing their coaching effectiveness
- Coaches identifying at-risk clients
- System generating risk alerts proactively
- Admin users viewing aggregate coach performance (future)

## Goals

### Must Have
1. Talk-time analysis per session (coach % vs client % speaking time)
2. Aggregate talk-time trends across sessions per client
3. Language cue detection (resistance, commitment, breakthrough, concern)
4. Sentiment tracking per session with trend analysis
5. Client engagement score calculation (composite metric)
6. Churn risk scoring based on multiple signals
7. Risk alert generation for high-risk clients
8. API endpoints for analytics retrieval
9. Celery tasks for batch analytics computation

### Should Have
- Coaching effectiveness metrics (question-to-statement ratio)
- Session-over-session comparison
- Client response latency patterns (from real-time sessions)
- Risk score explanation (which factors contributed)
- Configurable risk thresholds per coach
- Historical risk score tracking

### Won't Have (MVP)
- Real-time analytics during live sessions
- Coach comparison/benchmarking (privacy concerns)
- Predictive session outcomes
- Audio-based analytics (tone, pace) - text only
- Client self-service analytics access

## Technical Context

### Data Sources (from Spec 0003)

| Source | Table | Key Fields for Analytics |
|--------|-------|-------------------------|
| Transcription Jobs | `ai_backend.transcription_jobs` | client_id, coach_id, session_date, status |
| Transcripts | `ai_backend.transcripts` | duration_seconds, word_count, confidence_score |
| Utterances | `ai_backend.utterances` | speaker_label, text, start_time, end_time, intent, sentiment |
| Session Summaries | `ai_backend.session_summaries` | client_sentiment, engagement_score, coaching_moments |

### Analytics Computation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Analytics Pipeline                           │
│                                                                 │
│  1. Session Completion Trigger                                  │
│     └── Transcription job → status: completed                   │
│                                                                 │
│  2. Session Analytics (immediate)                               │
│     ├── Talk-time calculation                                   │
│     ├── Cue pattern detection                                   │
│     ├── Sentiment scoring                                       │
│     └── Store: ai_backend.session_analytics                     │
│                                                                 │
│  3. Client Analytics (batch, daily)                             │
│     ├── Aggregate talk-time trends                              │
│     ├── Sentiment trajectory                                    │
│     ├── Engagement trend                                        │
│     └── Store: ai_backend.client_analytics                      │
│                                                                 │
│  4. Risk Scoring (batch, daily)                                 │
│     ├── Multi-signal analysis                                   │
│     ├── Churn probability calculation                           │
│     ├── Risk alert generation                                   │
│     └── Store: ai_backend.risk_scores                           │
└─────────────────────────────────────────────────────────────────┘
```

### Integration with MVP

```
┌─────────────────────────────────────────────────────────────────┐
│                     Xenios MVP (Next.js)                        │
│                                                                 │
│  New endpoints to add:                                          │
│  GET /api/analytics/clients/{id}/sessions ──────────────────┼───┐
│  GET /api/analytics/clients/{id}/risk ──────────────────────┼───┤
│  GET /api/analytics/coach/at-risk-clients ──────────────────┼───┤
│                                                                 │
│  Dashboard components:                                          │
│  - Session analytics cards                                      │
│  - Client risk indicators                                       │
│  - At-risk client list                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Xenios AI Backend (This Spec)                 │
│                                                                 │
│  GET /api/v1/analytics/sessions/{job_id}                        │
│       → Single session analytics                                │
│                                                                 │
│  GET /api/v1/analytics/clients/{client_id}/summary              │
│       → Aggregate client analytics                              │
│                                                                 │
│  GET /api/v1/analytics/clients/{client_id}/sessions             │
│       → Session history with analytics                          │
│                                                                 │
│  GET /api/v1/analytics/clients/{client_id}/risk                 │
│       → Current risk score and factors                          │
│                                                                 │
│  GET /api/v1/analytics/risk/alerts                              │
│       → All at-risk clients for coach                           │
│                                                                 │
│  POST /api/v1/analytics/risk/acknowledge/{alert_id}             │
│       → Mark alert as reviewed                                  │
│                                                                 │
│  Celery Workers:                                                │
│       → compute_session_analytics (on transcription complete)   │
│       → compute_client_analytics (daily batch)                  │
│       → compute_risk_scores (daily batch)                       │
│       → generate_risk_alerts (daily batch)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Implementation

### Data Models

#### Session Analytics

```python
class SessionAnalytics(BaseModel):
    id: UUID
    job_id: UUID  # Reference to transcription job
    client_id: UUID
    coach_id: UUID
    session_date: date

    # Talk-time metrics
    total_duration_seconds: float
    coach_talk_time_seconds: float
    client_talk_time_seconds: float
    silence_time_seconds: float
    coach_talk_percentage: float  # 0-100
    client_talk_percentage: float  # 0-100

    # Turn-taking metrics
    total_turns: int
    coach_turns: int
    client_turns: int
    average_turn_duration_coach: float  # seconds
    average_turn_duration_client: float
    interruption_count: int  # overlapping speech

    # Coaching style metrics
    coach_question_count: int
    coach_statement_count: int
    question_to_statement_ratio: float
    open_question_count: int
    closed_question_count: int

    # Language cue counts
    cue_resistance_count: int
    cue_commitment_count: int
    cue_breakthrough_count: int
    cue_concern_count: int
    cue_deflection_count: int

    # Sentiment
    client_sentiment_score: float  # -1 to 1
    coach_sentiment_score: float
    sentiment_variance: float  # How much sentiment changed

    # Engagement
    engagement_score: float  # 0-100 composite
    response_elaboration_score: float  # How detailed are client responses

    # Metadata
    computed_at: datetime
    model_version: str  # For tracking algorithm changes


class LanguageCue(BaseModel):
    """Individual language cue detected in a session."""
    id: UUID
    session_analytics_id: UUID
    utterance_id: UUID  # Reference to specific utterance

    cue_type: CueType
    confidence: float  # 0-1
    text_excerpt: str  # The relevant text
    timestamp: float  # Seconds into session

    # Context
    preceding_context: str | None  # What was said before
    interpretation: str  # Why this is significant


class CueType(str, Enum):
    RESISTANCE = "resistance"      # "I don't think I can..." / "That won't work for me"
    COMMITMENT = "commitment"       # "I will..." / "I'm going to..."
    BREAKTHROUGH = "breakthrough"   # "I just realized..." / "That makes sense now"
    CONCERN = "concern"            # "I'm worried about..." / "What if..."
    DEFLECTION = "deflection"      # Changing subject, avoiding topic
    ENTHUSIASM = "enthusiasm"      # Positive energy, excitement
    DOUBT = "doubt"                # Uncertainty, hesitation
    AGREEMENT = "agreement"        # Aligned with coach
    GOAL_SETTING = "goal_setting"  # Setting concrete objectives
```

#### Client Analytics (Aggregate)

```python
class ClientAnalytics(BaseModel):
    id: UUID
    client_id: UUID
    coach_id: UUID

    # Time window
    window_start: date
    window_end: date
    window_type: str  # "30d", "90d", "all_time"

    # Session frequency
    total_sessions: int
    sessions_last_30_days: int
    average_days_between_sessions: float
    days_since_last_session: int
    session_frequency_trend: str  # "increasing", "stable", "decreasing"

    # Talk-time trends
    average_coach_talk_percentage: float
    average_client_talk_percentage: float
    talk_ratio_trend: str  # "coach_increasing", "balanced", "client_increasing"

    # Engagement trends
    average_engagement_score: float
    engagement_trend: str  # "improving", "stable", "declining"
    engagement_scores_history: list[float]  # Last N sessions

    # Sentiment trends
    average_sentiment_score: float
    sentiment_trend: str  # "improving", "stable", "declining"
    sentiment_scores_history: list[float]

    # Cue pattern summary
    total_resistance_cues: int
    total_commitment_cues: int
    total_breakthrough_cues: int
    resistance_to_commitment_ratio: float

    # Computed at
    computed_at: datetime
```

#### Risk Score

```python
class RiskScore(BaseModel):
    id: UUID
    client_id: UUID
    coach_id: UUID

    # Overall risk
    risk_score: float  # 0-100, higher = more at risk
    risk_level: RiskLevel  # low, medium, high, critical
    churn_probability: float  # 0-1 probability of disengagement

    # Risk factors (each contributes to overall score)
    factors: list[RiskFactor]

    # Trend
    previous_risk_score: float | None
    score_change: float  # Positive = increasing risk
    trend: str  # "improving", "stable", "worsening"

    # Recommendation
    recommended_action: str  # What coach should do

    # Metadata
    computed_at: datetime
    valid_until: datetime  # Scores expire after 7 days
    model_version: str


class RiskFactor(BaseModel):
    factor_type: str  # session_frequency, engagement, sentiment, etc.
    contribution: float  # 0-100 how much this affects total score
    value: float  # The actual metric value
    threshold: float  # The threshold for concern
    description: str  # Human-readable explanation


class RiskLevel(str, Enum):
    LOW = "low"          # 0-25
    MEDIUM = "medium"    # 26-50
    HIGH = "high"        # 51-75
    CRITICAL = "critical"  # 76-100


class RiskAlert(BaseModel):
    id: UUID
    client_id: UUID
    coach_id: UUID
    risk_score_id: UUID

    alert_type: str  # "new_high_risk", "risk_increased", "no_session_30d"
    severity: str  # "warning", "urgent"
    title: str
    message: str

    # Status
    status: str  # "pending", "acknowledged", "resolved"
    acknowledged_at: datetime | None
    resolved_at: datetime | None

    created_at: datetime
```

### Project Structure (Additions)

```
app/
├── services/
│   └── analytics/
│       ├── __init__.py
│       ├── session_analytics.py    # Per-session computation
│       ├── client_analytics.py     # Aggregate computation
│       ├── cue_detection.py        # Language cue detection
│       ├── risk_scoring.py         # Churn risk calculation
│       └── alerts.py               # Risk alert generation
│
├── workers/
│   └── tasks/
│       └── analytics.py            # Analytics Celery tasks
│
├── api/
│   └── v1/
│       └── analytics.py            # Analytics API endpoints
│
├── models/
│   └── analytics.py                # SQLAlchemy models
│
└── schemas/
    └── analytics.py                # Request/response schemas
```

### API Endpoints

```
GET /api/v1/analytics/sessions/{job_id}
  Response:
    - session_analytics: SessionAnalytics
    - cues: list[LanguageCue]
    - comparison: SessionComparison | None  # vs previous session

GET /api/v1/analytics/clients/{client_id}/summary
  Query params:
    - window: string (30d, 90d, all_time) - default 90d
  Response:
    - client_analytics: ClientAnalytics
    - session_count: int
    - latest_session_date: date
    - risk_level: string

GET /api/v1/analytics/clients/{client_id}/sessions
  Query params:
    - limit: int (default 20)
    - offset: int (default 0)
    - from_date: date (optional)
    - to_date: date (optional)
  Response:
    - sessions: list[SessionAnalyticsSummary]
    - total: int

GET /api/v1/analytics/clients/{client_id}/risk
  Response:
    - risk_score: RiskScore
    - history: list[RiskScoreHistory]  # Last 5 scores
    - alerts: list[RiskAlert]  # Active alerts

GET /api/v1/analytics/clients/{client_id}/trends
  Query params:
    - metrics: list[string]  # engagement, sentiment, talk_ratio
    - window: string (30d, 90d, all_time)
  Response:
    - trends: dict[metric, TrendData]
    - data_points: list[DataPoint]

GET /api/v1/analytics/risk/alerts
  Query params:
    - status: string (pending, acknowledged, all) - default pending
    - severity: string (warning, urgent, all) - default all
    - limit: int (default 50)
  Response:
    - alerts: list[RiskAlert]
    - total: int

POST /api/v1/analytics/risk/alerts/{alert_id}/acknowledge
  Request:
    - notes: string (optional)
  Response:
    - alert: RiskAlert (updated)

POST /api/v1/analytics/compute/{client_id}
  - Trigger manual recomputation of client analytics
  - Returns: 202 Accepted with task_id

GET /api/v1/analytics/coach/summary
  Response:
    - total_clients: int
    - clients_at_risk: int
    - average_engagement: float
    - sessions_this_month: int
    - risk_distribution: dict[RiskLevel, int]
```

### Analytics Computation

#### Talk-Time Analysis

```python
class TalkTimeAnalyzer:
    """Compute talk-time metrics from utterances."""

    def compute(self, utterances: list[Utterance]) -> TalkTimeMetrics:
        coach_time = sum(
            u.end_time - u.start_time
            for u in utterances
            if u.speaker_label == "coach"
        )
        client_time = sum(
            u.end_time - u.start_time
            for u in utterances
            if u.speaker_label == "client"
        )
        total_time = utterances[-1].end_time if utterances else 0
        silence_time = total_time - coach_time - client_time

        coach_turns = [u for u in utterances if u.speaker_label == "coach"]
        client_turns = [u for u in utterances if u.speaker_label == "client"]

        return TalkTimeMetrics(
            total_duration_seconds=total_time,
            coach_talk_time_seconds=coach_time,
            client_talk_time_seconds=client_time,
            silence_time_seconds=max(0, silence_time),
            coach_talk_percentage=(coach_time / total_time * 100) if total_time > 0 else 0,
            client_talk_percentage=(client_time / total_time * 100) if total_time > 0 else 0,
            total_turns=len(utterances),
            coach_turns=len(coach_turns),
            client_turns=len(client_turns),
            average_turn_duration_coach=self._avg_duration(coach_turns),
            average_turn_duration_client=self._avg_duration(client_turns),
            interruption_count=self._count_interruptions(utterances),
        )

    def _count_interruptions(self, utterances: list[Utterance]) -> int:
        """Count overlapping speech (approximation)."""
        count = 0
        for i in range(1, len(utterances)):
            # If next utterance starts before previous ends
            if utterances[i].start_time < utterances[i-1].end_time:
                count += 1
        return count
```

#### Language Cue Detection

```python
CUE_DETECTION_PROMPT = """Analyze this coaching session utterance for language cues.

UTTERANCE: {text}
SPEAKER: {speaker_label}
CONTEXT (previous 2 utterances): {context}

Identify any of these cue types present:
- resistance: Client pushing back, expressing doubt, or avoiding ("I don't think I can...", "That won't work")
- commitment: Client making promises or expressing determination ("I will...", "I'm going to...")
- breakthrough: Moments of insight or realization ("I just realized...", "That makes sense now")
- concern: Client expressing worry or anxiety ("I'm worried about...", "What if...")
- deflection: Changing subject, giving vague answers, avoiding specifics
- enthusiasm: Expressing excitement or positive energy
- doubt: Hesitation or uncertainty in responses
- goal_setting: Setting concrete, measurable objectives

For each cue found, provide:
1. cue_type: The type from the list above
2. confidence: 0.0-1.0 how confident you are
3. interpretation: Brief explanation of why this is significant

If no significant cues are present, return empty array.

Respond in JSON format:
{
  "cues": [
    {"cue_type": "...", "confidence": 0.X, "interpretation": "..."}
  ]
}
"""

class CueDetectionService:
    """Detect language cues using LLM analysis."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.min_confidence = 0.7  # Only keep high-confidence cues

    async def detect_cues(
        self,
        utterances: list[Utterance],
        batch_size: int = 10,
    ) -> list[LanguageCue]:
        """Detect cues in client utterances."""
        cues = []

        # Only analyze client utterances (coach utterances less relevant for risk)
        client_utterances = [u for u in utterances if u.speaker_label == "client"]

        for i, utterance in enumerate(client_utterances):
            # Get context (previous utterances)
            context = self._get_context(utterances, utterance)

            detected = await self._analyze_utterance(utterance, context)

            for cue in detected:
                if cue.confidence >= self.min_confidence:
                    cues.append(LanguageCue(
                        utterance_id=utterance.id,
                        cue_type=cue.cue_type,
                        confidence=cue.confidence,
                        text_excerpt=utterance.text[:200],
                        timestamp=utterance.start_time,
                        preceding_context=context,
                        interpretation=cue.interpretation,
                    ))

        return cues
```

#### Risk Scoring Algorithm

```python
class RiskScoringService:
    """Calculate client churn risk score."""

    # Factor weights (must sum to 100)
    WEIGHTS = {
        "session_frequency": 25,
        "engagement_trend": 25,
        "sentiment_trend": 20,
        "resistance_ratio": 15,
        "days_since_session": 15,
    }

    # Thresholds for concern
    THRESHOLDS = {
        "days_since_session_warning": 14,
        "days_since_session_critical": 30,
        "engagement_decline_warning": -10,  # percentage points
        "engagement_decline_critical": -20,
        "sentiment_decline_warning": -0.2,  # on -1 to 1 scale
        "sentiment_decline_critical": -0.4,
        "resistance_ratio_warning": 2.0,  # 2x more resistance than commitment
        "resistance_ratio_critical": 4.0,
        "session_frequency_decline_warning": 0.5,  # 50% fewer sessions
        "session_frequency_decline_critical": 0.75,
    }

    async def compute_risk_score(
        self,
        client_id: UUID,
        client_analytics: ClientAnalytics,
    ) -> RiskScore:
        factors = []

        # 1. Session frequency factor
        freq_factor = self._compute_frequency_factor(client_analytics)
        factors.append(freq_factor)

        # 2. Engagement trend factor
        engagement_factor = self._compute_engagement_factor(client_analytics)
        factors.append(engagement_factor)

        # 3. Sentiment trend factor
        sentiment_factor = self._compute_sentiment_factor(client_analytics)
        factors.append(sentiment_factor)

        # 4. Resistance ratio factor
        resistance_factor = self._compute_resistance_factor(client_analytics)
        factors.append(resistance_factor)

        # 5. Days since last session factor
        recency_factor = self._compute_recency_factor(client_analytics)
        factors.append(recency_factor)

        # Calculate weighted total
        total_score = sum(f.contribution for f in factors)

        # Determine risk level
        risk_level = self._score_to_level(total_score)

        # Generate recommendation
        recommendation = self._generate_recommendation(factors, risk_level)

        return RiskScore(
            client_id=client_id,
            risk_score=total_score,
            risk_level=risk_level,
            churn_probability=total_score / 100,  # Simple linear mapping
            factors=factors,
            recommended_action=recommendation,
        )

    def _compute_frequency_factor(self, analytics: ClientAnalytics) -> RiskFactor:
        """Score based on session frequency decline."""
        # Compare last 30 days to previous 30 days
        if analytics.session_frequency_trend == "decreasing":
            # Calculate severity
            if analytics.sessions_last_30_days == 0:
                contribution = self.WEIGHTS["session_frequency"]  # Max risk
            else:
                decline_ratio = 1 - (analytics.sessions_last_30_days / max(analytics.total_sessions / 3, 1))
                contribution = min(self.WEIGHTS["session_frequency"],
                                   decline_ratio * self.WEIGHTS["session_frequency"])
        else:
            contribution = 0

        return RiskFactor(
            factor_type="session_frequency",
            contribution=contribution,
            value=analytics.sessions_last_30_days,
            threshold=self.THRESHOLDS["session_frequency_decline_warning"],
            description=f"Client had {analytics.sessions_last_30_days} sessions in last 30 days",
        )

    def _score_to_level(self, score: float) -> RiskLevel:
        if score <= 25:
            return RiskLevel.LOW
        elif score <= 50:
            return RiskLevel.MEDIUM
        elif score <= 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_recommendation(
        self,
        factors: list[RiskFactor],
        level: RiskLevel
    ) -> str:
        """Generate actionable recommendation based on risk factors."""
        top_factors = sorted(factors, key=lambda f: f.contribution, reverse=True)[:2]

        recommendations = {
            ("session_frequency", RiskLevel.HIGH): "Schedule a check-in call to re-engage",
            ("session_frequency", RiskLevel.CRITICAL): "Urgent: Reach out immediately to prevent churn",
            ("engagement_trend", RiskLevel.HIGH): "Try new engagement strategies in next session",
            ("sentiment_trend", RiskLevel.HIGH): "Address underlying concerns in next conversation",
            ("resistance_ratio", RiskLevel.HIGH): "Review recent sessions for resistance patterns",
            ("days_since_session", RiskLevel.CRITICAL): "Client hasn't been seen in 30+ days - follow up",
        }

        for factor in top_factors:
            key = (factor.factor_type, level)
            if key in recommendations:
                return recommendations[key]

        if level == RiskLevel.LOW:
            return "Continue current coaching approach"
        elif level == RiskLevel.MEDIUM:
            return "Monitor closely and consider proactive check-in"
        else:
            return "Take immediate action to re-engage client"
```

### Database Schema

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

    -- Language cue counts
    cue_resistance_count INTEGER DEFAULT 0,
    cue_commitment_count INTEGER DEFAULT 0,
    cue_breakthrough_count INTEGER DEFAULT 0,
    cue_concern_count INTEGER DEFAULT 0,
    cue_deflection_count INTEGER DEFAULT 0,

    -- Sentiment
    client_sentiment_score DECIMAL(4,3),  -- -1 to 1
    coach_sentiment_score DECIMAL(4,3),
    sentiment_variance DECIMAL(4,3),

    -- Engagement
    engagement_score DECIMAL(5,2),  -- 0-100
    response_elaboration_score DECIMAL(5,2),

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
    window_type VARCHAR(20) NOT NULL,  -- 30d, 90d, all_time

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
    engagement_scores_history JSONB,  -- Array of floats

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

    risk_score DECIMAL(5,2) NOT NULL,  -- 0-100
    risk_level VARCHAR(20) NOT NULL,
    churn_probability DECIMAL(4,3) NOT NULL,  -- 0-1

    factors JSONB NOT NULL,  -- Array of RiskFactor

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

### Celery Tasks

```python
# app/workers/tasks/analytics.py

@celery_app.task(bind=True, max_retries=3)
def compute_session_analytics(
    self,
    job_id: str,
) -> dict:
    """
    Compute analytics for a completed transcription.
    Triggered when transcription job status -> completed.
    """
    try:
        # 1. Fetch transcript and utterances
        # 2. Compute talk-time metrics
        # 3. Detect language cues
        # 4. Calculate engagement score
        # 5. Store SessionAnalytics
        # 6. Trigger client analytics update
        pass
    except Exception as e:
        self.retry(exc=e, countdown=60)


@celery_app.task
def compute_client_analytics_batch():
    """
    Daily batch job to compute aggregate client analytics.
    Scheduled via Celery Beat at 2:00 AM.
    """
    # For each active client:
    # 1. Aggregate session analytics
    # 2. Calculate trends
    # 3. Update ClientAnalytics
    pass


@celery_app.task
def compute_risk_scores_batch():
    """
    Daily batch job to compute risk scores.
    Scheduled via Celery Beat at 3:00 AM (after client analytics).
    """
    # For each active client:
    # 1. Get latest ClientAnalytics
    # 2. Compute risk score
    # 3. Store RiskScore
    # 4. Generate alerts if needed
    pass


@celery_app.task
def generate_risk_alerts():
    """
    Generate alerts for high-risk clients.
    Called after risk score computation.
    """
    # Find clients with:
    # - New high/critical risk level
    # - Significant risk increase
    # - No session in 30+ days
    # Create RiskAlert records
    pass
```

### Celery Beat Schedule

```python
CELERY_BEAT_SCHEDULE = {
    "compute-client-analytics-daily": {
        "task": "app.workers.tasks.analytics.compute_client_analytics_batch",
        "schedule": crontab(hour=2, minute=0),  # 2:00 AM
    },
    "compute-risk-scores-daily": {
        "task": "app.workers.tasks.analytics.compute_risk_scores_batch",
        "schedule": crontab(hour=3, minute=0),  # 3:00 AM
    },
    "generate-risk-alerts-daily": {
        "task": "app.workers.tasks.analytics.generate_risk_alerts",
        "schedule": crontab(hour=3, minute=30),  # 3:30 AM
    },
}
```

## Security & Authorization

### Authentication
All endpoints require:
1. **X-API-Key header**: Backend-to-backend authentication (from MVP)
2. **Authorization header**: Bearer JWT from Supabase Auth

### Authorization Rules

| Endpoint | Who Can Access |
|----------|----------------|
| GET /sessions/{job_id} | Coach who owns the job |
| GET /clients/{client_id}/* | Coach with coach-client relationship |
| GET /risk/alerts | Coach (sees only their alerts) |
| POST /risk/alerts/{id}/acknowledge | Coach who owns the alert |
| POST /compute/{client_id} | Coach with coach-client relationship |
| GET /coach/summary | Coach (sees only their data) |

**Ownership enforcement**:
- `coach_id` extracted from JWT claims
- Client access verified via `coach_clients` relationship table
- 404 returned for unauthorized access (not 403)

## Error Handling

### Computation Failures

| Failure | Behavior |
|---------|----------|
| LLM cue detection fails | Log error, skip cues (rest of analytics still computed) |
| Database write fails | Retry 3x with exponential backoff |
| Missing utterances | Return partial analytics with warning flag |
| No sessions found | Return empty analytics (not error) |

### Edge Cases

| Case | Handling |
|------|----------|
| Client with 1 session | No trends (need 2+ sessions), risk score still computed |
| Client with 0 sessions | Return 404 for analytics endpoints |
| Very short session (<2 min) | Flag as potentially incomplete, still compute |
| Single speaker session | Talk-time 100% for one speaker, engagement scored differently |

## Acceptance Criteria

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
- [ ] Context preserved for each cue

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

## Test Plan

### Unit Tests
- Talk-time calculator with various utterance patterns
- Cue detection prompt formatting
- Risk score calculation
- Trend calculation (increasing/stable/declining)
- Alert generation logic

### Integration Tests
- Full session analytics computation flow
- Client analytics aggregation
- Risk score computation end-to-end
- API endpoint tests
- Celery task execution

### Test Scenarios
- Client with improving engagement (risk should be low)
- Client with declining sentiment (risk should increase)
- Client with no sessions in 30 days (alert generated)
- New client with 1 session (no trends, baseline risk)
- Client with high resistance cues (risk factor triggered)

### Manual Testing
- Review cue detection accuracy on real sessions
- Validate risk scores against coach intuition
- Test alert acknowledgment flow

## Dependencies

- **Spec 0001**: AI Backend Foundation (Celery, Redis, LLM client)
- **Spec 0003**: Transcription & Session Processing (utterance data)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM cue detection inaccurate | Medium | Medium | Confidence thresholds; human review option |
| Risk score doesn't match reality | Medium | High | Validate with coaches; tunable thresholds |
| Batch jobs too slow | Low | Medium | Parallelize; incremental updates |
| Alert fatigue | Medium | Medium | Severity levels; configurable thresholds |
| Privacy concerns with analytics | Low | High | Clear data access policies; audit logging |

## Performance Requirements

| Operation | Target | Maximum |
|-----------|--------|---------|
| Session analytics computation | 15 seconds | 30 seconds |
| Cue detection (per session) | 20 seconds | 45 seconds |
| Client analytics query | 100ms | 300ms |
| Risk score query | 50ms | 150ms |
| Batch job (1000 clients) | 30 minutes | 60 minutes |

## Cost Considerations

### LLM Usage (Cue Detection)
- ~20 utterances analyzed per session
- ~500 tokens per analysis
- ~$0.05 per session
- 100 sessions/month = ~$5

### Database Storage
- SessionAnalytics: ~2KB per session
- LanguageCues: ~500 bytes per cue (~10 per session = 5KB)
- ClientAnalytics: ~1KB per client
- RiskScores: ~500 bytes per score (daily)

**Storage estimate**: 100 clients × 10 sessions/month = ~1MB/month

## Open Questions

1. **Risk threshold tuning**: Should coaches be able to adjust their own risk thresholds?
2. **Alert preferences**: Should coaches set preferences for which alerts they receive?
3. **Benchmark data**: Should we eventually provide anonymized benchmarks for coach comparison?

## Future Considerations

- Real-time analytics during live sessions (WebSocket)
- Coach effectiveness benchmarking (opt-in)
- Predictive session preparation (suggest topics before session)
- Integration with client goals (risk based on goal progress)
- Voice-based analytics (tone, pace, energy from audio)
- Multi-client pattern detection (coach-level insights)

---

**Spec Status**: Ready for review
**Author**: Architect
**Created**: 2026-01-27
