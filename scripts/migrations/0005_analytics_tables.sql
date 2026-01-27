-- Migration: 0005_analytics_tables.sql
-- Description: Create tables for coaching analytics and risk detection (Spec 0005)
-- Author: Builder
-- Date: 2026-01-27

-- ============================================================================
-- Session Analytics Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.session_analytics (
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
    client_sentiment_score DECIMAL(4,3),  -- -1 to 1
    coach_sentiment_score DECIMAL(4,3),
    sentiment_variance DECIMAL(4,3),

    -- Engagement
    engagement_score DECIMAL(5,2),  -- 0-100
    response_elaboration_score DECIMAL(5,2),

    -- Quality flags
    quality_warning BOOLEAN DEFAULT FALSE,
    quality_warnings JSONB DEFAULT '[]',

    -- Metadata
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    model_version VARCHAR(50) NOT NULL,

    UNIQUE(job_id)
);

CREATE INDEX IF NOT EXISTS idx_session_analytics_client ON ai_backend.session_analytics(client_id);
CREATE INDEX IF NOT EXISTS idx_session_analytics_coach ON ai_backend.session_analytics(coach_id);
CREATE INDEX IF NOT EXISTS idx_session_analytics_date ON ai_backend.session_analytics(session_date DESC);
CREATE INDEX IF NOT EXISTS idx_session_analytics_coach_client ON ai_backend.session_analytics(coach_id, client_id);

-- ============================================================================
-- Language Cues Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.language_cues (
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

CREATE INDEX IF NOT EXISTS idx_language_cues_session ON ai_backend.language_cues(session_analytics_id);
CREATE INDEX IF NOT EXISTS idx_language_cues_type ON ai_backend.language_cues(cue_type);

-- ============================================================================
-- Client Analytics Table (Aggregate)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.client_analytics (
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

CREATE INDEX IF NOT EXISTS idx_client_analytics_client ON ai_backend.client_analytics(client_id);
CREATE INDEX IF NOT EXISTS idx_client_analytics_coach ON ai_backend.client_analytics(coach_id);
CREATE INDEX IF NOT EXISTS idx_client_analytics_coach_client ON ai_backend.client_analytics(coach_id, client_id);

-- ============================================================================
-- Risk Scores Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.risk_scores (
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

CREATE INDEX IF NOT EXISTS idx_risk_scores_client ON ai_backend.risk_scores(client_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_coach ON ai_backend.risk_scores(coach_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_level ON ai_backend.risk_scores(risk_level);
CREATE INDEX IF NOT EXISTS idx_risk_scores_computed ON ai_backend.risk_scores(computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_scores_valid ON ai_backend.risk_scores(valid_until);

-- ============================================================================
-- Risk Alerts Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.risk_alerts (
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

CREATE INDEX IF NOT EXISTS idx_risk_alerts_coach ON ai_backend.risk_alerts(coach_id);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_client ON ai_backend.risk_alerts(client_id);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_status ON ai_backend.risk_alerts(status);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_severity ON ai_backend.risk_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_created ON ai_backend.risk_alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_coach_status ON ai_backend.risk_alerts(coach_id, status);

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE ai_backend.session_analytics IS 'Per-session analytics computed from transcripts';
COMMENT ON TABLE ai_backend.language_cues IS 'Language cues detected in client utterances';
COMMENT ON TABLE ai_backend.client_analytics IS 'Aggregate analytics per client over time windows';
COMMENT ON TABLE ai_backend.risk_scores IS 'Churn risk scores for clients';
COMMENT ON TABLE ai_backend.risk_alerts IS 'Risk alerts generated for coaches';

COMMENT ON COLUMN ai_backend.session_analytics.model_version IS 'Version of analytics algorithm used';
COMMENT ON COLUMN ai_backend.session_analytics.quality_warning IS 'True if data quality issues detected';
COMMENT ON COLUMN ai_backend.session_analytics.quality_warnings IS 'Array of quality warning types';
COMMENT ON COLUMN ai_backend.language_cues.cue_type IS 'resistance, commitment, breakthrough, concern, deflection, enthusiasm, doubt, agreement, goal_setting';
COMMENT ON COLUMN ai_backend.client_analytics.window_type IS '30d, 90d, or all_time';
COMMENT ON COLUMN ai_backend.client_analytics.session_frequency_trend IS 'increasing, stable, or decreasing';
COMMENT ON COLUMN ai_backend.client_analytics.engagement_trend IS 'improving, stable, or declining';
COMMENT ON COLUMN ai_backend.risk_scores.risk_level IS 'low, medium, high, or critical';
COMMENT ON COLUMN ai_backend.risk_scores.valid_until IS 'Risk scores expire after 7 days';
COMMENT ON COLUMN ai_backend.risk_alerts.alert_type IS 'new_high_risk, risk_increased, no_session_30d';
COMMENT ON COLUMN ai_backend.risk_alerts.severity IS 'warning or urgent';
COMMENT ON COLUMN ai_backend.risk_alerts.status IS 'pending, acknowledged, or resolved';
