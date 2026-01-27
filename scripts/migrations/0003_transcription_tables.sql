-- Migration: 0003_transcription_tables.sql
-- Description: Create tables for transcription and session processing (Spec 0003)
-- Author: Builder
-- Date: 2025-01-27

-- ============================================================================
-- Transcription Jobs Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.transcription_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,
    conversation_id UUID,  -- Link to MVP conversations table

    -- Audio file
    audio_url TEXT NOT NULL,
    audio_filename VARCHAR(255) NOT NULL,
    audio_format VARCHAR(20) NOT NULL,
    audio_size_bytes INTEGER NOT NULL,
    audio_duration_seconds DECIMAL(10,2),

    -- Processing status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    transcription_completed_at TIMESTAMPTZ,
    summary_completed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Session metadata
    session_date DATE,
    session_type VARCHAR(50),
    session_title VARCHAR(255),

    -- Webhook
    webhook_url TEXT,
    webhook_secret TEXT,
    webhook_sent_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for transcription_jobs
CREATE INDEX IF NOT EXISTS idx_transcription_jobs_client
    ON ai_backend.transcription_jobs(client_id);
CREATE INDEX IF NOT EXISTS idx_transcription_jobs_coach
    ON ai_backend.transcription_jobs(coach_id);
CREATE INDEX IF NOT EXISTS idx_transcription_jobs_status
    ON ai_backend.transcription_jobs(status);
CREATE INDEX IF NOT EXISTS idx_transcription_jobs_created
    ON ai_backend.transcription_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transcription_jobs_coach_client
    ON ai_backend.transcription_jobs(coach_id, client_id);
CREATE INDEX IF NOT EXISTS idx_transcription_jobs_coach_status
    ON ai_backend.transcription_jobs(coach_id, status);

-- ============================================================================
-- Transcripts Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ai_backend.transcription_jobs(id) ON DELETE CASCADE,

    -- Full transcript
    full_text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    duration_seconds DECIMAL(10,2) NOT NULL,

    -- Deepgram metadata
    deepgram_request_id VARCHAR(100),
    model_used VARCHAR(50),
    confidence_score DECIMAL(4,3),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for transcripts
CREATE INDEX IF NOT EXISTS idx_transcripts_job
    ON ai_backend.transcripts(job_id);

-- Full-text search index on transcripts
CREATE INDEX IF NOT EXISTS idx_transcripts_fulltext
    ON ai_backend.transcripts USING gin(to_tsvector('english', full_text));

-- ============================================================================
-- Utterances Table (Speaker turns)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.utterances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_id UUID NOT NULL REFERENCES ai_backend.transcripts(id) ON DELETE CASCADE,

    -- Speaker identification
    speaker_number INTEGER NOT NULL,
    speaker_label VARCHAR(20),  -- 'coach', 'client', 'speaker', 'participant_N'
    speaker_confidence DECIMAL(4,3),

    -- Content
    text TEXT NOT NULL,
    start_time DECIMAL(10,3) NOT NULL,
    end_time DECIMAL(10,3) NOT NULL,
    confidence DECIMAL(4,3),

    -- Classification
    intent VARCHAR(50),
    sentiment VARCHAR(20),

    -- Ordering
    sequence_number INTEGER NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for utterances
CREATE INDEX IF NOT EXISTS idx_utterances_transcript
    ON ai_backend.utterances(transcript_id);
CREATE INDEX IF NOT EXISTS idx_utterances_speaker
    ON ai_backend.utterances(transcript_id, speaker_label);
CREATE INDEX IF NOT EXISTS idx_utterances_sequence
    ON ai_backend.utterances(transcript_id, sequence_number);

-- ============================================================================
-- Session Summaries Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.session_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ai_backend.transcription_jobs(id) ON DELETE CASCADE,
    transcript_id UUID NOT NULL REFERENCES ai_backend.transcripts(id) ON DELETE CASCADE,

    -- Summary content
    executive_summary TEXT NOT NULL,
    key_topics JSONB NOT NULL DEFAULT '[]',
    client_concerns JSONB NOT NULL DEFAULT '[]',
    coach_recommendations JSONB NOT NULL DEFAULT '[]',
    action_items JSONB NOT NULL DEFAULT '[]',
    goals_discussed JSONB NOT NULL DEFAULT '[]',
    coaching_moments JSONB NOT NULL DEFAULT '[]',

    -- Analysis results
    session_type_detected VARCHAR(50),
    client_sentiment VARCHAR(20),
    engagement_score DECIMAL(3,2),

    -- LLM metadata
    llm_model VARCHAR(100),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for session_summaries
CREATE INDEX IF NOT EXISTS idx_session_summaries_job
    ON ai_backend.session_summaries(job_id);
CREATE INDEX IF NOT EXISTS idx_session_summaries_transcript
    ON ai_backend.session_summaries(transcript_id);

-- ============================================================================
-- Updated_at Trigger
-- ============================================================================

-- Create trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION ai_backend.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add trigger to transcription_jobs
DROP TRIGGER IF EXISTS update_transcription_jobs_updated_at ON ai_backend.transcription_jobs;
CREATE TRIGGER update_transcription_jobs_updated_at
    BEFORE UPDATE ON ai_backend.transcription_jobs
    FOR EACH ROW
    EXECUTE FUNCTION ai_backend.update_updated_at_column();

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE ai_backend.transcription_jobs IS 'Audio transcription job tracking';
COMMENT ON TABLE ai_backend.transcripts IS 'Full transcripts from audio files';
COMMENT ON TABLE ai_backend.utterances IS 'Individual speaker turns with timing';
COMMENT ON TABLE ai_backend.session_summaries IS 'AI-generated session summaries';

COMMENT ON COLUMN ai_backend.transcription_jobs.status IS 'pending, uploading, transcribing, diarizing, summarizing, completed, partial, failed';
COMMENT ON COLUMN ai_backend.utterances.speaker_label IS 'coach, client, speaker (monologue), participant_N (multi-party)';
COMMENT ON COLUMN ai_backend.utterances.intent IS 'Intent classification: question_open, question_closed, reflection, advice, etc.';
COMMENT ON COLUMN ai_backend.session_summaries.session_type_detected IS 'nutrition, training, mindset, accountability, general';
COMMENT ON COLUMN ai_backend.session_summaries.client_sentiment IS 'positive, neutral, negative, mixed';
