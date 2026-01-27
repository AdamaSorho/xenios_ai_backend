-- Migration: 0004_rag_tables.sql
-- Description: Create tables for RAG system - embeddings, chat history, insight generation (Spec 0004)
-- Author: Builder
-- Date: 2025-01-27

-- ============================================================================
-- Enable pgvector Extension
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Embeddings Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,

    -- Source reference
    source_type VARCHAR(50) NOT NULL,
    source_id TEXT NOT NULL,  -- Deterministic composite key (see Source ID Scheme)
    source_table VARCHAR(100),

    -- Content
    content_text TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,  -- SHA256 for deduplication

    -- Embedding vector (1536 dimensions for ada-002)
    embedding vector(1536) NOT NULL,

    -- Metadata for filtering
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE(client_id, source_type, source_id)
);

-- Indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_embeddings_client
    ON ai_backend.embeddings(client_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_source_type
    ON ai_backend.embeddings(client_id, source_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_hash
    ON ai_backend.embeddings(client_id, content_hash);

-- Vector similarity index (IVFFlat for approximate nearest neighbor)
-- Note: IVFFlat requires data to be present for optimal index building
-- After initial data load, consider running: REINDEX INDEX idx_embeddings_vector
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON ai_backend.embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================================
-- Chat History Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    role VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,

    -- RAG metadata
    sources_used JSONB,  -- List of source citations used in response
    tokens_used INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for chat_history
CREATE INDEX IF NOT EXISTS idx_chat_history_conversation
    ON ai_backend.chat_history(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_client
    ON ai_backend.chat_history(client_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_history_coach_client
    ON ai_backend.chat_history(coach_id, client_id, created_at DESC);

-- ============================================================================
-- Insight Generation Log Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_backend.insight_generation_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    trigger VARCHAR(50) NOT NULL,
    triggering_data JSONB NOT NULL,

    -- Result
    insight_id UUID,  -- FK to public.insights if generated
    insight_type VARCHAR(50),
    title TEXT,
    title_embedding vector(1536),  -- For deduplication similarity check
    status VARCHAR(20) NOT NULL,  -- generated, duplicate, failed
    error_message TEXT,

    -- Metrics
    context_items_used INTEGER,
    tokens_used INTEGER,
    generation_time_ms INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for insight_generation_log
CREATE INDEX IF NOT EXISTS idx_insight_gen_log_client
    ON ai_backend.insight_generation_log(client_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_insight_gen_log_dedup
    ON ai_backend.insight_generation_log(client_id, insight_type, created_at DESC)
    WHERE status = 'generated';
CREATE INDEX IF NOT EXISTS idx_insight_gen_log_coach
    ON ai_backend.insight_generation_log(coach_id, created_at DESC);

-- ============================================================================
-- Updated_at Trigger for Embeddings
-- ============================================================================

-- Reuse the trigger function from 0003 migration if it exists
-- Otherwise create it
CREATE OR REPLACE FUNCTION ai_backend.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add trigger to embeddings table
DROP TRIGGER IF EXISTS update_embeddings_updated_at ON ai_backend.embeddings;
CREATE TRIGGER update_embeddings_updated_at
    BEFORE UPDATE ON ai_backend.embeddings
    FOR EACH ROW
    EXECUTE FUNCTION ai_backend.update_updated_at_column();

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE ai_backend.embeddings IS 'Vector embeddings for client health data (RAG system)';
COMMENT ON TABLE ai_backend.chat_history IS 'Chat conversation history for context continuity';
COMMENT ON TABLE ai_backend.insight_generation_log IS 'Log of insight generation attempts for deduplication and analytics';

COMMENT ON COLUMN ai_backend.embeddings.source_type IS 'Type: health_profile, health_metric_summary, health_goal, lab_result, session_summary, checkin_summary, message_thread';
COMMENT ON COLUMN ai_backend.embeddings.source_id IS 'Deterministic ID: UUID for single records, composite key for aggregated (e.g., {client_id}:metric:{type}:{week})';
COMMENT ON COLUMN ai_backend.embeddings.content_hash IS 'SHA256 hash of content_text for deduplication';

COMMENT ON COLUMN ai_backend.chat_history.role IS 'Message role: user or assistant';
COMMENT ON COLUMN ai_backend.chat_history.sources_used IS 'JSONB array of SourceCitation objects used in response';

COMMENT ON COLUMN ai_backend.insight_generation_log.trigger IS 'Trigger type: scheduled, metric_change, goal_progress, checkin_submitted, session_completed';
COMMENT ON COLUMN ai_backend.insight_generation_log.status IS 'Generation status: generated, duplicate, failed';
COMMENT ON COLUMN ai_backend.insight_generation_log.title_embedding IS 'Embedding of insight title for similarity-based deduplication';
