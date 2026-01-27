-- Initialize AI Backend schema in Supabase PostgreSQL
-- Run this script once to set up the ai_backend schema

-- Create AI backend schema (separate from main Xenios MVP schema)
CREATE SCHEMA IF NOT EXISTS ai_backend;

-- Inference logs table - tracks all LLM API calls
CREATE TABLE IF NOT EXISTS ai_backend.inference_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    task VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying by user
CREATE INDEX IF NOT EXISTS idx_inference_logs_user
ON ai_backend.inference_logs(user_id, created_at DESC);

-- Index for analytics by task type
CREATE INDEX IF NOT EXISTS idx_inference_logs_task
ON ai_backend.inference_logs(task, created_at DESC);

-- Job queue metadata table - tracks Celery job status
CREATE TABLE IF NOT EXISTS ai_backend.job_queue_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    celery_task_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_data JSONB,
    result_data JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Index for querying jobs by user
CREATE INDEX IF NOT EXISTS idx_job_queue_user
ON ai_backend.job_queue_metadata(user_id, created_at DESC);

-- Index for querying jobs by status
CREATE INDEX IF NOT EXISTS idx_job_queue_status
ON ai_backend.job_queue_metadata(status, created_at DESC);

-- Document extraction cache table
CREATE TABLE IF NOT EXISTS ai_backend.extraction_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_hash VARCHAR(64) NOT NULL,  -- SHA-256 hash of document
    document_name VARCHAR(255),
    mime_type VARCHAR(100),
    extracted_text TEXT,
    extracted_metadata JSONB,
    extraction_model VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '30 days')
);

-- Index for cache lookup by document hash
CREATE UNIQUE INDEX IF NOT EXISTS idx_extraction_cache_hash
ON ai_backend.extraction_cache(document_hash);

-- Index for cache expiration cleanup
CREATE INDEX IF NOT EXISTS idx_extraction_cache_expires
ON ai_backend.extraction_cache(expires_at);

-- Embeddings table (for future RAG implementation)
-- Uses pgvector extension if available
DO $$
BEGIN
    -- Try to create pgvector extension if not exists
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector extension not available, skipping embeddings table';
END $$;

-- Only create embeddings table if pgvector is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE TABLE IF NOT EXISTS ai_backend.embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_type VARCHAR(50) NOT NULL,  -- 'session_note', 'document', etc.
            source_id UUID NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            embedding vector(1536),  -- OpenAI ada-002 dimension
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Index for vector similarity search
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector
        ON ai_backend.embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

        -- Index for source lookup
        CREATE INDEX IF NOT EXISTS idx_embeddings_source
        ON ai_backend.embeddings(source_type, source_id);
    END IF;
END $$;

-- Grant permissions to the application user
-- Replace 'authenticator' with your Supabase service role if different
-- GRANT USAGE ON SCHEMA ai_backend TO authenticator;
-- GRANT ALL ON ALL TABLES IN SCHEMA ai_backend TO authenticator;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA ai_backend TO authenticator;

-- Add comment to schema
COMMENT ON SCHEMA ai_backend IS 'Schema for Xenios AI Backend service tables';
