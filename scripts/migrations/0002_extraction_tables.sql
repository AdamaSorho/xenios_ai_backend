-- Spec 0002: Document Extraction Pipeline
-- Migration to create extraction-related tables

-- Ensure ai_backend schema exists (created in 0001)
CREATE SCHEMA IF NOT EXISTS ai_backend;

-- Extraction job status enum
DO $$ BEGIN
    CREATE TYPE ai_backend.extraction_status AS ENUM (
        'pending',
        'processing',
        'completed',
        'failed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Extraction jobs table
-- Tracks each document extraction request and its results
CREATE TABLE IF NOT EXISTS ai_backend.extraction_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Ownership
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    -- Source file information
    file_url TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,  -- pdf, csv, json, xml
    file_size INTEGER NOT NULL,

    -- Processing information
    document_type VARCHAR(50),  -- inbody, lab_results, garmin, whoop, apple_health, nutrition
    status ai_backend.extraction_status NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Results
    extracted_data JSONB,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    validation_errors JSONB,
    error_message TEXT,

    -- Retry tracking
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Webhook notification
    webhook_url TEXT,
    webhook_sent_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_client
    ON ai_backend.extraction_jobs(client_id);
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_coach
    ON ai_backend.extraction_jobs(coach_id);
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_status
    ON ai_backend.extraction_jobs(status);
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_document_type
    ON ai_backend.extraction_jobs(document_type);
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_created
    ON ai_backend.extraction_jobs(created_at DESC);

-- Compound index for listing jobs by client with status filter
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_client_status
    ON ai_backend.extraction_jobs(client_id, status, created_at DESC);

-- Extraction results cache table
-- Stores denormalized key metrics for quick access
CREATE TABLE IF NOT EXISTS ai_backend.extraction_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ai_backend.extraction_jobs(id) ON DELETE CASCADE,
    document_type VARCHAR(50) NOT NULL,
    extraction_date DATE NOT NULL,

    -- Denormalized key metrics for quick access
    metrics JSONB NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_extraction_cache_job
    ON ai_backend.extraction_cache(job_id);
CREATE INDEX IF NOT EXISTS idx_extraction_cache_type_date
    ON ai_backend.extraction_cache(document_type, extraction_date DESC);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION ai_backend.update_extraction_job_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_extraction_job_timestamp ON ai_backend.extraction_jobs;
CREATE TRIGGER update_extraction_job_timestamp
    BEFORE UPDATE ON ai_backend.extraction_jobs
    FOR EACH ROW
    EXECUTE FUNCTION ai_backend.update_extraction_job_timestamp();

-- Comments for documentation
COMMENT ON TABLE ai_backend.extraction_jobs IS 'Tracks document extraction jobs and their results';
COMMENT ON COLUMN ai_backend.extraction_jobs.file_url IS 'S3/R2 URL of the uploaded file';
COMMENT ON COLUMN ai_backend.extraction_jobs.document_type IS 'Detected or specified document type';
COMMENT ON COLUMN ai_backend.extraction_jobs.extracted_data IS 'Structured JSON of extracted data';
COMMENT ON COLUMN ai_backend.extraction_jobs.confidence_score IS 'Overall extraction confidence (0-1)';
COMMENT ON TABLE ai_backend.extraction_cache IS 'Denormalized extraction results for quick access';
