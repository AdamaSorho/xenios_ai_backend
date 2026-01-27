# Xenios AI Backend

AI-powered backend services for the Xenios coaching platform.

## Overview

This service handles CPU-intensive AI operations for Xenios:
- Document extraction (IBM Docling)
- Async job processing (transcription, analysis)
- LLM orchestration with streaming support
- Heavy AI workloads that don't fit serverless constraints

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- uv package manager (`pip install uv`)
- Redis (for local development without Docker)

### Local Development

1. **Clone and setup**
   ```bash
   cd xenios_ai_backend
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

2. **Install dependencies**
   ```bash
   uv sync --dev
   ```

3. **Run the API server**
   ```bash
   make dev
   # Or: uv run uvicorn app.main:app --reload
   ```

4. **Run with Docker Compose** (recommended)
   ```bash
   make up
   # API: http://localhost:8000
   # Flower: http://localhost:5555
   ```

### Environment Variables

See `.env.example` for all required variables. Key ones:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Supabase PostgreSQL connection string |
| `SUPABASE_JWT_SECRET` | JWT secret from Supabase dashboard |
| `XENIOS_BACKEND_API_KEY` | API key for backend-to-backend auth |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM access |

## API Endpoints

### Health Checks

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health check |
| `GET /health/ready` | Readiness probe (checks DB, Redis) |
| `GET /health/live` | Liveness probe |

### V1 API

All V1 endpoints require `X-API-Key` header.

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/v1/status` | API Key | Service status |
| `GET /api/v1/llm/tasks` | API Key | List available LLM tasks |
| `POST /api/v1/llm/complete` | API Key + JWT | LLM completion |
| `POST /api/v1/llm/stream` | API Key + JWT | Streaming completion (SSE) |
| `POST /api/v1/llm/classify-intent` | API Key + JWT | Intent classification |
| `POST /api/v1/llm/extract-entities` | API Key + JWT | Entity extraction |

## Development

### Running Tests

```bash
make test          # Run all tests
make test-cov      # Run with coverage report
```

### Code Quality

```bash
make lint          # Run linter
make format        # Format code
make typecheck     # Type checking
make verify        # All checks
```

### Docker Commands

```bash
make build         # Build images
make up            # Start services
make down          # Stop services
make logs          # View logs
make clean         # Remove everything
```

## Deployment

### VPS Provisioning (First Time)

```bash
# Provision a new server
make provision ENV=staging
```

### Application Deployment

```bash
# Deploy to staging
make deploy-staging

# Deploy to production
make deploy-prod

# Rollback to previous version
make rollback ENV=staging
```

### Managing Secrets

Secrets are encrypted with Ansible Vault:

```bash
# Edit encrypted secrets
make edit-secrets ENV=staging

# Encrypt a new file
make encrypt FILE=path/to/file
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Xenios MVP                               │
│                      (Next.js/Supabase)                         │
│                                                                  │
│  /api/ai-assistant/[...path] ──────────────────────────────────┐│
└────────────────────────────────────────────────────────────────┘│
                                                                   │
                                                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AI Backend (FastAPI)                        │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │  Health  │  │   Auth   │  │   LLM    │  │  Job Queue API   ││
│  │ Endpoints│  │Middleware│  │Endpoints │  │    (Future)      ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    LLM Service                             │  │
│  │  OpenRouter → Opus 4.5 / Sonnet 4 (task-based routing)    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Celery Workers                              │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  ┌────────┐│
│  │ Default  │  │ Extraction   │  │ Transcription │  │  LLM   ││
│  │ Worker   │  │   Worker     │  │    Worker     │  │ Worker ││
│  └──────────┘  └──────────────┘  └───────────────┘  └────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────┐  ┌───────────────────────────────────────────────────┐
│  Redis   │  │              Supabase PostgreSQL                   │
│ (Broker) │  │  (Shared with MVP + ai_backend schema)            │
└──────────┘  └───────────────────────────────────────────────────┘
```

## Model Configuration

Models are selected based on task type:

| Task | Primary Model | Fallback Model |
|------|--------------|----------------|
| session_summary | Opus 4.5 | Sonnet 4 |
| insight_generation | Opus 4.5 | Sonnet 4 |
| chat | Opus 4.5 | Sonnet 4 |
| intent_classification | Sonnet 4 | GPT-4o-mini |
| entity_extraction | Sonnet 4 | GPT-4o-mini |

## License

Proprietary - Xenios Inc.
