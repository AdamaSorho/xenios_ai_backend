# Plan 0001: AI Backend Foundation

**Spec**: [codev/specs/0001-ai-backend-foundation.md](../specs/0001-ai-backend-foundation.md)
**Status**: Ready for implementation
**Estimated Phases**: 7

---

## Implementation Strategy

Build the foundation in layers, starting with the project skeleton and progressively adding infrastructure components. Each phase produces working, testable code.

**Key Principles:**
- Each phase should be independently testable
- Docker Compose should work after Phase 5
- Ansible deployment after Phase 6
- Full test suite after Phase 7

---

## Phase 1: Project Setup & Structure

**Goal**: Initialize Python project with proper structure, dependencies, and configuration.

### Tasks

1.1 **Initialize Python project with `uv`**
```bash
uv init xenios_ai_backend
cd xenios_ai_backend
```

1.2 **Add core dependencies to `pyproject.toml`**
```toml
[project]
name = "xenios-ai-backend"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "celery[redis]>=5.3.0",
    "redis>=5.0.0",
    "httpx>=0.26.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "asyncpg>=0.29.0",
    "sqlalchemy>=2.0.0",
    "python-jose[cryptography]>=3.3.0",
    "structlog>=24.1.0",
    "python-multipart>=0.0.6",
    "sse-starlette>=1.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.26.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]
```

1.3 **Create directory structure** (as per spec)
```
app/
├── __init__.py
├── main.py
├── config.py
├── dependencies.py
├── api/
│   ├── __init__.py
│   ├── router.py
│   ├── health.py
│   └── v1/
│       ├── __init__.py
│       └── router.py
├── core/
│   ├── __init__.py
│   ├── auth.py
│   ├── database.py
│   ├── redis.py
│   └── logging.py
├── services/
│   ├── __init__.py
│   └── llm/
│       ├── __init__.py
│       ├── client.py
│       ├── models.py
│       └── prompts.py
├── workers/
│   ├── __init__.py
│   ├── celery_app.py
│   └── tasks/
│       ├── __init__.py
│       └── base.py
└── schemas/
    ├── __init__.py
    ├── health.py
    └── common.py
```

1.4 **Implement configuration management** (`app/config.py`)
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    debug: bool = False

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Authentication
    supabase_jwt_secret: str
    xenios_backend_api_key: str

    # LLM
    openrouter_api_key: str

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

1.5 **Create `.env.example`** with all variables

1.6 **Create `.gitignore`** for Python/Docker

1.7 **Initialize basic `app/main.py`**
```python
from fastapi import FastAPI
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Xenios AI Backend",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
)

@app.get("/")
async def root():
    return {"service": "xenios-ai-backend", "version": "0.1.0"}
```

### Acceptance Criteria Coverage
- AC1: Project Setup (partial)

### Verification
```bash
uv sync
uv run uvicorn app.main:app --reload
curl http://localhost:8000/
```

---

## Phase 2: Core Infrastructure

**Goal**: Implement database, Redis, and Celery connections with health checks.

### Tasks

2.1 **Implement structured logging** (`app/core/logging.py`)
```python
import structlog
import logging
from app.config import get_settings

def setup_logging():
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
                if settings.environment == "production"
                else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
```

2.2 **Implement database connection** (`app/core/database.py`)
- Async connection using `asyncpg` + SQLAlchemy async
- Connection pool configuration
- Health check query function
- Schema initialization for `ai_backend`

2.3 **Implement Redis connection** (`app/core/redis.py`)
- Redis client factory
- Connection pool
- Health check (ping)

2.4 **Implement Celery app** (`app/workers/celery_app.py`)
- Celery configuration as per spec
- Task routing for queues
- Time limits and retry policies

2.5 **Implement base task class** (`app/workers/tasks/base.py`)
- Logging integration
- Error handling
- Retry configuration

2.6 **Implement health endpoints** (`app/api/health.py`)
```python
from fastapi import APIRouter, Depends
from app.core.database import check_db_health
from app.core.redis import check_redis_health

router = APIRouter(tags=["health"])

@router.get("/health")
async def health():
    return {"status": "healthy"}

@router.get("/health/ready")
async def readiness():
    db_ok = await check_db_health()
    redis_ok = await check_redis_health()

    if db_ok and redis_ok:
        return {"status": "ready", "database": "ok", "redis": "ok"}

    return JSONResponse(
        status_code=503,
        content={"status": "not ready", "database": db_ok, "redis": redis_ok}
    )

@router.get("/health/live")
async def liveness():
    return {"status": "alive"}
```

2.7 **Wire up routers in main.py**

### Acceptance Criteria Coverage
- AC3: Health Endpoints
- AC5: Database Connection
- AC6: Celery Job Queue (partial)
- AC8: Logging

### Verification
```bash
# Start Redis locally
docker run -d -p 6379:6379 redis:7-alpine

# Run app
uv run uvicorn app.main:app --reload

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/live
```

---

## Phase 3: Authentication

**Goal**: Implement API key and JWT verification middleware.

### Tasks

3.1 **Implement API key verification** (`app/core/auth.py`)
```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    settings = get_settings()
    if not api_key or api_key != settings.xenios_backend_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return api_key
```

3.2 **Implement JWT verification**
```python
from jose import jwt, JWTError
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

class UserContext(BaseModel):
    user_id: str
    role: str
    email: str | None = None

security = HTTPBearer(auto_error=False)

async def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    _: str = Depends(verify_api_key)  # API key required first
) -> UserContext:
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization")

    try:
        settings = get_settings()
        payload = jwt.decode(
            credentials.credentials,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated"
        )
        return UserContext(
            user_id=payload.get("sub"),
            role=payload.get("role", "authenticated"),
            email=payload.get("email")
        )
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
```

3.3 **Create auth dependencies** (`app/dependencies.py`)
- `get_current_user` - requires valid JWT
- `get_optional_user` - JWT optional (for public endpoints)
- `require_role(role)` - role-based access

3.4 **Add correlation ID middleware**
```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
```

3.5 **Add CORS middleware** for Xenios MVP origin

### Acceptance Criteria Coverage
- AC4: Authentication

### Verification
```bash
# Without API key - should fail
curl http://localhost:8000/api/v1/status

# With API key only - should work for public endpoints
curl -H "X-API-Key: test-key" http://localhost:8000/api/v1/status

# With API key + JWT - should work for protected endpoints
curl -H "X-API-Key: test-key" \
     -H "Authorization: Bearer <jwt>" \
     http://localhost:8000/api/v1/protected
```

---

## Phase 4: LLM Integration

**Goal**: Implement OpenRouter client with model routing and streaming support.

### Tasks

4.1 **Implement model configuration** (`app/services/llm/models.py`)
```python
from pydantic import BaseModel
from typing import Literal

class ModelConfig(BaseModel):
    primary: str
    fallback: str
    temperature: float
    max_tokens: int
    streaming: bool = False

TASK_MODELS: dict[str, ModelConfig] = {
    "session_summary": ModelConfig(
        primary="anthropic/claude-opus-4-20250514",
        fallback="anthropic/claude-sonnet-4-20250514",
        temperature=0.3,
        max_tokens=4000,
    ),
    "insight_generation": ModelConfig(
        primary="anthropic/claude-opus-4-20250514",
        fallback="anthropic/claude-sonnet-4-20250514",
        temperature=0.5,
        max_tokens=2000,
    ),
    "chat": ModelConfig(
        primary="anthropic/claude-opus-4-20250514",
        fallback="anthropic/claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=1500,
        streaming=True,
    ),
    "intent_classification": ModelConfig(
        primary="anthropic/claude-sonnet-4-20250514",
        fallback="openai/gpt-4o-mini",
        temperature=0.0,
        max_tokens=100,
    ),
    "entity_extraction": ModelConfig(
        primary="anthropic/claude-sonnet-4-20250514",
        fallback="openai/gpt-4o-mini",
        temperature=0.0,
        max_tokens=500,
    ),
}

def get_model_for_task(task: str) -> ModelConfig:
    if task not in TASK_MODELS:
        raise ValueError(f"Unknown task: {task}")
    return TASK_MODELS[task]
```

4.2 **Implement OpenRouter client** (`app/services/llm/client.py`)
```python
import httpx
from typing import AsyncIterator
from app.config import get_settings
from app.services.llm.models import get_model_for_task, ModelConfig

class LLMClient:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.settings = get_settings()

    async def complete(
        self,
        task: str,
        messages: list[dict],
        use_fallback: bool = False
    ) -> dict:
        config = get_model_for_task(task)
        model = config.fallback if use_fallback else config.primary

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "HTTP-Referer": "https://xenios.app",
                        "X-Title": "Xenios AI Backend",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if not use_fallback:
                    # Retry with fallback model
                    return await self.complete(task, messages, use_fallback=True)
                raise

    async def stream(
        self,
        task: str,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        config = get_model_for_task(task)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                    "HTTP-Referer": "https://xenios.app",
                },
                json={
                    "model": config.primary,
                    "messages": messages,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "stream": True,
                },
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]
```

4.3 **Implement LLM API endpoints** (`app/api/v1/llm.py`)
```python
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from app.services.llm.client import LLMClient
from app.core.auth import verify_jwt, UserContext

router = APIRouter(prefix="/llm", tags=["llm"])

@router.post("/complete")
async def llm_complete(
    request: LLMCompleteRequest,
    user: UserContext = Depends(verify_jwt)
):
    client = LLMClient()
    result = await client.complete(request.task, request.messages)
    return result

@router.post("/stream")
async def llm_stream(
    request: LLMCompleteRequest,
    user: UserContext = Depends(verify_jwt)
):
    client = LLMClient()
    return EventSourceResponse(client.stream(request.task, request.messages))
```

4.4 **Create request/response schemas** (`app/schemas/llm.py`)

4.5 **Add inference logging** - Log all LLM calls to `ai_backend.inference_logs`

### Acceptance Criteria Coverage
- AC7: LLM Integration

### Verification
```bash
# Sync completion
curl -X POST http://localhost:8000/api/v1/llm/complete \
  -H "X-API-Key: test-key" \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{"task": "intent_classification", "messages": [{"role": "user", "content": "Test"}]}'

# Streaming
curl -X POST http://localhost:8000/api/v1/llm/stream \
  -H "X-API-Key: test-key" \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{"task": "chat", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Phase 5: Docker Configuration

**Goal**: Create Dockerfiles and Docker Compose for local development and production.

### Tasks

5.1 **Create API Dockerfile** (`docker/Dockerfile`)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app/ ./app/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

5.2 **Create Worker Dockerfile** (`docker/Dockerfile.worker`)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Docling deps for future)
RUN apt-get update && apt-get install -y \
    curl \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app/ ./app/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uv", "run", "celery", "-A", "app.workers.celery_app", "worker", "-l", "info"]
```

5.3 **Create development Docker Compose** (`docker/docker-compose.yml`)
- As specified in spec
- Volume mounts for hot reload
- All services with health checks

5.4 **Create production Docker Compose template** (`docker/docker-compose.prod.yml`)
- No volume mounts
- Resource limits
- Restart policies
- Production environment variables

5.5 **Create `.dockerignore`**

5.6 **Update Makefile with Docker commands**
```makefile
.PHONY: dev up down logs build

# Local development
dev:
	uv run uvicorn app.main:app --reload

# Docker commands
up:
	docker compose -f docker/docker-compose.yml up -d

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f

build:
	docker compose -f docker/docker-compose.yml build

# Run specific worker
worker:
	docker compose -f docker/docker-compose.yml up -d worker

# View Flower
flower:
	open http://localhost:5555
```

### Acceptance Criteria Coverage
- AC2: Docker Compose
- AC10: Production Docker Compose

### Verification
```bash
# Build and start all services
make build
make up

# Check all services running
docker compose -f docker/docker-compose.yml ps

# Test health endpoint
curl http://localhost:8000/health/ready

# Check Flower
open http://localhost:5555

# Check logs
make logs
```

---

## Phase 6: Ansible Deployment

**Goal**: Create Ansible playbooks for VPS provisioning and application deployment.

### Tasks

6.1 **Create Ansible configuration** (`infrastructure/ansible.cfg`)
```ini
[defaults]
inventory = inventory/hosts.yml
roles_path = roles
host_key_checking = False
retry_files_enabled = False

[ssh_connection]
pipelining = True
```

6.2 **Create inventory structure**
- `infrastructure/inventory/hosts.yml`
- `infrastructure/inventory/group_vars/all.yml`
- `infrastructure/inventory/group_vars/staging.yml` (vault encrypted)
- `infrastructure/inventory/group_vars/production.yml` (vault encrypted)

6.3 **Create `common` role**
- Install base packages
- Create deploy user with SSH key
- Configure SSH hardening
- Set timezone

6.4 **Create `docker` role**
- Install Docker CE from official repo
- Install Docker Compose v2 plugin
- Add deploy user to docker group
- Configure Docker daemon (log rotation)

6.5 **Create `firewall` role**
- Install UFW
- Allow SSH (22), HTTP (80), HTTPS (443)
- Default deny incoming
- Enable UFW

6.6 **Create `caddy` role**
- Install Caddy
- Create Caddyfile from template
- Enable and start Caddy service

6.7 **Create `app` role**
- Create app directory
- Copy docker-compose.prod.yml template
- Copy .env template with secrets
- Pull and start containers
- Health check verification
- Prune old images

6.8 **Create provision playbook** (`infrastructure/playbooks/provision.yml`)
- Runs common, firewall, docker, caddy roles
- One-time setup for new VPS

6.9 **Create deploy playbook** (`infrastructure/playbooks/deploy.yml`)
- Runs app role
- Used for regular deployments

6.10 **Create rollback playbook** (`infrastructure/playbooks/rollback.yml`)
- Store previous image tag
- Revert to previous version
- Health check

6.11 **Create Jinja2 templates**
- `infrastructure/templates/docker-compose.prod.yml.j2`
- `infrastructure/templates/Caddyfile.j2`
- `infrastructure/templates/.env.j2`

6.12 **Update Makefile with Ansible commands**
```makefile
# Ansible commands
provision:
	cd infrastructure && ansible-playbook playbooks/provision.yml --limit $(ENV)

deploy:
	cd infrastructure && ansible-playbook playbooks/deploy.yml --limit $(ENV) --ask-vault-pass

deploy-staging:
	$(MAKE) deploy ENV=staging

deploy-prod:
	$(MAKE) deploy ENV=production

rollback:
	cd infrastructure && ansible-playbook playbooks/rollback.yml --limit $(ENV) --ask-vault-pass

edit-secrets:
	cd infrastructure && ansible-vault edit inventory/group_vars/$(ENV).yml

lint-ansible:
	cd infrastructure && ansible-lint playbooks/*.yml roles/*/tasks/*.yml
```

### Acceptance Criteria Coverage
- AC9: Ansible Deployment Automation

### Verification
```bash
# Lint playbooks
make lint-ansible

# Provision test server (use a test droplet)
make provision ENV=staging

# Deploy to staging
make deploy-staging

# Verify deployment
curl https://staging.ai.xenios.app/health

# Test rollback
make rollback ENV=staging
```

---

## Phase 7: Testing & Documentation

**Goal**: Comprehensive test suite and documentation.

### Tasks

7.1 **Create pytest configuration** (`pyproject.toml`)
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --cov=app --cov-report=term-missing"
```

7.2 **Create test fixtures** (`tests/conftest.py`)
- Test client fixture
- Mock settings fixture
- Mock Redis fixture
- Mock database fixture

7.3 **Implement unit tests**
- `tests/test_config.py` - Configuration loading
- `tests/test_auth.py` - API key and JWT validation
- `tests/test_llm_models.py` - Model routing logic

7.4 **Implement integration tests**
- `tests/test_health.py` - Health endpoints
- `tests/test_llm_endpoints.py` - LLM API endpoints

7.5 **Create README.md**
- Project overview
- Quick start guide
- Environment setup
- Docker commands
- Deployment guide
- API documentation link

7.6 **Create database init script** (`scripts/init_db.sql`)
```sql
-- Create AI backend schema
CREATE SCHEMA IF NOT EXISTS ai_backend;

-- Inference logs table
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
```

7.7 **Final Makefile consolidation**
- All commands organized
- Help target documenting commands

### Acceptance Criteria Coverage
- AC1: Project Setup (complete)
- All remaining acceptance criteria verification

### Verification
```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Lint code
make lint

# Full verification
make verify  # runs lint + test + type-check
```

---

## Implementation Order Summary

```
Phase 1: Project Setup ──────────────────────► Working Python project
    │
    ▼
Phase 2: Core Infrastructure ────────────────► DB, Redis, Celery, Health checks
    │
    ▼
Phase 3: Authentication ─────────────────────► API key + JWT middleware
    │
    ▼
Phase 4: LLM Integration ────────────────────► OpenRouter client, streaming
    │
    ▼
Phase 5: Docker Configuration ───────────────► docker compose up works
    │
    ▼
Phase 6: Ansible Deployment ─────────────────► Automated VPS deployment
    │
    ▼
Phase 7: Testing & Documentation ────────────► Production ready
```

---

## Files to Create (Summary)

### Phase 1
- `pyproject.toml`
- `.env.example`
- `.gitignore`
- `app/__init__.py`
- `app/main.py`
- `app/config.py`
- All `__init__.py` files for packages

### Phase 2
- `app/core/logging.py`
- `app/core/database.py`
- `app/core/redis.py`
- `app/workers/celery_app.py`
- `app/workers/tasks/base.py`
- `app/api/health.py`
- `app/api/router.py`

### Phase 3
- `app/core/auth.py`
- `app/dependencies.py`
- `app/middleware/correlation.py`

### Phase 4
- `app/services/llm/models.py`
- `app/services/llm/client.py`
- `app/services/llm/prompts.py`
- `app/api/v1/llm.py`
- `app/api/v1/router.py`
- `app/schemas/llm.py`

### Phase 5
- `docker/Dockerfile`
- `docker/Dockerfile.worker`
- `docker/docker-compose.yml`
- `docker/docker-compose.prod.yml`
- `.dockerignore`
- `Makefile`

### Phase 6
- `infrastructure/ansible.cfg`
- `infrastructure/inventory/hosts.yml`
- `infrastructure/inventory/group_vars/*.yml`
- `infrastructure/playbooks/*.yml`
- `infrastructure/roles/*/tasks/main.yml`
- `infrastructure/roles/*/handlers/main.yml`
- `infrastructure/templates/*.j2`

### Phase 7
- `tests/conftest.py`
- `tests/test_*.py`
- `README.md`
- `scripts/init_db.sql`

---

## Estimated Effort

| Phase | Complexity | Notes |
|-------|------------|-------|
| Phase 1 | Low | Boilerplate setup |
| Phase 2 | Medium | Database/Redis integration |
| Phase 3 | Medium | JWT validation nuances |
| Phase 4 | Medium | Streaming SSE handling |
| Phase 5 | Low | Docker configuration |
| Phase 6 | High | Ansible roles and testing |
| Phase 7 | Medium | Test coverage |

---

**Plan Status**: Ready for approval
**Author**: Architect
**Created**: 2025-01-27
