# Spec 0001: AI Backend Foundation

## Overview

**What**: Build the foundational infrastructure for Xenios AI Backend - a Python/FastAPI service that handles AI-intensive operations for the Xenios coaching platform.

**Why**: The existing Xenios MVP (Next.js/Supabase) needs a dedicated backend for:
- CPU-intensive document extraction (IBM Docling)
- Async job processing (transcription, analysis)
- LLM orchestration with streaming support
- Heavy AI workloads that don't fit serverless constraints

**Who**: Coaches and clients of the Xenios platform (indirectly - this is backend infrastructure)

## Goals

### Must Have
1. FastAPI application with proper project structure
2. Celery + Redis job queue infrastructure with worker routing
3. Docker Compose configuration for local development and MVP deployment
4. Supabase PostgreSQL connection (shared database with MVP)
5. Authentication middleware (verify JWT from Xenios MVP frontend)
6. API key authentication for backend-to-backend calls
7. OpenRouter integration with model routing (Opus 4.5 primary, Sonnet 4 for simple tasks)
8. Health check and status endpoints
9. Structured logging with correlation IDs
10. Environment-based configuration management
11. **Ansible deployment automation** for reproducible VPS provisioning and application deployment

### Should Have
- Flower dashboard for Celery monitoring
- Request/response validation with Pydantic
- Rate limiting middleware
- CORS configuration for Xenios MVP origin
- Ansible vault for secrets management

### Won't Have (MVP)
- Kubernetes deployment (Docker Compose is sufficient)
- Multi-region deployment
- Custom authentication (we verify JWT from Supabase)
- Admin UI

## Technical Context

### Integration with Xenios MVP

The Xenios MVP (at `/Users/adamasorho/Desktop/projects/xenios/xenios_mvp_003`) has:
- A proxy endpoint at `/api/ai-assistant/[...path]` that forwards to `XENIOS_BACKEND_API_URL`
- Supabase PostgreSQL with 60+ tables (we share this database)
- Supabase Auth with JWT tokens

**Environment variables expected by MVP:**
```
XENIOS_BACKEND_API_URL=https://ai.xenios.app  # or localhost:8000 for dev
XENIOS_BACKEND_API_KEY=<secret>
```

### Database Strategy

We share the Supabase PostgreSQL database but use a separate schema:

```sql
CREATE SCHEMA IF NOT EXISTS ai_backend;

-- AI Backend specific tables
ai_backend.job_queue_metadata     -- Celery job tracking
ai_backend.inference_logs         -- LLM call audit trail
ai_backend.extraction_cache       -- Document extraction results cache
ai_backend.embeddings             -- pgvector for RAG (future)
```

### LLM Model Configuration

Based on user preference for Opus 4.5's stronger reasoning:

```python
MODEL_CONFIG = {
    # Complex reasoning tasks - use Opus 4.5
    "session_summary": {
        "primary": "anthropic/claude-opus-4-20250514",
        "fallback": "anthropic/claude-sonnet-4-20250514",
        "temperature": 0.3,
        "max_tokens": 4000
    },
    "insight_generation": {
        "primary": "anthropic/claude-opus-4-20250514",
        "fallback": "anthropic/claude-sonnet-4-20250514",
        "temperature": 0.5,
        "max_tokens": 2000
    },
    "chat": {
        "primary": "anthropic/claude-opus-4-20250514",
        "fallback": "anthropic/claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 1500,
        "streaming": True
    },

    # Simpler tasks - Sonnet 4 is sufficient and faster
    "intent_classification": {
        "primary": "anthropic/claude-sonnet-4-20250514",
        "fallback": "openai/gpt-4o-mini",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 100
    },
    "entity_extraction": {
        "primary": "anthropic/claude-sonnet-4-20250514",
        "fallback": "openai/gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 500
    }
}
```

## Technical Implementation

### Project Structure

```
xenios_ai_backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Settings and environment
│   ├── dependencies.py         # Dependency injection
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py           # Main API router
│   │   ├── health.py           # Health check endpoints
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── router.py       # V1 API routes
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── auth.py             # JWT verification, API key auth
│   │   ├── database.py         # Supabase connection
│   │   ├── redis.py            # Redis connection
│   │   └── logging.py          # Structured logging
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── client.py       # OpenRouter client
│   │       ├── models.py       # Model configuration
│   │       └── prompts.py      # Prompt templates
│   │
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── celery_app.py       # Celery configuration
│   │   └── tasks/
│   │       ├── __init__.py
│   │       └── base.py         # Base task classes
│   │
│   └── schemas/
│       ├── __init__.py
│       ├── health.py           # Health check schemas
│       └── common.py           # Shared schemas
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_health.py
│   └── test_auth.py
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.worker
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml # Production overrides
│
├── infrastructure/
│   ├── ansible.cfg             # Ansible configuration
│   ├── inventory/
│   │   ├── hosts.yml           # Inventory (staging, production)
│   │   └── group_vars/
│   │       ├── all.yml         # Common variables
│   │       ├── staging.yml     # Staging-specific vars
│   │       └── production.yml  # Production-specific vars
│   │
│   ├── playbooks/
│   │   ├── provision.yml       # Initial VPS setup
│   │   ├── deploy.yml          # Application deployment
│   │   ├── rollback.yml        # Rollback to previous version
│   │   └── secrets.yml         # Secrets management
│   │
│   ├── roles/
│   │   ├── common/             # Base packages, users, SSH hardening
│   │   ├── docker/             # Docker + Docker Compose installation
│   │   ├── firewall/           # UFW configuration
│   │   ├── caddy/              # Caddy reverse proxy + SSL
│   │   ├── app/                # Application deployment
│   │   └── monitoring/         # Logging, metrics (optional)
│   │
│   └── templates/
│       ├── docker-compose.prod.yml.j2
│       ├── Caddyfile.j2
│       └── .env.j2
│
├── scripts/
│   └── init_db.sql             # AI backend schema setup
│
├── .env.example
├── .gitignore
├── pyproject.toml              # Poetry or uv
├── README.md
└── Makefile                    # Common commands
```

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Request from Xenios MVP                      │
│                                                                  │
│  Headers:                                                        │
│    Authorization: Bearer <supabase_jwt>                         │
│    X-API-Key: <backend_api_key>                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI Backend Auth Middleware                    │
│                                                                  │
│  1. Verify X-API-Key matches XENIOS_BACKEND_API_KEY            │
│  2. Decode JWT using Supabase JWT secret                        │
│  3. Extract user_id, role from JWT claims                       │
│  4. Attach user context to request state                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Request Handler (with user context)
```

### Celery Worker Configuration

```python
# app/workers/celery_app.py

from celery import Celery

celery_app = Celery("xenios_ai")

celery_app.conf.update(
    broker_url=settings.REDIS_URL,
    result_backend=settings.REDIS_URL,

    # Task routing
    task_routes={
        "app.workers.tasks.transcription.*": {"queue": "transcription"},
        "app.workers.tasks.extraction.*": {"queue": "extraction"},
        "app.workers.tasks.llm.*": {"queue": "llm"},
        "app.workers.tasks.analytics.*": {"queue": "analytics"},
    },

    # Retry policy
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,       # 10 minutes hard limit
)
```

### Docker Compose Configuration

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=${DATABASE_URL}
      - SUPABASE_JWT_SECRET=${SUPABASE_JWT_SECRET}
      - XENIOS_BACKEND_API_KEY=${XENIOS_BACKEND_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ../app:/app/app
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    command: celery -A app.workers.celery_app worker -l info -Q default,llm -c 4
    environment:
      - ENVIRONMENT=development
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=${DATABASE_URL}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ../app:/app/app
    depends_on:
      - redis

  extraction-worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    command: celery -A app.workers.celery_app worker -l info -Q extraction -c 2
    environment:
      - ENVIRONMENT=development
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ../app:/app/app
    depends_on:
      - redis

  transcription-worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    command: celery -A app.workers.celery_app worker -l info -Q transcription -c 3
    environment:
      - ENVIRONMENT=development
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=${DATABASE_URL}
      - DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}
    volumes:
      - ../app:/app/app
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  flower:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    command: celery -A app.workers.celery_app flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - worker

volumes:
  redis_data:
```

### Ansible Deployment Automation

Ansible provides reproducible, idempotent deployment automation for the AI backend.

#### Inventory Structure

```yaml
# infrastructure/inventory/hosts.yml
all:
  children:
    staging:
      hosts:
        staging-ai:
          ansible_host: staging.ai.xenios.app
          ansible_user: deploy
          environment: staging

    production:
      hosts:
        prod-ai-01:
          ansible_host: ai.xenios.app
          ansible_user: deploy
          environment: production
```

#### Provision Playbook (Initial VPS Setup)

```yaml
# infrastructure/playbooks/provision.yml
---
- name: Provision AI Backend Server
  hosts: all
  become: yes

  vars:
    deploy_user: deploy
    ssh_port: 22

  roles:
    - role: common
      tags: [common, always]
    - role: firewall
      tags: [firewall, security]
    - role: docker
      tags: [docker]
    - role: caddy
      tags: [caddy, proxy]

# Roles do the following:
# common:
#   - Update apt packages
#   - Install base packages (curl, git, htop, etc.)
#   - Create deploy user with SSH key
#   - Configure SSH hardening (disable root, key-only auth)
#   - Set timezone, locale
#
# firewall:
#   - Install UFW
#   - Allow SSH (22), HTTP (80), HTTPS (443)
#   - Allow internal ports only from localhost
#   - Enable UFW
#
# docker:
#   - Install Docker CE
#   - Install Docker Compose v2
#   - Add deploy user to docker group
#   - Configure Docker daemon (log rotation, etc.)
#
# caddy:
#   - Install Caddy
#   - Configure reverse proxy to app:8000
#   - Auto-SSL via Let's Encrypt
#   - Basic security headers
```

#### Deploy Playbook (Application Deployment)

```yaml
# infrastructure/playbooks/deploy.yml
---
- name: Deploy AI Backend Application
  hosts: all
  become: yes
  become_user: deploy

  vars:
    app_dir: /opt/xenios-ai-backend
    docker_registry: ghcr.io/xenios
    image_tag: "{{ lookup('env', 'IMAGE_TAG') | default('latest') }}"

  tasks:
    - name: Create application directory
      file:
        path: "{{ app_dir }}"
        state: directory
        owner: deploy
        group: deploy
        mode: '0755'

    - name: Copy docker-compose file
      template:
        src: templates/docker-compose.prod.yml.j2
        dest: "{{ app_dir }}/docker-compose.yml"
        owner: deploy
        group: deploy
        mode: '0644'
      notify: Restart application

    - name: Copy environment file
      template:
        src: templates/.env.j2
        dest: "{{ app_dir }}/.env"
        owner: deploy
        group: deploy
        mode: '0600'
      no_log: true
      notify: Restart application

    - name: Pull latest images
      community.docker.docker_compose_v2:
        project_src: "{{ app_dir }}"
        pull: always
        state: present
      register: pull_result

    - name: Start application
      community.docker.docker_compose_v2:
        project_src: "{{ app_dir }}"
        state: present
      register: compose_result

    - name: Wait for health check
      uri:
        url: "http://localhost:8000/health"
        status_code: 200
      register: health_check
      until: health_check.status == 200
      retries: 30
      delay: 2

    - name: Prune old Docker images
      community.docker.docker_prune:
        images: yes
        images_filters:
          dangling: true

  handlers:
    - name: Restart application
      community.docker.docker_compose_v2:
        project_src: "{{ app_dir }}"
        state: restarted
```

#### Secrets Management with Ansible Vault

```yaml
# infrastructure/inventory/group_vars/production.yml
# Encrypted with: ansible-vault encrypt inventory/group_vars/production.yml

# Database
database_url: "{{ vault_database_url }}"
supabase_jwt_secret: "{{ vault_supabase_jwt_secret }}"

# API Keys
xenios_backend_api_key: "{{ vault_xenios_backend_api_key }}"
openrouter_api_key: "{{ vault_openrouter_api_key }}"
deepgram_api_key: "{{ vault_deepgram_api_key }}"

# Redis (local, no secret needed for MVP)
redis_url: "redis://redis:6379/0"
```

#### Makefile Commands for Ansible

```makefile
# Makefile (additions for Ansible)

# Ansible commands
.PHONY: provision deploy rollback

# Provision a new server (run once per server)
provision:
	cd infrastructure && ansible-playbook playbooks/provision.yml -i inventory/hosts.yml --limit $(ENV)

# Deploy application
deploy:
	cd infrastructure && ansible-playbook playbooks/deploy.yml -i inventory/hosts.yml --limit $(ENV) --ask-vault-pass

# Deploy to staging
deploy-staging:
	$(MAKE) deploy ENV=staging

# Deploy to production
deploy-prod:
	$(MAKE) deploy ENV=production

# Rollback to previous version
rollback:
	cd infrastructure && ansible-playbook playbooks/rollback.yml -i inventory/hosts.yml --limit $(ENV) --ask-vault-pass

# Edit encrypted secrets
edit-secrets:
	cd infrastructure && ansible-vault edit inventory/group_vars/$(ENV).yml

# Encrypt a file
encrypt:
	cd infrastructure && ansible-vault encrypt $(FILE)

# Check playbook syntax
lint-ansible:
	cd infrastructure && ansible-lint playbooks/*.yml
```

#### Deployment Flow

```
Developer Machine                         VPS (Hetzner/DigitalOcean)
─────────────────                         ─────────────────────────

1. git push (triggers CI)
        │
        ▼
2. CI builds Docker image
   └── Push to ghcr.io/xenios/ai-backend:sha-xxxxx
        │
        ▼
3. make deploy-staging
        │
        ├── Ansible connects via SSH
        ├── Renders docker-compose.prod.yml.j2
        ├── Renders .env.j2 (with vault secrets)
        ├── docker compose pull
        ├── docker compose up -d
        ├── Waits for health check
        └── Prunes old images
        │
        ▼
4. Staging validation ✓
        │
        ▼
5. make deploy-prod
        │
        └── Same as above, production inventory
```

#### Production Docker Compose Template

```yaml
# infrastructure/templates/docker-compose.prod.yml.j2
version: '3.8'

services:
  api:
    image: {{ docker_registry }}/ai-backend:{{ image_tag }}
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL={{ redis_url }}
      - DATABASE_URL={{ database_url }}
      - SUPABASE_JWT_SECRET={{ supabase_jwt_secret }}
      - XENIOS_BACKEND_API_KEY={{ xenios_backend_api_key }}
      - OPENROUTER_API_KEY={{ openrouter_api_key }}
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 512M

  worker:
    image: {{ docker_registry }}/ai-backend:{{ image_tag }}
    command: celery -A app.workers.celery_app worker -l info -Q default,llm -c 4
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - REDIS_URL={{ redis_url }}
      - DATABASE_URL={{ database_url }}
      - OPENROUTER_API_KEY={{ openrouter_api_key }}
    depends_on:
      - redis
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 1G

  extraction-worker:
    image: {{ docker_registry }}/ai-backend:{{ image_tag }}
    command: celery -A app.workers.celery_app worker -l info -Q extraction -c 2
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - REDIS_URL={{ redis_url }}
      - DATABASE_URL={{ database_url }}
    depends_on:
      - redis
    deploy:
      resources:
        limits:
          memory: 8G  # Docling needs more memory
        reservations:
          memory: 2G

  transcription-worker:
    image: {{ docker_registry }}/ai-backend:{{ image_tag }}
    command: celery -A app.workers.celery_app worker -l info -Q transcription -c 3
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - REDIS_URL={{ redis_url }}
      - DATABASE_URL={{ database_url }}
      - DEEPGRAM_API_KEY={{ deepgram_api_key }}
    depends_on:
      - redis
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 512M

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 256M

  flower:
    image: {{ docker_registry }}/ai-backend:{{ image_tag }}
    command: celery -A app.workers.celery_app flower --port=5555 --basic_auth={{ flower_user }}:{{ flower_password }}
    restart: unless-stopped
    ports:
      - "127.0.0.1:5555:5555"
    environment:
      - REDIS_URL={{ redis_url }}
    depends_on:
      - redis

volumes:
  redis_data:
```

#### Caddy Configuration Template

```
# infrastructure/templates/Caddyfile.j2
{{ domain }} {
    reverse_proxy localhost:8000

    # Security headers
    header {
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        Referrer-Policy strict-origin-when-cross-origin
        -Server
    }

    # Logging
    log {
        output file /var/log/caddy/access.log
        format json
    }
}

# Flower dashboard (optional, can be removed in production)
flower.{{ domain }} {
    reverse_proxy localhost:5555

    # Basic auth handled by Flower itself
}
```

### API Endpoints (Foundation)

```
GET  /health                    # Basic health check
GET  /health/ready              # Readiness (DB, Redis, etc.)
GET  /health/live               # Liveness probe

GET  /api/v1/status             # Service status with versions
POST /api/v1/llm/complete       # Sync LLM completion (for testing)
POST /api/v1/llm/stream         # Streaming LLM completion (SSE)
```

### Environment Variables

```bash
# .env.example

# Environment
ENVIRONMENT=development  # development | staging | production

# Database (Supabase)
DATABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:5432/postgres

# Authentication
SUPABASE_JWT_SECRET=your-jwt-secret
XENIOS_BACKEND_API_KEY=generate-secure-key

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM Providers
OPENROUTER_API_KEY=sk-or-...

# External Services (for future specs)
DEEPGRAM_API_KEY=...

# Monitoring
LOG_LEVEL=INFO
SENTRY_DSN=  # Optional
```

## Acceptance Criteria

### AC1: Project Setup
- [ ] FastAPI application starts without errors
- [ ] Project structure matches specification
- [ ] All dependencies install via `uv sync` or `poetry install`
- [ ] `.env.example` contains all required variables

### AC2: Docker Compose
- [ ] `docker compose up` starts all services (api, worker, redis, flower)
- [ ] API accessible at `http://localhost:8000`
- [ ] Flower dashboard accessible at `http://localhost:5555`
- [ ] Services restart on file changes (development mode)

### AC3: Health Endpoints
- [ ] `GET /health` returns 200 with `{"status": "healthy"}`
- [ ] `GET /health/ready` checks database and Redis connectivity
- [ ] `GET /health/live` returns 200 (liveness probe)

### AC4: Authentication
- [ ] Requests without `X-API-Key` header return 401
- [ ] Requests with invalid API key return 401
- [ ] Valid API key + valid JWT passes authentication
- [ ] User context (user_id, role) available in request handlers

### AC5: Database Connection
- [ ] Can connect to Supabase PostgreSQL
- [ ] `ai_backend` schema created if not exists
- [ ] Can execute queries against shared tables

### AC6: Celery Job Queue
- [ ] Celery workers connect to Redis
- [ ] Jobs can be enqueued and executed
- [ ] Different queues route to appropriate workers
- [ ] Job status trackable via Flower

### AC7: LLM Integration
- [ ] OpenRouter client configured with model routing
- [ ] `POST /api/v1/llm/complete` returns LLM response
- [ ] `POST /api/v1/llm/stream` returns streaming SSE response
- [ ] Model selection based on task type works correctly
- [ ] Fallback to secondary model on primary failure

### AC8: Logging
- [ ] Structured JSON logs in production
- [ ] Correlation IDs trace requests across services
- [ ] Log level configurable via environment

### AC9: Ansible Deployment Automation
- [ ] `ansible-playbook playbooks/provision.yml` provisions fresh VPS without errors
- [ ] Provision installs Docker, Caddy, configures firewall, creates deploy user
- [ ] `ansible-playbook playbooks/deploy.yml` deploys application successfully
- [ ] Deploy pulls images, renders templates with secrets, starts containers
- [ ] Secrets encrypted with Ansible Vault, decrypted only during deployment
- [ ] Health check passes after deployment
- [ ] `ansible-playbook playbooks/rollback.yml` restores previous version
- [ ] Caddy auto-provisions SSL certificate for domain
- [ ] All playbooks are idempotent (can run multiple times safely)

### AC10: Production Docker Compose
- [ ] Production compose file includes resource limits for all services
- [ ] Redis configured with AOF persistence
- [ ] All containers restart automatically (`unless-stopped`)
- [ ] Ports bound to localhost only (Caddy handles external traffic)
- [ ] Flower protected with basic auth in production

## Test Plan

### Unit Tests
- Config loading from environment
- JWT token validation
- API key validation
- LLM model routing logic

### Integration Tests
- Health endpoints
- Database connectivity
- Redis connectivity
- Celery task execution

### Manual Testing
- Docker Compose full stack startup
- End-to-end LLM request (both sync and streaming)
- Flower dashboard functionality

### Ansible Testing
- Provision playbook on fresh VM (use DigitalOcean/Hetzner test droplet)
- Deploy playbook to staging environment
- Verify SSL certificate provisioned by Caddy
- Test rollback playbook restores previous version
- Ansible-lint passes on all playbooks

## Dependencies

None - this is the foundation project.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Supabase JWT validation issues | Medium | High | Test with real JWT from MVP; document JWT secret configuration |
| OpenRouter rate limits | Low | Medium | Implement exponential backoff; configure fallback models |
| Redis connection drops | Low | Medium | Celery handles reconnection; add health checks |
| Docker networking issues | Low | Low | Use explicit network configuration; document troubleshooting |
| Ansible vault password management | Medium | High | Use password file in CI, document secure storage practices |
| VPS provider issues during deploy | Low | High | Implement rollback playbook; test deployment in staging first |
| SSL certificate provisioning failure | Low | Medium | Caddy auto-retries; manual DNS verification documented |

## Open Questions

1. **Supabase JWT Secret**: Need to verify how to obtain this from Supabase dashboard
2. **CORS Origins**: Confirm exact domains that need access (likely just xenios MVP domain)
3. **Rate Limits**: Define initial rate limits for LLM endpoints

## Future Considerations

- Add OpenTelemetry tracing for distributed tracing
- Add Prometheus metrics endpoint
- Consider connection pooling with pgBouncer for high load
- GPU worker support for self-hosted models
- **GitHub Actions CI/CD**: Automate `make deploy-staging` on push to main, require manual approval for production
- **Blue-green deployments**: Run two versions side-by-side, switch traffic via Caddy
- **Multi-server scaling**: Extend Ansible inventory for multiple workers with Docker Swarm

---

**Spec Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-27
