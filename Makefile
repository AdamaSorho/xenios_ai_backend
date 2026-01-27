# Xenios AI Backend Makefile
.PHONY: help dev test lint format build up down logs clean test-transcription worker-transcription

# Default target
help:
	@echo "Xenios AI Backend - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Run API server locally with hot reload"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linter (ruff)"
	@echo "  make format       - Format code (ruff)"
	@echo "  make typecheck    - Run type checker (mypy)"
	@echo "  make verify       - Run lint + typecheck + test"
	@echo ""
	@echo "Docker:"
	@echo "  make build        - Build Docker images"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo "  make logs         - View service logs"
	@echo "  make restart      - Restart all services"
	@echo "  make clean        - Remove containers, volumes, and images"
	@echo ""
	@echo "Workers:"
	@echo "  make worker                 - Start default worker only"
	@echo "  make worker-extraction      - Start extraction worker"
	@echo "  make worker-transcription   - Start transcription worker"
	@echo "  make flower                 - Open Flower dashboard"
	@echo ""
	@echo "Transcription:"
	@echo "  make test-transcription     - Run transcription tests"
	@echo "  make migrate-transcription  - Apply transcription DB migration"
	@echo ""
	@echo "Ansible Deployment:"
	@echo "  make provision ENV=staging    - Provision VPS"
	@echo "  make deploy ENV=staging       - Deploy application"
	@echo "  make deploy-staging           - Deploy to staging"
	@echo "  make deploy-prod              - Deploy to production"
	@echo "  make rollback ENV=staging     - Rollback to previous version"
	@echo "  make edit-secrets ENV=staging - Edit encrypted secrets"
	@echo "  make lint-ansible             - Lint Ansible playbooks"

# ============================================================================
# Development
# ============================================================================

# Run API server locally
dev:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=app --cov-report=term-missing --cov-report=html

# Run linter
lint:
	uv run ruff check app tests

# Format code
format:
	uv run ruff format app tests
	uv run ruff check --fix app tests

# Type checking
typecheck:
	uv run mypy app

# Full verification (lint + typecheck + test)
verify: lint typecheck test

# Install dependencies
install:
	uv sync

# Install with dev dependencies
install-dev:
	uv sync --dev

# ============================================================================
# Extraction (Spec 0002)
# ============================================================================

# Run extraction tests only
test-extraction:
	uv run pytest tests/test_extraction_*.py tests/test_*extractor*.py tests/test_wearable*.py tests/test_lab*.py -v

# Start extraction worker only
worker-extraction:
	docker compose -f docker/docker-compose.yml up -d worker-extraction

# ============================================================================
# Transcription (Spec 0003)
# ============================================================================

# Run transcription tests only
test-transcription:
	uv run pytest tests/test_transcription_*.py -v

# Start transcription worker only
worker-transcription:
	docker compose -f docker/docker-compose.yml up -d worker-transcription

# Run transcription worker locally (for development)
worker-transcription-local:
	uv run celery -A app.workers.celery_app worker -l info -Q transcription -c 2

# Apply transcription database migration
migrate-transcription:
	psql $${DATABASE_URL} -f scripts/migrations/0003_transcription_tables.sql

# ============================================================================
# Docker
# ============================================================================

# Build Docker images
build:
	docker compose -f docker/docker-compose.yml build

# Start all services
up:
	docker compose -f docker/docker-compose.yml up -d

# Stop all services
down:
	docker compose -f docker/docker-compose.yml down

# View logs
logs:
	docker compose -f docker/docker-compose.yml logs -f

# Restart services
restart:
	docker compose -f docker/docker-compose.yml restart

# Clean up Docker resources
clean:
	docker compose -f docker/docker-compose.yml down -v --rmi local

# Start only the worker
worker:
	docker compose -f docker/docker-compose.yml up -d worker

# Open Flower dashboard
flower:
	@echo "Opening Flower dashboard at http://localhost:5555"
	@open http://localhost:5555 2>/dev/null || xdg-open http://localhost:5555 2>/dev/null || echo "Open http://localhost:5555 in your browser"

# ============================================================================
# Database
# ============================================================================

# Initialize database schema
init-db:
	uv run python -c "import asyncio; from app.core.database import init_ai_backend_schema; asyncio.run(init_ai_backend_schema())"

# Run database migrations (placeholder for future use)
migrate:
	@echo "No migrations configured yet"

# ============================================================================
# Ansible Deployment
# ============================================================================

# Provision a new server (run once per server)
provision:
ifndef ENV
	$(error ENV is required. Usage: make provision ENV=staging)
endif
	cd infrastructure && ansible-playbook playbooks/provision.yml -i inventory/hosts.yml --limit $(ENV)

# Deploy application
deploy:
ifndef ENV
	$(error ENV is required. Usage: make deploy ENV=staging)
endif
	cd infrastructure && ansible-playbook playbooks/deploy.yml -i inventory/hosts.yml --limit $(ENV) --ask-vault-pass

# Deploy to staging
deploy-staging:
	$(MAKE) deploy ENV=staging

# Deploy to production
deploy-prod:
	$(MAKE) deploy ENV=production

# Rollback to previous version
rollback:
ifndef ENV
	$(error ENV is required. Usage: make rollback ENV=staging)
endif
	cd infrastructure && ansible-playbook playbooks/rollback.yml -i inventory/hosts.yml --limit $(ENV) --ask-vault-pass

# Edit encrypted secrets
edit-secrets:
ifndef ENV
	$(error ENV is required. Usage: make edit-secrets ENV=staging)
endif
	cd infrastructure && ansible-vault edit inventory/group_vars/$(ENV).yml

# Encrypt a file
encrypt:
ifndef FILE
	$(error FILE is required. Usage: make encrypt FILE=path/to/file)
endif
	cd infrastructure && ansible-vault encrypt $(FILE)

# Lint Ansible playbooks
lint-ansible:
	cd infrastructure && ansible-lint playbooks/*.yml roles/*/tasks/*.yml 2>/dev/null || echo "ansible-lint not installed or no files to lint"
