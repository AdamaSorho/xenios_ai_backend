"""Xenios AI Backend - FastAPI Application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.config import get_settings
from app.core.logging import setup_logging
from app.middleware.audit_log import AuditLogMiddleware
from app.middleware.correlation import CorrelationIdMiddleware
from app.middleware.rate_limit import RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    setup_logging()

    yield

    # Shutdown (cleanup if needed)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Xenios AI Backend",
        description="AI-powered backend services for the Xenios coaching platform",
        version=__version__,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # Add middleware (order matters - first added = last executed)
    # Execution order: CORS -> Correlation -> RateLimit -> AuditLog -> Request
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(RateLimitMiddleware)  # Rate limit before audit (Spec 0004)
    app.add_middleware(AuditLogMiddleware)  # Audit logging (Spec 0004)

    # Include routers
    app.include_router(api_router)

    return app


# Create app instance
app = create_app()
