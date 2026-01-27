"""Health check endpoint schemas."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for basic health check endpoints."""

    status: Literal["healthy", "alive"]


class HealthCheckDetail(BaseModel):
    """Individual health check result."""

    database: Literal["ok", "failed"]
    redis: Literal["ok", "failed"]


class ReadinessResponse(BaseModel):
    """Response for readiness probe endpoint."""

    status: Literal["ready", "not ready"]
    checks: HealthCheckDetail
