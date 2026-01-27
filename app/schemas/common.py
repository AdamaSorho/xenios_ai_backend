"""Common schemas used across the application."""

from typing import Literal

from pydantic import BaseModel


class ServiceStatus(BaseModel):
    """Service status response."""

    service: str
    version: str
    environment: Literal["development", "staging", "production"]
    status: Literal["operational", "degraded", "down"]


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: str | None = None


class PaginatedResponse(BaseModel):
    """Base class for paginated responses."""

    total: int
    page: int
    page_size: int
    has_more: bool
