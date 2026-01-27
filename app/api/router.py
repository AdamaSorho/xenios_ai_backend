"""Main API router aggregating all route modules."""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.v1.router import router as v1_router

api_router = APIRouter()

# Health check endpoints (no prefix, no auth)
api_router.include_router(health_router)

# V1 API endpoints
api_router.include_router(v1_router, prefix="/api/v1")
