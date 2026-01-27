"""Health check endpoints for monitoring and orchestration."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.database import check_db_health
from app.core.redis import check_redis_health
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns 200 if the service is running. Does not check dependencies.
    Used for basic "is the process alive" checks.
    """
    return HealthResponse(status="healthy")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness() -> JSONResponse:
    """
    Readiness probe endpoint.

    Checks all dependencies (database, Redis) and returns their status.
    Returns 200 only if all dependencies are healthy.
    Used by load balancers to determine if the service can accept traffic.
    """
    db_healthy = await check_db_health()
    redis_healthy = await check_redis_health()

    all_healthy = db_healthy and redis_healthy
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not ready",
            "checks": {
                "database": "ok" if db_healthy else "failed",
                "redis": "ok" if redis_healthy else "failed",
            },
        },
    )


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """
    Liveness probe endpoint.

    Returns 200 if the service is running and not deadlocked.
    Used by Kubernetes/orchestrators to determine if the process should be restarted.
    """
    return HealthResponse(status="alive")
