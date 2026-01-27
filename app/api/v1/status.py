"""Service status endpoint."""

from fastapi import APIRouter, Depends

from app import __version__
from app.config import get_settings
from app.core.auth import verify_api_key
from app.schemas.common import ServiceStatus

router = APIRouter(tags=["status"])


@router.get(
    "/status",
    response_model=ServiceStatus,
    dependencies=[Depends(verify_api_key)],
)
async def get_status() -> ServiceStatus:
    """
    Get service status and version information.

    Requires API key authentication.
    """
    settings = get_settings()

    return ServiceStatus(
        service="xenios-ai-backend",
        version=__version__,
        environment=settings.environment,
        status="operational",
    )
