"""V1 API router."""

from fastapi import APIRouter

from app.api.v1.llm import router as llm_router
from app.api.v1.status import router as status_router

router = APIRouter()

# Status endpoint
router.include_router(status_router)

# LLM endpoints
router.include_router(llm_router)
