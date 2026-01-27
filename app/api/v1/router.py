"""V1 API router."""

from fastapi import APIRouter

from app.api.v1.analytics import router as analytics_router
from app.api.v1.extraction import router as extraction_router
from app.api.v1.llm import router as llm_router
from app.api.v1.status import router as status_router
from app.api.v1.transcription import router as transcription_router

router = APIRouter()

# Status endpoint
router.include_router(status_router)

# LLM endpoints
router.include_router(llm_router)

# Document extraction endpoints
router.include_router(extraction_router)

# Transcription endpoints (Spec 0003)
router.include_router(transcription_router)

# Analytics endpoints (Spec 0005)
router.include_router(analytics_router)
