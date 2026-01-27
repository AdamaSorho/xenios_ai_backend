"""V1 API router."""

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.extraction import router as extraction_router
from app.api.v1.insights import router as insights_router
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

# RAG endpoints (Spec 0004)
router.include_router(embeddings_router)
router.include_router(chat_router)
router.include_router(insights_router)
