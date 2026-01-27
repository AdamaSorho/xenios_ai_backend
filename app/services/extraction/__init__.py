"""Document extraction services for health documents."""

from app.services.extraction.base import BaseExtractor, ExtractionResult
from app.services.extraction.inbody import InBodyExtractor
from app.services.extraction.router import DocumentRouter, get_document_router
from app.services.extraction.storage import StorageService

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "DocumentRouter",
    "InBodyExtractor",
    "StorageService",
    "get_document_router",
    "register_extractors",
]


def register_extractors() -> None:
    """Register all available extractors with the document router."""
    router = get_document_router()

    # Register InBody extractor
    router.register_extractor(InBodyExtractor())

    # Additional extractors will be registered as they are implemented:
    # - LabResultsExtractor (Phase 4)
    # - GarminNormalizer, WhoopNormalizer, AppleHealthNormalizer (Phase 5)
