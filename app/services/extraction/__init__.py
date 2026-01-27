"""Document extraction services for health documents."""

from app.services.extraction.base import BaseExtractor, ExtractionResult
from app.services.extraction.inbody import InBodyExtractor
from app.services.extraction.lab_results import LabResultsExtractor
from app.services.extraction.router import DocumentRouter, get_document_router
from app.services.extraction.storage import StorageService
from app.services.extraction.wearables import (
    AppleHealthNormalizer,
    GarminNormalizer,
    WhoopNormalizer,
)

__all__ = [
    "AppleHealthNormalizer",
    "BaseExtractor",
    "DocumentRouter",
    "ExtractionResult",
    "GarminNormalizer",
    "InBodyExtractor",
    "LabResultsExtractor",
    "StorageService",
    "WhoopNormalizer",
    "get_document_router",
    "register_extractors",
]


def register_extractors() -> None:
    """Register all available extractors with the document router."""
    router = get_document_router()

    # Register document extractors
    router.register_extractor(InBodyExtractor())
    router.register_extractor(LabResultsExtractor())

    # Register wearable normalizers
    router.register_extractor(GarminNormalizer())
    router.register_extractor(WhoopNormalizer())
    router.register_extractor(AppleHealthNormalizer())
