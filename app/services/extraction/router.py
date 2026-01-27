"""Document type routing for extraction pipeline."""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.extraction.base import BaseExtractor

# Document type constants
DOCUMENT_TYPES = {
    "inbody": "InBody body composition scan",
    "lab_results": "Lab results (Quest, LabCorp, etc.)",
    "garmin": "Garmin wearable export",
    "whoop": "WHOOP wearable export",
    "apple_health": "Apple Health export",
    "nutrition": "Nutrition log",
}


class DocumentRouter:
    """
    Route documents to appropriate extractors based on content and metadata.

    Uses file extension, MIME type, and content heuristics to determine
    the correct extractor for a given document.
    """

    # MIME type to possible document types
    MIME_TYPE_MAP: dict[str, list[str]] = {
        "application/pdf": ["inbody", "lab_results"],
        "text/csv": ["lab_results", "garmin", "whoop", "nutrition"],
        "application/json": ["garmin", "apple_health"],
        "text/xml": ["apple_health"],
        "application/xml": ["apple_health"],
    }

    # File extension to MIME type
    EXTENSION_MAP: dict[str, str] = {
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".json": "application/json",
        ".xml": "text/xml",
    }

    # Content patterns for document type detection
    CONTENT_PATTERNS: dict[str, list[re.Pattern]] = {
        "inbody": [
            re.compile(r"InBody\s*(570|770|S10|230|270)", re.IGNORECASE),
            re.compile(r"Body\s*Composition\s*Analysis", re.IGNORECASE),
            re.compile(r"Skeletal\s*Muscle\s*Mass", re.IGNORECASE),
            re.compile(r"Percent\s*Body\s*Fat|PBF", re.IGNORECASE),
        ],
        "lab_results": [
            re.compile(r"Quest\s*Diagnostics", re.IGNORECASE),
            re.compile(r"LabCorp|Laboratory\s*Corporation", re.IGNORECASE),
            re.compile(r"Lipid\s*Panel|Metabolic\s*Panel", re.IGNORECASE),
            re.compile(r"(LDL|HDL)\s*Cholesterol", re.IGNORECASE),
            re.compile(r"Reference\s*Range", re.IGNORECASE),
        ],
        "garmin": [
            re.compile(r"Garmin", re.IGNORECASE),
            re.compile(r"Activity\s*Type.*Steps.*Distance", re.IGNORECASE),
            re.compile(r"Floors\s*Climbed", re.IGNORECASE),
        ],
        "whoop": [
            re.compile(r"WHOOP", re.IGNORECASE),
            re.compile(r"Recovery\s*score\s*%", re.IGNORECASE),
            re.compile(r"Strain.*HRV", re.IGNORECASE),
        ],
        "apple_health": [
            re.compile(r"HealthKit", re.IGNORECASE),
            re.compile(r"HKQuantityTypeIdentifier", re.IGNORECASE),
            re.compile(r'<HealthData[^>]*locale="', re.IGNORECASE),
        ],
        "nutrition": [
            re.compile(r"(Calories|Protein|Carbs|Fat).*\d+\s*(g|kcal)", re.IGNORECASE),
            re.compile(r"MyFitnessPal|Cronometer|MacroFactor", re.IGNORECASE),
        ],
    }

    def __init__(self) -> None:
        """Initialize the document router."""
        self._extractors: dict[str, "BaseExtractor"] = {}

    def register_extractor(self, extractor: "BaseExtractor") -> None:
        """
        Register an extractor for a document type.

        Args:
            extractor: The extractor instance to register
        """
        self._extractors[extractor.document_type] = extractor

    def get_extractor(self, document_type: str) -> "BaseExtractor | None":
        """
        Get the extractor for a document type.

        Args:
            document_type: The document type

        Returns:
            The registered extractor or None
        """
        return self._extractors.get(document_type)

    def detect_document_type(
        self,
        file_content: bytes,
        filename: str,
        hint: str | None = None,
    ) -> str | None:
        """
        Detect the document type from content and filename.

        Args:
            file_content: The file content as bytes
            filename: The original filename
            hint: Optional hint about the document type

        Returns:
            Detected document type or None if undetected
        """
        # If hint is provided and valid, use it
        if hint and hint in DOCUMENT_TYPES:
            return hint

        # Get file extension
        ext = Path(filename).suffix.lower()
        mime_type = self.EXTENSION_MAP.get(ext)

        if not mime_type:
            return None

        # Get possible types for this MIME type
        possible_types = self.MIME_TYPE_MAP.get(mime_type, [])

        if not possible_types:
            return None

        # If only one possibility, return it
        if len(possible_types) == 1:
            return possible_types[0]

        # Use content heuristics to narrow down
        try:
            # Try to decode content for text-based formats
            if mime_type in ("text/csv", "text/xml", "application/json"):
                text_content = file_content.decode("utf-8", errors="ignore")
            else:
                # For PDFs, we'll check the raw bytes for patterns
                text_content = file_content.decode("latin-1", errors="ignore")
        except Exception:
            text_content = ""

        # Score each possible type by pattern matches
        scores: dict[str, int] = {}
        for doc_type in possible_types:
            patterns = self.CONTENT_PATTERNS.get(doc_type, [])
            score = sum(1 for p in patterns if p.search(text_content))
            if score > 0:
                scores[doc_type] = score

        if scores:
            # Return the type with the highest score
            return max(scores, key=lambda k: scores[k])

        # Default to first possible type if no patterns matched
        return possible_types[0] if possible_types else None

    def get_mime_type(self, filename: str) -> str | None:
        """
        Get MIME type from filename extension.

        Args:
            filename: The filename

        Returns:
            MIME type or None
        """
        ext = Path(filename).suffix.lower()
        return self.EXTENSION_MAP.get(ext)

    @property
    def supported_types(self) -> list[str]:
        """Get list of supported document types."""
        return list(DOCUMENT_TYPES.keys())

    @property
    def registered_extractors(self) -> list[str]:
        """Get list of registered extractor types."""
        return list(self._extractors.keys())


# Global router instance
_router: DocumentRouter | None = None


def get_document_router() -> DocumentRouter:
    """Get the global document router instance."""
    global _router
    if _router is None:
        _router = DocumentRouter()
    return _router
