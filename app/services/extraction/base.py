"""Base extractor class for document extraction."""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """Result of a document extraction operation."""

    success: bool
    data: dict[str, Any] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    extraction_time_ms: int | None = None


class BaseExtractor(ABC):
    """
    Base class for all document extractors.

    Subclasses must implement:
    - document_type: The type of document this extractor handles
    - extract(): Extract data from the document
    - validate(): Validate extracted data
    """

    document_type: str

    @abstractmethod
    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract data from a document.

        Args:
            file_path: Path to the document file

        Returns:
            ExtractionResult with extracted data or errors
        """
        pass

    @abstractmethod
    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate extracted data.

        Args:
            data: Extracted data dictionary

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        pass

    def calculate_confidence(self, field_confidences: dict[str, float]) -> float:
        """
        Calculate overall confidence from individual field confidences.

        Args:
            field_confidences: Dict mapping field names to confidence scores

        Returns:
            Weighted average confidence score
        """
        if not field_confidences:
            return 0.0
        return sum(field_confidences.values()) / len(field_confidences)

    def _parse_date(self, date_str: str | None) -> date | None:
        """
        Parse a date string to a date object.

        Handles common date formats.
        """
        if not date_str:
            return None

        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m/%d/%y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%Y%m%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue

        return None

    def _convert_to_kg(self, value: float, unit: str) -> float:
        """Convert weight to kilograms."""
        unit_lower = unit.lower().strip()
        if unit_lower in ("lb", "lbs", "pounds"):
            return value * 0.453592
        return value

    def _convert_to_cm(self, value: float, unit: str) -> float:
        """Convert height to centimeters."""
        unit_lower = unit.lower().strip()
        if unit_lower in ("in", "inches"):
            return value * 2.54
        if unit_lower in ("ft", "feet"):
            return value * 30.48
        return value
