"""Tests for extraction base classes and utilities."""

import pytest
from datetime import date

from app.services.extraction.base import BaseExtractor, ExtractionResult
from app.services.extraction.validation import (
    VALIDATION_RANGES,
    flag_biomarker_value,
    validate_date_not_future,
    validate_range,
    validate_required_fields,
)
from app.services.extraction.router import DocumentRouter, get_document_router


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_successful_result(self):
        """Test creating a successful extraction result."""
        result = ExtractionResult(
            success=True,
            data={"weight_kg": 75.5},
            confidence=0.95,
            errors=[],
            warnings=[],
        )
        assert result.success
        assert result.data["weight_kg"] == 75.5
        assert result.confidence == 0.95

    def test_failed_result(self):
        """Test creating a failed extraction result."""
        result = ExtractionResult(
            success=False,
            data=None,
            confidence=0.0,
            errors=["Could not extract weight"],
            warnings=["Low quality scan"],
        )
        assert not result.success
        assert result.data is None
        assert "Could not extract weight" in result.errors

    def test_confidence_bounds(self):
        """Test that confidence is bounded 0-1."""
        # Valid confidence
        result = ExtractionResult(success=True, confidence=0.5)
        assert result.confidence == 0.5

        # Test boundary values
        result_min = ExtractionResult(success=True, confidence=0.0)
        assert result_min.confidence == 0.0

        result_max = ExtractionResult(success=True, confidence=1.0)
        assert result_max.confidence == 1.0


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_range_valid(self):
        """Test valid range validation."""
        is_valid, error = validate_range("weight_kg", 75.0)
        assert is_valid
        assert error is None

    def test_validate_range_invalid_low(self):
        """Test range validation for value too low."""
        is_valid, error = validate_range("weight_kg", 15.0)
        assert not is_valid
        assert "outside valid range" in error

    def test_validate_range_invalid_high(self):
        """Test range validation for value too high."""
        is_valid, error = validate_range("weight_kg", 600.0)
        assert not is_valid
        assert "outside valid range" in error

    def test_validate_range_none_value(self):
        """Test range validation with None value."""
        is_valid, error = validate_range("weight_kg", None)
        assert is_valid
        assert error is None

    def test_validate_range_unknown_field(self):
        """Test range validation for unknown field."""
        is_valid, error = validate_range("unknown_field", 100.0)
        assert is_valid
        assert error is None

    def test_validate_required_fields_present(self):
        """Test validation when required fields are present."""
        data = {"weight_kg": 75.0, "body_fat_percent": 18.5}
        errors = validate_required_fields(data, ["weight_kg", "body_fat_percent"])
        assert len(errors) == 0

    def test_validate_required_fields_missing(self):
        """Test validation when required fields are missing."""
        data = {"weight_kg": 75.0}
        errors = validate_required_fields(data, ["weight_kg", "body_fat_percent"])
        assert len(errors) == 1
        assert "body_fat_percent" in errors[0]

    def test_validate_date_not_future_valid(self):
        """Test date validation for past date."""
        is_valid, error = validate_date_not_future("scan_date", date(2024, 1, 15))
        assert is_valid
        assert error is None

    def test_validate_date_not_future_invalid(self):
        """Test date validation for future date."""
        from datetime import timedelta

        future_date = date.today() + timedelta(days=30)
        is_valid, error = validate_date_not_future("scan_date", future_date)
        assert not is_valid
        assert "future" in error

    def test_flag_biomarker_high(self):
        """Test biomarker flagging for high values."""
        flag = flag_biomarker_value("ldl_cholesterol", 150)
        assert flag == "high"

    def test_flag_biomarker_low(self):
        """Test biomarker flagging for low values."""
        flag = flag_biomarker_value("hdl_cholesterol", 30)
        assert flag == "low"

    def test_flag_biomarker_normal(self):
        """Test biomarker flagging for normal values."""
        flag = flag_biomarker_value("glucose", 90)
        assert flag == "normal"

    def test_flag_biomarker_unknown(self):
        """Test biomarker flagging for unknown biomarker."""
        flag = flag_biomarker_value("unknown_marker", 100)
        assert flag is None


class TestDocumentRouter:
    """Tests for document type routing."""

    def test_detect_csv_type(self):
        """Test document type detection for CSV."""
        router = DocumentRouter()
        content = b"Date,Steps,Calories\n2024-01-15,8000,2000"

        detected = router.detect_document_type(content, "data.csv")
        assert detected in ["lab_results", "garmin", "whoop", "nutrition"]

    def test_detect_with_hint(self):
        """Test that hint overrides detection."""
        router = DocumentRouter()
        content = b"some random content"

        detected = router.detect_document_type(content, "file.csv", hint="garmin")
        assert detected == "garmin"

    def test_detect_garmin_content(self):
        """Test Garmin detection from content."""
        router = DocumentRouter()
        content = b"Garmin Connect Export\nDate,Steps,Floors Climbed"

        detected = router.detect_document_type(content, "export.csv")
        assert detected == "garmin"

    def test_detect_whoop_content(self):
        """Test WHOOP detection from content."""
        router = DocumentRouter()
        content = b"WHOOP Recovery\nRecovery score %,Strain,HRV"

        detected = router.detect_document_type(content, "export.csv")
        assert detected == "whoop"

    def test_get_mime_type(self):
        """Test MIME type detection from filename."""
        router = DocumentRouter()

        assert router.get_mime_type("file.pdf") == "application/pdf"
        assert router.get_mime_type("data.csv") == "text/csv"
        assert router.get_mime_type("export.json") == "application/json"
        assert router.get_mime_type("health.xml") == "text/xml"

    def test_supported_types(self):
        """Test listing supported document types."""
        router = DocumentRouter()
        types = router.supported_types

        assert "inbody" in types
        assert "lab_results" in types
        assert "garmin" in types
        assert "whoop" in types
        assert "apple_health" in types

    def test_register_and_get_extractor(self):
        """Test registering and retrieving extractors."""
        from app.services.extraction import register_extractors

        router = get_document_router()
        register_extractors()

        # Check extractors are registered
        assert router.get_extractor("inbody") is not None
        assert router.get_extractor("lab_results") is not None
        assert router.get_extractor("garmin") is not None
        assert router.get_extractor("whoop") is not None
        assert router.get_extractor("apple_health") is not None
