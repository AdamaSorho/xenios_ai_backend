"""Tests for lab results extractor."""

from pathlib import Path

import pytest

from app.services.extraction.lab_results import LabResultsExtractor, BiomarkerValue


# Get path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLabResultsExtractor:
    """Tests for lab results extraction."""

    @pytest.fixture
    def extractor(self):
        return LabResultsExtractor()

    @pytest.fixture
    def sample_csv_path(self):
        return str(FIXTURES_DIR / "sample_lab_results.csv")

    @pytest.mark.asyncio
    async def test_extract_from_csv(self, extractor, sample_csv_path):
        """Test extracting biomarkers from CSV."""
        result = await extractor.extract(sample_csv_path)

        assert result.success
        assert result.confidence >= 0.8
        assert result.data is not None
        assert "biomarkers" in result.data

    @pytest.mark.asyncio
    async def test_extract_biomarker_values(self, extractor, sample_csv_path):
        """Test that biomarker values are correctly extracted."""
        result = await extractor.extract(sample_csv_path)

        biomarkers = result.data["biomarkers"]
        assert len(biomarkers) > 0

        # Find total cholesterol
        tc = next((b for b in biomarkers if "cholesterol" in b["name"].lower() and "total" in b["name"].lower()), None)
        assert tc is not None
        assert tc["value"] == 195
        assert tc["unit"] == "mg/dL"

    @pytest.mark.asyncio
    async def test_extract_biomarker_flags(self, extractor, sample_csv_path):
        """Test that biomarker flags are extracted."""
        result = await extractor.extract(sample_csv_path)

        biomarkers = result.data["biomarkers"]

        # Find LDL which is flagged as high
        ldl = next((b for b in biomarkers if "ldl" in b["name"].lower()), None)
        assert ldl is not None
        # Either extracted from CSV or calculated
        assert ldl.get("flag") in ["High", "high", "borderline"]

    @pytest.mark.asyncio
    async def test_extract_glucose(self, extractor, sample_csv_path):
        """Test glucose extraction."""
        result = await extractor.extract(sample_csv_path)

        biomarkers = result.data["biomarkers"]
        glucose = next((b for b in biomarkers if "glucose" in b["name"].lower()), None)

        assert glucose is not None
        assert glucose["value"] == 92
        assert glucose["flag"] == "normal"

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, extractor, tmp_path):
        """Test handling of unsupported file type."""
        bad_file = tmp_path / "data.txt"
        bad_file.write_text("some data")

        result = await extractor.extract(str(bad_file))

        assert not result.success
        assert "Unsupported file type" in result.errors[0]

    @pytest.mark.asyncio
    async def test_empty_csv(self, extractor, tmp_path):
        """Test handling of empty CSV."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("Test Name,Result,Unit\n")

        result = await extractor.extract(str(empty_file))

        # Should fail or return empty biomarkers
        if result.success:
            assert len(result.data["biomarkers"]) == 0
        else:
            assert "empty" in result.errors[0].lower() or "No valid" in result.errors[0]

    def test_validate_with_biomarkers(self, extractor):
        """Test validation with valid biomarkers."""
        data = {
            "biomarkers": [
                {"name": "LDL", "value": 100, "unit": "mg/dL"}
            ]
        }
        is_valid, errors = extractor.validate(data)
        assert is_valid
        assert len(errors) == 0

    def test_validate_without_biomarkers(self, extractor):
        """Test validation with no biomarkers."""
        data = {"biomarkers": []}
        is_valid, errors = extractor.validate(data)
        assert not is_valid
        assert "No biomarkers" in errors[0]

    def test_validate_unreasonable_value(self, extractor):
        """Test validation catches unreasonable values."""
        data = {
            "biomarkers": [
                {"name": "LDL", "value": -100, "unit": "mg/dL"}
            ]
        }
        is_valid, errors = extractor.validate(data)
        assert not is_valid
        assert "Unreasonable" in errors[0]


class TestBiomarkerValue:
    """Tests for BiomarkerValue model."""

    def test_create_biomarker(self):
        """Test creating a biomarker value."""
        biomarker = BiomarkerValue(
            name="LDL Cholesterol",
            value=118.0,
            unit="mg/dL",
            flag="high",
            confidence=0.95,
        )

        assert biomarker.name == "LDL Cholesterol"
        assert biomarker.value == 118.0
        assert biomarker.flag == "high"

    def test_create_biomarker_with_code(self):
        """Test creating a biomarker with LOINC code."""
        biomarker = BiomarkerValue(
            name="Glucose",
            code="2345-7",
            value=92.0,
            unit="mg/dL",
            reference_range="70-100",
            confidence=0.9,
        )

        assert biomarker.code == "2345-7"
        assert biomarker.reference_range == "70-100"

    def test_default_confidence(self):
        """Test default confidence value."""
        biomarker = BiomarkerValue(
            name="Test",
            value=100.0,
            unit="mg/dL",
        )

        assert biomarker.confidence == 0.9


class TestBiomarkerAliases:
    """Tests for biomarker name standardization."""

    @pytest.fixture
    def extractor(self):
        return LabResultsExtractor()

    def test_standardize_ldl(self, extractor):
        """Test LDL name standardization."""
        std = extractor._standardize_biomarker_name("LDL Cholesterol")
        assert std == "ldl_cholesterol"

        std = extractor._standardize_biomarker_name("LDL-C")
        assert std == "ldl_cholesterol"

    def test_standardize_glucose(self, extractor):
        """Test glucose name standardization."""
        std = extractor._standardize_biomarker_name("Fasting Glucose")
        assert std == "glucose"

        std = extractor._standardize_biomarker_name("Blood Glucose")
        assert std == "glucose"

    def test_standardize_hba1c(self, extractor):
        """Test HbA1c name standardization."""
        std = extractor._standardize_biomarker_name("Hemoglobin A1c")
        assert std == "hemoglobin_a1c"

        std = extractor._standardize_biomarker_name("HbA1c")
        assert std == "hemoglobin_a1c"

    def test_standardize_unknown(self, extractor):
        """Test unknown biomarker returns None."""
        std = extractor._standardize_biomarker_name("Unknown Marker XYZ")
        assert std is None
