"""Tests for wearable data normalizers."""

import os
from pathlib import Path

import pytest

from app.services.extraction.wearables import (
    DailyHealthMetrics,
    GarminNormalizer,
    WhoopNormalizer,
    AppleHealthNormalizer,
)


# Get path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestGarminNormalizer:
    """Tests for Garmin data normalizer."""

    @pytest.fixture
    def normalizer(self):
        return GarminNormalizer()

    @pytest.fixture
    def sample_csv_path(self):
        return str(FIXTURES_DIR / "sample_garmin.csv")

    @pytest.mark.asyncio
    async def test_extract_from_csv(self, normalizer, sample_csv_path):
        """Test extracting data from Garmin CSV."""
        result = await normalizer.extract(sample_csv_path)

        assert result.success
        assert result.confidence >= 0.9
        assert result.data is not None
        assert result.data["source"] == "garmin"
        assert result.data["total_days"] == 5

    @pytest.mark.asyncio
    async def test_extract_daily_metrics(self, normalizer, sample_csv_path):
        """Test that daily metrics are correctly extracted."""
        result = await normalizer.extract(sample_csv_path)

        daily_metrics = result.data["daily_metrics"]
        assert len(daily_metrics) == 5

        # Check first day
        first_day = daily_metrics[0]
        assert first_day["steps"] == 8542
        assert first_day["calories_burned"] == 2150
        assert first_day["resting_hr"] == 62

    @pytest.mark.asyncio
    async def test_extract_date_range(self, normalizer, sample_csv_path):
        """Test date range extraction."""
        result = await normalizer.extract(sample_csv_path)

        assert result.data["date_range_start"] == "2024-01-15"
        assert result.data["date_range_end"] == "2024-01-19"

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, normalizer, tmp_path):
        """Test handling of unsupported file type."""
        # Create a file with unsupported extension
        bad_file = tmp_path / "data.txt"
        bad_file.write_text("some data")

        result = await normalizer.extract(str(bad_file))

        assert not result.success
        assert "Unsupported file type" in result.errors[0]

    def test_metric_types(self, normalizer):
        """Test listing available metric types."""
        types = normalizer.get_metric_types()

        assert "steps" in types
        assert "calories" in types
        assert "resting_hr" in types

    def test_validate_valid_data(self, normalizer):
        """Test validation of valid data."""
        data = {
            "daily_metrics": [
                {"date": "2024-01-15", "steps": 8000}
            ]
        }
        is_valid, errors = normalizer.validate(data)
        assert is_valid
        assert len(errors) == 0

    def test_validate_empty_data(self, normalizer):
        """Test validation of empty data."""
        data = {"daily_metrics": []}
        is_valid, errors = normalizer.validate(data)
        assert not is_valid
        assert "No daily metrics" in errors[0]


class TestWhoopNormalizer:
    """Tests for WHOOP data normalizer."""

    @pytest.fixture
    def normalizer(self):
        return WhoopNormalizer()

    @pytest.fixture
    def sample_csv_path(self):
        return str(FIXTURES_DIR / "sample_whoop.csv")

    @pytest.mark.asyncio
    async def test_extract_from_csv(self, normalizer, sample_csv_path):
        """Test extracting data from WHOOP CSV."""
        result = await normalizer.extract(sample_csv_path)

        assert result.success
        assert result.confidence >= 0.9
        assert result.data is not None
        assert result.data["source"] == "whoop"
        assert result.data["total_days"] == 5

    @pytest.mark.asyncio
    async def test_extract_recovery_strain(self, normalizer, sample_csv_path):
        """Test that WHOOP-specific metrics are extracted."""
        result = await normalizer.extract(sample_csv_path)

        daily_metrics = result.data["daily_metrics"]
        first_day = daily_metrics[0]

        # WHOOP-specific fields
        assert first_day["recovery_score"] == 72
        assert first_day["strain"] == 12.5
        assert first_day["hrv"] == 45

    @pytest.mark.asyncio
    async def test_extract_sleep_metrics(self, normalizer, sample_csv_path):
        """Test sleep metric extraction."""
        result = await normalizer.extract(sample_csv_path)

        daily_metrics = result.data["daily_metrics"]
        first_day = daily_metrics[0]

        assert first_day["sleep_duration_hours"] == 7.2
        assert first_day["sleep_score"] == 85

    def test_validate_recovery_range(self, normalizer):
        """Test validation catches invalid recovery scores."""
        data = {
            "daily_metrics": [
                {"recovery_score": 150}  # Invalid: > 100
            ]
        }
        is_valid, errors = normalizer.validate(data)
        assert not is_valid
        assert "recovery score" in errors[0].lower()

    def test_validate_strain_range(self, normalizer):
        """Test validation catches invalid strain scores."""
        data = {
            "daily_metrics": [
                {"strain": 30}  # Invalid: > 25
            ]
        }
        is_valid, errors = normalizer.validate(data)
        assert not is_valid
        assert "strain" in errors[0].lower()


class TestAppleHealthNormalizer:
    """Tests for Apple Health data normalizer."""

    @pytest.fixture
    def normalizer(self):
        return AppleHealthNormalizer()

    @pytest.fixture
    def sample_xml_path(self):
        return str(FIXTURES_DIR / "sample_apple_health.xml")

    @pytest.mark.asyncio
    async def test_extract_from_xml(self, normalizer, sample_xml_path):
        """Test extracting data from Apple Health XML."""
        result = await normalizer.extract(sample_xml_path)

        assert result.success
        assert result.confidence >= 0.9
        assert result.data is not None
        assert result.data["source"] == "apple_health"

    @pytest.mark.asyncio
    async def test_extract_steps(self, normalizer, sample_xml_path):
        """Test step count extraction."""
        result = await normalizer.extract(sample_xml_path)

        daily_metrics = result.data["daily_metrics"]
        # Find the first day
        first_day = next((m for m in daily_metrics if m["date"] == "2024-01-15"), None)

        assert first_day is not None
        assert first_day["steps"] == 8542

    @pytest.mark.asyncio
    async def test_extract_heart_metrics(self, normalizer, sample_xml_path):
        """Test heart rate metrics extraction."""
        result = await normalizer.extract(sample_xml_path)

        daily_metrics = result.data["daily_metrics"]
        first_day = next((m for m in daily_metrics if m["date"] == "2024-01-15"), None)

        assert first_day is not None
        assert first_day["resting_hr"] == 62
        assert first_day["hrv"] == 45.5

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, normalizer, tmp_path):
        """Test handling of unsupported file type."""
        bad_file = tmp_path / "data.txt"
        bad_file.write_text("some data")

        result = await normalizer.extract(str(bad_file))

        assert not result.success
        assert "Unsupported file type" in result.errors[0]

    def test_metric_types(self, normalizer):
        """Test listing available metric types."""
        types = normalizer.get_metric_types()

        assert "steps" in types
        assert "resting_hr" in types
        assert "hrv" in types


class TestDailyHealthMetrics:
    """Tests for DailyHealthMetrics model."""

    def test_create_minimal_metrics(self):
        """Test creating metrics with minimal fields."""
        from datetime import date

        metrics = DailyHealthMetrics(
            date=date(2024, 1, 15),
            source="garmin",
        )

        assert metrics.date == date(2024, 1, 15)
        assert metrics.source == "garmin"
        assert metrics.steps is None
        assert metrics.calories_burned is None

    def test_create_full_metrics(self):
        """Test creating metrics with all fields."""
        from datetime import date

        metrics = DailyHealthMetrics(
            date=date(2024, 1, 15),
            source="whoop",
            steps=10000,
            calories_burned=2500,
            resting_hr=58,
            avg_hr=72,
            hrv=55.0,
            recovery_score=85.0,
            strain=14.5,
            sleep_duration_hours=7.8,
        )

        assert metrics.steps == 10000
        assert metrics.recovery_score == 85.0
        assert metrics.strain == 14.5
