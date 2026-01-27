"""Tests for trend calculation utilities."""

import pytest
from datetime import date, timedelta

from app.services.analytics.trends import (
    calculate_trend,
    calculate_talk_ratio_trend,
    calculate_session_frequency_trend,
)


class TestCalculateTrend:
    """Tests for generic trend calculation."""

    def test_empty_values(self):
        """Test with empty values list."""
        trend = calculate_trend([], higher_is_better=True)
        assert trend == "stable"

    def test_single_value(self):
        """Test with single value - insufficient data."""
        trend = calculate_trend([50.0], higher_is_better=True)
        assert trend == "stable"

    def test_improving_trend_higher_is_better(self):
        """Test improving trend when higher values are better."""
        values = [50.0, 55.0, 60.0, 65.0]  # Increasing
        trend = calculate_trend(values, higher_is_better=True)
        assert trend == "improving"

    def test_declining_trend_higher_is_better(self):
        """Test declining trend when higher values are better."""
        values = [70.0, 65.0, 60.0, 55.0]  # Decreasing
        trend = calculate_trend(values, higher_is_better=True)
        assert trend == "declining"

    def test_improving_trend_lower_is_better(self):
        """Test improving trend when lower values are better."""
        values = [70.0, 65.0, 60.0, 55.0]  # Decreasing = improving
        trend = calculate_trend(values, higher_is_better=False)
        assert trend == "improving"

    def test_declining_trend_lower_is_better(self):
        """Test declining trend when lower values are better."""
        values = [50.0, 55.0, 60.0, 65.0]  # Increasing = declining
        trend = calculate_trend(values, higher_is_better=False)
        assert trend == "declining"

    def test_stable_trend(self):
        """Test stable trend with minimal change."""
        values = [50.0, 51.0, 49.0, 50.0]  # Small fluctuations
        trend = calculate_trend(values, higher_is_better=True)
        assert trend == "stable"

    def test_two_values_comparison(self):
        """Test with exactly two values."""
        # Improving
        assert calculate_trend([40.0, 60.0], higher_is_better=True) == "improving"
        # Declining
        assert calculate_trend([60.0, 40.0], higher_is_better=True) == "declining"


class TestCalculateTalkRatioTrend:
    """Tests for talk ratio trend calculation."""

    def test_empty_values(self):
        """Test with empty values."""
        trend = calculate_talk_ratio_trend([])
        assert trend == "balanced"

    def test_single_value(self):
        """Test with single value."""
        trend = calculate_talk_ratio_trend([40.0])
        assert trend == "balanced"

    def test_increasing_client_talk(self):
        """Test when client is talking more over time."""
        values = [30.0, 35.0, 40.0, 45.0]  # Client talk increasing
        trend = calculate_talk_ratio_trend(values)
        assert trend == "client_increasing"

    def test_decreasing_client_talk(self):
        """Test when client is talking less over time."""
        values = [50.0, 45.0, 40.0, 35.0]  # Client talk decreasing
        trend = calculate_talk_ratio_trend(values)
        assert trend == "client_decreasing"

    def test_balanced(self):
        """Test balanced/stable talk ratio."""
        values = [40.0, 41.0, 39.0, 40.0]  # Small fluctuations
        trend = calculate_talk_ratio_trend(values)
        assert trend == "balanced"


class TestCalculateSessionFrequencyTrend:
    """Tests for session frequency trend calculation."""

    def test_empty_dates(self):
        """Test with empty dates list."""
        trend = calculate_session_frequency_trend([])
        assert trend == "stable"

    def test_single_session(self):
        """Test with single session."""
        trend = calculate_session_frequency_trend([date.today()])
        assert trend == "stable"

    def test_two_sessions(self):
        """Test with two sessions."""
        dates = [date.today() - timedelta(days=14), date.today()]
        trend = calculate_session_frequency_trend(dates)
        assert trend == "stable"  # Need more data

    def test_increasing_frequency(self):
        """Test increasing session frequency."""
        # Sessions getting more frequent (gaps shrinking)
        today = date.today()
        dates = [
            today - timedelta(days=60),  # Long gap
            today - timedelta(days=30),  # 30 day gap
            today - timedelta(days=15),  # 15 day gap
            today - timedelta(days=7),   # 8 day gap
            today,                        # 7 day gap
        ]
        trend = calculate_session_frequency_trend(dates)
        assert trend == "increasing"

    def test_decreasing_frequency(self):
        """Test decreasing session frequency."""
        # Sessions getting less frequent (gaps growing)
        today = date.today()
        dates = [
            today - timedelta(days=90),  # 7 day gap
            today - timedelta(days=83),  # 13 day gap
            today - timedelta(days=70),  # 20 day gap
            today - timedelta(days=50),  # 30 day gap
            today - timedelta(days=20),
        ]
        trend = calculate_session_frequency_trend(dates)
        assert trend == "decreasing"

    def test_stable_frequency(self):
        """Test stable session frequency."""
        # Regular weekly sessions
        today = date.today()
        dates = [
            today - timedelta(days=28),
            today - timedelta(days=21),
            today - timedelta(days=14),
            today - timedelta(days=7),
            today,
        ]
        trend = calculate_session_frequency_trend(dates)
        assert trend == "stable"
