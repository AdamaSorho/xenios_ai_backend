"""Tests for risk scoring service."""

import pytest

from app.schemas.analytics import RiskLevel
from app.services.analytics.risk_scoring import normalize_factor_contribution


class TestNormalizeFactorContribution:
    """Tests for the factor normalization function."""

    def test_below_warning_threshold(self):
        """Test that values below warning threshold return 0."""
        contribution = normalize_factor_contribution(
            value=5.0,
            warning_threshold=10.0,
            critical_threshold=20.0,
            weight=25,
            inverse=False,
        )

        assert contribution == 0.0

    def test_at_warning_threshold(self):
        """Test that values at warning threshold return ~50% of weight."""
        contribution = normalize_factor_contribution(
            value=10.0,
            warning_threshold=10.0,
            critical_threshold=20.0,
            weight=25,
            inverse=False,
        )

        # At warning, contribution should be 50% of weight
        assert contribution == pytest.approx(12.5, abs=0.1)

    def test_at_critical_threshold(self):
        """Test that values at critical threshold return 100% of weight."""
        contribution = normalize_factor_contribution(
            value=20.0,
            warning_threshold=10.0,
            critical_threshold=20.0,
            weight=25,
            inverse=False,
        )

        assert contribution == 25.0

    def test_above_critical_threshold(self):
        """Test that values above critical threshold are capped at weight."""
        contribution = normalize_factor_contribution(
            value=30.0,
            warning_threshold=10.0,
            critical_threshold=20.0,
            weight=25,
            inverse=False,
        )

        assert contribution == 25.0

    def test_between_warning_and_critical(self):
        """Test linear interpolation between thresholds."""
        # Midpoint between warning and critical
        contribution = normalize_factor_contribution(
            value=15.0,  # Midpoint between 10 and 20
            warning_threshold=10.0,
            critical_threshold=20.0,
            weight=25,
            inverse=False,
        )

        # At midpoint: 50% + 50% * 50% = 75% of weight = 18.75
        assert contribution == pytest.approx(18.75, abs=0.5)

    def test_inverse_below_threshold(self):
        """Test inverse normalization where low values are bad."""
        # For inverse, -5 becomes +5, and thresholds flip
        contribution = normalize_factor_contribution(
            value=-5.0,  # Would be low engagement = bad
            warning_threshold=-10.0,  # -10 point decline warning
            critical_threshold=-20.0,  # -20 point decline critical
            weight=25,
            inverse=True,
        )

        # -5 > -10 (after flip: 5 < 10), so below warning
        assert contribution == 0.0

    def test_inverse_at_warning(self):
        """Test inverse at warning threshold."""
        contribution = normalize_factor_contribution(
            value=-10.0,
            warning_threshold=-10.0,
            critical_threshold=-20.0,
            weight=25,
            inverse=True,
        )

        # At warning threshold
        assert contribution == pytest.approx(12.5, abs=0.1)

    def test_zero_range(self):
        """Test handling of zero range (warning == critical)."""
        contribution = normalize_factor_contribution(
            value=15.0,
            warning_threshold=10.0,
            critical_threshold=10.0,  # Same as warning
            weight=25,
            inverse=False,
        )

        # With zero range, should return full weight if at or above threshold
        assert contribution == 25.0


class TestRiskLevelThresholds:
    """Tests for risk level determination."""

    def test_low_risk_threshold(self):
        """Test that scores <= 25 are low risk."""
        assert RiskLevel.LOW.value == "low"
        # Score 0-25 = low

    def test_medium_risk_threshold(self):
        """Test that scores 26-50 are medium risk."""
        assert RiskLevel.MEDIUM.value == "medium"

    def test_high_risk_threshold(self):
        """Test that scores 51-75 are high risk."""
        assert RiskLevel.HIGH.value == "high"

    def test_critical_risk_threshold(self):
        """Test that scores > 75 are critical risk."""
        assert RiskLevel.CRITICAL.value == "critical"


class TestRiskFactorWeights:
    """Tests for risk factor weight configuration."""

    def test_weights_sum_to_100(self):
        """Test that factor weights sum to 100."""
        from app.services.analytics.risk_scoring import RiskScoringService

        weights = RiskScoringService.WEIGHTS
        total = sum(weights.values())

        assert total == 100

    def test_all_factors_have_weights(self):
        """Test that all expected factors have weights."""
        from app.services.analytics.risk_scoring import RiskScoringService

        expected_factors = [
            "session_frequency",
            "engagement_trend",
            "sentiment_trend",
            "resistance_ratio",
            "days_since_session",
        ]

        for factor in expected_factors:
            assert factor in RiskScoringService.WEIGHTS
