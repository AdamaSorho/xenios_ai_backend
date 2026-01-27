"""Tests for sentiment analysis utilities."""

import pytest

from app.services.analytics.sentiment import (
    sentiment_label_to_score,
    aggregate_sentiment,
    calculate_sentiment_variance,
    extract_sentiment_scores,
)


class TestSentimentLabelToScore:
    """Tests for sentiment label to score conversion."""

    def test_positive_labels(self):
        """Test positive sentiment labels."""
        assert sentiment_label_to_score("positive") == pytest.approx(0.5, abs=0.3)
        assert sentiment_label_to_score("very_positive") == pytest.approx(1.0, abs=0.2)

    def test_negative_labels(self):
        """Test negative sentiment labels."""
        assert sentiment_label_to_score("negative") == pytest.approx(-0.5, abs=0.3)
        assert sentiment_label_to_score("very_negative") == pytest.approx(-1.0, abs=0.2)

    def test_neutral_label(self):
        """Test neutral sentiment label."""
        assert sentiment_label_to_score("neutral") == pytest.approx(0.0, abs=0.1)

    def test_unknown_label(self):
        """Test unknown sentiment label returns neutral."""
        assert sentiment_label_to_score("unknown") == 0.0
        assert sentiment_label_to_score("") == 0.0
        assert sentiment_label_to_score(None) == 0.0

    def test_case_insensitivity(self):
        """Test case insensitive label handling."""
        assert sentiment_label_to_score("POSITIVE") == sentiment_label_to_score("positive")
        assert sentiment_label_to_score("Negative") == sentiment_label_to_score("negative")


class TestAggregateSentiment:
    """Tests for sentiment aggregation."""

    def test_empty_utterances(self):
        """Test with empty utterance list."""
        result = aggregate_sentiment([], speaker="client")
        assert result == 0.0

    def test_client_sentiment_only(self):
        """Test aggregation for client utterances only."""
        utterances = [
            {"speaker_label": "coach", "sentiment": "positive"},
            {"speaker_label": "client", "sentiment": "positive"},
            {"speaker_label": "client", "sentiment": "positive"},
            {"speaker_label": "coach", "sentiment": "negative"},
        ]

        result = aggregate_sentiment(utterances, speaker="client")

        # Should only consider client utterances (both positive)
        assert result > 0

    def test_coach_sentiment_only(self):
        """Test aggregation for coach utterances only."""
        utterances = [
            {"speaker_label": "coach", "sentiment": "negative"},
            {"speaker_label": "client", "sentiment": "positive"},
        ]

        result = aggregate_sentiment(utterances, speaker="coach")

        # Should only consider coach (negative)
        assert result < 0

    def test_mixed_sentiment(self):
        """Test with mixed positive and negative sentiment."""
        utterances = [
            {"speaker_label": "client", "sentiment": "positive"},
            {"speaker_label": "client", "sentiment": "negative"},
        ]

        result = aggregate_sentiment(utterances, speaker="client")

        # Should average out to roughly neutral
        assert result == pytest.approx(0.0, abs=0.2)

    def test_missing_sentiment(self):
        """Test handling of missing sentiment values."""
        utterances = [
            {"speaker_label": "client", "sentiment": "positive"},
            {"speaker_label": "client"},  # No sentiment
            {"speaker_label": "client", "sentiment": None},
        ]

        result = aggregate_sentiment(utterances, speaker="client")

        # Should handle missing values gracefully
        assert result > 0


class TestCalculateSentimentVariance:
    """Tests for sentiment variance calculation."""

    def test_empty_scores(self):
        """Test with empty scores list."""
        variance = calculate_sentiment_variance([])
        assert variance == 0.0

    def test_single_score(self):
        """Test with single score."""
        variance = calculate_sentiment_variance([0.5])
        assert variance == 0.0

    def test_uniform_scores(self):
        """Test with all identical scores."""
        variance = calculate_sentiment_variance([0.5, 0.5, 0.5, 0.5])
        assert variance == 0.0

    def test_high_variance(self):
        """Test with high variance (mixed extreme scores)."""
        variance = calculate_sentiment_variance([1.0, -1.0, 1.0, -1.0])

        # This should have high variance
        assert variance > 0.5

    def test_low_variance(self):
        """Test with low variance (similar scores)."""
        variance = calculate_sentiment_variance([0.4, 0.5, 0.6, 0.5])

        # This should have low variance
        assert variance < 0.2


class TestExtractSentimentScores:
    """Tests for extracting sentiment scores from utterances."""

    def test_empty_utterances(self):
        """Test with empty utterance list."""
        scores = extract_sentiment_scores([], speaker="client")
        assert scores == []

    def test_extract_client_scores(self):
        """Test extracting scores for client only."""
        utterances = [
            {"speaker_label": "coach", "sentiment": "positive"},
            {"speaker_label": "client", "sentiment": "positive"},
            {"speaker_label": "client", "sentiment": "negative"},
        ]

        scores = extract_sentiment_scores(utterances, speaker="client")

        # Should have 2 scores for client
        assert len(scores) == 2

    def test_skip_missing_sentiment(self):
        """Test that utterances without sentiment are skipped."""
        utterances = [
            {"speaker_label": "client", "sentiment": "positive"},
            {"speaker_label": "client"},  # No sentiment
            {"speaker_label": "client", "sentiment": "negative"},
        ]

        scores = extract_sentiment_scores(utterances, speaker="client")

        # Should have 2 scores (skipping the one without sentiment)
        assert len(scores) == 2
