"""Tests for engagement score calculation."""

import pytest

from app.services.analytics.engagement import (
    calculate_engagement_score,
    calculate_response_elaboration_score,
    count_words_in_utterances,
)


class TestEngagementScore:
    """Tests for engagement score calculation."""

    def test_perfect_engagement(self):
        """Test calculation with ideal engagement metrics."""
        score = calculate_engagement_score(
            client_talk_percentage=50.0,  # Balanced talk ratio
            duration_minutes=60.0,  # Full hour session
            client_turns=30,  # Good turn count
            total_client_words=500,  # Substantial responses
            sentiment_score=0.8,  # Positive sentiment
            commitment_cue_count=5,  # Shows commitment
            resistance_cue_count=0,  # No resistance
        )

        # Perfect engagement should be high
        assert score >= 80
        assert score <= 100

    def test_low_engagement(self):
        """Test calculation with low engagement metrics."""
        score = calculate_engagement_score(
            client_talk_percentage=10.0,  # Very low participation
            duration_minutes=5.0,  # Very short session
            client_turns=2,  # Few turns
            total_client_words=10,  # Minimal responses
            sentiment_score=-0.5,  # Negative sentiment
            commitment_cue_count=0,  # No commitment
            resistance_cue_count=5,  # High resistance
        )

        # Low engagement should be low score
        assert score < 40
        assert score >= 0

    def test_moderate_engagement(self):
        """Test calculation with moderate engagement."""
        score = calculate_engagement_score(
            client_talk_percentage=30.0,
            duration_minutes=30.0,
            client_turns=15,
            total_client_words=200,
            sentiment_score=0.0,  # Neutral
            commitment_cue_count=1,
            resistance_cue_count=1,
        )

        # Should be in middle range
        assert score >= 30
        assert score <= 70

    def test_score_bounded_0_to_100(self):
        """Test that score is always bounded between 0 and 100."""
        # Extreme positive
        score_high = calculate_engagement_score(
            client_talk_percentage=100.0,
            duration_minutes=120.0,
            client_turns=100,
            total_client_words=5000,
            sentiment_score=1.0,
            commitment_cue_count=20,
            resistance_cue_count=0,
        )

        assert score_high <= 100

        # Extreme negative
        score_low = calculate_engagement_score(
            client_talk_percentage=0.0,
            duration_minutes=0.0,
            client_turns=0,
            total_client_words=0,
            sentiment_score=-1.0,
            commitment_cue_count=0,
            resistance_cue_count=20,
        )

        assert score_low >= 0

    def test_resistance_lowers_score(self):
        """Test that resistance cues lower the score."""
        base_score = calculate_engagement_score(
            client_talk_percentage=40.0,
            duration_minutes=45.0,
            client_turns=20,
            total_client_words=300,
            sentiment_score=0.3,
            commitment_cue_count=2,
            resistance_cue_count=0,
        )

        score_with_resistance = calculate_engagement_score(
            client_talk_percentage=40.0,
            duration_minutes=45.0,
            client_turns=20,
            total_client_words=300,
            sentiment_score=0.3,
            commitment_cue_count=2,
            resistance_cue_count=5,  # Added resistance
        )

        assert score_with_resistance < base_score

    def test_commitment_raises_score(self):
        """Test that commitment cues raise the score."""
        base_score = calculate_engagement_score(
            client_talk_percentage=40.0,
            duration_minutes=45.0,
            client_turns=20,
            total_client_words=300,
            sentiment_score=0.3,
            commitment_cue_count=0,  # No commitment
            resistance_cue_count=1,
        )

        score_with_commitment = calculate_engagement_score(
            client_talk_percentage=40.0,
            duration_minutes=45.0,
            client_turns=20,
            total_client_words=300,
            sentiment_score=0.3,
            commitment_cue_count=5,  # Added commitment
            resistance_cue_count=1,
        )

        assert score_with_commitment > base_score


class TestResponseElaborationScore:
    """Tests for response elaboration score calculation."""

    def test_empty_utterances(self):
        """Test with empty utterance list."""
        score = calculate_response_elaboration_score([])
        assert score == 0

    def test_short_responses(self):
        """Test with very short responses."""
        utterances = [
            {"text": "Yes"},
            {"text": "No"},
            {"text": "Okay"},
        ]

        score = calculate_response_elaboration_score(utterances)

        # Average of ~1 word = low score
        assert score < 20

    def test_detailed_responses(self):
        """Test with detailed, elaborate responses."""
        utterances = [
            {"text": "I really think this approach makes a lot of sense because it addresses the core issues we discussed last time."},
            {"text": "The main challenges I'm facing are time management and maintaining motivation when things get difficult."},
        ]

        score = calculate_response_elaboration_score(utterances)

        # Average ~15-20 words = moderate-high score
        assert score >= 40

    def test_very_long_responses(self):
        """Test that score is capped at 100."""
        utterances = [
            {"text": " ".join(["word"] * 100)},  # 100 words
        ]

        score = calculate_response_elaboration_score(utterances)

        # Should cap at 100
        assert score == 100

    def test_mixed_response_lengths(self):
        """Test with mixed length responses."""
        utterances = [
            {"text": "Yes"},  # 1 word
            {"text": "I've been thinking about what we discussed and I have some ideas."},  # ~12 words
            {"text": "Okay"},  # 1 word
        ]

        score = calculate_response_elaboration_score(utterances)

        # Average ~5 words
        assert score >= 10
        assert score <= 30


class TestCountWords:
    """Tests for word counting in utterances."""

    def test_empty_utterances(self):
        """Test with empty list."""
        count = count_words_in_utterances([])
        assert count == 0

    def test_count_words(self):
        """Test basic word counting."""
        utterances = [
            {"text": "Hello world"},  # 2 words
            {"text": "One two three"},  # 3 words
        ]

        count = count_words_in_utterances(utterances)
        assert count == 5

    def test_handles_empty_text(self):
        """Test handling of empty text."""
        utterances = [
            {"text": ""},
            {"text": "Some words"},
        ]

        count = count_words_in_utterances(utterances)
        assert count == 2

    def test_handles_missing_text(self):
        """Test handling of missing text field."""
        utterances = [
            {},
            {"text": "Has text"},
        ]

        count = count_words_in_utterances(utterances)
        assert count == 2
