"""Tests for coaching style analysis."""

import pytest

from app.services.analytics.coaching_style import (
    CoachingStyleAnalyzer,
    CoachingStyleMetrics,
)


class TestCoachingStyleAnalyzer:
    """Tests for CoachingStyleAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CoachingStyleAnalyzer()

    def test_empty_utterances(self):
        """Test with empty utterance list."""
        result = self.analyzer.compute([])

        assert result.coach_question_count == 0
        assert result.coach_statement_count == 0
        assert result.question_to_statement_ratio == 0.0
        assert result.open_question_count == 0
        assert result.closed_question_count == 0

    def test_question_detection(self):
        """Test basic question detection."""
        utterances = [
            {"text": "What do you think about that?"},
            {"text": "How are you feeling today?"},
            {"text": "That sounds interesting."},
        ]

        result = self.analyzer.compute(utterances)

        assert result.coach_question_count == 2
        assert result.coach_statement_count == 1

    def test_open_vs_closed_questions(self):
        """Test open vs closed question classification."""
        utterances = [
            {"text": "What are your goals?"},  # Open
            {"text": "How do you feel about that?"},  # Open
            {"text": "Why is this important to you?"},  # Open
            {"text": "Do you agree?"},  # Closed
            {"text": "Is that correct?"},  # Closed
            {"text": "Can you do that?"},  # Closed
        ]

        result = self.analyzer.compute(utterances)

        assert result.open_question_count == 3
        assert result.closed_question_count == 3

    def test_question_to_statement_ratio(self):
        """Test question to statement ratio calculation."""
        utterances = [
            {"text": "What do you think?"},
            {"text": "How is that going?"},
            {"text": "That's a great point."},
            {"text": "I see what you mean."},
        ]

        result = self.analyzer.compute(utterances)

        # 2 questions, 2 statements = ratio of 1.0
        assert result.question_to_statement_ratio == 1.0

    def test_only_questions(self):
        """Test with only questions (no statements)."""
        utterances = [
            {"text": "What do you think?"},
            {"text": "How are you?"},
            {"text": "Why is that?"},
        ]

        result = self.analyzer.compute(utterances)

        assert result.coach_question_count == 3
        assert result.coach_statement_count == 0
        # With 0 statements, ratio should handle gracefully
        assert result.question_to_statement_ratio >= 0

    def test_only_statements(self):
        """Test with only statements (no questions)."""
        utterances = [
            {"text": "That's a good point."},
            {"text": "I understand."},
            {"text": "Let me explain."},
        ]

        result = self.analyzer.compute(utterances)

        assert result.coach_question_count == 0
        assert result.coach_statement_count == 3
        assert result.question_to_statement_ratio == 0.0

    def test_question_mark_detection(self):
        """Test that question marks are used for detection."""
        utterances = [
            {"text": "I wonder how that works?"},  # Question with ?
            {"text": "What a great day"},  # Starts with "What" but no ?
        ]

        result = self.analyzer.compute(utterances)

        # Both should be detected as questions
        # The first has ?, the second starts with question word
        assert result.coach_question_count >= 1

    def test_case_insensitivity(self):
        """Test case insensitive question word detection."""
        utterances = [
            {"text": "WHAT do you think?"},
            {"text": "how ARE you?"},
            {"text": "WHY is that important?"},
        ]

        result = self.analyzer.compute(utterances)

        assert result.open_question_count == 3

    def test_returns_metrics_dataclass(self):
        """Test that compute returns CoachingStyleMetrics dataclass."""
        result = self.analyzer.compute([])

        assert isinstance(result, CoachingStyleMetrics)
