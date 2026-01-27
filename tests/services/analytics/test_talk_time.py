"""Tests for talk-time analysis."""

import pytest

from app.services.analytics.talk_time import TalkTimeAnalyzer, TalkTimeMetrics


class TestTalkTimeAnalyzer:
    """Tests for TalkTimeAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TalkTimeAnalyzer()

    def test_compute_with_empty_utterances(self):
        """Test computation with empty utterance list."""
        result = self.analyzer.compute([])

        assert result.total_duration_seconds == 0
        assert result.coach_talk_time_seconds == 0
        assert result.client_talk_time_seconds == 0
        assert result.coach_talk_percentage == 0
        assert result.client_talk_percentage == 0
        assert result.total_turns == 0

    def test_compute_with_coach_only(self):
        """Test computation with only coach utterances."""
        utterances = [
            {"speaker_label": "coach", "start_time": 0.0, "end_time": 10.0},
            {"speaker_label": "coach", "start_time": 12.0, "end_time": 20.0},
        ]

        result = self.analyzer.compute(utterances)

        assert result.coach_talk_time_seconds == 18.0
        assert result.client_talk_time_seconds == 0.0
        assert result.coach_talk_percentage == 100.0
        assert result.client_talk_percentage == 0.0
        assert result.coach_turns == 2
        assert result.client_turns == 0

    def test_compute_with_client_only(self):
        """Test computation with only client utterances."""
        utterances = [
            {"speaker_label": "client", "start_time": 0.0, "end_time": 15.0},
        ]

        result = self.analyzer.compute(utterances)

        assert result.coach_talk_time_seconds == 0.0
        assert result.client_talk_time_seconds == 15.0
        assert result.coach_talk_percentage == 0.0
        assert result.client_talk_percentage == 100.0

    def test_compute_balanced_conversation(self):
        """Test computation with balanced coach/client conversation."""
        utterances = [
            {"speaker_label": "coach", "start_time": 0.0, "end_time": 10.0},
            {"speaker_label": "client", "start_time": 11.0, "end_time": 21.0},
            {"speaker_label": "coach", "start_time": 22.0, "end_time": 32.0},
            {"speaker_label": "client", "start_time": 33.0, "end_time": 43.0},
        ]

        result = self.analyzer.compute(utterances)

        assert result.coach_talk_time_seconds == 20.0
        assert result.client_talk_time_seconds == 20.0
        assert result.coach_talk_percentage == 50.0
        assert result.client_talk_percentage == 50.0
        assert result.coach_turns == 2
        assert result.client_turns == 2
        assert result.total_turns == 4

    def test_compute_average_turn_duration(self):
        """Test average turn duration calculation."""
        utterances = [
            {"speaker_label": "coach", "start_time": 0.0, "end_time": 10.0},
            {"speaker_label": "coach", "start_time": 12.0, "end_time": 22.0},
            {"speaker_label": "client", "start_time": 24.0, "end_time": 30.0},
        ]

        result = self.analyzer.compute(utterances)

        assert result.average_turn_duration_coach == 10.0  # (10 + 10) / 2
        assert result.average_turn_duration_client == 6.0  # 6 / 1

    def test_compute_silence_time(self):
        """Test silence time calculation."""
        utterances = [
            {"speaker_label": "coach", "start_time": 0.0, "end_time": 10.0},
            {"speaker_label": "client", "start_time": 15.0, "end_time": 20.0},  # 5s gap
        ]

        result = self.analyzer.compute(utterances)

        # Total duration is 20s, speech is 15s, so silence is 5s
        assert result.total_duration_seconds == 20.0
        assert result.silence_time_seconds == 5.0

    def test_compute_interruption_detection(self):
        """Test interruption detection with overlapping speech."""
        utterances = [
            {"speaker_label": "coach", "start_time": 0.0, "end_time": 15.0},
            {"speaker_label": "client", "start_time": 14.0, "end_time": 20.0},  # Overlaps
            {"speaker_label": "coach", "start_time": 19.0, "end_time": 25.0},  # Overlaps
        ]

        result = self.analyzer.compute(utterances)

        # Client started at 14.0 while coach was speaking until 15.0 - interruption
        # Coach started at 19.0 while client was speaking until 20.0 - interruption
        assert result.interruption_count == 2

    def test_compute_no_interruptions(self):
        """Test no interruptions with clean turn-taking."""
        utterances = [
            {"speaker_label": "coach", "start_time": 0.0, "end_time": 10.0},
            {"speaker_label": "client", "start_time": 11.0, "end_time": 20.0},
        ]

        result = self.analyzer.compute(utterances)

        assert result.interruption_count == 0

    def test_compute_case_insensitive_labels(self):
        """Test case-insensitive speaker label handling."""
        utterances = [
            {"speaker_label": "COACH", "start_time": 0.0, "end_time": 10.0},
            {"speaker_label": "Client", "start_time": 12.0, "end_time": 20.0},
        ]

        result = self.analyzer.compute(utterances)

        assert result.coach_turns == 1
        assert result.client_turns == 1

    def test_compute_unknown_speaker_ignored(self):
        """Test that unknown speaker labels are handled."""
        utterances = [
            {"speaker_label": "coach", "start_time": 0.0, "end_time": 10.0},
            {"speaker_label": "unknown", "start_time": 12.0, "end_time": 15.0},
            {"speaker_label": "client", "start_time": 16.0, "end_time": 20.0},
        ]

        result = self.analyzer.compute(utterances)

        assert result.coach_turns == 1
        assert result.client_turns == 1
        assert result.total_turns == 3  # All turns counted

    def test_compute_returns_metrics_dataclass(self):
        """Test that compute returns a TalkTimeMetrics dataclass."""
        utterances = [
            {"speaker_label": "coach", "start_time": 0.0, "end_time": 10.0},
        ]

        result = self.analyzer.compute(utterances)

        assert isinstance(result, TalkTimeMetrics)
