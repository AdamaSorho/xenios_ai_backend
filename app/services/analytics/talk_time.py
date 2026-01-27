"""Talk-time analysis for coaching sessions."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TalkTimeMetrics:
    """Metrics computed from talk-time analysis."""

    total_duration_seconds: float
    coach_talk_time_seconds: float
    client_talk_time_seconds: float
    silence_time_seconds: float
    coach_talk_percentage: float
    client_talk_percentage: float
    total_turns: int
    coach_turns: int
    client_turns: int
    average_turn_duration_coach: float
    average_turn_duration_client: float
    interruption_count: int


class TalkTimeAnalyzer:
    """
    Compute talk-time metrics from session utterances.

    Analyzes:
    - Speaking time per speaker
    - Turn counts and durations
    - Interruption detection
    """

    def compute(self, utterances: list[dict[str, Any]]) -> TalkTimeMetrics:
        """
        Compute talk-time metrics from utterances.

        Args:
            utterances: List of utterance dicts with:
                - speaker_label: "coach" or "client"
                - start_time: float (seconds)
                - end_time: float (seconds)
                - text: str

        Returns:
            TalkTimeMetrics with all computed values
        """
        if not utterances:
            return TalkTimeMetrics(
                total_duration_seconds=0.0,
                coach_talk_time_seconds=0.0,
                client_talk_time_seconds=0.0,
                silence_time_seconds=0.0,
                coach_talk_percentage=0.0,
                client_talk_percentage=0.0,
                total_turns=0,
                coach_turns=0,
                client_turns=0,
                average_turn_duration_coach=0.0,
                average_turn_duration_client=0.0,
                interruption_count=0,
            )

        # Calculate total duration from first to last utterance
        sorted_utterances = sorted(utterances, key=lambda u: u.get("start_time", 0))
        total_duration = sorted_utterances[-1].get("end_time", 0) if sorted_utterances else 0

        # Separate by speaker
        coach_utterances = [
            u for u in sorted_utterances
            if u.get("speaker_label", "").lower() == "coach"
        ]
        client_utterances = [
            u for u in sorted_utterances
            if u.get("speaker_label", "").lower() == "client"
        ]

        # Calculate talk times
        coach_time = sum(
            u.get("end_time", 0) - u.get("start_time", 0)
            for u in coach_utterances
        )
        client_time = sum(
            u.get("end_time", 0) - u.get("start_time", 0)
            for u in client_utterances
        )

        # Silence is the remainder (but can't be negative due to overlaps)
        silence_time = max(0, total_duration - coach_time - client_time)

        # Calculate percentages (avoid division by zero)
        if total_duration > 0:
            coach_percentage = (coach_time / total_duration) * 100
            client_percentage = (client_time / total_duration) * 100
        else:
            coach_percentage = 0.0
            client_percentage = 0.0

        # Turn counts
        coach_turns = len(coach_utterances)
        client_turns = len(client_utterances)
        total_turns = len(sorted_utterances)

        # Average turn durations
        avg_coach_duration = self._avg_duration(coach_utterances)
        avg_client_duration = self._avg_duration(client_utterances)

        # Count interruptions
        interruption_count = self._count_interruptions(sorted_utterances)

        return TalkTimeMetrics(
            total_duration_seconds=round(total_duration, 2),
            coach_talk_time_seconds=round(coach_time, 2),
            client_talk_time_seconds=round(client_time, 2),
            silence_time_seconds=round(silence_time, 2),
            coach_talk_percentage=round(coach_percentage, 2),
            client_talk_percentage=round(client_percentage, 2),
            total_turns=total_turns,
            coach_turns=coach_turns,
            client_turns=client_turns,
            average_turn_duration_coach=round(avg_coach_duration, 2),
            average_turn_duration_client=round(avg_client_duration, 2),
            interruption_count=interruption_count,
        )

    def _avg_duration(self, utterances: list[dict[str, Any]]) -> float:
        """Calculate average duration of utterances."""
        if not utterances:
            return 0.0

        durations = [
            u.get("end_time", 0) - u.get("start_time", 0)
            for u in utterances
        ]
        return sum(durations) / len(durations)

    def _count_interruptions(self, utterances: list[dict[str, Any]]) -> int:
        """
        Count overlapping speech (interruptions).

        An interruption is detected when:
        - Current utterance starts before previous ends
        - Speaker is different from previous
        """
        if len(utterances) < 2:
            return 0

        count = 0
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]

            prev_end = prev.get("end_time", 0)
            curr_start = curr.get("start_time", 0)

            # Check for overlap
            if curr_start < prev_end:
                # Only count if different speaker
                prev_speaker = prev.get("speaker_label", "").lower()
                curr_speaker = curr.get("speaker_label", "").lower()
                if prev_speaker != curr_speaker:
                    count += 1

        return count
