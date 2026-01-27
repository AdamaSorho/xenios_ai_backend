"""Speaker diarization service for identifying coach vs client."""

from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SpeakerStats:
    """Statistics for a speaker."""

    speaker_number: int
    utterance_count: int
    total_words: int
    question_count: int
    coaching_term_count: int
    directive_count: int
    is_first_speaker: bool


class DiarizationService:
    """
    Assign speaker labels based on conversation patterns.

    Uses heuristics to identify which speaker is the coach vs client.
    """

    # Coaching-specific vocabulary
    COACHING_TERMS = [
        "goal", "goals", "progress", "plan", "tracking", "consistency",
        "nutrition", "training", "recovery", "sleep", "stress",
        "habit", "habits", "routine", "accountability", "check-in", "review",
        "how", "what", "why", "tell me", "describe", "explain",
        "let's", "we", "together", "support", "help", "focus",
    ]

    # Directive phrases (coach giving instructions)
    DIRECTIVE_STARTS = [
        "try", "make sure", "remember", "focus", "let's",
        "you should", "i want you", "consider", "think about",
        "start", "keep", "continue", "don't forget", "be sure",
    ]

    # Confidence threshold for coach/client labeling
    SPEAKER_CONFIDENCE_THRESHOLD = 0.7

    def assign_speaker_labels(
        self,
        utterances: list[dict],
    ) -> list[dict]:
        """
        Assign coach/client labels to speakers.

        Heuristics:
        1. Question ratio - coaches ask more questions
        2. Coaching terms - coaches use specific vocabulary
        3. First speaker - often the coach
        4. Directive language - coaches give instructions

        Args:
            utterances: List of utterance dicts with speaker_number and text

        Returns:
            Utterances with speaker_label and speaker_confidence added
        """
        if not utterances:
            return utterances

        # Count unique speakers
        speakers = set(u["speaker_number"] for u in utterances)
        speaker_count = len(speakers)

        if speaker_count == 1:
            # Monologue - label as "speaker"
            return self._handle_single_speaker(utterances)

        if speaker_count > 2:
            # Multi-party - identify coach, label others as participants
            return self._handle_multi_speaker(utterances, speakers)

        # Two speakers - identify coach vs client
        return self._handle_two_speakers(utterances, speakers)

    def _handle_single_speaker(self, utterances: list[dict]) -> list[dict]:
        """Handle single-speaker audio (monologue)."""
        for u in utterances:
            u["speaker_label"] = "speaker"
            u["speaker_confidence"] = 1.0
        return utterances

    def _handle_two_speakers(
        self,
        utterances: list[dict],
        speakers: set[int],
    ) -> list[dict]:
        """Handle two-speaker conversation."""
        speaker_scores = self._calculate_speaker_scores(utterances, speakers)

        # Highest score is likely coach
        coach_speaker = max(speaker_scores, key=speaker_scores.get)
        coach_confidence = speaker_scores[coach_speaker]

        # Normalize confidence to 0-1 range
        total_score = sum(speaker_scores.values())
        if total_score > 0:
            normalized_confidence = coach_confidence / total_score
        else:
            normalized_confidence = 0.5

        logger.info(
            "Speaker identification completed",
            coach_speaker=coach_speaker,
            confidence=normalized_confidence,
            scores=speaker_scores,
        )

        for u in utterances:
            if normalized_confidence >= self.SPEAKER_CONFIDENCE_THRESHOLD:
                u["speaker_label"] = "coach" if u["speaker_number"] == coach_speaker else "client"
                u["speaker_confidence"] = normalized_confidence
            else:
                # Low confidence - use generic labels
                u["speaker_label"] = f"speaker_{u['speaker_number']}"
                u["speaker_confidence"] = normalized_confidence

        return utterances

    def _handle_multi_speaker(
        self,
        utterances: list[dict],
        speakers: set[int],
    ) -> list[dict]:
        """Handle 3+ speaker conversations."""
        speaker_scores = self._calculate_speaker_scores(utterances, speakers)

        # Highest score is likely coach
        coach_speaker = max(speaker_scores, key=speaker_scores.get)
        coach_confidence = speaker_scores[coach_speaker]

        # Normalize
        total_score = sum(speaker_scores.values())
        normalized_confidence = coach_confidence / total_score if total_score > 0 else 0.5

        # Build label mapping
        speaker_labels = {}
        participant_num = 1
        for speaker in sorted(speakers):
            if speaker == coach_speaker:
                speaker_labels[speaker] = "coach"
            else:
                speaker_labels[speaker] = f"participant_{participant_num}"
                participant_num += 1

        for u in utterances:
            u["speaker_label"] = speaker_labels[u["speaker_number"]]
            u["speaker_confidence"] = normalized_confidence

        return utterances

    def _calculate_speaker_scores(
        self,
        utterances: list[dict],
        speakers: set[int],
    ) -> dict[int, float]:
        """Calculate coach likelihood score for each speaker."""
        # Gather stats for each speaker
        speaker_stats: dict[int, SpeakerStats] = {}

        first_speaker = utterances[0]["speaker_number"] if utterances else None

        for speaker in speakers:
            speaker_utterances = [u for u in utterances if u["speaker_number"] == speaker]
            total_text = " ".join(u["text"] for u in speaker_utterances)
            total_words = len(total_text.split())

            speaker_stats[speaker] = SpeakerStats(
                speaker_number=speaker,
                utterance_count=len(speaker_utterances),
                total_words=total_words,
                question_count=self._count_questions(speaker_utterances),
                coaching_term_count=self._count_coaching_terms(total_text),
                directive_count=self._count_directives(speaker_utterances),
                is_first_speaker=speaker == first_speaker,
            )

        # Calculate scores
        scores = {}
        for speaker, stats in speaker_stats.items():
            # Avoid division by zero
            utt_count = max(stats.utterance_count, 1)
            word_count = max(stats.total_words, 1)

            # Calculate ratios
            question_ratio = stats.question_count / utt_count
            coaching_term_ratio = stats.coaching_term_count / word_count * 100  # Scale up
            directive_ratio = stats.directive_count / utt_count
            first_speaker_bonus = 1.0 if stats.is_first_speaker else 0.0

            # Weighted score (weights sum to 1.0)
            score = (
                question_ratio * 0.30 +
                coaching_term_ratio * 0.30 +
                first_speaker_bonus * 0.15 +
                directive_ratio * 0.25
            )

            scores[speaker] = score

        return scores

    def _count_questions(self, utterances: list[dict]) -> int:
        """Count utterances that are questions."""
        return sum(1 for u in utterances if "?" in u.get("text", ""))

    def _count_coaching_terms(self, text: str) -> int:
        """Count coaching-specific terms in text."""
        text_lower = text.lower()
        return sum(1 for term in self.COACHING_TERMS if term in text_lower)

    def _count_directives(self, utterances: list[dict]) -> int:
        """Count directive statements."""
        count = 0
        for u in utterances:
            text_lower = u.get("text", "").lower().strip()
            if any(text_lower.startswith(d) for d in self.DIRECTIVE_STARTS):
                count += 1
        return count


# Global service instance
_diarization_service: DiarizationService | None = None


def get_diarization_service() -> DiarizationService:
    """Get the global diarization service instance."""
    global _diarization_service
    if _diarization_service is None:
        _diarization_service = DiarizationService()
    return _diarization_service
