"""Coaching style analysis for coach utterances."""

from dataclasses import dataclass
from typing import Any

# Word sets for classification
QUESTION_STARTERS = {"what", "how", "why", "where", "when", "who", "which"}
OPEN_STARTERS = {"what", "how", "why", "tell"}
CLOSED_STARTERS = {"do", "did", "is", "are", "can", "could", "will", "would", "have", "has"}


@dataclass
class CoachingStyleMetrics:
    """Metrics computed from coaching style analysis."""

    coach_question_count: int
    coach_statement_count: int
    question_to_statement_ratio: float
    open_question_count: int
    closed_question_count: int


class CoachingStyleAnalyzer:
    """
    Analyze coaching style from coach utterances.

    Uses pattern-based detection (no LLM required) to identify:
    - Questions vs statements
    - Open vs closed questions
    """

    def compute(self, coach_utterances: list[dict[str, Any]]) -> CoachingStyleMetrics:
        """
        Compute coaching style metrics from coach utterances.

        Args:
            coach_utterances: List of utterance dicts with:
                - text: str

        Returns:
            CoachingStyleMetrics with all computed values

        Heuristics:
        - Question: Ends with '?' OR starts with question word
        - Statement: Everything else
        - Open question: Starts with "what", "how", "why", "tell me"
        - Closed question: Starts with "do", "did", "is", "are", "can", "will"
        """
        question_count = 0
        statement_count = 0
        open_question_count = 0
        closed_question_count = 0

        for utterance in coach_utterances:
            text = utterance.get("text", "").strip().lower()
            if not text:
                continue

            words = text.split()
            first_word = words[0] if words else ""

            is_question = text.endswith("?") or first_word in QUESTION_STARTERS

            if is_question:
                question_count += 1

                # Classify question type
                if first_word in OPEN_STARTERS or text.startswith("tell me"):
                    open_question_count += 1
                elif first_word in CLOSED_STARTERS:
                    closed_question_count += 1
                # Note: Some questions don't fit either category
            else:
                statement_count += 1

        # Calculate ratio (avoid division by zero)
        if statement_count > 0:
            ratio = question_count / statement_count
        else:
            ratio = float(question_count) if question_count > 0 else 0.0

        return CoachingStyleMetrics(
            coach_question_count=question_count,
            coach_statement_count=statement_count,
            question_to_statement_ratio=round(ratio, 3),
            open_question_count=open_question_count,
            closed_question_count=closed_question_count,
        )
