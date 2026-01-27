"""Sentiment analysis utilities for coaching sessions."""

from typing import Any


# Mapping from sentiment labels to numeric scores
SENTIMENT_SCORES = {
    "positive": 0.5,
    "neutral": 0.0,
    "negative": -0.5,
    "mixed": 0.0,  # Treat mixed as neutral for averaging
}


def sentiment_label_to_score(label: str | None) -> float:
    """
    Convert sentiment label to numeric score.

    Args:
        label: Sentiment label (positive, neutral, negative, mixed)

    Returns:
        Score from -1 to 1 (default 0 for unknown)
    """
    if label is None:
        return 0.0
    return SENTIMENT_SCORES.get(label.lower(), 0.0)


def aggregate_sentiment(
    utterances: list[dict[str, Any]],
    speaker_label: str | None = None,
) -> float:
    """
    Calculate average sentiment score for utterances.

    Args:
        utterances: List of utterance dicts with 'sentiment' and 'speaker_label'
        speaker_label: Optional filter by speaker (e.g., "client")

    Returns:
        Average sentiment score (-1 to 1), 0 if no utterances
    """
    if speaker_label:
        filtered = [
            u for u in utterances
            if u.get("speaker_label", "").lower() == speaker_label.lower()
        ]
    else:
        filtered = utterances

    if not filtered:
        return 0.0

    scores = [
        sentiment_label_to_score(u.get("sentiment"))
        for u in filtered
    ]

    return round(sum(scores) / len(scores), 3)


def calculate_sentiment_variance(utterance_sentiments: list[float]) -> float:
    """
    Calculate variance of sentiment across utterances.

    High variance (>0.3): Client emotions fluctuated significantly
    Low variance (<0.1): Consistent emotional tone throughout

    Args:
        utterance_sentiments: List of sentiment scores (-1 to 1)

    Returns:
        Standard deviation of sentiment scores (0 to ~1)
    """
    if len(utterance_sentiments) < 2:
        return 0.0

    mean = sum(utterance_sentiments) / len(utterance_sentiments)
    variance = sum((s - mean) ** 2 for s in utterance_sentiments) / len(utterance_sentiments)

    return round(variance ** 0.5, 3)  # Standard deviation


def extract_sentiment_scores(
    utterances: list[dict[str, Any]],
    speaker_label: str | None = None,
) -> list[float]:
    """
    Extract sentiment scores from utterances.

    Args:
        utterances: List of utterance dicts with 'sentiment' field
        speaker_label: Optional filter by speaker

    Returns:
        List of sentiment scores
    """
    if speaker_label:
        filtered = [
            u for u in utterances
            if u.get("speaker_label", "").lower() == speaker_label.lower()
        ]
    else:
        filtered = utterances

    return [sentiment_label_to_score(u.get("sentiment")) for u in filtered]
