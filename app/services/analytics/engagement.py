"""Engagement score calculation for coaching sessions."""

from typing import Any


def calculate_engagement_score(
    client_talk_percentage: float,
    duration_minutes: float,
    client_turns: int,
    total_client_words: int,
    sentiment_score: float,
    commitment_cue_count: int,
    resistance_cue_count: int,
) -> float:
    """
    Calculate engagement score (0-100) from multiple factors.

    Engagement Score = weighted sum of normalized components.

    Components (weights sum to 100):
    - Participation balance (25): How close to 50/50 talk time
    - Response depth (25): Words per response vs baseline
    - Interaction density (20): Turns per minute
    - Emotional engagement (15): Sentiment positivity
    - Commitment signals (15): Commitment vs resistance cues

    Args:
        client_talk_percentage: (client_talk_time / total_duration) * 100
        duration_minutes: total_duration_seconds / 60
        client_turns: count of client utterances
        total_client_words: sum of word counts across client utterances
        sentiment_score: average sentiment (-1 to 1) for client utterances
        commitment_cue_count: number of commitment cues detected
        resistance_cue_count: number of resistance cues detected

    Returns:
        Engagement score 0-100
    """
    # Calculate response_elaboration (average words per client turn)
    response_elaboration = total_client_words / max(client_turns, 1)

    # 1. Participation balance (25 points max)
    # Optimal: 50% client talk. Penalty for deviation.
    deviation = abs(client_talk_percentage - 50)
    participation_score = max(0, 25 - deviation * 0.5)

    # 2. Response depth (25 points max)
    # Baseline: 15 words/response. Max score at 30+ words.
    depth_score = min(25, (response_elaboration / 30) * 25)

    # 3. Interaction density (20 points max)
    # Baseline: 2 turns/minute is good engagement
    turns_per_minute = client_turns / max(duration_minutes, 1)
    density_score = min(20, turns_per_minute * 10)

    # 4. Emotional engagement (15 points max)
    # Map -1..1 sentiment to 0..15
    emotion_score = (sentiment_score + 1) / 2 * 15

    # 5. Commitment signals (15 points max)
    # More commitments than resistance = positive
    total_cues = commitment_cue_count + resistance_cue_count
    if total_cues == 0:
        commitment_score = 7.5  # Neutral
    else:
        ratio = commitment_cue_count / total_cues
        commitment_score = ratio * 15

    total = (
        participation_score +
        depth_score +
        density_score +
        emotion_score +
        commitment_score
    )

    return round(min(100, max(0, total)), 2)


def calculate_response_elaboration_score(client_utterances: list[dict[str, Any]]) -> float:
    """
    Calculate response elaboration score (0-100).

    Based on average words per client utterance:
    - 0-10 words: Low elaboration (0-33 score)
    - 10-20 words: Medium elaboration (33-66 score)
    - 20-30+ words: High elaboration (66-100 score)

    Score = min(100, (avg_words / 30) * 100)

    Args:
        client_utterances: List of utterance dicts with 'text' field

    Returns:
        Elaboration score 0-100
    """
    if not client_utterances:
        return 0.0

    total_words = sum(
        len(u.get("text", "").split())
        for u in client_utterances
    )
    avg_words = total_words / len(client_utterances)

    return round(min(100.0, (avg_words / 30) * 100), 2)


def count_words_in_utterances(utterances: list[dict[str, Any]]) -> int:
    """
    Count total words in a list of utterances.

    Args:
        utterances: List of utterance dicts with 'text' field

    Returns:
        Total word count
    """
    return sum(len(u.get("text", "").split()) for u in utterances)
