"""Trend calculation utilities for analytics."""

from typing import Any


def calculate_trend(
    values: list[float],
    threshold: float = 0.1,
    higher_is_better: bool = True,
) -> str:
    """
    Determine trend direction from a series of values.

    Method: Compare first half average to second half average.

    Args:
        values: List of values ordered oldest to newest
        threshold: Minimum change ratio (default 10%) to be significant
        higher_is_better: If True, increasing values = "improving"
                         If False, decreasing values = "improving"

    Returns:
        "improving", "stable", or "declining"
    """
    if len(values) < 2:
        return "stable"

    # Split into first half and second half
    midpoint = len(values) // 2
    if midpoint == 0:
        midpoint = 1

    first_half = values[:midpoint]
    second_half = values[midpoint:]

    first_half_avg = sum(first_half) / len(first_half)
    second_half_avg = sum(second_half) / len(second_half)

    # Avoid division by zero
    if first_half_avg == 0:
        if second_half_avg > 0:
            change_ratio = 1.0  # Significant increase from zero
        elif second_half_avg < 0:
            change_ratio = -1.0  # Significant decrease to negative
        else:
            return "stable"  # Both zero
    else:
        change_ratio = (second_half_avg - first_half_avg) / abs(first_half_avg)

    # Determine trend based on direction and threshold
    if abs(change_ratio) < threshold:
        return "stable"

    if higher_is_better:
        return "improving" if change_ratio > 0 else "declining"
    else:
        return "improving" if change_ratio < 0 else "declining"


def calculate_talk_ratio_trend(
    client_percentages: list[float],
    threshold: float = 0.1,
) -> str:
    """
    Calculate trend for talk ratio.

    Optimal is around 50% client talk. Trend indicates if moving
    toward or away from balance.

    Args:
        client_percentages: List of client talk percentages (0-100)
        threshold: Minimum change to be significant

    Returns:
        "coach_increasing", "balanced", or "client_increasing"
    """
    if len(client_percentages) < 2:
        return "balanced"

    midpoint = len(client_percentages) // 2
    if midpoint == 0:
        midpoint = 1

    first_half_avg = sum(client_percentages[:midpoint]) / midpoint
    second_half_avg = sum(client_percentages[midpoint:]) / len(client_percentages[midpoint:])

    change = second_half_avg - first_half_avg

    if abs(change) < threshold * 100:  # threshold is ratio, percentages are 0-100
        return "balanced"
    elif change > 0:
        return "client_increasing"
    else:
        return "coach_increasing"


def calculate_session_frequency_trend(
    session_dates: list[Any],
    threshold: float = 0.25,
) -> str:
    """
    Calculate trend for session frequency.

    Compares gap between recent sessions to earlier sessions.

    Args:
        session_dates: List of session dates (sorted oldest to newest)
        threshold: Minimum change ratio to be significant

    Returns:
        "increasing", "stable", or "decreasing"
    """
    if len(session_dates) < 3:
        return "stable"

    # Calculate gaps between sessions
    gaps = []
    for i in range(1, len(session_dates)):
        gap = (session_dates[i] - session_dates[i - 1]).days
        gaps.append(gap)

    if not gaps:
        return "stable"

    # Compare average gap in first half vs second half
    # Larger gaps = less frequent = decreasing
    # Smaller gaps = more frequent = increasing
    midpoint = len(gaps) // 2
    if midpoint == 0:
        midpoint = 1

    first_half_avg = sum(gaps[:midpoint]) / midpoint
    second_half_avg = sum(gaps[midpoint:]) / len(gaps[midpoint:])

    if first_half_avg == 0:
        return "stable"

    change_ratio = (second_half_avg - first_half_avg) / first_half_avg

    if abs(change_ratio) < threshold:
        return "stable"
    elif change_ratio > 0:
        return "decreasing"  # Larger gaps = less frequent
    else:
        return "increasing"  # Smaller gaps = more frequent
