"""Wearable data normalizers for various fitness devices."""

from abc import abstractmethod
from datetime import date

from pydantic import BaseModel

from app.services.extraction.base import BaseExtractor


class DailyHealthMetrics(BaseModel):
    """Normalized daily health metrics from wearable data."""

    date: date
    source: str

    # Activity
    steps: int | None = None
    calories_burned: int | None = None
    active_calories: int | None = None
    active_minutes: int | None = None
    distance_km: float | None = None
    floors_climbed: int | None = None

    # Heart
    resting_hr: int | None = None
    avg_hr: int | None = None
    max_hr: int | None = None
    hrv: float | None = None

    # Recovery (WHOOP-specific)
    recovery_score: float | None = None
    strain: float | None = None

    # Sleep
    sleep_duration_hours: float | None = None
    sleep_score: float | None = None
    rem_hours: float | None = None
    deep_sleep_hours: float | None = None
    light_sleep_hours: float | None = None
    awake_hours: float | None = None


class WearableData(BaseModel):
    """Structured data extracted from wearable exports."""

    source: str  # garmin, whoop, apple_health
    date_range_start: date | None = None
    date_range_end: date | None = None
    total_days: int
    daily_metrics: list[DailyHealthMetrics]


class WearableNormalizer(BaseExtractor):
    """
    Base class for wearable data normalizers.

    Subclasses implement specific parsing for each device type
    but output the same normalized DailyHealthMetrics format.
    """

    @abstractmethod
    def get_metric_types(self) -> list[str]:
        """Return list of metric types this normalizer produces."""
        pass

    def aggregate_daily(
        self,
        metrics: list[DailyHealthMetrics],
    ) -> dict[str, float | int]:
        """
        Calculate aggregate statistics across all days.

        Returns:
            Dictionary with avg_, min_, max_ prefixed stats
        """
        if not metrics:
            return {}

        stats: dict[str, float | int] = {}

        # Calculate averages for numeric fields
        numeric_fields = [
            "steps",
            "calories_burned",
            "active_minutes",
            "resting_hr",
            "avg_hr",
            "hrv",
            "sleep_duration_hours",
        ]

        for field in numeric_fields:
            values = [getattr(m, field) for m in metrics if getattr(m, field) is not None]
            if values:
                stats[f"avg_{field}"] = sum(values) / len(values)
                stats[f"min_{field}"] = min(values)
                stats[f"max_{field}"] = max(values)
                stats[f"total_{field}"] = sum(values)

        stats["total_days"] = len(metrics)
        return stats


# Import normalizers for registration
from app.services.extraction.wearables.garmin import GarminNormalizer
from app.services.extraction.wearables.whoop import WhoopNormalizer
from app.services.extraction.wearables.apple_health import AppleHealthNormalizer

__all__ = [
    "DailyHealthMetrics",
    "WearableData",
    "WearableNormalizer",
    "GarminNormalizer",
    "WhoopNormalizer",
    "AppleHealthNormalizer",
]
