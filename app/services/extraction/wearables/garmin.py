"""Garmin Connect data normalizer for CSV/JSON exports."""

import csv
import io
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.services.extraction.base import ExtractionResult
from app.services.extraction.wearables import (
    DailyHealthMetrics,
    WearableData,
    WearableNormalizer,
)

logger = get_logger(__name__)


class GarminNormalizer(WearableNormalizer):
    """
    Normalize Garmin Connect export data.

    Supports CSV exports from Garmin Connect web interface
    and JSON exports from third-party tools.
    """

    document_type = "garmin"

    # CSV column mappings - Garmin uses various column names
    COLUMN_MAP: dict[str, list[str]] = {
        "date": ["Date", "date", "Activity Date", "Start", "Start Time"],
        "steps": ["Steps", "steps", "Total Steps", "Daily Steps"],
        "calories": ["Calories", "calories", "Total Calories", "Calories (kcal)"],
        "active_calories": ["Active Calories", "Activity Calories", "active_calories"],
        "distance": ["Distance", "distance", "Total Distance", "Distance (km)", "Distance (mi)"],
        "floors": ["Floors Climbed", "Floors", "floors", "Floors Ascended"],
        "active_minutes": [
            "Active Minutes",
            "Minutes Active",
            "Moderate Activity (min)",
            "Vigorous Activity (min)",
        ],
        "sedentary_minutes": ["Minutes Sedentary", "Sedentary Minutes", "sedentary"],
        "resting_hr": [
            "Resting Heart Rate",
            "Resting HR",
            "resting_hr",
            "Avg Resting Heart Rate",
        ],
        "avg_hr": ["Average Heart Rate", "Avg HR", "avg_hr", "Avg Heart Rate"],
        "max_hr": ["Max Heart Rate", "Max HR", "max_hr"],
        "hrv": ["HRV", "Heart Rate Variability", "HRV Status"],
        "sleep_duration": [
            "Sleep Duration",
            "Total Sleep",
            "sleep_duration",
            "Sleep Time (hours)",
        ],
        "sleep_score": ["Sleep Score", "sleep_score"],
        "deep_sleep": ["Deep Sleep", "Deep Sleep (hours)", "deep_sleep"],
        "light_sleep": ["Light Sleep", "Light Sleep (hours)", "light_sleep"],
        "rem_sleep": ["REM Sleep", "REM Sleep (hours)", "rem_sleep"],
        "awake": ["Awake Time", "Awake (hours)", "awake"],
    }

    def get_metric_types(self) -> list[str]:
        """Return list of metric types this normalizer produces."""
        return [
            "steps",
            "calories",
            "distance",
            "floors",
            "active_minutes",
            "resting_hr",
            "avg_hr",
            "max_hr",
            "hrv",
            "sleep_duration",
            "sleep_score",
        ]

    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract and normalize Garmin data from export file.

        Args:
            file_path: Path to Garmin export file (CSV or JSON)

        Returns:
            ExtractionResult with normalized WearableData
        """
        import time

        start_time = time.time()
        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            return await self._extract_from_csv(file_path, start_time)
        elif ext == ".json":
            return await self._extract_from_json(file_path, start_time)
        else:
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"Unsupported file type: {ext}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    async def _extract_from_csv(self, file_path: str, start_time: float) -> ExtractionResult:
        """Extract data from Garmin CSV export."""
        warnings: list[str] = []

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

            # Detect delimiter
            delimiter = self._detect_delimiter(content)

            reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
            rows = list(reader)

            if not rows:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["CSV file is empty"],
                    warnings=[],
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            # Map columns
            column_map = self._map_columns(reader.fieldnames or [])

            if "date" not in column_map:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["Could not find date column in CSV"],
                    warnings=[],
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            # Parse rows into daily metrics
            daily_metrics: list[DailyHealthMetrics] = []
            dates_seen: set[date] = set()

            for row in rows:
                try:
                    # Parse date
                    date_str = row.get(column_map["date"], "")
                    parsed_date = self._parse_date(date_str)
                    if not parsed_date or parsed_date in dates_seen:
                        continue
                    dates_seen.add(parsed_date)

                    # Extract metrics
                    metrics = DailyHealthMetrics(
                        date=parsed_date,
                        source="garmin",
                        steps=self._get_int(row, column_map.get("steps")),
                        calories_burned=self._get_int(row, column_map.get("calories")),
                        active_calories=self._get_int(row, column_map.get("active_calories")),
                        distance_km=self._get_float(
                            row, column_map.get("distance"), convert_miles=True
                        ),
                        floors_climbed=self._get_int(row, column_map.get("floors")),
                        active_minutes=self._get_int(row, column_map.get("active_minutes")),
                        resting_hr=self._get_int(row, column_map.get("resting_hr")),
                        avg_hr=self._get_int(row, column_map.get("avg_hr")),
                        max_hr=self._get_int(row, column_map.get("max_hr")),
                        hrv=self._get_float(row, column_map.get("hrv")),
                        sleep_duration_hours=self._get_float(row, column_map.get("sleep_duration")),
                        sleep_score=self._get_float(row, column_map.get("sleep_score")),
                        deep_sleep_hours=self._get_float(row, column_map.get("deep_sleep")),
                        light_sleep_hours=self._get_float(row, column_map.get("light_sleep")),
                        rem_hours=self._get_float(row, column_map.get("rem_sleep")),
                        awake_hours=self._get_float(row, column_map.get("awake")),
                    )
                    daily_metrics.append(metrics)

                except Exception as e:
                    warnings.append(f"Failed to parse row: {str(e)}")
                    continue

            # Sort by date
            daily_metrics.sort(key=lambda m: m.date)

            if not daily_metrics:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["No valid data rows found"],
                    warnings=warnings,
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            # Build output
            data = WearableData(
                source="garmin",
                date_range_start=daily_metrics[0].date,
                date_range_end=daily_metrics[-1].date,
                total_days=len(daily_metrics),
                daily_metrics=daily_metrics,
            ).model_dump(mode="json")

            return ExtractionResult(
                success=True,
                data=data,
                confidence=0.95,
                errors=[],
                warnings=warnings,
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error("Garmin CSV extraction failed", error=str(e), exc_info=True)
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"CSV parsing error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    async def _extract_from_json(self, file_path: str, start_time: float) -> ExtractionResult:
        """Extract data from Garmin JSON export."""
        warnings: list[str] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "activities" in data:
                records = data["activities"]
            elif isinstance(data, dict) and "dailySummaries" in data:
                records = data["dailySummaries"]
            else:
                records = [data]

            daily_metrics: list[DailyHealthMetrics] = []
            dates_seen: set[date] = set()

            for record in records:
                try:
                    # Try to extract date
                    date_str = (
                        record.get("date")
                        or record.get("startTimeLocal")
                        or record.get("calendarDate")
                    )
                    parsed_date = self._parse_date(date_str)

                    if not parsed_date or parsed_date in dates_seen:
                        continue
                    dates_seen.add(parsed_date)

                    metrics = DailyHealthMetrics(
                        date=parsed_date,
                        source="garmin",
                        steps=record.get("steps") or record.get("totalSteps"),
                        calories_burned=record.get("calories") or record.get("totalKilocalories"),
                        active_calories=record.get("activeCalories"),
                        distance_km=self._convert_distance(record.get("distance")),
                        floors_climbed=record.get("floorsClimbed"),
                        active_minutes=record.get("activeMinutes")
                        or record.get("moderateIntensityMinutes"),
                        resting_hr=record.get("restingHeartRate"),
                        avg_hr=record.get("averageHeartRate"),
                        max_hr=record.get("maxHeartRate"),
                        sleep_duration_hours=self._convert_sleep_duration(
                            record.get("sleepTimeSeconds")
                        ),
                    )
                    daily_metrics.append(metrics)

                except Exception as e:
                    warnings.append(f"Failed to parse record: {str(e)}")
                    continue

            daily_metrics.sort(key=lambda m: m.date)

            if not daily_metrics:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["No valid records found in JSON"],
                    warnings=warnings,
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            output = WearableData(
                source="garmin",
                date_range_start=daily_metrics[0].date,
                date_range_end=daily_metrics[-1].date,
                total_days=len(daily_metrics),
                daily_metrics=daily_metrics,
            ).model_dump(mode="json")

            return ExtractionResult(
                success=True,
                data=output,
                confidence=0.95,
                errors=[],
                warnings=warnings,
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error("Garmin JSON extraction failed", error=str(e), exc_info=True)
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"JSON parsing error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate extracted Garmin data."""
        errors: list[str] = []

        daily_metrics = data.get("daily_metrics", [])
        if not daily_metrics:
            errors.append("No daily metrics extracted")

        return len(errors) == 0, errors

    def _map_columns(self, fieldnames: list[str]) -> dict[str, str]:
        """Map CSV column names to standard field names."""
        column_map: dict[str, str] = {}
        lower_fields = {f.lower().strip(): f for f in fieldnames}

        for std_name, aliases in self.COLUMN_MAP.items():
            for alias in aliases:
                if alias.lower() in lower_fields:
                    column_map[std_name] = lower_fields[alias.lower()]
                    break

        return column_map

    def _detect_delimiter(self, content: str) -> str:
        """Detect CSV delimiter."""
        first_line = content.split("\n")[0] if "\n" in content else content
        delimiters = [",", "\t", ";"]
        counts = {d: first_line.count(d) for d in delimiters}
        return max(counts, key=lambda d: counts[d])

    def _get_int(self, row: dict, col_name: str | None) -> int | None:
        """Get integer value from row."""
        if not col_name:
            return None
        try:
            val = row.get(col_name, "").strip()
            if not val:
                return None
            return int(float(val.replace(",", "")))
        except (ValueError, TypeError):
            return None

    def _get_float(
        self,
        row: dict,
        col_name: str | None,
        convert_miles: bool = False,
    ) -> float | None:
        """Get float value from row."""
        if not col_name:
            return None
        try:
            val = row.get(col_name, "").strip()
            if not val:
                return None
            result = float(val.replace(",", ""))
            if convert_miles and "mi" in col_name.lower():
                result *= 1.60934  # Convert miles to km
            return result
        except (ValueError, TypeError):
            return None

    def _convert_distance(self, distance: Any) -> float | None:
        """Convert distance to km."""
        if distance is None:
            return None
        try:
            # Garmin sometimes stores in meters
            meters = float(distance)
            if meters > 1000:  # Likely meters
                return meters / 1000
            return meters
        except (ValueError, TypeError):
            return None

    def _convert_sleep_duration(self, seconds: Any) -> float | None:
        """Convert sleep duration from seconds to hours."""
        if seconds is None:
            return None
        try:
            return float(seconds) / 3600
        except (ValueError, TypeError):
            return None
