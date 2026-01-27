"""WHOOP recovery and strain data normalizer for CSV exports."""

import csv
import io
from datetime import date
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


class WhoopNormalizer(WearableNormalizer):
    """
    Normalize WHOOP export data.

    WHOOP exports recovery, strain, sleep, and physiological data
    in CSV format from the app or web interface.
    """

    document_type = "whoop"

    # CSV column mappings for WHOOP exports
    COLUMN_MAP: dict[str, list[str]] = {
        "date": [
            "Cycle start time",
            "Date",
            "date",
            "Sleep onset",
            "Cycle Start",
        ],
        "recovery_score": [
            "Recovery score %",
            "Recovery Score",
            "recovery_score",
            "Recovery",
        ],
        "strain": ["Strain", "strain", "Day Strain", "Strain Score"],
        "resting_hr": [
            "Resting heart rate (bpm)",
            "Resting HR",
            "resting_hr",
            "RHR",
        ],
        "hrv": [
            "Heart rate variability (ms)",
            "HRV",
            "hrv",
            "Heart Rate Variability",
        ],
        "avg_hr": [
            "Average heart rate (bpm)",
            "Avg HR",
            "avg_hr",
            "Average HR",
        ],
        "max_hr": [
            "Max heart rate (bpm)",
            "Max HR",
            "max_hr",
        ],
        "calories": [
            "Calories (kcal)",
            "Calories",
            "calories",
            "Kilojoules",
        ],
        "sleep_duration": [
            "Sleep duration (hours)",
            "Sleep Duration",
            "Total Sleep",
            "Asleep duration (hours)",
        ],
        "sleep_score": [
            "Sleep performance %",
            "Sleep Score",
            "sleep_score",
            "Sleep Performance",
        ],
        "rem_sleep": [
            "REM sleep (hours)",
            "REM Sleep",
            "rem_sleep",
            "REM duration (hours)",
        ],
        "deep_sleep": [
            "Deep (SWS) sleep (hours)",
            "Deep Sleep",
            "SWS duration (hours)",
            "deep_sleep",
        ],
        "light_sleep": [
            "Light sleep (hours)",
            "Light Sleep",
            "light_sleep",
        ],
        "awake": [
            "Awake (hours)",
            "Awake duration (hours)",
            "Awake",
            "awake",
        ],
        "respiratory_rate": [
            "Respiratory rate (rpm)",
            "Respiratory Rate",
            "respiratory_rate",
        ],
        "spo2": [
            "Blood oxygen %",
            "SpO2",
            "spo2",
        ],
        "skin_temp": [
            "Skin temp (celsius)",
            "Skin Temperature",
            "skin_temp",
        ],
    }

    def get_metric_types(self) -> list[str]:
        """Return list of metric types this normalizer produces."""
        return [
            "recovery_score",
            "strain",
            "resting_hr",
            "hrv",
            "avg_hr",
            "sleep_duration",
            "sleep_score",
        ]

    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract and normalize WHOOP data from CSV export.

        Args:
            file_path: Path to WHOOP export file (CSV)

        Returns:
            ExtractionResult with normalized WearableData
        """
        import time

        start_time = time.time()
        ext = Path(file_path).suffix.lower()

        if ext != ".csv":
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"WHOOP export should be CSV, got: {ext}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

        return await self._extract_from_csv(file_path, start_time)

    async def _extract_from_csv(self, file_path: str, start_time: float) -> ExtractionResult:
        """Extract data from WHOOP CSV export."""
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

                    # Extract metrics - WHOOP-specific fields
                    metrics = DailyHealthMetrics(
                        date=parsed_date,
                        source="whoop",
                        # WHOOP-specific
                        recovery_score=self._get_float(row, column_map.get("recovery_score")),
                        strain=self._get_float(row, column_map.get("strain")),
                        # Heart
                        resting_hr=self._get_int(row, column_map.get("resting_hr")),
                        avg_hr=self._get_int(row, column_map.get("avg_hr")),
                        max_hr=self._get_int(row, column_map.get("max_hr")),
                        hrv=self._get_float(row, column_map.get("hrv")),
                        # Activity
                        calories_burned=self._get_int(row, column_map.get("calories")),
                        # Sleep
                        sleep_duration_hours=self._get_float(row, column_map.get("sleep_duration")),
                        sleep_score=self._get_float(row, column_map.get("sleep_score")),
                        rem_hours=self._get_float(row, column_map.get("rem_sleep")),
                        deep_sleep_hours=self._get_float(row, column_map.get("deep_sleep")),
                        light_sleep_hours=self._get_float(row, column_map.get("light_sleep")),
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
                source="whoop",
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
            logger.error("WHOOP CSV extraction failed", error=str(e), exc_info=True)
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"CSV parsing error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate extracted WHOOP data."""
        errors: list[str] = []

        daily_metrics = data.get("daily_metrics", [])
        if not daily_metrics:
            errors.append("No daily metrics extracted")

        # Validate WHOOP-specific ranges
        for metrics in daily_metrics:
            # Recovery should be 0-100%
            recovery = metrics.get("recovery_score")
            if recovery is not None and (recovery < 0 or recovery > 100):
                errors.append(f"Invalid recovery score: {recovery}")

            # Strain should be 0-21 (WHOOP's scale)
            strain = metrics.get("strain")
            if strain is not None and (strain < 0 or strain > 25):
                errors.append(f"Invalid strain score: {strain}")

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

    def _get_float(self, row: dict, col_name: str | None) -> float | None:
        """Get float value from row."""
        if not col_name:
            return None
        try:
            val = row.get(col_name, "").strip()
            if not val:
                return None
            # Handle percentage values
            if val.endswith("%"):
                val = val[:-1]
            return float(val.replace(",", ""))
        except (ValueError, TypeError):
            return None
