"""Apple Health data normalizer for XML/JSON exports."""

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from app.core.logging import get_logger
from app.services.extraction.base import ExtractionResult
from app.services.extraction.wearables import (
    DailyHealthMetrics,
    WearableData,
    WearableNormalizer,
)

logger = get_logger(__name__)


class AppleHealthNormalizer(WearableNormalizer):
    """
    Normalize Apple Health export data.

    Supports:
    - XML exports from Apple Health app (export.xml)
    - JSON exports from third-party apps
    """

    document_type = "apple_health"

    # Apple Health record type mappings to our schema
    RECORD_TYPE_MAP: dict[str, str] = {
        # Steps
        "HKQuantityTypeIdentifierStepCount": "steps",
        # Heart Rate
        "HKQuantityTypeIdentifierHeartRate": "heart_rate",
        "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv",
        # Activity
        "HKQuantityTypeIdentifierActiveEnergyBurned": "active_calories",
        "HKQuantityTypeIdentifierBasalEnergyBurned": "basal_calories",
        "HKQuantityTypeIdentifierDistanceWalkingRunning": "distance",
        "HKQuantityTypeIdentifierFlightsClimbed": "floors",
        "HKQuantityTypeIdentifierAppleExerciseTime": "active_minutes",
        # Sleep
        "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
    }

    # Sleep analysis value mapping
    SLEEP_VALUES: dict[str, str] = {
        "HKCategoryValueSleepAnalysisInBed": "in_bed",
        "HKCategoryValueSleepAnalysisAsleep": "asleep",
        "HKCategoryValueSleepAnalysisAwake": "awake",
        "HKCategoryValueSleepAnalysisAsleepCore": "light",
        "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
        "HKCategoryValueSleepAnalysisAsleepREM": "rem",
    }

    def get_metric_types(self) -> list[str]:
        """Return list of metric types this normalizer produces."""
        return [
            "steps",
            "resting_hr",
            "hrv",
            "active_calories",
            "distance",
            "active_minutes",
            "sleep_duration",
        ]

    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract and normalize Apple Health data.

        Args:
            file_path: Path to Apple Health export file (XML or JSON)

        Returns:
            ExtractionResult with normalized WearableData
        """
        import time

        start_time = time.time()
        ext = Path(file_path).suffix.lower()

        if ext == ".xml":
            return await self._extract_from_xml(file_path, start_time)
        elif ext == ".json":
            return await self._extract_from_json(file_path, start_time)
        else:
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"Unsupported file type: {ext}. Expected XML or JSON."],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    async def _extract_from_xml(self, file_path: str, start_time: float) -> ExtractionResult:
        """Extract data from Apple Health XML export."""
        import time

        warnings: list[str] = []

        try:
            # Parse XML - use iterparse for large files
            daily_data: dict[date, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )

            context = ET.iterparse(file_path, events=("end",))

            record_count = 0
            for event, elem in context:
                if elem.tag == "Record":
                    record_type = elem.get("type", "")
                    metric_type = self.RECORD_TYPE_MAP.get(record_type)

                    if metric_type:
                        try:
                            # Parse date
                            date_str = elem.get("startDate", "")
                            record_date = self._parse_date(date_str)
                            if not record_date:
                                continue

                            # Parse value
                            value = elem.get("value")
                            if value and metric_type != "sleep":
                                daily_data[record_date][metric_type].append(float(value))
                                record_count += 1

                        except (ValueError, TypeError):
                            continue

                    # Clear element to save memory
                    elem.clear()

                elif elem.tag == "SleepAnalysis" or (
                    elem.tag == "Record"
                    and elem.get("type") == "HKCategoryTypeIdentifierSleepAnalysis"
                ):
                    # Handle sleep data
                    try:
                        start_str = elem.get("startDate", "")
                        end_str = elem.get("endDate", "")
                        value = elem.get("value", "")

                        start_date = self._parse_date(start_str)
                        if not start_date:
                            continue

                        # Calculate duration in hours
                        start_dt = self._parse_datetime(start_str)
                        end_dt = self._parse_datetime(end_str)
                        if start_dt and end_dt:
                            duration_hours = (end_dt - start_dt).total_seconds() / 3600

                            sleep_type = self.SLEEP_VALUES.get(value, "asleep")
                            daily_data[start_date][f"sleep_{sleep_type}"].append(duration_hours)
                            record_count += 1

                    except (ValueError, TypeError):
                        continue

                    elem.clear()

            if record_count == 0:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["No valid health records found in XML"],
                    warnings=warnings,
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            # Aggregate daily data
            daily_metrics = self._aggregate_daily_data(daily_data)

            if not daily_metrics:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["No daily metrics could be computed"],
                    warnings=warnings,
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            # Build output
            data = WearableData(
                source="apple_health",
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

        except ET.ParseError as e:
            logger.error("Apple Health XML parse error", error=str(e))
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"XML parsing error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Apple Health XML extraction failed", error=str(e), exc_info=True)
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"Extraction error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    async def _extract_from_json(self, file_path: str, start_time: float) -> ExtractionResult:
        """Extract data from Apple Health JSON export (third-party apps)."""
        import time

        warnings: list[str] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, dict):
                records = data.get("data", data.get("records", [data]))
            else:
                records = data

            daily_data: dict[date, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )

            for record in records:
                try:
                    record_type = record.get("type", "")
                    metric_type = self.RECORD_TYPE_MAP.get(record_type)

                    if not metric_type:
                        continue

                    # Parse date
                    date_str = record.get("startDate") or record.get("date")
                    record_date = self._parse_date(date_str)
                    if not record_date:
                        continue

                    # Parse value
                    value = record.get("value") or record.get("quantity")
                    if value is not None:
                        daily_data[record_date][metric_type].append(float(value))

                except (ValueError, TypeError):
                    continue

            # Aggregate daily data
            daily_metrics = self._aggregate_daily_data(daily_data)

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
                source="apple_health",
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
            logger.error("Apple Health JSON extraction failed", error=str(e), exc_info=True)
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"JSON parsing error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate extracted Apple Health data."""
        errors: list[str] = []

        daily_metrics = data.get("daily_metrics", [])
        if not daily_metrics:
            errors.append("No daily metrics extracted")

        return len(errors) == 0, errors

    def _aggregate_daily_data(
        self,
        daily_data: dict[date, dict[str, list[float]]],
    ) -> list[DailyHealthMetrics]:
        """Aggregate raw records into daily metrics."""
        daily_metrics: list[DailyHealthMetrics] = []

        for day, metrics in sorted(daily_data.items()):
            # Sum cumulative metrics, average instantaneous ones
            steps = sum(metrics.get("steps", [])) if "steps" in metrics else None
            active_calories = (
                sum(metrics.get("active_calories", [])) if "active_calories" in metrics else None
            )
            basal_calories = (
                sum(metrics.get("basal_calories", [])) if "basal_calories" in metrics else None
            )
            total_calories = None
            if active_calories is not None or basal_calories is not None:
                total_calories = int((active_calories or 0) + (basal_calories or 0))

            distance_values = metrics.get("distance", [])
            distance_km = sum(distance_values) / 1000 if distance_values else None  # Convert m to km

            floors = sum(metrics.get("floors", [])) if "floors" in metrics else None
            active_minutes = sum(metrics.get("active_minutes", [])) if "active_minutes" in metrics else None

            # Average heart rate metrics
            hr_values = metrics.get("heart_rate", [])
            avg_hr = int(sum(hr_values) / len(hr_values)) if hr_values else None
            max_hr = int(max(hr_values)) if hr_values else None

            resting_hr_values = metrics.get("resting_hr", [])
            resting_hr = int(sum(resting_hr_values) / len(resting_hr_values)) if resting_hr_values else None

            hrv_values = metrics.get("hrv", [])
            hrv = sum(hrv_values) / len(hrv_values) if hrv_values else None

            # Sleep metrics
            sleep_asleep = sum(metrics.get("sleep_asleep", []))
            sleep_light = sum(metrics.get("sleep_light", []))
            sleep_deep = sum(metrics.get("sleep_deep", []))
            sleep_rem = sum(metrics.get("sleep_rem", []))
            sleep_awake = sum(metrics.get("sleep_awake", []))

            # Total sleep is asleep + light + deep + rem (not awake time)
            total_sleep = sleep_asleep + sleep_light + sleep_deep + sleep_rem
            if total_sleep == 0:
                total_sleep = None

            daily_metrics.append(
                DailyHealthMetrics(
                    date=day,
                    source="apple_health",
                    steps=int(steps) if steps else None,
                    calories_burned=total_calories,
                    active_calories=int(active_calories) if active_calories else None,
                    distance_km=round(distance_km, 2) if distance_km else None,
                    floors_climbed=int(floors) if floors else None,
                    active_minutes=int(active_minutes) if active_minutes else None,
                    resting_hr=resting_hr,
                    avg_hr=avg_hr,
                    max_hr=max_hr,
                    hrv=round(hrv, 1) if hrv else None,
                    sleep_duration_hours=round(total_sleep, 2) if total_sleep else None,
                    rem_hours=round(sleep_rem, 2) if sleep_rem > 0 else None,
                    deep_sleep_hours=round(sleep_deep, 2) if sleep_deep > 0 else None,
                    light_sleep_hours=round(sleep_light, 2) if sleep_light > 0 else None,
                    awake_hours=round(sleep_awake, 2) if sleep_awake > 0 else None,
                )
            )

        return daily_metrics

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse Apple Health datetime string."""
        if not dt_str:
            return None

        # Apple Health format: 2024-01-15 08:30:00 -0500
        formats = [
            "%Y-%m-%d %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(dt_str.strip(), fmt)
            except ValueError:
                continue

        return None
