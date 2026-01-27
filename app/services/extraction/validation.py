"""Common validation utilities for document extraction."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationRange:
    """Defines a valid range for a numeric value."""

    min_value: float
    max_value: float
    unit: str
    description: str


# Validation ranges for common health metrics
VALIDATION_RANGES: dict[str, ValidationRange] = {
    # Body composition
    "weight_kg": ValidationRange(20.0, 500.0, "kg", "Body weight"),
    "body_fat_percent": ValidationRange(3.0, 60.0, "%", "Body fat percentage"),
    "skeletal_muscle_mass_kg": ValidationRange(10.0, 100.0, "kg", "Skeletal muscle mass"),
    "basal_metabolic_rate_kcal": ValidationRange(800.0, 4000.0, "kcal", "Basal metabolic rate"),
    "body_water_percent": ValidationRange(30.0, 80.0, "%", "Body water percentage"),
    "visceral_fat_level": ValidationRange(1.0, 30.0, "level", "Visceral fat level"),
    "lean_body_mass_kg": ValidationRange(20.0, 150.0, "kg", "Lean body mass"),
    "bone_mineral_content_kg": ValidationRange(1.0, 10.0, "kg", "Bone mineral content"),
    # Vital signs
    "resting_hr": ValidationRange(30.0, 200.0, "bpm", "Resting heart rate"),
    "avg_hr": ValidationRange(30.0, 220.0, "bpm", "Average heart rate"),
    "hrv": ValidationRange(1.0, 300.0, "ms", "Heart rate variability"),
    # Activity
    "steps": ValidationRange(0.0, 100000.0, "steps", "Daily steps"),
    "calories_burned": ValidationRange(0.0, 10000.0, "kcal", "Calories burned"),
    "active_minutes": ValidationRange(0.0, 1440.0, "min", "Active minutes"),
    "distance_km": ValidationRange(0.0, 200.0, "km", "Distance"),
    # Sleep
    "sleep_duration_hours": ValidationRange(0.0, 24.0, "hours", "Sleep duration"),
    "sleep_score": ValidationRange(0.0, 100.0, "score", "Sleep score"),
    # Recovery
    "recovery_score": ValidationRange(0.0, 100.0, "%", "Recovery score"),
    "strain": ValidationRange(0.0, 25.0, "score", "Strain score"),
}


def validate_range(
    field_name: str,
    value: float | None,
    custom_range: ValidationRange | None = None,
) -> tuple[bool, str | None]:
    """
    Validate that a value falls within expected range.

    Args:
        field_name: Name of the field being validated
        value: Value to validate
        custom_range: Optional custom range to use instead of defaults

    Returns:
        Tuple of (is_valid, error_message or None)
    """
    if value is None:
        return True, None

    range_def = custom_range or VALIDATION_RANGES.get(field_name)
    if not range_def:
        return True, None

    if not (range_def.min_value <= value <= range_def.max_value):
        return False, (
            f"{field_name} value {value} {range_def.unit} outside valid range "
            f"({range_def.min_value}-{range_def.max_value})"
        )

    return True, None


def validate_required_fields(
    data: dict[str, Any],
    required_fields: list[str],
) -> list[str]:
    """
    Validate that required fields are present and non-null.

    Args:
        data: Data dictionary to validate
        required_fields: List of required field names

    Returns:
        List of error messages for missing fields
    """
    errors = []
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    return errors


def validate_date_not_future(
    field_name: str,
    date_value: Any,
) -> tuple[bool, str | None]:
    """
    Validate that a date is not in the future.

    Args:
        field_name: Name of the date field
        date_value: Date value to validate

    Returns:
        Tuple of (is_valid, error_message or None)
    """
    if date_value is None:
        return True, None

    from datetime import date

    if hasattr(date_value, "date"):
        # It's a datetime
        date_value = date_value.date()

    if isinstance(date_value, date) and date_value > date.today():
        return False, f"{field_name} cannot be in the future"

    return True, None


# Reference ranges for common biomarkers
BIOMARKER_REFERENCE_RANGES: dict[str, dict[str, Any]] = {
    # Lipid Panel
    "total_cholesterol": {"low": 0, "optimal": 200, "high": 240, "unit": "mg/dL"},
    "ldl_cholesterol": {"low": 0, "optimal": 100, "high": 130, "unit": "mg/dL"},
    "hdl_cholesterol": {"low": 40, "optimal": 60, "high": 999, "unit": "mg/dL"},
    "triglycerides": {"low": 0, "optimal": 150, "high": 200, "unit": "mg/dL"},
    # Metabolic Panel
    "glucose": {"low": 70, "optimal": 100, "high": 126, "unit": "mg/dL"},
    "hemoglobin_a1c": {"low": 0, "optimal": 5.7, "high": 6.5, "unit": "%"},
    "creatinine": {"low": 0.5, "optimal": 1.2, "high": 1.5, "unit": "mg/dL"},
    "bun": {"low": 7, "optimal": 20, "high": 25, "unit": "mg/dL"},
    # Thyroid
    "tsh": {"low": 0.4, "optimal": 4.0, "high": 10.0, "unit": "mIU/L"},
    "t3": {"low": 80, "optimal": 200, "high": 220, "unit": "ng/dL"},
    "t4": {"low": 4.5, "optimal": 12.0, "high": 13.0, "unit": "mcg/dL"},
    # Hormones
    "testosterone": {"low": 300, "optimal": 800, "high": 1200, "unit": "ng/dL"},
    "estradiol": {"low": 10, "optimal": 50, "high": 400, "unit": "pg/mL"},
    "cortisol": {"low": 5, "optimal": 25, "high": 30, "unit": "mcg/dL"},
    # Vitamins & Minerals
    "vitamin_d": {"low": 20, "optimal": 50, "high": 100, "unit": "ng/mL"},
    "vitamin_b12": {"low": 200, "optimal": 900, "high": 1500, "unit": "pg/mL"},
    "iron": {"low": 60, "optimal": 170, "high": 200, "unit": "mcg/dL"},
    "ferritin": {"low": 20, "optimal": 200, "high": 500, "unit": "ng/mL"},
    # Inflammation
    "crp": {"low": 0, "optimal": 1.0, "high": 3.0, "unit": "mg/L"},
    "esr": {"low": 0, "optimal": 20, "high": 30, "unit": "mm/hr"},
}


def flag_biomarker_value(
    biomarker_name: str,
    value: float,
    unit: str | None = None,
) -> str | None:
    """
    Flag a biomarker value as high, low, or normal.

    Args:
        biomarker_name: Standardized biomarker name
        value: Measured value
        unit: Optional unit to verify against reference

    Returns:
        "high", "low", "normal", or None if biomarker not recognized
    """
    ref = BIOMARKER_REFERENCE_RANGES.get(biomarker_name.lower())
    if not ref:
        return None

    if value < ref["low"]:
        return "low"
    elif value > ref["high"]:
        return "high"
    elif value <= ref["optimal"]:
        return "normal"
    else:
        # Between optimal and high
        return "borderline"
