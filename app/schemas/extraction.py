"""Schemas for document extraction API."""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ExtractionStatus(str, Enum):
    """Status of an extraction job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported document types for extraction."""

    INBODY = "inbody"
    LAB_RESULTS = "lab_results"
    GARMIN = "garmin"
    WHOOP = "whoop"
    APPLE_HEALTH = "apple_health"
    NUTRITION = "nutrition"


# ============================================================================
# Request Schemas
# ============================================================================


class ExtractionUploadRequest(BaseModel):
    """Request schema for document upload (metadata only, file is multipart)."""

    client_id: UUID
    document_type: DocumentType | None = Field(
        None,
        description="Optional hint for document type. If not provided, will be auto-detected.",
    )
    webhook_url: str | None = Field(
        None,
        description="Optional webhook URL to notify on completion.",
    )


class ExtractionReprocessRequest(BaseModel):
    """Request to reprocess a failed extraction job."""

    force: bool = Field(
        False,
        description="Force reprocessing even if job is not in failed state.",
    )


# ============================================================================
# Response Schemas
# ============================================================================


class ExtractionJobResponse(BaseModel):
    """Response schema for an extraction job."""

    id: UUID
    client_id: UUID
    coach_id: UUID
    file_name: str
    file_type: str
    file_size: int
    document_type: str | None
    status: ExtractionStatus
    created_at: datetime

    class Config:
        from_attributes = True


class ExtractionStatusResponse(BaseModel):
    """Response schema for extraction job status with results."""

    id: UUID
    client_id: UUID
    coach_id: UUID
    file_name: str
    file_type: str
    file_size: int
    document_type: str | None
    status: ExtractionStatus
    started_at: datetime | None
    completed_at: datetime | None
    confidence_score: float | None
    validation_errors: list[str] | None
    error_message: str | None
    extracted_data: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExtractionListResponse(BaseModel):
    """Response schema for listing extraction jobs."""

    jobs: list[ExtractionJobResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# Extracted Data Schemas
# ============================================================================


class InBodyExtractedData(BaseModel):
    """Structured data extracted from InBody scans."""

    scan_date: date | None = None
    device_model: str | None = None

    # Core metrics with confidence
    weight_kg: float
    weight_confidence: float = Field(ge=0.0, le=1.0)

    body_fat_percent: float
    body_fat_confidence: float = Field(ge=0.0, le=1.0)

    skeletal_muscle_mass_kg: float
    smm_confidence: float = Field(ge=0.0, le=1.0)

    basal_metabolic_rate_kcal: int
    bmr_confidence: float = Field(ge=0.0, le=1.0)

    # Optional metrics
    body_water_percent: float | None = None
    visceral_fat_level: int | None = None
    lean_body_mass_kg: float | None = None
    bone_mineral_content_kg: float | None = None

    # Segmental analysis (if available)
    segmental_lean: dict[str, float] | None = None
    segmental_fat: dict[str, float] | None = None


class BiomarkerValue(BaseModel):
    """A single biomarker measurement from lab results."""

    name: str
    code: str | None = None  # LOINC code if available
    value: float
    unit: str
    reference_range: str | None = None
    flag: str | None = None  # "high", "low", "normal", "critical"
    confidence: float = Field(ge=0.0, le=1.0)


class LabResultsExtractedData(BaseModel):
    """Structured data extracted from lab results."""

    lab_provider: str | None = None
    collection_date: date | None = None
    report_date: date | None = None
    patient_name: str | None = None  # May be partially redacted
    biomarkers: list[BiomarkerValue]


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


class WearableExtractedData(BaseModel):
    """Structured data extracted from wearable exports."""

    source: str  # garmin, whoop, apple_health
    date_range_start: date | None = None
    date_range_end: date | None = None
    total_days: int
    daily_metrics: list[DailyHealthMetrics]


class NutritionEntry(BaseModel):
    """A single nutrition log entry."""

    date: date
    meal: str | None = None  # breakfast, lunch, dinner, snack
    food_name: str | None = None
    calories: int | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    fiber_g: float | None = None
    sugar_g: float | None = None
    sodium_mg: float | None = None


class NutritionExtractedData(BaseModel):
    """Structured data extracted from nutrition logs."""

    source: str | None = None  # MyFitnessPal, Cronometer, etc.
    date_range_start: date | None = None
    date_range_end: date | None = None
    total_entries: int
    entries: list[NutritionEntry]

    # Daily summaries
    avg_daily_calories: float | None = None
    avg_daily_protein_g: float | None = None
    avg_daily_carbs_g: float | None = None
    avg_daily_fat_g: float | None = None


# ============================================================================
# Webhook Schemas
# ============================================================================


class ExtractionWebhookPayload(BaseModel):
    """Payload sent to webhook URL on extraction completion."""

    event: str = "extraction.completed"
    job_id: UUID
    client_id: UUID
    coach_id: UUID
    document_type: str | None
    status: ExtractionStatus
    confidence_score: float | None
    extracted_data: dict[str, Any] | None
    error_message: str | None
    completed_at: datetime
