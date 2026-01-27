# Plan 0002: Document Extraction Pipeline

**Spec**: [codev/specs/0002-document-extraction-pipeline.md](../specs/0002-document-extraction-pipeline.md)
**Status**: Ready for implementation
**Estimated Phases**: 6

---

## Implementation Strategy

Build extractors incrementally, starting with the most common document type (InBody), then adding lab results and wearable normalizers. Each phase produces working, testable extraction capability.

**Key Principles:**
- InBody extractor first (highest value, most structured)
- Each extractor independently testable
- Confidence scoring from the start
- Test documents required for each type

---

## Phase 1: Infrastructure & Base Classes

**Goal**: Set up extraction infrastructure, S3 integration, database schema, and base extractor class.

### Tasks

1.1 **Add extraction dependencies to `pyproject.toml`**
```toml
dependencies = [
    # ... existing ...
    "docling>=2.0.0",
    "pandas>=2.0.0",
    "boto3>=1.34.0",
    "aioboto3>=12.0.0",
    "python-magic>=0.4.27",
    "openpyxl>=3.1.0",  # Excel support
    "lxml>=5.0.0",      # XML parsing for Apple Health
]
```

1.2 **Create extraction service structure**
```
app/services/extraction/
├── __init__.py
├── base.py           # BaseExtractor abstract class
├── router.py         # Document type routing
├── storage.py        # S3/R2 file operations
└── validation.py     # Common validation utilities
```

1.3 **Implement BaseExtractor class** (`app/services/extraction/base.py`)
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any

class ExtractionResult(BaseModel):
    success: bool
    data: dict | None
    confidence: float
    errors: list[str]
    warnings: list[str]

class BaseExtractor(ABC):
    """Base class for all document extractors."""

    document_type: str

    @abstractmethod
    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract data from document."""
        pass

    @abstractmethod
    def validate(self, data: dict) -> tuple[bool, list[str]]:
        """Validate extracted data."""
        pass

    def calculate_confidence(self, field_confidences: dict[str, float]) -> float:
        """Calculate overall confidence from field confidences."""
        if not field_confidences:
            return 0.0
        return sum(field_confidences.values()) / len(field_confidences)
```

1.4 **Implement S3 storage service** (`app/services/extraction/storage.py`)
```python
import aioboto3
from app.config import get_settings

class StorageService:
    """Handle file uploads and downloads to S3/R2."""

    async def upload_file(
        self,
        file_content: bytes,
        client_id: str,
        filename: str,
        content_type: str,
    ) -> str:
        """Upload file and return URL."""
        pass

    async def download_file(self, file_url: str) -> bytes:
        """Download file from URL."""
        pass

    async def get_presigned_url(self, file_url: str, expires: int = 3600) -> str:
        """Get presigned URL for file access."""
        pass
```

1.5 **Add S3 configuration to settings**
```python
# app/config.py additions
s3_endpoint_url: str = ""  # For R2 compatibility
s3_bucket: str = "xenios-extractions"
s3_access_key_id: str = ""
s3_secret_access_key: str = ""
s3_region: str = "auto"
```

1.6 **Create database migration** (`scripts/migrations/0002_extraction_tables.sql`)
```sql
-- Extraction jobs table
CREATE TABLE ai_backend.extraction_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,
    file_url TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER NOT NULL,
    document_type VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    extracted_data JSONB,
    confidence_score DECIMAL(3,2),
    validation_errors JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_extraction_jobs_client ON ai_backend.extraction_jobs(client_id);
CREATE INDEX idx_extraction_jobs_status ON ai_backend.extraction_jobs(status);
CREATE INDEX idx_extraction_jobs_created ON ai_backend.extraction_jobs(created_at DESC);
```

1.7 **Create extraction schemas** (`app/schemas/extraction.py`)
- `ExtractionUploadRequest`
- `ExtractionJobResponse`
- `ExtractionStatusResponse`
- `ExtractionResultResponse`

### Acceptance Criteria Coverage
- AC7: Data Storage (schema)

### Verification
```bash
# Run migration
psql $DATABASE_URL -f scripts/migrations/0002_extraction_tables.sql

# Test S3 connection
uv run python -c "from app.services.extraction.storage import StorageService; ..."
```

---

## Phase 2: API Endpoints & Celery Task

**Goal**: Implement upload endpoint and extraction Celery task with status tracking.

### Tasks

2.1 **Implement extraction API router** (`app/api/v1/extraction.py`)
```python
from fastapi import APIRouter, UploadFile, File, Depends
from app.core.auth import get_current_user

router = APIRouter(prefix="/extraction", tags=["extraction"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str | None = None,
    user: UserContext = Depends(get_current_user),
) -> ExtractionJobResponse:
    """
    Upload a document for extraction.

    Returns job_id immediately. Poll /status/{job_id} for results.
    """
    # 1. Validate file (size, type)
    # 2. Upload to S3
    # 3. Create extraction_job record
    # 4. Queue Celery task
    # 5. Return job_id
    pass

@router.get("/status/{job_id}")
async def get_extraction_status(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
) -> ExtractionStatusResponse:
    """Get extraction job status and results."""
    pass

@router.get("/jobs")
async def list_extraction_jobs(
    client_id: UUID | None = None,
    status: str | None = None,
    limit: int = 20,
    user: UserContext = Depends(get_current_user),
) -> list[ExtractionJobResponse]:
    """List extraction jobs."""
    pass

@router.post("/reprocess/{job_id}")
async def reprocess_extraction(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
) -> ExtractionJobResponse:
    """Retry a failed extraction."""
    pass
```

2.2 **Implement document type detection** (`app/services/extraction/router.py`)
```python
import magic
from pathlib import Path

class DocumentRouter:
    """Route documents to appropriate extractors."""

    MIME_TYPE_MAP = {
        "application/pdf": ["inbody", "lab_results"],
        "text/csv": ["lab_results", "garmin", "whoop", "nutrition"],
        "application/json": ["garmin", "apple_health"],
        "text/xml": ["apple_health"],
    }

    def detect_document_type(
        self,
        file_content: bytes,
        filename: str,
        hint: str | None = None,
    ) -> str:
        """Detect document type from content and filename."""
        if hint and hint in self.SUPPORTED_TYPES:
            return hint

        mime_type = magic.from_buffer(file_content, mime=True)
        # Use heuristics to determine specific type
        # ...
```

2.3 **Implement extraction Celery task** (`app/workers/tasks/extraction.py`)
```python
from app.workers.celery_app import celery_app
from app.workers.tasks.base import BaseTask
from app.services.extraction.router import DocumentRouter
from app.services.extraction.storage import StorageService

@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="extraction",
    max_retries=3,
    soft_time_limit=300,
)
def process_extraction(self, job_id: str) -> dict:
    """
    Process document extraction.

    1. Fetch job from database
    2. Download file from S3
    3. Detect/confirm document type
    4. Route to appropriate extractor
    5. Run extraction
    6. Validate results
    7. Store results
    8. Update job status
    """
    pass
```

2.4 **Add extraction router to v1** (`app/api/v1/router.py`)
```python
from app.api.v1.extraction import router as extraction_router
router.include_router(extraction_router)
```

2.5 **Implement job status updates**
- Update job status at each stage
- Store timestamps (started_at, completed_at)
- Handle errors gracefully

### Acceptance Criteria Coverage
- AC5: API Endpoints
- AC6: Async Processing

### Verification
```bash
# Test upload endpoint
curl -X POST http://localhost:8000/api/v1/extraction/upload \
  -H "X-API-Key: $API_KEY" \
  -H "Authorization: Bearer $JWT" \
  -F "file=@test_inbody.pdf" \
  -F "document_type=inbody"

# Check status
curl http://localhost:8000/api/v1/extraction/status/{job_id} \
  -H "X-API-Key: $API_KEY" \
  -H "Authorization: Bearer $JWT"
```

---

## Phase 3: InBody Extractor

**Goal**: Implement InBody PDF extraction using Docling with >95% accuracy.

### Tasks

3.1 **Install and configure Docling**
- Add to Docker worker image
- Test basic PDF extraction
- Configure for optimal table extraction

3.2 **Create InBody extractor** (`app/services/extraction/inbody.py`)
```python
from docling.document_converter import DocumentConverter
from app.services.extraction.base import BaseExtractor, ExtractionResult
from pydantic import BaseModel

class InBodyData(BaseModel):
    """Structured InBody extraction results."""
    scan_date: date | None
    device_model: str | None

    # Core metrics
    weight_kg: float
    weight_confidence: float

    body_fat_percent: float
    body_fat_confidence: float

    skeletal_muscle_mass_kg: float
    smm_confidence: float

    basal_metabolic_rate_kcal: int
    bmr_confidence: float

    # Optional metrics
    body_water_percent: float | None = None
    visceral_fat_level: int | None = None
    lean_body_mass_kg: float | None = None

class InBodyExtractor(BaseExtractor):
    """Extract body composition data from InBody scan PDFs."""

    document_type = "inbody"

    # Patterns for different InBody models
    FIELD_PATTERNS = {
        "weight": [
            r"(?:Body\s*Weight|Weight)\s*[:\s]*(\d+\.?\d*)\s*(kg|lbs?)",
            r"(\d+\.?\d*)\s*(kg|lb)\s*(?:Body\s*Weight)",
        ],
        "body_fat_percent": [
            r"(?:Percent\s*Body\s*Fat|PBF|Body\s*Fat\s*%?)\s*[:\s]*(\d+\.?\d*)\s*%?",
            r"(\d+\.?\d*)\s*%\s*(?:Body\s*Fat|PBF)",
        ],
        # ... more patterns
    }

    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract data from InBody PDF."""
        try:
            # 1. Use Docling to extract document
            converter = DocumentConverter()
            result = converter.convert(file_path)

            # 2. Find results table
            tables = self._extract_tables(result)

            # 3. Extract each field with confidence
            extracted = {}
            confidences = {}

            for field, patterns in self.FIELD_PATTERNS.items():
                value, confidence = self._extract_field(tables, patterns)
                extracted[field] = value
                confidences[field] = confidence

            # 4. Validate
            is_valid, errors = self.validate(extracted)

            return ExtractionResult(
                success=is_valid,
                data=InBodyData(**extracted).model_dump() if is_valid else None,
                confidence=self.calculate_confidence(confidences),
                errors=errors,
                warnings=[],
            )

        except Exception as e:
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[str(e)],
                warnings=[],
            )

    def validate(self, data: dict) -> tuple[bool, list[str]]:
        """Validate InBody extraction."""
        errors = []

        # Weight validation (20-500 kg)
        weight = data.get("weight_kg")
        if weight is not None and not (20 <= weight <= 500):
            errors.append(f"Weight {weight} kg outside valid range (20-500)")

        # Body fat validation (3-60%)
        bf = data.get("body_fat_percent")
        if bf is not None and not (3 <= bf <= 60):
            errors.append(f"Body fat {bf}% outside valid range (3-60)")

        # SMM validation (10-100 kg)
        smm = data.get("skeletal_muscle_mass_kg")
        if smm is not None and not (10 <= smm <= 100):
            errors.append(f"SMM {smm} kg outside valid range (10-100)")

        # BMR validation (800-4000 kcal)
        bmr = data.get("basal_metabolic_rate_kcal")
        if bmr is not None and not (800 <= bmr <= 4000):
            errors.append(f"BMR {bmr} kcal outside valid range (800-4000)")

        return len(errors) == 0, errors
```

3.3 **Create InBody test suite**
- Use 3+ anonymized sample PDFs
- Test each field extraction
- Test validation logic
- Test different InBody models (570, 770, S10)

3.4 **Implement confidence scoring**
- Pattern match confidence
- Multi-source validation (if value appears multiple times)
- Format consistency scoring

### Acceptance Criteria Coverage
- AC1: IBM Docling Integration
- AC2: InBody Extraction
- AC8: Validation

### Verification
```bash
# Run InBody extraction tests
uv run pytest tests/test_inbody_extractor.py -v

# Test with real file
uv run python -c "
from app.services.extraction.inbody import InBodyExtractor
import asyncio
extractor = InBodyExtractor()
result = asyncio.run(extractor.extract('tests/fixtures/sample_inbody.pdf'))
print(result)
"
```

---

## Phase 4: Lab Results Extractor

**Goal**: Extract biomarkers from lab result documents (Quest, LabCorp, generic CSV).

### Tasks

4.1 **Create lab results extractor** (`app/services/extraction/lab_results.py`)
```python
class LabResultsExtractor(BaseExtractor):
    """Extract biomarkers from lab result documents."""

    document_type = "lab_results"

    # Common biomarker patterns
    BIOMARKER_PATTERNS = {
        "ldl_cholesterol": [
            r"LDL[\s\-]?(?:Cholesterol|C)?\s*[:\s]*(\d+\.?\d*)\s*(mg/dL|mmol/L)",
        ],
        "hdl_cholesterol": [...],
        "total_cholesterol": [...],
        "triglycerides": [...],
        "glucose": [...],
        "hemoglobin_a1c": [...],
        "testosterone": [...],
        # ... more biomarkers
    }

    # Reference ranges for flagging
    REFERENCE_RANGES = {
        "ldl_cholesterol": {"low": 0, "normal": 100, "high": 130, "unit": "mg/dL"},
        "hdl_cholesterol": {"low": 40, "normal": 60, "high": 999, "unit": "mg/dL"},
        # ...
    }

    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract biomarkers from lab results."""
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return await self._extract_from_pdf(file_path)
        elif ext == ".csv":
            return await self._extract_from_csv(file_path)
        else:
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"Unsupported file type: {ext}"],
                warnings=[],
            )

    async def _extract_from_pdf(self, file_path: str) -> ExtractionResult:
        """Extract from PDF using Docling."""
        converter = DocumentConverter()
        result = converter.convert(file_path)
        # Parse tables and text for biomarkers
        pass

    async def _extract_from_csv(self, file_path: str) -> ExtractionResult:
        """Extract from CSV file."""
        df = pd.read_csv(file_path)
        # Map columns to biomarkers
        pass

    def _flag_value(self, name: str, value: float) -> str | None:
        """Flag value as high/low/normal."""
        ref = self.REFERENCE_RANGES.get(name)
        if not ref:
            return None
        if value < ref["low"]:
            return "low"
        elif value > ref["high"]:
            return "high"
        return "normal"
```

4.2 **Implement Quest format parser**
- Specific patterns for Quest Diagnostics layout
- Table structure detection

4.3 **Implement LabCorp format parser**
- Specific patterns for LabCorp layout
- Handle variations in format

4.4 **Implement generic CSV parser**
- Column name detection
- Unit normalization
- Date parsing

4.5 **Create comprehensive biomarker list**
- LOINC code mapping (where available)
- Reference ranges by age/sex (future)
- Unit conversion helpers

### Acceptance Criteria Coverage
- AC3: Lab Results Extraction

### Verification
```bash
# Test lab extraction
uv run pytest tests/test_lab_extractor.py -v
```

---

## Phase 5: Wearable Data Normalizers

**Goal**: Normalize wearable export data (Garmin, WHOOP, Apple Health) to common schema.

### Tasks

5.1 **Create wearable normalizer base** (`app/services/extraction/wearables/__init__.py`)
```python
from abc import abstractmethod
from app.services.extraction.base import BaseExtractor

class WearableNormalizer(BaseExtractor):
    """Base class for wearable data normalizers."""

    @abstractmethod
    def get_daily_summary(self, data: dict) -> dict:
        """Aggregate to daily summary."""
        pass

    @abstractmethod
    def get_metric_types(self) -> list[str]:
        """Return list of metric types this normalizer produces."""
        pass
```

5.2 **Implement Garmin normalizer** (`app/services/extraction/wearables/garmin.py`)
```python
class GarminNormalizer(WearableNormalizer):
    """Normalize Garmin export data."""

    document_type = "garmin"

    # Garmin CSV column mappings
    COLUMN_MAP = {
        "Date": "date",
        "Calories": "calories_burned",
        "Steps": "steps",
        "Distance": "distance_km",
        "Floors Climbed": "floors_climbed",
        "Minutes Sedentary": "sedentary_minutes",
        "Minutes Lightly Active": "light_active_minutes",
        "Minutes Fairly Active": "moderate_active_minutes",
        "Minutes Very Active": "vigorous_active_minutes",
        "Activity Calories": "activity_calories",
    }

    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract and normalize Garmin data."""
        df = pd.read_csv(file_path)
        df = df.rename(columns=self.COLUMN_MAP)

        # Normalize dates
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Calculate daily summaries
        daily = self.get_daily_summary(df)

        return ExtractionResult(
            success=True,
            data={"daily_metrics": daily, "source": "garmin"},
            confidence=0.95,
            errors=[],
            warnings=[],
        )
```

5.3 **Implement WHOOP normalizer** (`app/services/extraction/wearables/whoop.py`)
```python
class WhoopNormalizer(WearableNormalizer):
    """Normalize WHOOP export data."""

    document_type = "whoop"

    COLUMN_MAP = {
        "Cycle start time": "date",
        "Recovery score %": "recovery_score",
        "Resting heart rate (bpm)": "resting_hr",
        "Heart rate variability (ms)": "hrv",
        "Strain": "strain",
        "Average heart rate (bpm)": "avg_hr",
        "Calories (kcal)": "calories",
    }
```

5.4 **Implement Apple Health normalizer** (`app/services/extraction/wearables/apple_health.py`)
```python
import xml.etree.ElementTree as ET

class AppleHealthNormalizer(WearableNormalizer):
    """Normalize Apple Health export data."""

    document_type = "apple_health"

    # Apple Health record type mappings
    RECORD_TYPE_MAP = {
        "HKQuantityTypeIdentifierStepCount": "steps",
        "HKQuantityTypeIdentifierHeartRate": "heart_rate",
        "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv",
        "HKQuantityTypeIdentifierActiveEnergyBurned": "active_calories",
        "HKQuantityTypeIdentifierBasalEnergyBurned": "basal_calories",
        "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
    }

    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract from Apple Health XML export."""
        tree = ET.parse(file_path)
        root = tree.getroot()

        records = []
        for record in root.findall(".//Record"):
            record_type = record.get("type")
            if record_type in self.RECORD_TYPE_MAP:
                records.append({
                    "type": self.RECORD_TYPE_MAP[record_type],
                    "value": float(record.get("value", 0)),
                    "date": record.get("startDate"),
                    "unit": record.get("unit"),
                })

        # Aggregate to daily summaries
        daily = self._aggregate_daily(records)

        return ExtractionResult(
            success=True,
            data={"daily_metrics": daily, "source": "apple_health"},
            confidence=0.95,
            errors=[],
            warnings=[],
        )
```

5.5 **Implement common health metrics schema**
```python
class DailyHealthMetrics(BaseModel):
    """Normalized daily health metrics."""
    date: date
    source: str

    # Activity
    steps: int | None = None
    calories_burned: int | None = None
    active_minutes: int | None = None
    distance_km: float | None = None

    # Heart
    resting_hr: int | None = None
    avg_hr: int | None = None
    hrv: float | None = None

    # Recovery
    recovery_score: float | None = None
    strain: float | None = None

    # Sleep
    sleep_duration_hours: float | None = None
    sleep_score: float | None = None
```

### Acceptance Criteria Coverage
- AC4: Wearable Data Normalization

### Verification
```bash
uv run pytest tests/test_wearable_normalizers.py -v
```

---

## Phase 6: Testing, Integration & Documentation

**Goal**: Comprehensive tests, integration with MVP database, and documentation.

### Tasks

6.1 **Create test fixtures**
- Anonymized InBody PDFs (3+ samples)
- Sample lab CSVs (Quest, LabCorp formats)
- Sample wearable exports (Garmin, WHOOP, Apple Health)

6.2 **Write unit tests**
- `tests/test_inbody_extractor.py`
- `tests/test_lab_extractor.py`
- `tests/test_wearable_normalizers.py`
- `tests/test_extraction_api.py`
- `tests/test_document_router.py`

6.3 **Write integration tests**
- Full upload → extraction → storage flow
- S3 upload/download
- Celery task execution
- Database storage

6.4 **Implement MVP database integration**
- Write extracted metrics to `healthMetrics` table
- Write lab values to `labValues` table
- Update client health profile

6.5 **Add extraction monitoring**
- Extraction job metrics (success rate, latency)
- Error tracking by document type
- Confidence score distribution

6.6 **Update Docker worker image**
- Add Docling dependencies
- Add system libraries for PDF processing
- Test image build

6.7 **Update documentation**
- API documentation for extraction endpoints
- Supported document formats
- Troubleshooting guide

6.8 **Update Makefile**
```makefile
# Extraction-specific commands
test-extraction:
	uv run pytest tests/test_*extract*.py tests/test_*wearable*.py -v

extract-sample:
	uv run python -m app.scripts.extract_sample $(FILE)
```

### Acceptance Criteria Coverage
- All remaining ACs
- Integration verification

### Verification
```bash
# Full test suite
make test

# Integration test
make up
curl -X POST http://localhost:8000/api/v1/extraction/upload \
  -F "file=@tests/fixtures/sample_inbody.pdf"
# Wait and check results
```

---

## Implementation Order Summary

```
Phase 1: Infrastructure ──────────────────► Base classes, S3, DB schema
    │
    ▼
Phase 2: API & Celery ────────────────────► Upload endpoint, async task
    │
    ▼
Phase 3: InBody Extractor ────────────────► PDF extraction with Docling
    │
    ▼
Phase 4: Lab Results ─────────────────────► Quest, LabCorp, CSV parsers
    │
    ▼
Phase 5: Wearable Normalizers ────────────► Garmin, WHOOP, Apple Health
    │
    ▼
Phase 6: Testing & Integration ───────────► Full test suite, MVP integration
```

---

## Files to Create (Summary)

### Phase 1
- `app/services/extraction/__init__.py`
- `app/services/extraction/base.py`
- `app/services/extraction/router.py`
- `app/services/extraction/storage.py`
- `app/services/extraction/validation.py`
- `app/schemas/extraction.py`
- `scripts/migrations/0002_extraction_tables.sql`

### Phase 2
- `app/api/v1/extraction.py`
- `app/workers/tasks/extraction.py`

### Phase 3
- `app/services/extraction/inbody.py`
- `tests/test_inbody_extractor.py`
- `tests/fixtures/sample_inbody_*.pdf`

### Phase 4
- `app/services/extraction/lab_results.py`
- `tests/test_lab_extractor.py`
- `tests/fixtures/sample_lab_*.csv`

### Phase 5
- `app/services/extraction/wearables/__init__.py`
- `app/services/extraction/wearables/garmin.py`
- `app/services/extraction/wearables/whoop.py`
- `app/services/extraction/wearables/apple_health.py`
- `tests/test_wearable_normalizers.py`
- `tests/fixtures/sample_garmin.csv`
- `tests/fixtures/sample_whoop.csv`
- `tests/fixtures/sample_apple_health.xml`

### Phase 6
- Integration tests
- Documentation updates
- Docker image updates

---

## Test Documents Required

| Document | Format | Source |
|----------|--------|--------|
| InBody 570 scan | PDF | Anonymized sample |
| InBody 770 scan | PDF | Anonymized sample |
| InBody S10 scan | PDF | Anonymized sample |
| Quest Diagnostics results | PDF | Anonymized sample |
| LabCorp results | PDF | Anonymized sample |
| Generic lab CSV | CSV | Generated sample |
| Garmin export | CSV | Generated sample |
| WHOOP export | CSV | Generated sample |
| Apple Health export | XML | Generated sample |

**Note**: Builder should request sample documents or create synthetic test data that matches real format structures.

---

## Estimated Effort

| Phase | Complexity | Notes |
|-------|------------|-------|
| Phase 1 | Medium | S3 integration, schema design |
| Phase 2 | Medium | API + Celery task wiring |
| Phase 3 | High | Docling + pattern matching |
| Phase 4 | High | Multiple format parsers |
| Phase 5 | Medium | Standard CSV/XML parsing |
| Phase 6 | Medium | Testing + integration |

---

**Plan Status**: Ready for approval
**Author**: Architect
**Created**: 2025-01-27
