# Spec 0002: Document Extraction Pipeline

## Overview

**What**: Build an automated document extraction pipeline that processes health documents (InBody scans, lab results, wearable exports) and extracts structured data for the Xenios coaching platform.

**Why**: Coaches and clients upload various health documents in different formats (PDFs, CSVs, JSON). Manual data entry is time-consuming and error-prone. Automated extraction enables:
- Instant data availability after upload
- Consistent, validated data extraction
- AI-powered insights from extracted metrics
- 60-second target from upload to structured data (per PRD)

**Who**:
- Coaches uploading client health documents
- Clients uploading their own data
- System processing wearable sync exports

## Goals

### Must Have
1. IBM Docling integration for PDF document extraction
2. InBody scan PDF parser (body composition: weight, body fat %, muscle mass, BMR)
3. Lab results parser (CSV and PDF formats from Quest, LabCorp, generic labs)
4. Wearable data normalizers (Garmin, WHOOP, Apple Health CSV/JSON exports)
5. Celery task for async document processing
6. Extraction results stored in database with source tracking
7. Validation and confidence scoring for extracted values
8. API endpoints for upload and extraction status
9. Webhook/callback on extraction completion

### Should Have
- OCR fallback for scanned/image PDFs
- Multiple InBody model support (570, 770, S10)
- Biomarker reference ranges and flagging (high/low/normal)
- Extraction retry with exponential backoff
- Batch upload support

### Won't Have (MVP)
- Real-time wearable API integrations (handled by MVP frontend)
- Custom document type training
- Multi-language support
- Video/image analysis

## Technical Context

### Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                     Xenios MVP (Next.js)                         │
│                                                                  │
│  POST /api/client/documents/upload ─────────────────────────────┼───┐
│                                                                  │   │
│  Existing tables:                                               │   │
│  - clientDocuments (file metadata)                              │   │
│  - healthMetrics (extracted values go here)                     │   │
│  - labResults, labValues (lab data)                             │   │
└─────────────────────────────────────────────────────────────────┘   │
                                                                       │
                              ┌────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Xenios AI Backend (This Spec)                  │
│                                                                  │
│  POST /api/v1/extraction/upload                                 │
│       → Upload file to S3/R2                                    │
│       → Queue extraction job                                    │
│       → Return job_id                                           │
│                                                                  │
│  GET /api/v1/extraction/status/{job_id}                         │
│       → Return extraction status and results                    │
│                                                                  │
│  Celery Worker (extraction queue):                              │
│       → Download file from S3                                   │
│       → Detect document type                                    │
│       → Route to appropriate extractor                          │
│       → Validate and score results                              │
│       → Store in database                                       │
│       → Notify via webhook                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Document Types and Extractors

| Document Type | Format | Extractor | Output |
|---------------|--------|-----------|--------|
| InBody Scan | PDF | `InBodyExtractor` | Weight, body fat %, SMM, BMR, etc. |
| Lab Results (Quest) | PDF/CSV | `LabResultsExtractor` | Biomarkers with values, units, flags |
| Lab Results (LabCorp) | PDF/CSV | `LabResultsExtractor` | Biomarkers with values, units, flags |
| Lab Results (Generic) | CSV | `GenericLabExtractor` | Biomarkers with values, units |
| Garmin Export | CSV/JSON | `GarminNormalizer` | Heart rate, steps, sleep, etc. |
| WHOOP Export | CSV | `WhoopNormalizer` | HRV, recovery, strain, sleep |
| Apple Health | XML/JSON | `AppleHealthNormalizer` | Various health metrics |
| Nutrition Log | CSV | `NutritionExtractor` | Calories, macros, meals |

### IBM Docling Integration

IBM Docling is a self-hosted document extraction library that provides:
- Table extraction with 97% TEDS accuracy
- Layout-aware text extraction
- PDF parsing without cloud dependencies
- Free (self-hosted)

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Access extracted content
for table in result.document.tables:
    # Process tables (InBody data is in tables)
    pass

for text_block in result.document.texts:
    # Process text (lab results narrative)
    pass
```

### Data Models

#### Extraction Job

```python
class ExtractionJob(BaseModel):
    id: UUID
    client_id: UUID
    coach_id: UUID

    # Source file
    file_url: str
    file_name: str
    file_type: str  # pdf, csv, json, xml
    file_size: int

    # Processing
    document_type: str | None  # inbody, lab_results, garmin, whoop, etc.
    status: ExtractionStatus  # pending, processing, completed, failed
    started_at: datetime | None
    completed_at: datetime | None

    # Results
    extracted_data: dict | None  # Structured extraction results
    confidence_score: float | None  # 0-1 overall confidence
    validation_errors: list[str] | None

    # Metadata
    created_at: datetime
    updated_at: datetime
```

#### Extracted Metrics (InBody Example)

```python
class InBodyExtraction(BaseModel):
    # Identification
    scan_date: date
    device_model: str | None  # InBody 570, 770, S10

    # Body Composition
    weight_kg: float
    weight_confidence: float

    body_fat_percent: float
    body_fat_confidence: float

    skeletal_muscle_mass_kg: float
    smm_confidence: float

    basal_metabolic_rate_kcal: int
    bmr_confidence: float

    # Optional fields (may not be on all scans)
    body_water_percent: float | None
    visceral_fat_level: int | None
    bone_mineral_content_kg: float | None

    # Segmental (if available)
    segmental_lean: dict | None  # {"left_arm": 3.2, "right_arm": 3.4, ...}
    segmental_fat: dict | None
```

#### Extracted Lab Results

```python
class LabResultExtraction(BaseModel):
    # Identification
    lab_provider: str  # Quest, LabCorp, etc.
    collection_date: date
    report_date: date | None

    # Biomarkers
    biomarkers: list[BiomarkerValue]

class BiomarkerValue(BaseModel):
    name: str  # e.g., "LDL Cholesterol"
    code: str | None  # LOINC code if available
    value: float
    unit: str
    reference_range: str | None  # e.g., "<100 mg/dL"
    flag: str | None  # "high", "low", "normal", "critical"
    confidence: float
```

## Technical Implementation

### Project Structure (Additions)

```
app/
├── services/
│   └── extraction/
│       ├── __init__.py
│       ├── base.py              # BaseExtractor class
│       ├── router.py            # Document type routing
│       ├── inbody.py            # InBody PDF extractor
│       ├── lab_results.py       # Lab results extractor
│       ├── wearables/
│       │   ├── __init__.py
│       │   ├── garmin.py        # Garmin CSV/JSON normalizer
│       │   ├── whoop.py         # WHOOP CSV normalizer
│       │   └── apple_health.py  # Apple Health XML normalizer
│       ├── nutrition.py         # Nutrition log extractor
│       └── validation.py        # Value validation and scoring
│
├── workers/
│   └── tasks/
│       └── extraction.py        # Extraction Celery tasks
│
├── api/
│   └── v1/
│       └── extraction.py        # Extraction API endpoints
│
└── schemas/
    └── extraction.py            # Extraction request/response schemas
```

### API Endpoints

```
POST /api/v1/extraction/upload
  - Upload file (multipart/form-data)
  - Optional: document_type hint
  - Returns: { job_id, status: "pending" }

GET /api/v1/extraction/status/{job_id}
  - Returns job status and results when complete

GET /api/v1/extraction/jobs
  - List extraction jobs for client
  - Filters: status, document_type, date_range

POST /api/v1/extraction/reprocess/{job_id}
  - Retry failed extraction

DELETE /api/v1/extraction/{job_id}
  - Cancel pending job or delete results
```

### Extraction Pipeline Flow

```
1. Upload Request
   │
   ├── Validate file (size, type)
   ├── Upload to S3/R2 with presigned URL
   ├── Create ExtractionJob record (status: pending)
   ├── Queue Celery task
   └── Return job_id immediately

2. Celery Worker (extraction queue)
   │
   ├── Download file from S3
   ├── Detect document type (if not provided)
   │   └── Use file extension + content heuristics
   │
   ├── Route to appropriate extractor
   │   ├── InBodyExtractor
   │   ├── LabResultsExtractor
   │   ├── GarminNormalizer
   │   └── etc.
   │
   ├── Extract structured data
   │   └── Docling for PDFs, pandas for CSVs
   │
   ├── Validate extracted values
   │   ├── Type checking
   │   ├── Range validation (weight: 20-500 kg)
   │   ├── Required field checking
   │   └── Assign confidence scores
   │
   ├── Store results
   │   ├── Update ExtractionJob with results
   │   ├── Insert into healthMetrics table
   │   └── Insert into labValues if lab results
   │
   └── Notify completion
       └── Webhook to MVP backend (optional)

3. Polling/Webhook
   │
   └── Client polls status or receives webhook
       └── Results available for display
```

### InBody Extractor Implementation

```python
class InBodyExtractor(BaseExtractor):
    """Extract body composition data from InBody scan PDFs."""

    SUPPORTED_MODELS = ["InBody 570", "InBody 770", "InBody S10"]

    # Field patterns for table extraction
    FIELD_PATTERNS = {
        "weight": r"(?:Body\s*Weight|Weight)\s*[:\s]*(\d+\.?\d*)\s*(kg|lbs?)",
        "body_fat_percent": r"(?:Percent\s*Body\s*Fat|PBF|Body\s*Fat\s*%)\s*[:\s]*(\d+\.?\d*)\s*%?",
        "smm": r"(?:Skeletal\s*Muscle\s*Mass|SMM)\s*[:\s]*(\d+\.?\d*)\s*(kg|lbs?)",
        "bmr": r"(?:Basal\s*Metabolic\s*Rate|BMR)\s*[:\s]*(\d+)\s*(?:kcal)?",
    }

    async def extract(self, file_path: str) -> InBodyExtraction:
        # Use Docling to extract tables and text
        converter = DocumentConverter()
        result = converter.convert(file_path)

        # Find the main results table
        main_table = self._find_results_table(result.document.tables)

        # Extract values with confidence scoring
        extracted = {}
        for field, pattern in self.FIELD_PATTERNS.items():
            value, confidence = self._extract_field(main_table, pattern)
            extracted[field] = value
            extracted[f"{field}_confidence"] = confidence

        # Validate and return
        return self._validate_and_build(extracted)
```

### File Storage Strategy

```python
# Use S3/R2 for file storage
# Files stored with structure: {client_id}/{year}/{month}/{filename}

async def upload_file(
    file: UploadFile,
    client_id: UUID,
) -> str:
    """Upload file to S3 and return URL."""
    settings = get_settings()

    # Generate storage path
    now = datetime.utcnow()
    key = f"extractions/{client_id}/{now.year}/{now.month:02d}/{uuid4()}_{file.filename}"

    # Upload to S3
    s3_client = get_s3_client()
    await s3_client.upload_fileobj(
        file.file,
        settings.s3_bucket,
        key,
        ExtraArgs={"ContentType": file.content_type}
    )

    return f"s3://{settings.s3_bucket}/{key}"
```

### Database Schema Additions

```sql
-- Extraction jobs tracking
CREATE TABLE ai_backend.extraction_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES public.clients(id),
    coach_id UUID NOT NULL REFERENCES public.coaches(id),

    -- Source file
    file_url TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER NOT NULL,

    -- Processing
    document_type VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Results
    extracted_data JSONB,
    confidence_score DECIMAL(3,2),
    validation_errors JSONB,
    error_message TEXT,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_extraction_jobs_client ON ai_backend.extraction_jobs(client_id);
CREATE INDEX idx_extraction_jobs_status ON ai_backend.extraction_jobs(status);

-- Extraction results cache (for quick lookups)
CREATE TABLE ai_backend.extraction_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ai_backend.extraction_jobs(id),
    document_type VARCHAR(50) NOT NULL,
    extraction_date DATE NOT NULL,

    -- Denormalized key metrics for quick access
    metrics JSONB NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Acceptance Criteria

### AC1: IBM Docling Integration
- [ ] Docling installed and configured in worker Docker image
- [ ] Can extract tables from PDF documents
- [ ] Can extract text blocks from PDF documents
- [ ] Handles corrupt/invalid PDFs gracefully

### AC2: InBody Extraction
- [ ] Extracts weight with >95% accuracy on test documents
- [ ] Extracts body fat % with >95% accuracy
- [ ] Extracts skeletal muscle mass with >95% accuracy
- [ ] Extracts BMR with >95% accuracy
- [ ] Handles InBody 570, 770, and S10 formats
- [ ] Returns confidence scores for each field
- [ ] Flags missing or uncertain values

### AC3: Lab Results Extraction
- [ ] Parses Quest Diagnostics PDF format
- [ ] Parses LabCorp PDF format
- [ ] Parses generic CSV lab exports
- [ ] Extracts biomarker name, value, unit, reference range
- [ ] Flags out-of-range values (high/low)
- [ ] Handles common biomarkers (lipid panel, metabolic panel, hormones)

### AC4: Wearable Data Normalization
- [ ] Parses Garmin CSV export format
- [ ] Parses WHOOP CSV export format
- [ ] Parses Apple Health XML export format
- [ ] Normalizes to common schema (healthMetrics)
- [ ] Handles date/timezone conversions correctly
- [ ] Aggregates to daily summaries where appropriate

### AC5: API Endpoints
- [ ] POST /upload accepts multipart file upload
- [ ] Returns job_id within 2 seconds of upload
- [ ] GET /status returns current job status
- [ ] GET /status returns extracted data when complete
- [ ] Proper error responses for invalid files

### AC6: Async Processing
- [ ] Extraction runs in Celery worker (extraction queue)
- [ ] Job status updates visible via API
- [ ] Failed jobs have error messages
- [ ] Retry logic for transient failures
- [ ] Processing completes within 60 seconds for typical documents

### AC7: Data Storage
- [ ] Extracted data stored in extraction_jobs table
- [ ] Results also written to appropriate MVP tables (healthMetrics, labValues)
- [ ] File URLs stored securely
- [ ] Confidence scores recorded for audit

### AC8: Validation
- [ ] Weight validated: 20-500 kg range
- [ ] Body fat validated: 3-60% range
- [ ] Lab values validated against reasonable ranges
- [ ] Dates validated (not in future)
- [ ] Required fields flagged if missing

## Test Plan

### Unit Tests
- InBody extractor with sample PDFs (anonymized)
- Lab results extractor with sample CSVs
- Wearable normalizers with sample exports
- Validation logic for each document type
- Document type detection

### Integration Tests
- Full upload → extraction → storage flow
- API endpoint tests with mock files
- Celery task execution
- S3 upload/download

### Test Documents
- 3+ InBody scan samples (different models)
- 2+ Quest lab result samples
- 2+ LabCorp lab result samples
- Garmin export sample
- WHOOP export sample
- Apple Health export sample

## Dependencies

- **Spec 0001**: AI Backend Foundation (Celery, Redis, database connection)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| InBody format changes | Medium | High | Version detection; multiple parser patterns |
| Docling extraction errors | Medium | Medium | Fallback to regex patterns; manual review queue |
| Large file uploads | Low | Medium | Size limits (50MB); streaming upload |
| Lab format variations | High | Medium | Generic CSV fallback; confidence scoring |
| PHI in extraction logs | Medium | High | Sanitize logs; no PII in error messages |

## Performance Requirements

| Operation | Target | Maximum |
|-----------|--------|---------|
| File upload (10MB) | 5 seconds | 15 seconds |
| InBody extraction | 10 seconds | 30 seconds |
| Lab CSV extraction | 5 seconds | 15 seconds |
| Wearable normalization | 5 seconds | 15 seconds |
| End-to-end (upload to results) | 30 seconds | 60 seconds |

## Open Questions

1. **S3 vs R2**: Which object storage to use? R2 is cheaper, S3 has more tooling
2. **Webhook format**: Define webhook payload schema with MVP team
3. **Retention policy**: How long to keep source files? (Suggest 90 days)

## Future Considerations

- Azure Document Intelligence fallback (if Docling accuracy insufficient)
- Custom model training for unusual formats
- Real-time extraction progress updates via WebSocket
- Batch extraction for bulk imports
- PDF annotation showing extracted regions

---

**Spec Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-27
