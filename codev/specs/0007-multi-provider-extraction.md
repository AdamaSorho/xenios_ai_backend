# Spec 0007: Multi-Provider Document Extraction

## Overview

**What**: Add Reducto.ai as an alternative extraction backend alongside Docling, with runtime provider selection.

**Why**:
- **Flexibility**: Different providers excel at different document types
- **Fallback**: If Docling fails, Reducto can serve as backup
- **Accuracy**: Reducto.ai offers cloud-based ML extraction that may outperform local Docling on complex documents
- **Cost/Performance tradeoff**: Docling is free/local, Reducto is cloud-based with better accuracy

**Who**:
- API consumers selecting extraction provider preference
- System administrators configuring defaults
- Documents with complex layouts that benefit from cloud ML

## Goals

### Must Have
1. Reducto.ai API integration for PDF extraction
2. Runtime provider selection via `X-Extraction-Provider` header
3. Configurable default provider via environment variable
4. Provider-specific API key configuration
5. Unified extraction result format regardless of provider
6. Graceful fallback when preferred provider fails

### Should Have
- Provider health checks
- Per-provider timeout configuration
- Usage tracking by provider
- Provider-specific error handling

### Won't Have (MVP)
- UI for provider selection (header-based only)
- Dynamic provider addition without code changes
- Load balancing across providers
- Cost-based routing

## Technical Context

### Current Architecture

```
app/services/extraction/
├── base.py           # BaseExtractor abstract class, ExtractionResult
├── inbody.py         # InBodyExtractor - uses Docling directly
├── lab_results.py    # LabResultsExtractor - uses Docling directly
├── router.py         # DocumentRouter - routes to extractors
├── storage.py        # S3 storage service
└── validation.py     # Validation ranges and flagging
```

**Current Docling Usage** (in extractors):
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert(file_path)
text_content = result.document.export_to_markdown()
```

### Proposed Architecture

```
app/services/extraction/
├── providers/
│   ├── __init__.py
│   ├── base.py           # Abstract DocumentProvider interface
│   ├── docling.py        # Docling provider (local)
│   └── reducto.py        # Reducto.ai provider (cloud)
├── base.py               # BaseExtractor (unchanged)
├── inbody.py             # Updated to use provider abstraction
├── lab_results.py        # Updated to use provider abstraction
├── router.py             # Unchanged
├── storage.py            # Unchanged
└── validation.py         # Unchanged
```

## Technical Implementation

### Document Provider Interface

```python
# app/services/extraction/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DocumentContent:
    """Normalized document extraction result."""
    text: str                          # Full text content
    markdown: str                      # Markdown formatted content
    tables: list[dict]                 # Extracted tables as dicts
    metadata: dict                     # Document metadata
    pages: int                         # Page count
    provider: str                      # Provider that extracted this


class DocumentProvider(ABC):
    """Abstract base class for document extraction providers."""

    name: str  # Provider identifier

    @abstractmethod
    async def extract(self, file_path: Path) -> DocumentContent:
        """Extract content from document file."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass

    @abstractmethod
    def supports_file_type(self, extension: str) -> bool:
        """Check if provider supports the file type."""
        pass
```

### Docling Provider

```python
# app/services/extraction/providers/docling.py
from pathlib import Path
from docling.document_converter import DocumentConverter
from .base import DocumentProvider, DocumentContent

class DoclingProvider(DocumentProvider):
    """Local Docling-based document extraction."""

    name = "docling"

    def __init__(self):
        self._converter = None

    @property
    def converter(self) -> DocumentConverter:
        """Lazy-load converter (heavy import)."""
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter

    async def extract(self, file_path: Path) -> DocumentContent:
        """Extract using Docling."""
        import asyncio

        # Docling is sync, run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._extract_sync,
            file_path,
        )
        return result

    def _extract_sync(self, file_path: Path) -> DocumentContent:
        """Synchronous extraction."""
        result = self.converter.convert(str(file_path))
        doc = result.document

        # Extract text content
        text_parts = []
        for item in doc.iterate_items():
            if hasattr(item, "text"):
                text_parts.append(item.text)

        # Extract tables
        tables = []
        for table in doc.tables:
            tables.append(table.export_to_dataframe().to_dict())

        return DocumentContent(
            text="\n".join(text_parts),
            markdown=doc.export_to_markdown(),
            tables=tables,
            metadata={
                "source": str(file_path),
                "format": file_path.suffix,
            },
            pages=len(doc.pages) if hasattr(doc, "pages") else 1,
            provider=self.name,
        )

    def is_available(self) -> bool:
        """Docling is always available (local)."""
        return True

    def supports_file_type(self, extension: str) -> bool:
        """Docling supports PDFs and images."""
        return extension.lower() in [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]
```

### Reducto Provider

```python
# app/services/extraction/providers/reducto.py
import httpx
from pathlib import Path
from .base import DocumentProvider, DocumentContent

class ReductoProvider(DocumentProvider):
    """Reducto.ai cloud-based document extraction."""

    name = "reducto"
    BASE_URL = "https://api.reducto.ai"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def extract(self, file_path: Path) -> DocumentContent:
        """Extract using Reducto.ai API."""
        # Upload file and extract
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/pdf")}
            response = await self._client.post(
                "/v1/parse",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files=files,
                data={
                    "output_format": "markdown",
                    "extract_tables": "true",
                },
            )
            response.raise_for_status()
            data = response.json()

        return DocumentContent(
            text=data.get("text", ""),
            markdown=data.get("markdown", data.get("text", "")),
            tables=data.get("tables", []),
            metadata={
                "source": str(file_path),
                "format": file_path.suffix,
                "reducto_job_id": data.get("job_id"),
            },
            pages=data.get("page_count", 1),
            provider=self.name,
        )

    def is_available(self) -> bool:
        """Available if API key is configured."""
        return bool(self._api_key)

    def supports_file_type(self, extension: str) -> bool:
        """Reducto supports PDFs."""
        return extension.lower() in [".pdf"]

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()
```

### Provider Manager

```python
# app/services/extraction/providers/__init__.py
from app.config import get_settings
from .base import DocumentProvider, DocumentContent
from .docling import DoclingProvider
from .reducto import ReductoProvider

_providers: dict[str, DocumentProvider] = {}

def init_providers() -> dict[str, DocumentProvider]:
    """Initialize available document providers."""
    global _providers
    if _providers:
        return _providers

    settings = get_settings()

    # Docling is always available (local)
    _providers["docling"] = DoclingProvider()

    # Reducto requires API key
    if settings.reducto_api_key:
        _providers["reducto"] = ReductoProvider(settings.reducto_api_key)

    return _providers

def get_provider(name: str | None = None) -> DocumentProvider:
    """Get provider by name, falling back to default."""
    providers = init_providers()
    settings = get_settings()

    provider_name = name or settings.extraction_default_provider

    if provider_name not in providers:
        # Fallback to any available provider
        if providers:
            provider_name = "docling"  # Docling as ultimate fallback
        else:
            raise RuntimeError("No document providers available")

    return providers[provider_name]

def get_available_providers() -> list[str]:
    """Get list of available provider names."""
    return list(init_providers().keys())
```

### Updated Extractors

```python
# app/services/extraction/inbody.py (updated extract method)
from app.services.extraction.providers import get_provider, DocumentContent

class InBodyExtractor(BaseExtractor):
    async def extract(
        self,
        file_path: Path,
        provider: str | None = None,
    ) -> ExtractionResult:
        """Extract InBody data using specified provider."""
        start_time = time.time()

        try:
            # Get document provider
            doc_provider = get_provider(provider)

            # Extract document content
            content: DocumentContent = await doc_provider.extract(file_path)

            # Parse InBody-specific data from content
            extracted_data = self._parse_inbody_content(content)

            # Validate
            validation_result = self.validate(extracted_data)

            return ExtractionResult(
                success=True,
                data=extracted_data,
                confidence=self._calculate_confidence(extracted_data),
                errors=validation_result.errors,
                warnings=validation_result.warnings,
                extraction_time_ms=int((time.time() - start_time) * 1000),
                metadata={"provider": content.provider},
            )

        except Exception as e:
            # Try fallback provider if available
            if provider and provider != "docling":
                return await self.extract(file_path, provider="docling")
            raise
```

### Configuration

```python
# app/config.py (additions)
class Settings(BaseSettings):
    # Existing extraction settings
    s3_bucket: str = "xenios-extractions"
    extraction_max_file_size_mb: int = 50

    # New provider settings
    reducto_api_key: str = ""
    extraction_default_provider: str = "docling"  # docling | reducto
```

### Request Header Integration

```python
# app/api/v1/extraction.py (updated)
from fastapi import Header

@router.post("/upload")
async def upload_document(
    file: UploadFile,
    request: ExtractionUploadRequest,
    x_extraction_provider: str | None = Header(None, alias="X-Extraction-Provider"),
    # ... other params
):
    """Upload document for extraction."""
    valid_providers = ["docling", "reducto"]
    provider = None
    if x_extraction_provider and x_extraction_provider.lower() in valid_providers:
        provider = x_extraction_provider.lower()

    # Pass provider to task
    task = process_extraction.delay(
        job_id=str(job_id),
        provider=provider,
    )
```

### Environment Variables

```bash
# .env
REDUCTO_API_KEY=rsk_xxx                    # Reducto.ai API key
EXTRACTION_DEFAULT_PROVIDER=docling        # Options: docling, reducto
```

## API Changes

### Request Headers

| Header | Values | Default | Description |
|--------|--------|---------|-------------|
| `X-Extraction-Provider` | `docling`, `reducto` | From config | Override extraction provider |

### Example Usage

```bash
# Use default provider (Docling)
curl -X POST /api/v1/extraction/upload \
  -H "X-API-Key: ..." \
  -F "file=@inbody.pdf"

# Force Reducto.ai
curl -X POST /api/v1/extraction/upload \
  -H "X-API-Key: ..." \
  -H "X-Extraction-Provider: reducto" \
  -F "file=@inbody.pdf"
```

### Response Changes

Add provider info to extraction response:

```json
{
  "job_id": "xxx",
  "status": "completed",
  "extracted_data": {...},
  "metadata": {
    "provider": "reducto",
    "extraction_time_ms": 1234
  }
}
```

## Project Structure Changes

```
app/services/extraction/
├── providers/               # NEW
│   ├── __init__.py          # Provider manager
│   ├── base.py              # Abstract interface
│   ├── docling.py           # Docling implementation
│   └── reducto.py           # Reducto implementation
├── base.py                  # Unchanged
├── inbody.py                # MODIFY - use provider abstraction
├── lab_results.py           # MODIFY - use provider abstraction
├── router.py                # Unchanged
├── storage.py               # Unchanged
└── validation.py            # Unchanged
```

## Acceptance Criteria

### AC1: Provider Abstraction
- [ ] Abstract `DocumentProvider` base class defined
- [ ] `DoclingProvider` implements interface
- [ ] `ReductoProvider` implements interface
- [ ] Both providers return normalized `DocumentContent`

### AC2: Configuration
- [ ] `REDUCTO_API_KEY` environment variable supported
- [ ] `EXTRACTION_DEFAULT_PROVIDER` environment variable supported
- [ ] Missing Reducto API key disables that provider gracefully

### AC3: Runtime Selection
- [ ] `X-Extraction-Provider` header selects provider
- [ ] Invalid header value falls back to default
- [ ] Missing header uses default provider

### AC4: Fallback Behavior
- [ ] If selected provider fails, tries Docling as fallback
- [ ] Error includes which provider was attempted

### AC5: Extractor Updates
- [ ] InBodyExtractor uses provider abstraction
- [ ] LabResultsExtractor uses provider abstraction
- [ ] Extraction results include provider metadata

### AC6: Backward Compatibility
- [ ] Existing API calls work without changes
- [ ] No header = same behavior as before (Docling)

## Test Plan

### Unit Tests
- Provider initialization with/without API keys
- DocumentContent normalization
- Header parsing and validation
- Fallback logic when provider fails

### Integration Tests
- Docling extraction on sample PDF
- Reducto extraction on sample PDF (mocked)
- Fallback when Reducto unavailable

### Manual Testing
- Compare extraction quality between providers
- Verify extraction accuracy on InBody PDFs
- Verify extraction accuracy on lab results

## Dependencies

- **Spec 0002**: Document Extraction Pipeline (existing extractors)
- **httpx**: Already used, for Reducto API calls

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Reducto API latency | Medium | Medium | Timeout config, async processing |
| Different extraction quality | Medium | Medium | Validation layer catches issues |
| Reducto API costs | Low | Low | Default to free Docling |
| Reducto API changes | Low | Medium | Version pin, adapter pattern |

## Cost Considerations

### Docling
- Free (local processing)
- CPU-intensive
- ~2-5 seconds per document

### Reducto.ai
- Pay-per-page pricing
- Cloud processing (faster for complex docs)
- ~1-3 seconds per document

**Recommendation**: Default to Docling for cost savings, use Reducto for complex documents or when accuracy is critical.

## Future Considerations

- Add more providers (AWS Textract, Google Document AI)
- Automatic provider selection based on document complexity
- Quality scoring to choose best provider per document type
- Caching extracted content across providers

---

**Spec Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-28
