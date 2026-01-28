# Plan 0007: Multi-Provider Document Extraction

**Spec**: [codev/specs/0007-multi-provider-extraction.md](../specs/0007-multi-provider-extraction.md)
**Protocol**: SPIDER

## Overview

Implement multi-provider document extraction with Reducto.ai alongside Docling, runtime provider selection via headers, and automatic fallback.

## Implementation Phases

### Phase 1: Provider Abstraction Layer

**Goal**: Create the provider interface and directory structure.

**Files to create**:
- `app/services/extraction/providers/__init__.py`
- `app/services/extraction/providers/base.py`

**Tasks**:
1. Create `providers/` directory under `app/services/extraction/`
2. Define `DocumentContent` dataclass with:
   - `text`: Full text content
   - `markdown`: Markdown formatted content
   - `tables`: List of extracted tables
   - `metadata`: Document metadata
   - `pages`: Page count
   - `provider`: Provider name
3. Implement `DocumentProvider` abstract base class with:
   - `name` property
   - `extract()` async method
   - `is_available()` method
   - `supports_file_type()` method

**Acceptance**: Abstract class importable, dataclass works.

---

### Phase 2: Docling Provider

**Goal**: Extract existing Docling logic into provider class.

**Files to create**:
- `app/services/extraction/providers/docling.py`

**Tasks**:
1. Create `DoclingProvider` class implementing `DocumentProvider`
2. Implement lazy-loaded `DocumentConverter` (heavy import)
3. Implement `extract()` with thread pool executor (Docling is sync)
4. Extract text, markdown, and tables from Docling result
5. Return normalized `DocumentContent`

**Key implementation**:
```python
async def extract(self, file_path: Path) -> DocumentContent:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self._extract_sync, file_path)
```

**Acceptance**: Docling provider extracts PDFs correctly.

---

### Phase 3: Reducto Provider

**Goal**: Implement Reducto.ai cloud provider.

**Files to create**:
- `app/services/extraction/providers/reducto.py`

**Tasks**:
1. Create `ReductoProvider` class implementing `DocumentProvider`
2. Use `httpx.AsyncClient` for API calls
3. Implement file upload to `/v1/parse` endpoint
4. Parse Reducto response and normalize to `DocumentContent`
5. Handle API errors gracefully

**Reducto API**:
```
POST https://api.reducto.ai/v1/parse
Authorization: Bearer {api_key}
Content-Type: multipart/form-data

file: <binary>
output_format: markdown
extract_tables: true
```

**Acceptance**: Reducto provider extracts PDFs via API.

---

### Phase 4: Provider Manager

**Goal**: Create provider initialization and selection logic.

**Files to modify**:
- `app/services/extraction/providers/__init__.py`

**Tasks**:
1. Implement `init_providers()` to initialize available providers
2. Implement `get_provider(name)` with fallback logic
3. Implement `get_available_providers()` for listing
4. Handle missing Reducto API key gracefully

**Key logic**:
```python
def get_provider(name: str | None = None) -> DocumentProvider:
    providers = init_providers()
    provider_name = name or settings.extraction_default_provider
    if provider_name not in providers:
        provider_name = "docling"  # Ultimate fallback
    return providers[provider_name]
```

**Acceptance**: Provider manager initializes and selects correctly.

---

### Phase 5: Configuration Updates

**Goal**: Add new settings for Reducto.

**Files to modify**:
- `app/config.py`
- `.env.example`
- `infrastructure/ansible/templates/env.j2`

**Tasks**:
1. Add `reducto_api_key: str = ""` to Settings
2. Add `extraction_default_provider: str = "docling"` to Settings
3. Update `.env.example` with new variables
4. Update Ansible env template

**Acceptance**: Settings load correctly.

---

### Phase 6: Update Extractors

**Goal**: Refactor extractors to use provider abstraction.

**Files to modify**:
- `app/services/extraction/inbody.py`
- `app/services/extraction/lab_results.py`

**Tasks**:
1. Add `provider` parameter to `extract()` method signature
2. Replace direct Docling calls with `get_provider(provider).extract()`
3. Add fallback logic if provider fails
4. Include provider name in extraction result metadata

**Key change**:
```python
async def extract(self, file_path: Path, provider: str | None = None) -> ExtractionResult:
    doc_provider = get_provider(provider)
    content = await doc_provider.extract(file_path)
    # ... parse content as before
```

**Acceptance**: Extractors work with both providers.

---

### Phase 7: Update Celery Task

**Goal**: Pass provider through extraction pipeline.

**Files to modify**:
- `app/workers/tasks/extraction.py`

**Tasks**:
1. Add `provider` parameter to `process_extraction` task
2. Pass provider to extractor's `extract()` method
3. Log provider used in extraction

**Acceptance**: Provider selection flows through Celery task.

---

### Phase 8: Header Integration

**Goal**: Add X-Extraction-Provider header support.

**Files to modify**:
- `app/api/v1/extraction.py`
- `app/schemas/extraction.py`

**Tasks**:
1. Add header extraction to upload endpoint
2. Validate header value (docling, reducto, or None)
3. Pass provider to Celery task
4. Add provider to job response metadata
5. Update schema to include provider in response

**Acceptance**: Header changes provider, response shows provider used.

---

### Phase 9: Testing

**Goal**: Add comprehensive tests.

**Files to create**:
- `tests/unit/services/extraction/test_providers.py`
- `tests/integration/test_extraction_providers.py`

**Unit tests**:
1. Provider initialization with/without API keys
2. DocumentContent normalization
3. Header parsing
4. Fallback logic

**Integration tests** (with mocked Reducto API):
1. Docling extraction on sample PDF
2. Reducto extraction (mocked)
3. Fallback when Reducto unavailable

**Acceptance**: All tests pass.

---

## File Change Summary

| File | Action |
|------|--------|
| `app/services/extraction/providers/__init__.py` | Create |
| `app/services/extraction/providers/base.py` | Create |
| `app/services/extraction/providers/docling.py` | Create |
| `app/services/extraction/providers/reducto.py` | Create |
| `app/services/extraction/inbody.py` | Modify |
| `app/services/extraction/lab_results.py` | Modify |
| `app/workers/tasks/extraction.py` | Modify |
| `app/api/v1/extraction.py` | Modify |
| `app/schemas/extraction.py` | Modify |
| `app/config.py` | Modify |
| `.env.example` | Modify |
| `infrastructure/ansible/templates/env.j2` | Modify |
| `tests/unit/services/extraction/test_providers.py` | Create |
| `tests/integration/test_extraction_providers.py` | Create |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing extraction | Phase 6 maintains backward compatibility |
| Reducto API latency | Async processing, configurable timeout |
| Different extraction formats | Normalize in provider layer |

## Estimated Effort

- Phases 1-4: Provider layer (~200 lines)
- Phases 5-6: Config and extractor updates (~100 lines)
- Phases 7-8: Task and API updates (~50 lines)
- Phase 9: Tests (~150 lines)

**Total**: ~500 lines of code

---

**Plan Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-28
