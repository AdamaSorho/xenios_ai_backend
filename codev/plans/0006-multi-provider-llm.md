# Plan 0006: Multi-Provider LLM Support

**Spec**: [codev/specs/0006-multi-provider-llm.md](../specs/0006-multi-provider-llm.md)
**Protocol**: SPIDER

## Overview

Implement multi-provider LLM support with direct Claude API alongside OpenRouter, runtime provider selection via headers, and automatic fallback.

## Implementation Phases

### Phase 1: Provider Abstraction Layer

**Goal**: Create the provider interface and directory structure.

**Files to create**:
- `app/services/llm/providers/__init__.py`
- `app/services/llm/providers/base.py`

**Tasks**:
1. Create `providers/` directory under `app/services/llm/`
2. Implement `LLMProvider` abstract base class with:
   - `name` property
   - `complete()` async method
   - `stream()` async method
   - `is_available()` method
3. Export from `__init__.py`

**Acceptance**: Abstract class importable, mypy passes.

---

### Phase 2: OpenRouter Provider Refactor

**Goal**: Extract existing OpenRouter logic into provider class.

**Files to modify**:
- `app/services/llm/providers/openrouter.py` (new)
- `app/services/llm/client.py` (temporary - will fully update in Phase 5)

**Tasks**:
1. Create `OpenRouterProvider` class implementing `LLMProvider`
2. Move HTTP client logic from `LLMClient` to provider
3. Implement `complete()`, `stream()`, `is_available()`
4. Keep response format as-is (already OpenAI-compatible)

**Acceptance**: OpenRouter provider works standalone, existing tests pass.

---

### Phase 3: Anthropic Provider

**Goal**: Implement direct Claude API provider.

**Files to create/modify**:
- `app/services/llm/providers/anthropic.py` (new)
- `pyproject.toml` (add anthropic dependency)

**Tasks**:
1. Add `anthropic>=0.40.0` to dependencies
2. Create `AnthropicProvider` class implementing `LLMProvider`
3. Implement `complete()` with response normalization
4. Implement `stream()` using SDK's streaming
5. Handle system message extraction (Anthropic uses separate `system` param)
6. Normalize response to OpenAI-compatible format

**Acceptance**: Anthropic provider completes requests, response format matches OpenRouter.

---

### Phase 4: Model Mapping

**Goal**: Add per-provider model name mappings.

**Files to modify**:
- `app/services/llm/models.py`

**Tasks**:
1. Add `OPENROUTER_MODELS` dict (existing model names)
2. Add `ANTHROPIC_MODELS` dict (direct model names without provider prefix)
3. Add `PROVIDER_MODELS` mapping provider name -> models dict
4. Add `get_model_for_task(task, provider)` function
5. Handle tasks without Anthropic equivalent (fallback to Sonnet)

**Model mappings**:
```python
# OpenRouter format: "provider/model"
OPENROUTER_MODELS = {
    "chat": "anthropic/claude-opus-4-20250514",
    "cue_detection": "openai/gpt-4o-mini",
    ...
}

# Anthropic format: "model" only
ANTHROPIC_MODELS = {
    "chat": "claude-opus-4-20250514",
    "cue_detection": "claude-sonnet-4-20250514",  # No GPT on Anthropic
    ...
}
```

**Acceptance**: `get_model_for_task()` returns correct model for each provider.

---

### Phase 5: Configuration Updates

**Goal**: Add new settings and update LLMClient.

**Files to modify**:
- `app/config.py`
- `.env.example`
- `infrastructure/ansible/templates/env.j2`

**Tasks**:
1. Add `anthropic_api_key: str = ""` to Settings
2. Add `llm_default_provider: str = "openrouter"` to Settings
3. Update `.env.example` with new variables
4. Update Ansible env template

**Acceptance**: Settings load correctly, validation works.

---

### Phase 6: LLMClient Refactor

**Goal**: Update LLMClient to use provider abstraction.

**Files to modify**:
- `app/services/llm/client.py`

**Tasks**:
1. Add `_init_providers()` to initialize available providers
2. Add `get_provider(name)` method
3. Update `complete()` to use provider and handle fallback
4. Update `stream()` to use provider
5. Keep `complete_with_json()` working (uses complete internally)
6. Add provider parameter to methods

**Key changes**:
```python
def __init__(self, provider: str | None = None):
    self._default_provider = provider or settings.llm_default_provider
    self._providers = self._init_providers()

async def complete(self, task, messages, provider=None, use_fallback=True):
    provider_name = provider or self._default_provider
    llm_provider = self.get_provider(provider_name)
    model = get_model_for_task(task, provider_name)
    # ... rest of logic
```

**Acceptance**: All existing LLM calls work, provider selection works.

---

### Phase 7: Header Integration

**Goal**: Add X-LLM-Provider header support.

**Files to modify**:
- `app/dependencies.py`
- `app/api/v1/llm.py`
- `app/api/v1/chat.py`

**Tasks**:
1. Create `get_llm_client()` dependency with header extraction
2. Update LLM endpoints to use dependency
3. Update chat endpoints to use dependency
4. Validate header value (openrouter, anthropic, or None)

**Header handling**:
```python
def get_llm_client(
    x_llm_provider: str | None = Header(None, alias="X-LLM-Provider"),
) -> LLMClient:
    valid = ["openrouter", "anthropic"]
    provider = x_llm_provider.lower() if x_llm_provider in valid else None
    return LLMClient(provider=provider)
```

**Acceptance**: Header changes provider, invalid header falls back to default.

---

### Phase 8: Testing

**Goal**: Add comprehensive tests.

**Files to create**:
- `tests/unit/services/llm/test_providers.py`
- `tests/unit/services/llm/test_models.py`
- `tests/integration/test_llm_providers.py`

**Unit tests**:
1. Provider initialization with/without API keys
2. Model mapping for all tasks and providers
3. Response normalization
4. Header parsing
5. Fallback logic

**Integration tests** (with mocked API):
1. OpenRouter complete/stream
2. Anthropic complete/stream
3. Provider fallback on failure

**Acceptance**: All tests pass, coverage adequate.

---

## File Change Summary

| File | Action |
|------|--------|
| `app/services/llm/providers/__init__.py` | Create |
| `app/services/llm/providers/base.py` | Create |
| `app/services/llm/providers/openrouter.py` | Create |
| `app/services/llm/providers/anthropic.py` | Create |
| `app/services/llm/client.py` | Modify |
| `app/services/llm/models.py` | Modify |
| `app/config.py` | Modify |
| `app/dependencies.py` | Modify |
| `app/api/v1/llm.py` | Modify |
| `app/api/v1/chat.py` | Modify |
| `.env.example` | Modify |
| `infrastructure/ansible/templates/env.j2` | Modify |
| `pyproject.toml` | Modify |
| `tests/unit/services/llm/test_providers.py` | Create |
| `tests/unit/services/llm/test_models.py` | Create |
| `tests/integration/test_llm_providers.py` | Create |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing LLM calls | Phase 6 maintains backward compatibility |
| Anthropic SDK version issues | Pin specific version |
| Different error formats | Normalize in provider layer |

## Estimated Effort

- Phases 1-4: Provider layer and models (~100 lines)
- Phases 5-6: Config and client refactor (~150 lines)
- Phase 7: Header integration (~30 lines)
- Phase 8: Tests (~200 lines)

**Total**: ~480 lines of code

---

**Plan Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-28
