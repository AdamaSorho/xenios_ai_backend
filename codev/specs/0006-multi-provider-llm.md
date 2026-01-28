# Spec 0006: Multi-Provider LLM Support

## Overview

**What**: Add direct Claude API support alongside OpenRouter, allowing runtime provider selection via request headers.

**Why**:
- **Flexibility**: Clients may prefer direct API for latency, reliability, or cost reasons
- **Fallback**: If OpenRouter is down, Claude direct can serve as backup
- **Cost optimization**: Direct API may have different pricing than OpenRouter
- **Compliance**: Some clients may require direct vendor relationships

**Who**:
- API consumers selecting provider preference
- System administrators configuring defaults
- Developers extending to additional providers

## Goals

### Must Have
1. Claude API direct integration (Anthropic SDK)
2. Runtime provider selection via `X-LLM-Provider` header
3. Configurable default provider via environment variable
4. Provider-specific API key configuration
5. Unified response format regardless of provider
6. Task-based model mapping per provider
7. Graceful fallback when preferred provider fails

### Should Have
- Provider health checks
- Per-provider rate limiting awareness
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
app/services/llm/
├── client.py      # LLMClient - hardcoded to OpenRouter
├── models.py      # TASK_MODELS - model names use OpenRouter format
└── prompts.py     # Prompt builders
```

**Current LLMClient** (`app/services/llm/client.py`):
- Base URL: `https://openrouter.ai/api/v1`
- Auth: `Authorization: Bearer {openrouter_api_key}`
- Format: OpenAI-compatible chat completions API

**Current Model Mapping** (`app/services/llm/models.py`):
```python
TASK_MODELS = {
    "chat": ModelConfig(primary="anthropic/claude-opus-4-20250514", ...),
    "session_summary": ModelConfig(primary="anthropic/claude-opus-4-20250514", ...),
    "intent_classification": ModelConfig(primary="anthropic/claude-sonnet-4-20250514", ...),
    ...
}
```

### Proposed Architecture

```
app/services/llm/
├── providers/
│   ├── __init__.py
│   ├── base.py           # Abstract LLMProvider interface
│   ├── openrouter.py     # OpenRouter implementation
│   └── anthropic.py      # Direct Claude API implementation
├── client.py             # Updated to use provider abstraction
├── models.py             # Extended with per-provider model mappings
└── prompts.py            # Unchanged
```

## Technical Implementation

### Provider Interface

```python
# app/services/llm/providers/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str  # Provider identifier (e.g., "openrouter", "anthropic")

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """Send completion request, return normalized response."""
        pass

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream completion, yield content chunks."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass
```

### Anthropic Provider

```python
# app/services/llm/providers/anthropic.py
import anthropic
from .base import LLMProvider

class AnthropicProvider(LLMProvider):
    """Direct Claude API provider using Anthropic SDK."""

    name = "anthropic"

    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self._api_key = api_key

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """Send completion to Claude API."""
        # Extract system message if present
        system_content = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)

        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_content,
            messages=chat_messages,
        )

        # Normalize to OpenAI-compatible format
        return self._normalize_response(response)

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream completion from Claude API."""
        system_content = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)

        async with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_content,
            messages=chat_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _normalize_response(self, response) -> dict:
        """Convert Anthropic response to OpenAI-compatible format."""
        return {
            "id": response.id,
            "object": "chat.completion",
            "model": response.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content[0].text,
                },
                "finish_reason": response.stop_reason,
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }
```

### OpenRouter Provider (Refactored)

```python
# app/services/llm/providers/openrouter.py
import httpx
from .base import LLMProvider

class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider (OpenAI-compatible)."""

    name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """Send completion to OpenRouter."""
        response = await self._client.post(
            "/chat/completions",
            headers=self._headers(),
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream completion from OpenRouter."""
        async with self._client.stream(
            "POST",
            "/chat/completions",
            headers=self._headers(),
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if content := chunk["choices"][0]["delta"].get("content"):
                        yield content

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://xenios.app",
            "X-Title": "Xenios AI Backend",
            "Content-Type": "application/json",
        }
```

### Model Mapping Per Provider

```python
# app/services/llm/models.py (updated)

# OpenRouter uses "provider/model" format
OPENROUTER_MODELS = {
    "chat": "anthropic/claude-opus-4-20250514",
    "session_summary": "anthropic/claude-opus-4-20250514",
    "insight_generation": "anthropic/claude-opus-4-20250514",
    "intent_classification": "anthropic/claude-sonnet-4-20250514",
    "entity_extraction": "anthropic/claude-sonnet-4-20250514",
    "cue_detection": "openai/gpt-4o-mini",
}

# Anthropic uses direct model names
ANTHROPIC_MODELS = {
    "chat": "claude-opus-4-20250514",
    "session_summary": "claude-opus-4-20250514",
    "insight_generation": "claude-opus-4-20250514",
    "intent_classification": "claude-sonnet-4-20250514",
    "entity_extraction": "claude-sonnet-4-20250514",
    "cue_detection": "claude-sonnet-4-20250514",  # Fallback - no GPT on Anthropic
}

PROVIDER_MODELS = {
    "openrouter": OPENROUTER_MODELS,
    "anthropic": ANTHROPIC_MODELS,
}

def get_model_for_task(task: str, provider: str) -> str:
    """Get the appropriate model name for a task and provider."""
    models = PROVIDER_MODELS.get(provider, OPENROUTER_MODELS)
    return models.get(task, models.get("chat"))
```

### Updated LLMClient

```python
# app/services/llm/client.py (updated)
from .providers.base import LLMProvider
from .providers.openrouter import OpenRouterProvider
from .providers.anthropic import AnthropicProvider
from .models import get_model_for_task, TASK_MODELS

class LLMClient:
    """LLM client with multi-provider support."""

    def __init__(
        self,
        provider: str | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._default_provider = provider or self._settings.llm_default_provider
        self._providers = self._init_providers()

    def _init_providers(self) -> dict[str, LLMProvider]:
        """Initialize available providers."""
        providers = {}

        if self._settings.openrouter_api_key:
            providers["openrouter"] = OpenRouterProvider(
                self._settings.openrouter_api_key
            )

        if self._settings.anthropic_api_key:
            providers["anthropic"] = AnthropicProvider(
                self._settings.anthropic_api_key
            )

        return providers

    def get_provider(self, name: str | None = None) -> LLMProvider:
        """Get provider by name, falling back to default."""
        provider_name = name or self._default_provider

        if provider_name not in self._providers:
            # Fallback to any available provider
            if self._providers:
                provider_name = next(iter(self._providers))
            else:
                raise LLMError("No LLM providers configured")

        return self._providers[provider_name]

    async def complete(
        self,
        task: str,
        messages: list[dict],
        provider: str | None = None,
        use_fallback: bool = True,
    ) -> dict:
        """Complete with optional provider override and fallback."""
        provider_name = provider or self._default_provider
        llm_provider = self.get_provider(provider_name)

        model = get_model_for_task(task, provider_name)
        config = TASK_MODELS.get(task, TASK_MODELS["chat"])

        try:
            return await llm_provider.complete(
                model=model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except Exception as e:
            if use_fallback:
                return await self._try_fallback(task, messages, provider_name, e)
            raise LLMError(f"LLM request failed: {e}") from e

    async def _try_fallback(
        self,
        task: str,
        messages: list[dict],
        failed_provider: str,
        original_error: Exception,
    ) -> dict:
        """Try other providers as fallback."""
        for name, provider in self._providers.items():
            if name == failed_provider:
                continue
            try:
                model = get_model_for_task(task, name)
                config = TASK_MODELS.get(task, TASK_MODELS["chat"])
                return await provider.complete(
                    model=model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
            except Exception:
                continue

        raise LLMError(f"All providers failed. Original: {original_error}")
```

### Request Header Integration

```python
# app/dependencies.py (updated)
from fastapi import Header
from app.services.llm.client import LLMClient

def get_llm_client(
    x_llm_provider: str | None = Header(None, alias="X-LLM-Provider"),
) -> LLMClient:
    """
    Get LLM client with optional provider override.

    Header: X-LLM-Provider: openrouter | anthropic
    """
    valid_providers = ["openrouter", "anthropic"]
    provider = None

    if x_llm_provider and x_llm_provider.lower() in valid_providers:
        provider = x_llm_provider.lower()

    return LLMClient(provider=provider)
```

### Configuration

```python
# app/config.py (updated)
class Settings(BaseSettings):
    # Existing
    openrouter_api_key: str = ""

    # New
    anthropic_api_key: str = ""
    llm_default_provider: str = "openrouter"  # or "anthropic"
```

### Environment Variables

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
LLM_DEFAULT_PROVIDER=openrouter  # Options: openrouter, anthropic
```

## API Changes

### Request Headers

| Header | Values | Default | Description |
|--------|--------|---------|-------------|
| `X-LLM-Provider` | `openrouter`, `anthropic` | From config | Override LLM provider for this request |

### Example Usage

```bash
# Use default provider (from LLM_DEFAULT_PROVIDER)
curl -X POST /api/v1/llm/complete \
  -H "X-API-Key: ..." \
  -d '{"task": "chat", "messages": [...]}'

# Force Anthropic direct
curl -X POST /api/v1/llm/complete \
  -H "X-API-Key: ..." \
  -H "X-LLM-Provider: anthropic" \
  -d '{"task": "chat", "messages": [...]}'

# Force OpenRouter
curl -X POST /api/v1/llm/complete \
  -H "X-API-Key: ..." \
  -H "X-LLM-Provider: openrouter" \
  -d '{"task": "chat", "messages": [...]}'
```

### Response Format

Response format remains unchanged - normalized to OpenAI-compatible format regardless of provider:

```json
{
  "id": "msg_xxx",
  "object": "chat.completion",
  "model": "claude-opus-4-20250514",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "..."
    },
    "finish_reason": "end_turn"
  }],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

## Project Structure Changes

```
app/
├── services/
│   └── llm/
│       ├── __init__.py
│       ├── providers/           # NEW
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract provider interface
│       │   ├── openrouter.py    # OpenRouter implementation
│       │   └── anthropic.py     # Anthropic implementation
│       ├── client.py            # UPDATED - uses providers
│       ├── models.py            # UPDATED - per-provider mappings
│       └── prompts.py           # Unchanged
├── config.py                    # UPDATED - new settings
└── dependencies.py              # UPDATED - header injection
```

## Acceptance Criteria

### AC1: Provider Abstraction
- [ ] Abstract `LLMProvider` base class defined
- [ ] `OpenRouterProvider` implements interface
- [ ] `AnthropicProvider` implements interface
- [ ] Both providers normalize responses to same format

### AC2: Configuration
- [ ] `ANTHROPIC_API_KEY` environment variable supported
- [ ] `LLM_DEFAULT_PROVIDER` environment variable supported
- [ ] Settings validation ensures at least one provider configured
- [ ] Missing provider API key disables that provider gracefully

### AC3: Runtime Selection
- [ ] `X-LLM-Provider` header selects provider
- [ ] Invalid header value falls back to default
- [ ] Missing header uses default provider
- [ ] Header is case-insensitive

### AC4: Fallback Behavior
- [ ] If selected provider fails, tries other providers
- [ ] Fallback can be disabled per-request
- [ ] Error includes which providers were tried

### AC5: Model Mapping
- [ ] Each provider has correct model name format
- [ ] Tasks without Anthropic equivalent use fallback model
- [ ] `cue_detection` task works on both providers

### AC6: Streaming
- [ ] Streaming works with OpenRouter
- [ ] Streaming works with Anthropic
- [ ] Stream chunks are normalized to same format

### AC7: Backward Compatibility
- [ ] Existing API calls work without changes
- [ ] No header = same behavior as before
- [ ] Response format unchanged

## Test Plan

### Unit Tests
- Provider initialization with/without API keys
- Model mapping for each task and provider
- Response normalization for both providers
- Header parsing and validation
- Fallback logic when provider fails

### Integration Tests
- Complete request to OpenRouter
- Complete request to Anthropic
- Streaming request to both providers
- Fallback when primary provider unavailable
- Task-based model selection

### Manual Testing
- Verify response quality from both providers
- Test streaming UX from both providers
- Verify usage tracking accuracy

## Dependencies

- **Spec 0001**: AI Backend Foundation (LLMClient base)
- **anthropic** Python SDK (new dependency)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Different response qualities | Low | Medium | Same models via both routes |
| Rate limiting differences | Medium | Low | Per-provider rate limit awareness |
| Cost tracking complexity | Medium | Medium | Log provider with each request |
| SDK version conflicts | Low | Low | Pin anthropic SDK version |

## Cost Considerations

### Anthropic Direct Pricing (as of Jan 2025)
- Claude Opus: $15/1M input, $75/1M output
- Claude Sonnet: $3/1M input, $15/1M output

### OpenRouter Pricing
- Typically same as direct + small margin
- May offer better rates on some models

**Recommendation**: Default to OpenRouter for cost flexibility, allow Anthropic direct for latency-sensitive or compliance requirements.

## Future Considerations

- Add OpenAI as third provider
- Provider health monitoring dashboard
- Automatic provider selection based on latency
- Cost-based routing (cheapest available)
- Provider-specific rate limit handling

---

**Spec Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-28
