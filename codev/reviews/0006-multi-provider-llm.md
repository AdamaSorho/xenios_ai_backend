# Review 0006: Multi-Provider LLM Support

**Spec**: [codev/specs/0006-multi-provider-llm.md](../specs/0006-multi-provider-llm.md)
**Plan**: [codev/plans/0006-multi-provider-llm.md](../plans/0006-multi-provider-llm.md)
**Status**: Complete
**Author**: Builder 0006

## Implementation Summary

Successfully implemented multi-provider LLM support allowing runtime selection between OpenRouter and direct Anthropic API. The implementation follows the Provider abstraction pattern with normalized responses.

### Key Changes

1. **Provider Abstraction Layer** (`app/services/llm/providers/`)
   - Abstract `LLMProvider` base class defining the interface
   - `OpenRouterProvider` - refactored from existing client
   - `AnthropicProvider` - new direct Claude API support using official SDK

2. **Model Mapping** (`app/services/llm/models.py`)
   - Per-provider model name mappings (OpenRouter uses `provider/model` format)
   - `cue_detection` task maps to Claude Sonnet on Anthropic (no GPT available)
   - New `get_model_for_task(task, provider)` function

3. **Configuration** (`app/config.py`)
   - Added `anthropic_api_key` setting
   - Added `llm_default_provider` setting (defaults to "openrouter")

4. **Header Integration** (`app/dependencies.py`)
   - `X-LLM-Provider` header for per-request provider selection
   - Case-insensitive, falls back to default for invalid values
   - `LLMClientDep` type alias for dependency injection

5. **LLMClient Refactor** (`app/services/llm/client.py`)
   - Multi-provider initialization based on configured API keys
   - Automatic fallback when primary provider fails
   - Provider selection via constructor or method parameters

## Acceptance Criteria Verification

| AC | Requirement | Status |
|----|-------------|--------|
| AC1 | Provider abstraction with normalized responses | ✅ Complete |
| AC2 | Configuration (API keys, default provider) | ✅ Complete |
| AC3 | Runtime selection via X-LLM-Provider header | ✅ Complete |
| AC4 | Fallback when provider fails | ✅ Complete |
| AC5 | Per-provider model mappings | ✅ Complete |
| AC6 | Streaming for both providers | ✅ Complete |
| AC7 | Backward compatibility | ✅ Complete |

## Test Coverage

Created comprehensive tests in:
- `tests/test_llm_models.py` - Extended with provider model mapping tests
- `tests/test_llm_providers.py` - New tests for providers and client

Test categories:
- Provider initialization with/without API keys
- Model mapping for each task and provider
- Response normalization
- Header parsing (case-insensitive, whitespace handling)
- Fallback logic when provider fails
- LLMClient dependency injection

**Total: 41 tests passing**

## External Review Summary

### Claude Review (impl-review)
**Verdict**: COMMENT
**Confidence**: HIGH

Key points raised:
1. **Fallback semantics clarification** - Model-level vs provider-level fallback interaction
2. **Streaming contract** - Both providers now yield OpenAI-compatible JSON strings
3. **Empty response handling** - Already implemented with safe iteration
4. **Invalid header behavior** - Silent fallback is intentional design choice

All concerns were either already addressed in implementation or are acceptable design decisions.

## Lessons Learned

### What Went Well
1. **Provider pattern** worked cleanly for abstraction
2. **Response normalization** to OpenAI format simplifies client code
3. **Dependency injection** for provider selection is clean FastAPI pattern
4. **Anthropic SDK** is straightforward to integrate

### Areas for Future Improvement
1. **Provider health checks** (Should Have) not implemented
2. **Usage tracking by provider** (Should Have) not implemented
3. **Rate limiting awareness** (Should Have) not implemented
4. Consider caching provider instances instead of creating per-request

### Technical Debt
- None introduced; existing fallback mechanism (model-level) still works alongside new provider-level fallback

## Files Changed

| File | Change |
|------|--------|
| `app/services/llm/providers/__init__.py` | New - exports |
| `app/services/llm/providers/base.py` | New - abstract base |
| `app/services/llm/providers/openrouter.py` | New - OpenRouter impl |
| `app/services/llm/providers/anthropic.py` | New - Anthropic impl |
| `app/services/llm/client.py` | Modified - multi-provider |
| `app/services/llm/models.py` | Modified - per-provider mappings |
| `app/services/llm/__init__.py` | Modified - exports |
| `app/config.py` | Modified - new settings |
| `app/dependencies.py` | Modified - LLM client dep |
| `app/api/v1/llm.py` | Modified - use dep |
| `app/api/v1/chat.py` | Modified - use dep |
| `app/services/rag/chat.py` | Modified - accept LLM client |
| `.env.example` | Modified - new vars |
| `infrastructure/ansible/templates/env.j2` | Modified - new vars |
| `pyproject.toml` | Modified - anthropic dep |
| `tests/test_llm_models.py` | Modified - provider tests |
| `tests/test_llm_providers.py` | New - provider tests |

## Conclusion

The implementation meets all Must Have requirements from the spec. The Should Have items (health checks, rate limiting, usage tracking) were explicitly scoped out for MVP per the spec. The code is clean, tested, and maintains backward compatibility.

---

**Review Status**: Complete
**Date**: 2026-01-28
