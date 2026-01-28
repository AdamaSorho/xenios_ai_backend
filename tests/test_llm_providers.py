"""Tests for multi-provider LLM support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm.client import LLMClient, LLMError
from app.services.llm.providers.anthropic import AnthropicProvider
from app.services.llm.providers.base import LLMProvider
from app.services.llm.providers.openrouter import OpenRouterProvider


class TestLLMProviderBase:
    """Tests for the LLMProvider abstract base class."""

    def test_provider_is_abstract(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_provider_requires_complete(self):
        """Test that subclasses must implement complete."""

        class IncompleteProvider(LLMProvider):
            name = "incomplete"

            async def stream(self, model, messages, temperature=0.7, max_tokens=4096):
                pass

            def is_available(self):
                return True

        with pytest.raises(TypeError):
            IncompleteProvider()


class TestOpenRouterProvider:
    """Tests for the OpenRouter provider implementation."""

    def test_provider_name(self):
        """Test that provider has correct name."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.name == "openrouter"

    def test_is_available_with_key(self):
        """Test is_available returns True when API key is set."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.is_available() is True

    def test_is_available_without_key(self):
        """Test is_available returns False when API key is empty."""
        provider = OpenRouterProvider(api_key="")
        assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_complete_builds_correct_request(self):
        """Test that complete builds the correct HTTP request."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "test-id",
            "choices": [{"message": {"content": "test response"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await provider.complete(
                model="anthropic/claude-opus-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=1000,
            )

            assert result["id"] == "test-id"


class TestAnthropicProvider:
    """Tests for the Anthropic provider implementation."""

    def test_provider_name(self):
        """Test that provider has correct name."""
        with patch("anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="test-key")
            assert provider.name == "anthropic"

    def test_is_available_with_key(self):
        """Test is_available returns True when API key is set."""
        with patch("anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="test-key")
            assert provider.is_available() is True

    def test_is_available_without_key(self):
        """Test is_available returns False when API key is empty."""
        with patch("anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="")
            assert provider.is_available() is False

    def test_normalize_response(self):
        """Test that Anthropic responses are normalized to OpenAI format."""
        with patch("anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="test-key")

            # Create a mock Anthropic response
            mock_response = MagicMock()
            mock_response.id = "msg_123"
            mock_response.model = "claude-opus-4-20250514"
            mock_response.stop_reason = "end_turn"
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50

            mock_content_block = MagicMock()
            mock_content_block.text = "Test response"
            mock_response.content = [mock_content_block]

            normalized = provider._normalize_response(mock_response)

            assert normalized["id"] == "msg_123"
            assert normalized["object"] == "chat.completion"
            assert normalized["model"] == "claude-opus-4-20250514"
            assert normalized["choices"][0]["message"]["role"] == "assistant"
            assert normalized["choices"][0]["message"]["content"] == "Test response"
            assert normalized["usage"]["prompt_tokens"] == 100
            assert normalized["usage"]["completion_tokens"] == 50
            assert normalized["usage"]["total_tokens"] == 150


class TestLLMClient:
    """Tests for the multi-provider LLM client."""

    def test_init_with_default_provider(self):
        """Test client initialization with default provider."""
        with patch("app.services.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.openrouter_api_key = "test-openrouter-key"
            mock_settings.return_value.anthropic_api_key = ""
            mock_settings.return_value.llm_default_provider = "openrouter"

            client = LLMClient()

            assert client._default_provider == "openrouter"
            assert "openrouter" in client.available_providers
            assert "anthropic" not in client.available_providers

    def test_init_with_provider_override(self):
        """Test client initialization with provider override."""
        with patch("app.services.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.openrouter_api_key = "test-key"
            mock_settings.return_value.anthropic_api_key = "test-key"
            mock_settings.return_value.llm_default_provider = "openrouter"

            client = LLMClient(provider="anthropic")

            assert client._default_provider == "anthropic"

    def test_init_with_both_providers(self):
        """Test client initialization with both providers configured."""
        with patch("app.services.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.openrouter_api_key = "test-openrouter-key"
            mock_settings.return_value.anthropic_api_key = "test-anthropic-key"
            mock_settings.return_value.llm_default_provider = "openrouter"

            client = LLMClient()

            assert "openrouter" in client.available_providers
            assert "anthropic" in client.available_providers

    def test_get_provider_default(self):
        """Test getting the default provider."""
        with patch("app.services.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.openrouter_api_key = "test-key"
            mock_settings.return_value.anthropic_api_key = ""
            mock_settings.return_value.llm_default_provider = "openrouter"

            client = LLMClient()
            provider = client.get_provider()

            assert provider.name == "openrouter"

    def test_get_provider_by_name(self):
        """Test getting a specific provider by name."""
        with patch("app.services.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.openrouter_api_key = "test-openrouter-key"
            mock_settings.return_value.anthropic_api_key = "test-anthropic-key"
            mock_settings.return_value.llm_default_provider = "openrouter"

            with patch("anthropic.AsyncAnthropic"):
                client = LLMClient()
                provider = client.get_provider("anthropic")

                assert provider.name == "anthropic"

    def test_get_provider_fallback_when_not_available(self):
        """Test provider fallback when requested provider is not available."""
        with patch("app.services.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.openrouter_api_key = "test-key"
            mock_settings.return_value.anthropic_api_key = ""
            mock_settings.return_value.llm_default_provider = "anthropic"  # Not available

            client = LLMClient()
            provider = client.get_provider()

            # Should fall back to openrouter since anthropic is not configured
            assert provider.name == "openrouter"

    def test_get_provider_raises_when_none_configured(self):
        """Test that LLMError is raised when no providers are configured."""
        with patch("app.services.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.openrouter_api_key = ""
            mock_settings.return_value.anthropic_api_key = ""
            mock_settings.return_value.llm_default_provider = "openrouter"

            client = LLMClient()

            with pytest.raises(LLMError) as exc_info:
                client.get_provider()

            assert "No LLM providers configured" in str(exc_info.value)


class TestLLMClientDependency:
    """Tests for the LLM client FastAPI dependency."""

    def test_get_llm_client_no_header(self):
        """Test get_llm_client with no provider header."""
        from app.dependencies import get_llm_client

        with patch("app.dependencies.LLMClient") as mock_client:
            get_llm_client(None)

            mock_client.assert_called_once_with(provider=None)

    def test_get_llm_client_valid_header(self):
        """Test get_llm_client with valid provider header."""
        from app.dependencies import get_llm_client

        with patch("app.dependencies.LLMClient") as mock_client:
            get_llm_client("anthropic")

            mock_client.assert_called_once_with(provider="anthropic")

    def test_get_llm_client_invalid_header(self):
        """Test get_llm_client with invalid provider header falls back to None."""
        from app.dependencies import get_llm_client

        with patch("app.dependencies.LLMClient") as mock_client:
            get_llm_client("invalid_provider")

            mock_client.assert_called_once_with(provider=None)

    def test_get_llm_client_case_insensitive(self):
        """Test get_llm_client handles case-insensitive header values."""
        from app.dependencies import get_llm_client

        with patch("app.dependencies.LLMClient") as mock_client:
            get_llm_client("ANTHROPIC")

            mock_client.assert_called_once_with(provider="anthropic")

    def test_get_llm_client_whitespace_handling(self):
        """Test get_llm_client handles whitespace in header values."""
        from app.dependencies import get_llm_client

        with patch("app.dependencies.LLMClient") as mock_client:
            get_llm_client("  openrouter  ")

            mock_client.assert_called_once_with(provider="openrouter")
