"""Multi-provider LLM client with task-based model routing and streaming support."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.models import get_model_for_task, get_task_config
from app.services.llm.providers.anthropic import AnthropicProvider
from app.services.llm.providers.base import LLMProvider
from app.services.llm.providers.openrouter import OpenRouterProvider

logger = get_logger(__name__)


class LLMError(Exception):
    """Base exception for LLM client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class LLMClient:
    """
    Multi-provider LLM client with task-based model routing.

    Supports multiple LLM providers (OpenRouter, Anthropic) with runtime
    provider selection via header or configuration. Provides automatic
    fallback when the preferred provider fails.
    """

    def __init__(
        self,
        provider: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            provider: Override the default provider ('openrouter' or 'anthropic')
            settings: Optional settings instance (defaults to global settings)
        """
        self._settings = settings or get_settings()
        self._default_provider = provider or self._settings.llm_default_provider
        self._providers = self._init_providers()

    def _init_providers(self) -> dict[str, LLMProvider]:
        """Initialize available providers based on configuration."""
        providers: dict[str, LLMProvider] = {}

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
        """
        Get provider by name, falling back to default or any available.

        Args:
            name: Provider name ('openrouter' or 'anthropic')

        Returns:
            The requested LLMProvider instance

        Raises:
            LLMError: If no providers are configured
        """
        provider_name = name or self._default_provider

        if provider_name not in self._providers:
            # Fallback to any available provider
            if self._providers:
                fallback_name = next(iter(self._providers))
                logger.warning(
                    "Requested provider not available, using fallback",
                    requested=provider_name,
                    fallback=fallback_name,
                )
                provider_name = fallback_name
            else:
                raise LLMError("No LLM providers configured")

        return self._providers[provider_name]

    @property
    def available_providers(self) -> list[str]:
        """Return list of available provider names."""
        return list(self._providers.keys())

    async def complete(
        self,
        task: str,
        messages: list[dict[str, Any]],
        provider: str | None = None,
        use_fallback: bool = True,
    ) -> dict[str, Any]:
        """
        Send a completion request with optional provider override and fallback.

        Args:
            task: The task type for model selection
            messages: List of message dicts with 'role' and 'content'
            provider: Optional provider override ('openrouter' or 'anthropic')
            use_fallback: If True, try other providers on failure

        Returns:
            Normalized API response as dict (OpenAI-compatible format)

        Raises:
            LLMError: On API errors after exhausting all providers
        """
        provider_name = provider or self._default_provider
        llm_provider = self.get_provider(provider_name)
        actual_provider_name = llm_provider.name

        model = get_model_for_task(task, actual_provider_name)
        config = get_task_config(task)

        logger.info(
            "Sending LLM completion request",
            task=task,
            model=model,
            provider=actual_provider_name,
            message_count=len(messages),
        )

        try:
            result = await llm_provider.complete(
                model=model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            logger.info(
                "LLM completion successful",
                task=task,
                model=model,
                provider=actual_provider_name,
                usage=result.get("usage"),
            )
            return result

        except httpx.HTTPStatusError as e:
            logger.warning(
                "LLM request failed",
                task=task,
                model=model,
                provider=actual_provider_name,
                status_code=e.response.status_code,
                error=str(e),
            )

            if use_fallback:
                return await self._try_fallback(
                    task, messages, actual_provider_name, e
                )

            raise LLMError(
                f"LLM request failed: {e}",
                status_code=e.response.status_code,
            ) from None

        except Exception as e:
            logger.error(
                "LLM request error",
                task=task,
                provider=actual_provider_name,
                error=str(e),
            )

            if use_fallback:
                return await self._try_fallback(
                    task, messages, actual_provider_name, e
                )

            raise LLMError(f"LLM request error: {e}") from None

    async def _try_fallback(
        self,
        task: str,
        messages: list[dict[str, Any]],
        failed_provider: str,
        original_error: Exception,
    ) -> dict[str, Any]:
        """Try other providers as fallback."""
        tried_providers = [failed_provider]

        for name, llm_provider in self._providers.items():
            if name == failed_provider:
                continue

            tried_providers.append(name)
            logger.info(
                "Trying fallback provider",
                task=task,
                provider=name,
                failed_provider=failed_provider,
            )

            try:
                model = get_model_for_task(task, name)
                config = get_task_config(task)
                result = await llm_provider.complete(
                    model=model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                logger.info(
                    "Fallback provider succeeded",
                    task=task,
                    provider=name,
                )
                return result
            except Exception as e:
                logger.warning(
                    "Fallback provider failed",
                    task=task,
                    provider=name,
                    error=str(e),
                )
                continue

        raise LLMError(
            f"All providers failed. Tried: {', '.join(tried_providers)}. "
            f"Original error: {original_error}"
        )

    async def stream(
        self,
        task: str,
        messages: list[dict[str, Any]],
        provider: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion response from the LLM.

        Args:
            task: The task type for model selection
            messages: List of message dicts with 'role' and 'content'
            provider: Optional provider override

        Yields:
            SSE data lines (JSON strings) from the streaming response

        Raises:
            LLMError: On API errors
        """
        provider_name = provider or self._default_provider
        llm_provider = self.get_provider(provider_name)
        actual_provider_name = llm_provider.name

        model = get_model_for_task(task, actual_provider_name)
        config = get_task_config(task)

        logger.info(
            "Starting LLM stream request",
            task=task,
            model=model,
            provider=actual_provider_name,
            message_count=len(messages),
        )

        try:
            async for chunk in llm_provider.stream(
                model=model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            ):
                yield chunk

            logger.info(
                "LLM stream completed",
                task=task,
                model=model,
                provider=actual_provider_name,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "LLM stream failed",
                task=task,
                model=model,
                provider=actual_provider_name,
                status_code=e.response.status_code,
            )
            raise LLMError(
                f"LLM stream failed: {e}",
                status_code=e.response.status_code,
            ) from None

        except Exception as e:
            logger.error(
                "LLM stream error",
                task=task,
                provider=actual_provider_name,
                error=str(e),
            )
            raise LLMError(f"LLM stream error: {e}") from None

    async def complete_with_json(
        self,
        task: str,
        messages: list[dict[str, Any]],
        provider: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a completion and parse the response as JSON.

        Useful for structured outputs like intent classification.

        Args:
            task: The task type for model selection
            messages: List of message dicts
            provider: Optional provider override

        Returns:
            Parsed JSON from the completion content

        Raises:
            LLMError: On API or parsing errors
        """
        result = await self.complete(task, messages, provider=provider)

        try:
            content = result["choices"][0]["message"]["content"]
            # Try to extract JSON from the content
            # Sometimes models wrap JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Failed to parse LLM JSON response", error=str(e))
            raise LLMError(f"Failed to parse JSON response: {e}") from None
