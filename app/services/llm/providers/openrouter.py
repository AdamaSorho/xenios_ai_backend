"""OpenRouter API provider implementation."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.logging import get_logger
from app.services.llm.providers.base import LLMProvider

logger = get_logger(__name__)


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter API provider (OpenAI-compatible).

    Uses the OpenRouter API which provides access to multiple LLM providers
    through a unified OpenAI-compatible interface.
    """

    name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_TIMEOUT = 60.0
    STREAM_TIMEOUT = 120.0

    def __init__(self, api_key: str) -> None:
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key
        """
        self._api_key = api_key

    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenRouter API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://xenios.app",
            "X-Title": "Xenios AI Backend",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send completion request to OpenRouter."""
        logger.debug(
            "OpenRouter complete request",
            model=model,
            message_count=len(messages),
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream completion from OpenRouter."""
        logger.debug(
            "OpenRouter stream request",
            model=model,
            message_count=len(messages),
        )

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                timeout=httpx.Timeout(self.STREAM_TIMEOUT, connect=10.0),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        yield data

    def is_available(self) -> bool:
        """Check if OpenRouter is configured."""
        return bool(self._api_key)
