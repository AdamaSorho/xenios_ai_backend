"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (OpenRouter, Anthropic, etc.) must implement this interface
    to ensure consistent behavior across different backends.
    """

    name: str  # Provider identifier (e.g., "openrouter", "anthropic")

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Send a completion request and return normalized response.

        Args:
            model: The model identifier to use
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Normalized response dict in OpenAI-compatible format:
            {
                "id": "...",
                "object": "chat.completion",
                "model": "...",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "..."},
                    "finish_reason": "..."
                }],
                "usage": {
                    "prompt_tokens": ...,
                    "completion_tokens": ...,
                    "total_tokens": ...
                }
            }
        """
        pass

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        Stream a completion response, yielding content chunks.

        Args:
            model: The model identifier to use
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Yields:
            Raw SSE data lines (JSON strings) from the streaming response
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is configured and available.

        Returns:
            True if the provider has valid credentials and can be used
        """
        pass
