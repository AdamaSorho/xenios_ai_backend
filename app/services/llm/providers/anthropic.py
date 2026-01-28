"""Anthropic (Claude) API provider implementation."""

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from app.core.logging import get_logger
from app.services.llm.providers.base import LLMProvider

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Direct Claude API provider using the Anthropic SDK.

    Provides direct access to Claude models without going through OpenRouter,
    useful for lower latency or when direct vendor relationship is required.
    """

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
        """
        self._api_key = api_key
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Send completion request to Claude API.

        Extracts system message if present and passes it separately,
        as Anthropic API uses a dedicated system parameter.
        """
        logger.debug(
            "Anthropic complete request",
            model=model,
            message_count=len(messages),
        )

        # Extract system message if present
        system_content = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system_content:
            request_kwargs["system"] = system_content

        response = await self._client.messages.create(**request_kwargs)

        return self._normalize_response(response)

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        Stream completion from Claude API.

        Yields SSE-compatible JSON strings matching OpenRouter format.
        """
        logger.debug(
            "Anthropic stream request",
            model=model,
            message_count=len(messages),
        )

        # Extract system message if present
        system_content = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system_content:
            request_kwargs["system"] = system_content

        async with self._client.messages.stream(**request_kwargs) as stream:
            async for event in stream:
                # Convert Anthropic events to OpenAI-compatible format
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        chunk = {
                            "choices": [{
                                "index": 0,
                                "delta": {"content": event.delta.text},
                                "finish_reason": None,
                            }]
                        }
                        yield json.dumps(chunk)

                elif event.type == "message_stop":
                    # Final chunk with finish_reason
                    chunk = {
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }]
                    }
                    yield json.dumps(chunk)

    def is_available(self) -> bool:
        """Check if Anthropic is configured."""
        return bool(self._api_key)

    def _normalize_response(self, response: anthropic.types.Message) -> dict[str, Any]:
        """
        Convert Anthropic response to OpenAI-compatible format.

        This ensures a unified response format regardless of provider.
        """
        # Extract text content from response
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

        return {
            "id": response.id,
            "object": "chat.completion",
            "model": response.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": response.stop_reason or "stop",
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }
