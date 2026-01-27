"""OpenRouter LLM client with model routing and streaming support."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import get_settings
from app.core.logging import get_logger
from app.services.llm.models import ModelConfig, get_model_for_task

logger = get_logger(__name__)


class LLMError(Exception):
    """Base exception for LLM client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class LLMClient:
    """
    Client for OpenRouter API with task-based model routing.

    Supports both synchronous completions and streaming responses.
    Automatically falls back to secondary model on primary failure.
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_TIMEOUT = 60.0
    STREAM_TIMEOUT = 120.0

    def __init__(self) -> None:
        self.settings = get_settings()

    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenRouter API requests."""
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "HTTP-Referer": "https://xenios.app",
            "X-Title": "Xenios AI Backend",
            "Content-Type": "application/json",
        }

    def _build_request_body(
        self,
        model: str,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request body for OpenRouter API."""
        return {
            "model": model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": stream,
        }

    async def complete(
        self,
        task: str,
        messages: list[dict[str, Any]],
        use_fallback: bool = False,
    ) -> dict[str, Any]:
        """
        Send a completion request to OpenRouter.

        Args:
            task: The task type for model selection
            messages: List of message dicts with 'role' and 'content'
            use_fallback: If True, use fallback model instead of primary

        Returns:
            OpenRouter API response as dict

        Raises:
            LLMError: On API errors after exhausting retries
        """
        config = get_model_for_task(task)
        model = config.fallback if use_fallback else config.primary

        logger.info(
            "Sending LLM completion request",
            task=task,
            model=model,
            message_count=len(messages),
            use_fallback=use_fallback,
        )

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=self._build_request_body(model, messages, config),
                    timeout=self.DEFAULT_TIMEOUT,
                )
                response.raise_for_status()

                result = response.json()
                logger.info(
                    "LLM completion successful",
                    task=task,
                    model=model,
                    usage=result.get("usage"),
                )
                return result

            except httpx.HTTPStatusError as e:
                logger.warning(
                    "LLM request failed",
                    task=task,
                    model=model,
                    status_code=e.response.status_code,
                    error=str(e),
                )

                # Try fallback model if not already using it
                if not use_fallback:
                    logger.info("Retrying with fallback model", fallback=config.fallback)
                    return await self.complete(task, messages, use_fallback=True)

                raise LLMError(
                    f"LLM request failed: {e}",
                    status_code=e.response.status_code,
                ) from None

            except httpx.RequestError as e:
                logger.error("LLM request error", task=task, error=str(e))

                if not use_fallback:
                    return await self.complete(task, messages, use_fallback=True)

                raise LLMError(f"LLM request error: {e}") from None

    async def stream(
        self,
        task: str,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """
        Stream a completion response from OpenRouter.

        Args:
            task: The task type for model selection
            messages: List of message dicts with 'role' and 'content'

        Yields:
            SSE data lines from the streaming response

        Raises:
            LLMError: On API errors
        """
        config = get_model_for_task(task)
        model = config.primary

        logger.info(
            "Starting LLM stream request",
            task=task,
            model=model,
            message_count=len(messages),
        )

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=self._build_request_body(model, messages, config, stream=True),
                    timeout=self.STREAM_TIMEOUT,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data == "[DONE]":
                                break
                            yield data

                logger.info("LLM stream completed", task=task, model=model)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "LLM stream failed",
                    task=task,
                    model=model,
                    status_code=e.response.status_code,
                )
                raise LLMError(
                    f"LLM stream failed: {e}",
                    status_code=e.response.status_code,
                ) from None

            except httpx.RequestError as e:
                logger.error("LLM stream error", task=task, error=str(e))
                raise LLMError(f"LLM stream error: {e}") from None

    async def complete_with_json(
        self,
        task: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Send a completion and parse the response as JSON.

        Useful for structured outputs like intent classification.

        Args:
            task: The task type for model selection
            messages: List of message dicts

        Returns:
            Parsed JSON from the completion content

        Raises:
            LLMError: On API or parsing errors
        """
        result = await self.complete(task, messages)

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
