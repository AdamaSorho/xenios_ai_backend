"""Multi-provider LLM service with OpenRouter and Anthropic support."""

from app.services.llm.client import LLMClient, LLMError
from app.services.llm.models import (
    ANTHROPIC_MODELS,
    OPENROUTER_MODELS,
    PROVIDER_MODELS,
    TASK_MODELS,
    ModelConfig,
    get_model_for_task,
    get_task_config,
    list_available_tasks,
)

__all__ = [
    "LLMClient",
    "LLMError",
    "ModelConfig",
    "TASK_MODELS",
    "OPENROUTER_MODELS",
    "ANTHROPIC_MODELS",
    "PROVIDER_MODELS",
    "get_model_for_task",
    "get_task_config",
    "list_available_tasks",
]
