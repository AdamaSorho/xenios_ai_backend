"""LLM model configuration for task-based routing with multi-provider support."""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for a model based on task type."""

    primary: str
    fallback: str
    temperature: float
    max_tokens: int
    streaming: bool = False


# Task-based model routing configuration
# Based on user preference for Opus 4.5's stronger reasoning
TASK_MODELS: dict[str, ModelConfig] = {
    # Complex reasoning tasks - use Opus 4.5
    "session_summary": ModelConfig(
        primary="anthropic/claude-opus-4-20250514",
        fallback="anthropic/claude-sonnet-4-20250514",
        temperature=0.3,
        max_tokens=4000,
    ),
    "insight_generation": ModelConfig(
        primary="anthropic/claude-opus-4-20250514",
        fallback="anthropic/claude-sonnet-4-20250514",
        temperature=0.5,
        max_tokens=2000,
    ),
    "chat": ModelConfig(
        primary="anthropic/claude-opus-4-20250514",
        fallback="anthropic/claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=1500,
        streaming=True,
    ),
    # Simpler tasks - Sonnet 4 is sufficient and faster
    "intent_classification": ModelConfig(
        primary="anthropic/claude-sonnet-4-20250514",
        fallback="openai/gpt-4o-mini",
        temperature=0.0,  # Deterministic
        max_tokens=100,
    ),
    "entity_extraction": ModelConfig(
        primary="anthropic/claude-sonnet-4-20250514",
        fallback="openai/gpt-4o-mini",
        temperature=0.0,
        max_tokens=500,
    ),
    # Language cue detection - cost-effective classification
    "cue_detection": ModelConfig(
        primary="openai/gpt-4o-mini",  # Cost-effective for classification per spec
        fallback="anthropic/claude-sonnet-4-20250514",
        temperature=0.0,  # Deterministic
        max_tokens=500,
    ),
}


# Per-provider model name mappings
# OpenRouter uses "provider/model" format
OPENROUTER_MODELS: dict[str, str] = {
    "chat": "anthropic/claude-opus-4-20250514",
    "session_summary": "anthropic/claude-opus-4-20250514",
    "insight_generation": "anthropic/claude-opus-4-20250514",
    "intent_classification": "anthropic/claude-sonnet-4-20250514",
    "entity_extraction": "anthropic/claude-sonnet-4-20250514",
    "cue_detection": "openai/gpt-4o-mini",
}

# Anthropic uses direct model names without provider prefix
ANTHROPIC_MODELS: dict[str, str] = {
    "chat": "claude-opus-4-20250514",
    "session_summary": "claude-opus-4-20250514",
    "insight_generation": "claude-opus-4-20250514",
    "intent_classification": "claude-sonnet-4-20250514",
    "entity_extraction": "claude-sonnet-4-20250514",
    "cue_detection": "claude-sonnet-4-20250514",  # Fallback - no GPT on Anthropic
}

# Provider to models mapping
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "openrouter": OPENROUTER_MODELS,
    "anthropic": ANTHROPIC_MODELS,
}


def get_model_for_task(task: str, provider: str | None = None) -> str:
    """
    Get the appropriate model name for a task and provider.

    Args:
        task: The task type (e.g., 'chat', 'session_summary')
        provider: The provider name ('openrouter' or 'anthropic')
                  If None, returns the OpenRouter model name for backward compatibility

    Returns:
        Model name string appropriate for the specified provider

    Raises:
        ValueError: If the task type is unknown
    """
    if task not in TASK_MODELS:
        available_tasks = ", ".join(sorted(TASK_MODELS.keys()))
        raise ValueError(f"Unknown task: {task}. Available tasks: {available_tasks}")

    # Get provider-specific model mapping
    models = PROVIDER_MODELS.get(provider or "openrouter", OPENROUTER_MODELS)
    return models.get(task, models.get("chat", ""))


def get_task_config(task: str) -> ModelConfig:
    """
    Get the model configuration for a specific task.

    Args:
        task: The task type (e.g., 'chat', 'session_summary')

    Returns:
        ModelConfig with temperature, max_tokens, and other settings

    Raises:
        ValueError: If the task type is unknown
    """
    if task not in TASK_MODELS:
        available_tasks = ", ".join(sorted(TASK_MODELS.keys()))
        raise ValueError(f"Unknown task: {task}. Available tasks: {available_tasks}")
    return TASK_MODELS[task]


def list_available_tasks() -> list[str]:
    """Return list of available task types."""
    return list(TASK_MODELS.keys())
