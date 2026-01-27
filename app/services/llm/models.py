"""LLM model configuration for task-based routing."""

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
}


def get_model_for_task(task: str) -> ModelConfig:
    """
    Get the model configuration for a specific task.

    Args:
        task: The task type (e.g., 'chat', 'session_summary')

    Returns:
        ModelConfig with primary/fallback model settings

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
