"""Tests for LLM model configuration and routing."""

import pytest


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig instance."""
        from app.services.llm.models import ModelConfig

        config = ModelConfig(
            primary="anthropic/claude-opus-4-20250514",
            fallback="anthropic/claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=1500,
            streaming=True,
        )

        assert config.primary == "anthropic/claude-opus-4-20250514"
        assert config.fallback == "anthropic/claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 1500
        assert config.streaming is True

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from app.services.llm.models import ModelConfig

        config = ModelConfig(
            primary="test-model",
            fallback="test-fallback",
            temperature=0.5,
            max_tokens=100,
        )

        assert config.streaming is False


class TestTaskModels:
    """Tests for task model routing."""

    def test_task_models_defined(self):
        """Test that all expected tasks are defined."""
        from app.services.llm.models import TASK_MODELS

        expected_tasks = [
            "session_summary",
            "insight_generation",
            "chat",
            "intent_classification",
            "entity_extraction",
        ]

        for task in expected_tasks:
            assert task in TASK_MODELS, f"Task '{task}' not found in TASK_MODELS"

    def test_get_model_for_task(self):
        """Test getting model config for a valid task."""
        from app.services.llm.models import get_model_for_task

        config = get_model_for_task("chat")

        assert config.primary == "anthropic/claude-opus-4-20250514"
        assert config.streaming is True

    def test_get_model_for_unknown_task_raises(self):
        """Test that unknown task raises ValueError."""
        from app.services.llm.models import get_model_for_task

        with pytest.raises(ValueError) as exc_info:
            get_model_for_task("unknown_task")

        assert "Unknown task: unknown_task" in str(exc_info.value)

    def test_list_available_tasks(self):
        """Test listing available tasks."""
        from app.services.llm.models import list_available_tasks

        tasks = list_available_tasks()

        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert "chat" in tasks
        assert "session_summary" in tasks


class TestModelRouting:
    """Tests for model routing logic."""

    def test_opus_used_for_complex_tasks(self):
        """Test that Opus is used for complex reasoning tasks."""
        from app.services.llm.models import TASK_MODELS

        complex_tasks = ["session_summary", "insight_generation", "chat"]

        for task in complex_tasks:
            config = TASK_MODELS[task]
            assert "opus" in config.primary.lower(), f"Task '{task}' should use Opus"

    def test_sonnet_used_for_simple_tasks(self):
        """Test that Sonnet is used for simpler tasks."""
        from app.services.llm.models import TASK_MODELS

        simple_tasks = ["intent_classification", "entity_extraction"]

        for task in simple_tasks:
            config = TASK_MODELS[task]
            assert "sonnet" in config.primary.lower(), f"Task '{task}' should use Sonnet"

    def test_deterministic_temperature_for_classification(self):
        """Test that classification tasks use deterministic temperature."""
        from app.services.llm.models import TASK_MODELS

        deterministic_tasks = ["intent_classification", "entity_extraction"]

        for task in deterministic_tasks:
            config = TASK_MODELS[task]
            assert config.temperature == 0.0, f"Task '{task}' should have temperature 0.0"

    def test_streaming_enabled_for_chat(self):
        """Test that chat tasks have streaming enabled."""
        from app.services.llm.models import TASK_MODELS

        config = TASK_MODELS["chat"]
        assert config.streaming is True
