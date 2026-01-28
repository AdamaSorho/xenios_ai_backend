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
            "cue_detection",
        ]

        for task in expected_tasks:
            assert task in TASK_MODELS, f"Task '{task}' not found in TASK_MODELS"

    def test_get_task_config(self):
        """Test getting model config for a valid task."""
        from app.services.llm.models import get_task_config

        config = get_task_config("chat")

        assert config.temperature == 0.7
        assert config.streaming is True

    def test_get_task_config_for_unknown_task_raises(self):
        """Test that unknown task raises ValueError."""
        from app.services.llm.models import get_task_config

        with pytest.raises(ValueError) as exc_info:
            get_task_config("unknown_task")

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


class TestProviderModelMappings:
    """Tests for per-provider model mappings."""

    def test_openrouter_models_defined(self):
        """Test that OpenRouter model mappings are defined for all tasks."""
        from app.services.llm.models import OPENROUTER_MODELS, TASK_MODELS

        for task in TASK_MODELS:
            assert task in OPENROUTER_MODELS, f"Task '{task}' missing from OPENROUTER_MODELS"

    def test_anthropic_models_defined(self):
        """Test that Anthropic model mappings are defined for all tasks."""
        from app.services.llm.models import ANTHROPIC_MODELS, TASK_MODELS

        for task in TASK_MODELS:
            assert task in ANTHROPIC_MODELS, f"Task '{task}' missing from ANTHROPIC_MODELS"

    def test_openrouter_models_have_provider_prefix(self):
        """Test that OpenRouter models use provider/model format."""
        from app.services.llm.models import OPENROUTER_MODELS

        for task, model in OPENROUTER_MODELS.items():
            assert "/" in model, f"OpenRouter model for '{task}' should have provider prefix"

    def test_anthropic_models_no_provider_prefix(self):
        """Test that Anthropic models don't have provider prefix."""
        from app.services.llm.models import ANTHROPIC_MODELS

        for task, model in ANTHROPIC_MODELS.items():
            # Anthropic models should start with 'claude-'
            assert model.startswith("claude-"), f"Anthropic model for '{task}' should be direct model name"

    def test_get_model_for_task_openrouter(self):
        """Test getting model for task with OpenRouter provider."""
        from app.services.llm.models import get_model_for_task

        model = get_model_for_task("chat", "openrouter")

        assert model == "anthropic/claude-opus-4-20250514"

    def test_get_model_for_task_anthropic(self):
        """Test getting model for task with Anthropic provider."""
        from app.services.llm.models import get_model_for_task

        model = get_model_for_task("chat", "anthropic")

        assert model == "claude-opus-4-20250514"

    def test_get_model_for_task_default_provider(self):
        """Test getting model for task with no provider (defaults to OpenRouter)."""
        from app.services.llm.models import get_model_for_task

        model = get_model_for_task("chat")

        assert model == "anthropic/claude-opus-4-20250514"

    def test_get_model_for_task_unknown_provider_fallback(self):
        """Test that unknown provider falls back to OpenRouter."""
        from app.services.llm.models import get_model_for_task

        model = get_model_for_task("chat", "unknown_provider")

        assert model == "anthropic/claude-opus-4-20250514"

    def test_cue_detection_differs_by_provider(self):
        """Test that cue_detection uses different models per provider."""
        from app.services.llm.models import get_model_for_task

        openrouter_model = get_model_for_task("cue_detection", "openrouter")
        anthropic_model = get_model_for_task("cue_detection", "anthropic")

        # OpenRouter uses GPT-4o-mini, Anthropic uses Sonnet as fallback
        assert "gpt-4o-mini" in openrouter_model
        assert "claude-sonnet" in anthropic_model
