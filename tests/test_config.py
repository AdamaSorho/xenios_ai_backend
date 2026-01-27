"""Tests for configuration management."""

import pytest
from pydantic import ValidationError


class TestSettings:
    """Tests for Settings class."""

    def test_settings_loads_from_env(self, mock_settings):
        """Test that settings are loaded from environment."""
        assert mock_settings.environment == "development"
        assert mock_settings.database_url == "postgresql://test:test@localhost:5432/test"
        assert mock_settings.redis_url == "redis://localhost:6379/0"

    def test_settings_defaults(self, mock_settings):
        """Test default values are applied."""
        assert mock_settings.log_level == "INFO"
        assert mock_settings.cors_origins == "http://localhost:3000"

    def test_cors_origins_list(self, mock_settings):
        """Test CORS origins are parsed correctly."""
        origins = mock_settings.cors_origins_list
        assert isinstance(origins, list)
        assert "http://localhost:3000" in origins

    def test_is_production(self, mock_settings):
        """Test is_production property."""
        assert mock_settings.is_production is False

    def test_is_development(self, mock_settings):
        """Test is_development property."""
        assert mock_settings.is_development is True


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_caching(self):
        """Test that settings are cached."""
        from app.config import get_settings

        # Clear cache
        get_settings.cache_clear()

        # Call twice
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance (cached)
        assert settings1 is settings2

    def test_settings_required_fields(self, monkeypatch):
        """Test that missing required fields raise an error."""
        from app.config import Settings, get_settings

        # Clear cache
        get_settings.cache_clear()

        # Remove required environment variable
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
        monkeypatch.delenv("XENIOS_BACKEND_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValidationError):
            Settings()
