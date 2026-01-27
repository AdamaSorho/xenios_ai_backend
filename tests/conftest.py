"""Pytest configuration and fixtures."""

import os
from collections.abc import AsyncIterator, Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Set test environment variables BEFORE any app imports
os.environ["ENVIRONMENT"] = "development"
os.environ["DEBUG"] = "true"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret-at-least-32-chars-long"
os.environ["XENIOS_BACKEND_API_KEY"] = "test-api-key"
os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"
# S3 settings for extraction tests
os.environ["S3_ENDPOINT_URL"] = ""
os.environ["S3_BUCKET"] = "test-bucket"
os.environ["S3_ACCESS_KEY_ID"] = "test-key"
os.environ["S3_SECRET_ACCESS_KEY"] = "test-secret"
os.environ["S3_REGION"] = "us-east-1"


@pytest.fixture(scope="session")
def test_settings():
    """Get test settings."""
    from app.config import Settings

    return Settings(
        environment="development",
        debug=True,
        database_url="postgresql://test:test@localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        supabase_jwt_secret="test-jwt-secret-at-least-32-chars-long",
        xenios_backend_api_key="test-api-key",
        openrouter_api_key="test-openrouter-key",
    )


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset settings cache before each test."""
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_settings(test_settings):
    """Mock get_settings to return test settings."""
    with patch("app.config.get_settings", return_value=test_settings):
        yield test_settings


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.close = AsyncMock()

    with patch("app.core.redis.get_redis", new_callable=AsyncMock, return_value=mock):
        with patch("app.core.redis.check_redis_health", new_callable=AsyncMock, return_value=True):
            yield mock


@pytest.fixture
def mock_database():
    """Mock database connection."""
    with patch("app.core.database.check_db_health", new_callable=AsyncMock, return_value=True):
        with patch("app.core.database.get_pool", new_callable=AsyncMock):
            yield


@pytest.fixture
def app(mock_redis, mock_database):
    """Create test application instance."""
    from app.main import create_app

    return create_app()


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(app) -> AsyncIterator[AsyncClient]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def valid_api_key() -> str:
    """Return valid API key for testing."""
    return "test-api-key"


@pytest.fixture
def valid_jwt_token() -> str:
    """Generate a valid JWT token for testing."""
    from jose import jwt

    payload = {
        "sub": "test-user-id",
        "role": "authenticated",
        "email": "test@example.com",
        "aud": "authenticated",
    }
    return jwt.encode(
        payload,
        "test-jwt-secret-at-least-32-chars-long",
        algorithm="HS256",
    )


@pytest.fixture
def auth_headers(valid_api_key, valid_jwt_token) -> dict[str, str]:
    """Return headers with valid API key and JWT."""
    return {
        "X-API-Key": valid_api_key,
        "Authorization": f"Bearer {valid_jwt_token}",
    }


@pytest.fixture
def api_key_headers(valid_api_key) -> dict[str, str]:
    """Return headers with valid API key only."""
    return {"X-API-Key": valid_api_key}
