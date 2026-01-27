"""Tests for auth state middleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jose import jwt


class TestAuthStateMiddleware:
    """Tests for AuthStateMiddleware."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.supabase_jwt_secret = "test-secret"
        return settings

    @pytest.fixture
    def valid_token(self, mock_settings):
        """Create a valid JWT token."""
        payload = {
            "sub": "user-123",
            "role": "authenticated",
            "email": "test@example.com",
            "aud": "authenticated",
        }
        return jwt.encode(payload, mock_settings.supabase_jwt_secret, algorithm="HS256")

    @pytest.mark.asyncio
    async def test_extracts_user_id_from_valid_jwt(self, mock_settings, valid_token):
        """Test that user_id is extracted from valid JWT."""
        with patch("app.middleware.auth_state.get_settings", return_value=mock_settings):
            from app.middleware.auth_state import AuthStateMiddleware

            middleware = AuthStateMiddleware(app=MagicMock())

            request = MagicMock()
            request.headers.get.side_effect = lambda k: {
                "Authorization": f"Bearer {valid_token}",
            }.get(k)
            request.state = MagicMock()

            call_next = AsyncMock(return_value=MagicMock())

            await middleware.dispatch(request, call_next)

            assert request.state.user_id == "user-123"
            assert call_next.called

    @pytest.mark.asyncio
    async def test_user_id_none_when_no_auth_header(self, mock_settings):
        """Test that user_id is None when no Authorization header."""
        with patch("app.middleware.auth_state.get_settings", return_value=mock_settings):
            from app.middleware.auth_state import AuthStateMiddleware

            middleware = AuthStateMiddleware(app=MagicMock())

            request = MagicMock()
            request.headers.get.return_value = None
            request.state = MagicMock()

            call_next = AsyncMock(return_value=MagicMock())

            await middleware.dispatch(request, call_next)

            assert request.state.user_id is None
            assert call_next.called

    @pytest.mark.asyncio
    async def test_user_id_none_with_invalid_jwt(self, mock_settings):
        """Test that user_id is None with invalid JWT."""
        with patch("app.middleware.auth_state.get_settings", return_value=mock_settings):
            from app.middleware.auth_state import AuthStateMiddleware

            middleware = AuthStateMiddleware(app=MagicMock())

            request = MagicMock()
            request.headers.get.side_effect = lambda k: {
                "Authorization": "Bearer invalid-token",
            }.get(k)
            request.state = MagicMock()

            call_next = AsyncMock(return_value=MagicMock())

            await middleware.dispatch(request, call_next)

            assert request.state.user_id is None
            assert call_next.called

    @pytest.mark.asyncio
    async def test_user_id_none_with_non_bearer_auth(self, mock_settings):
        """Test that user_id is None with non-Bearer auth."""
        with patch("app.middleware.auth_state.get_settings", return_value=mock_settings):
            from app.middleware.auth_state import AuthStateMiddleware

            middleware = AuthStateMiddleware(app=MagicMock())

            request = MagicMock()
            request.headers.get.side_effect = lambda k: {
                "Authorization": "Basic dXNlcjpwYXNz",
            }.get(k)
            request.state = MagicMock()

            call_next = AsyncMock(return_value=MagicMock())

            await middleware.dispatch(request, call_next)

            assert request.state.user_id is None
            assert call_next.called

    @pytest.mark.asyncio
    async def test_request_continues_even_with_jwt_error(self, mock_settings):
        """Test that request continues even if JWT parsing fails."""
        with patch("app.middleware.auth_state.get_settings", return_value=mock_settings):
            from app.middleware.auth_state import AuthStateMiddleware

            middleware = AuthStateMiddleware(app=MagicMock())

            request = MagicMock()
            request.headers.get.side_effect = lambda k: {
                "Authorization": "Bearer malformed.jwt.token",
            }.get(k)
            request.state = MagicMock()

            call_next = AsyncMock(return_value=MagicMock())

            response = await middleware.dispatch(request, call_next)

            # Request should still continue
            assert call_next.called
            assert response is not None
