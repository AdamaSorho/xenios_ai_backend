"""Tests for rate limiting middleware."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


class TestRateLimitingMiddleware:
    """Tests for rate limiting functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.incr = AsyncMock(return_value=1)
        redis.expire = AsyncMock()
        return redis

    @pytest.mark.asyncio
    async def test_rate_limit_increments_counter(self, mock_redis):
        """Test that requests increment the rate limit counter."""
        with patch("app.middleware.rate_limit.get_redis", return_value=mock_redis):
            from app.middleware.rate_limit import RateLimitMiddleware

            middleware = RateLimitMiddleware(app=MagicMock())

            # Simulate a request
            request = MagicMock()
            request.url.path = "/api/v1/chat/complete"
            request.state.user_id = str(uuid4())
            request.headers.get.return_value = None

            call_next = AsyncMock(return_value=MagicMock())

            await middleware.dispatch(request, call_next)

            # Counter should be incremented
            assert mock_redis.incr.called

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, mock_redis):
        """Test that exceeding limit returns 429 status."""
        # Setup: counter is already at limit
        mock_redis.incr.return_value = 101  # Over the 100/hour limit

        with patch("app.middleware.rate_limit.get_redis", return_value=mock_redis):
            from app.middleware.rate_limit import RateLimitMiddleware

            middleware = RateLimitMiddleware(app=MagicMock())

            request = MagicMock()
            request.url.path = "/api/v1/chat/complete"
            request.state.user_id = str(uuid4())
            request.headers.get.return_value = None

            call_next = AsyncMock()

            response = await middleware.dispatch(request, call_next)

            assert response.status_code == 429
            # call_next should not be called when rate limited
            assert not call_next.called

    @pytest.mark.asyncio
    async def test_system_key_bypasses_rate_limit(self, mock_redis):
        """Test that system API key bypasses rate limiting."""
        with patch("app.middleware.rate_limit.get_redis", return_value=mock_redis):
            with patch("app.middleware.rate_limit.get_settings") as mock_settings:
                mock_settings.return_value.xenios_backend_api_key = "system-key"

                from app.middleware.rate_limit import RateLimitMiddleware

                middleware = RateLimitMiddleware(app=MagicMock())

                request = MagicMock()
                request.url.path = "/api/v1/chat/complete"
                request.headers.get.side_effect = lambda k: {
                    "X-API-Key": "system-key",
                    "X-System-Request": "true",
                }.get(k)

                call_next = AsyncMock(return_value=MagicMock())

                await middleware.dispatch(request, call_next)

                # Should bypass rate limiting and call next
                assert call_next.called
                # Redis should not be called
                assert not mock_redis.incr.called

    @pytest.mark.asyncio
    async def test_rate_limit_headers_included(self, mock_redis):
        """Test that rate limit headers are included in response."""
        mock_redis.incr.return_value = 50  # Under limit

        with patch("app.middleware.rate_limit.get_redis", return_value=mock_redis):
            from app.middleware.rate_limit import RateLimitMiddleware

            middleware = RateLimitMiddleware(app=MagicMock())

            request = MagicMock()
            request.url.path = "/api/v1/chat/complete"
            request.state.user_id = str(uuid4())
            request.headers.get.return_value = None

            response = MagicMock()
            response.headers = {}
            call_next = AsyncMock(return_value=response)

            result = await middleware.dispatch(request, call_next)

            # Rate limit headers should be set
            assert "X-RateLimit-Limit" in result.headers
            assert "X-RateLimit-Remaining" in result.headers

    @pytest.mark.asyncio
    async def test_different_limits_per_endpoint(self, mock_redis):
        """Test that different endpoints have different limits."""
        from app.middleware.rate_limit import RateLimitMiddleware

        middleware = RateLimitMiddleware(app=MagicMock())

        # Chat endpoint: 100/hour
        assert middleware.LIMITS["/api/v1/chat/"]["requests"] == 100

        # Embeddings endpoint: 10/hour
        assert middleware.LIMITS["/api/v1/embeddings/"]["requests"] == 10

        # Insights endpoint: 50/day
        assert middleware.LIMITS["/api/v1/insights/"]["requests"] == 50


class TestRateLimitConfiguration:
    """Tests for rate limit configuration values."""

    def test_chat_rate_limit_100_per_hour(self):
        """Verify chat endpoints have 100 requests/hour limit."""
        from app.middleware.rate_limit import RateLimitMiddleware

        limits = RateLimitMiddleware.LIMITS
        assert limits["/api/v1/chat/"]["requests"] == 100
        assert limits["/api/v1/chat/"]["window"] == 3600

    def test_embeddings_rate_limit_10_per_hour(self):
        """Verify embeddings endpoints have 10 updates/hour limit."""
        from app.middleware.rate_limit import RateLimitMiddleware

        limits = RateLimitMiddleware.LIMITS
        assert limits["/api/v1/embeddings/"]["requests"] == 10
        assert limits["/api/v1/embeddings/"]["window"] == 3600

    def test_insights_rate_limit_50_per_day(self):
        """Verify insights endpoints have 50 generations/day limit."""
        from app.middleware.rate_limit import RateLimitMiddleware

        limits = RateLimitMiddleware.LIMITS
        assert limits["/api/v1/insights/"]["requests"] == 50
        assert limits["/api/v1/insights/"]["window"] == 86400
