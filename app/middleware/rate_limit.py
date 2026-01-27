"""Redis-based rate limiting middleware for RAG endpoints (Spec 0004)."""

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import get_settings
from app.core.logging import get_logger
from app.core.redis import get_redis

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis sliding window rate limiting middleware.

    Rate limits per spec AC8:
    - Chat endpoints: 100 requests per coach per hour
    - Embeddings endpoints: 10 updates per coach per hour
    - Insights endpoints: 50 generations per coach per day
    """

    # Rate limit configuration: path prefix -> (requests, window_seconds)
    LIMITS = {
        "/api/v1/chat/": {"requests": 100, "window": 3600},  # 100/hour
        "/api/v1/embeddings/": {"requests": 10, "window": 3600},  # 10/hour
        "/api/v1/insights/": {"requests": 50, "window": 86400},  # 50/day
    }

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Check rate limits before processing request."""
        settings = get_settings()

        # Skip for system/service API keys
        api_key = request.headers.get("X-API-Key")
        if api_key == settings.xenios_backend_api_key:
            # Check for system bypass header
            if request.headers.get("X-System-Request") == "true":
                return await call_next(request)

        # Find matching rate limit config
        path = request.url.path
        limit_config = None
        endpoint_key = None

        for prefix, config in self.LIMITS.items():
            if path.startswith(prefix):
                limit_config = config
                # Extract the endpoint category for the rate limit key
                endpoint_key = prefix.split("/")[-2]  # chat, embeddings, insights
                break

        # No rate limit for this endpoint
        if not limit_config:
            return await call_next(request)

        # Get user ID from request state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            # Try to get from JWT if auth hasn't run yet
            # For now, skip rate limiting if no user context
            return await call_next(request)

        # Check rate limit using Redis
        try:
            redis = await get_redis()

            # Sliding window key: ratelimit:{endpoint}:{user_id}:{window_start}
            window_start = int(time.time()) // limit_config["window"]
            key = f"ratelimit:{endpoint_key}:{user_id}:{window_start}"

            # Increment counter
            current = await redis.incr(key)

            # Set expiry if this is a new key
            if current == 1:
                await redis.expire(key, limit_config["window"])

            # Check if over limit
            if current > limit_config["requests"]:
                logger.warning(
                    "Rate limit exceeded",
                    endpoint=endpoint_key,
                    user_id=user_id,
                    current=current,
                    limit=limit_config["requests"],
                )

                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": limit_config["window"],
                    },
                    headers={"Retry-After": str(limit_config["window"])},
                )

            # Add rate limit headers to response
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(limit_config["requests"])
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, limit_config["requests"] - current)
            )
            response.headers["X-RateLimit-Reset"] = str(
                (window_start + 1) * limit_config["window"]
            )

            return response

        except Exception as e:
            # If Redis fails, allow the request but log the error
            logger.error("Rate limiting failed, allowing request", error=str(e))
            return await call_next(request)
