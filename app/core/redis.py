"""Redis connection and utilities."""

from collections.abc import AsyncIterator

import redis.asyncio as redis
from redis.asyncio import Redis

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global Redis client (initialized on first use)
_redis_client: Redis | None = None


async def get_redis() -> Redis:
    """Get or create the Redis client."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("Redis client created", url=settings.redis_url.split("@")[-1])
    return _redis_client


async def check_redis_health() -> bool:
    """Check Redis connectivity for health checks."""
    try:
        client = await get_redis()
        result = await client.ping()
        return result is True
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        return False


async def close_redis() -> None:
    """Close the Redis client connection."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis client closed")


async def get_redis_dependency() -> AsyncIterator[Redis]:
    """FastAPI dependency for Redis client."""
    client = await get_redis()
    yield client
