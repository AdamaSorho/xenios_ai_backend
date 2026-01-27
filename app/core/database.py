"""Database connection and utilities for Supabase PostgreSQL."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global connection pool (initialized on first use)
_pool: asyncpg.Pool | None = None


def get_async_database_url() -> str:
    """Convert database URL to async format for SQLAlchemy."""
    settings = get_settings()
    url = settings.database_url

    # Convert postgresql:// to postgresql+asyncpg://
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


# SQLAlchemy async engine (lazy initialization)
_engine = None
_session_factory = None


def get_engine():
    """Get or create the SQLAlchemy async engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            get_async_database_url(),
            echo=get_settings().debug,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Get a database session for use in request handlers."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_pool() -> asyncpg.Pool:
    """Get or create the asyncpg connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        logger.info("Database connection pool created")
    return _pool


async def check_db_health() -> bool:
    """Check database connectivity for health checks."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return result == 1
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False


async def init_ai_backend_schema() -> None:
    """Initialize the ai_backend schema if it doesn't exist."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE SCHEMA IF NOT EXISTS ai_backend")
            logger.info("ai_backend schema initialized")
    except Exception as e:
        logger.error("Failed to initialize ai_backend schema", error=str(e))
        raise


async def close_pool() -> None:
    """Close the database connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")
