"""Middleware modules for the application."""

from app.middleware.correlation import CorrelationIdMiddleware

__all__ = ["CorrelationIdMiddleware"]
