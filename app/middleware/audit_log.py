"""Audit logging middleware for compliance (Spec 0004)."""

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Audit logging middleware for compliance and security.

    Per Spec 0004 security requirements:
    - Log all embedding reads/writes with coach_id, client_id, timestamp
    - Log all chat completions with coach_id, client_id, tokens_used
    - Log insight generation attempts (success/duplicate/failure)
    - Audit logs retained 2 years for compliance
    """

    # Paths that require audit logging
    AUDIT_PATHS = [
        "/api/v1/chat/",
        "/api/v1/embeddings/",
        "/api/v1/insights/",
    ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Log audit entry for relevant endpoints."""
        path = request.url.path

        # Check if this path needs audit logging
        should_audit = any(path.startswith(p) for p in self.AUDIT_PATHS)

        if not should_audit:
            return await call_next(request)

        # Record start time
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Get user context from request state (set by auth)
        coach_id = getattr(request.state, "user_id", None)
        client_id = getattr(request.state, "client_id", None)

        # Determine operation type from path and method
        operation = self._get_operation_type(path, request.method)

        # Log audit entry
        # Note: PHI is automatically redacted by the logging processor
        logger.info(
            "audit",
            operation=operation,
            path=path,
            method=request.method,
            coach_id=coach_id,
            client_id=client_id,
            status_code=response.status_code,
            duration_ms=duration_ms,
            # Additional context that may be set by endpoints
            tokens_used=getattr(request.state, "tokens_used", None),
            source_type=getattr(request.state, "source_type", None),
        )

        return response

    def _get_operation_type(self, path: str, method: str) -> str:
        """Determine the operation type from path and method."""
        if "/chat/" in path:
            if "stream" in path:
                return "chat_stream"
            return "chat_complete"
        elif "/embeddings/" in path:
            if "search" in path:
                return "embedding_search"
            elif "batch" in path:
                return "embedding_batch_update"
            return "embedding_update"
        elif "/insights/" in path:
            if "pending" in path:
                return "insight_list"
            return "insight_generate"
        return f"{method.lower()}_{path}"
