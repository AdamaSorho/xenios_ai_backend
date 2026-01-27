"""Auth state middleware to extract user_id for downstream middleware."""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AuthStateMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract user_id from JWT and set request.state.user_id.

    This runs BEFORE rate limiting and audit logging middleware so they
    can access user_id. This middleware does NOT enforce authentication -
    that's still the responsibility of endpoint dependencies (get_current_user).

    If no valid JWT is present, user_id remains None and downstream
    middleware should handle that case appropriately.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Extract user_id from JWT if present."""
        settings = get_settings()

        # Initialize user_id as None
        request.state.user_id = None

        # Try to extract user_id from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix

            try:
                payload = jwt.decode(
                    token,
                    settings.supabase_jwt_secret,
                    algorithms=["HS256"],
                    audience="authenticated",
                )

                user_id = payload.get("sub")
                if user_id:
                    request.state.user_id = user_id

            except JWTError:
                # Invalid token - don't set user_id, let endpoint auth handle it
                pass
            except Exception as e:
                # Unexpected error - log but don't fail the request
                logger.debug("Failed to extract user_id from JWT", error=str(e))

        return await call_next(request)
