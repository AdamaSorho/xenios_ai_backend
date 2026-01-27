"""Authentication utilities for API key and JWT verification."""

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import Settings, get_settings

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class UserContext(BaseModel):
    """User context extracted from JWT token."""

    user_id: str
    role: str
    email: str | None = None


async def verify_api_key(
    api_key: Annotated[str | None, Security(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    """
    Verify the API key from X-API-Key header.

    This is required for all API requests from the Xenios MVP frontend.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.xenios_backend_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


async def verify_jwt(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UserContext:
    """
    Verify the JWT token from Authorization header.

    The JWT is issued by Supabase Auth and contains user information.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Decode JWT using Supabase secret
        payload = jwt.decode(
            credentials.credentials,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )

        # Extract user information from claims
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )

        return UserContext(
            user_id=user_id,
            role=payload.get("role", "authenticated"),
            email=payload.get("email"),
        )

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


async def get_current_user(
    _api_key: Annotated[str, Depends(verify_api_key)],
    user: Annotated[UserContext, Depends(verify_jwt)],
) -> UserContext:
    """
    Get the current authenticated user.

    Requires both API key and valid JWT.
    Use this dependency for protected endpoints.
    """
    return user


async def verify_coach_client_relationship(
    db,
    coach_id: str,
    client_id: str,
    raise_404: bool = False,
) -> bool:
    """
    Verify that a coach has access to a client.

    Args:
        db: Database session
        coach_id: The coach's user ID
        client_id: The client's ID
        raise_404: If True, raise 404 instead of 403 (prevents enumeration)

    Returns:
        True if relationship exists

    Raises:
        HTTPException: 404 or 403 if relationship doesn't exist
    """
    from sqlalchemy import text

    # Query the coach_clients relationship table
    query = text("""
        SELECT 1 FROM public.coach_clients
        WHERE coach_id = :coach_id
        AND client_id = :client_id
        AND status = 'active'
        LIMIT 1
    """)

    result = await db.execute(query, {"coach_id": coach_id, "client_id": client_id})
    relationship_exists = result.scalar_one_or_none() is not None

    if not relationship_exists:
        # Per spec: Return 404 instead of 403 to prevent client ID enumeration
        if raise_404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this client",
            )

    return True


async def get_optional_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UserContext | None:
    """
    Get the current user if authenticated, None otherwise.

    Does not require API key - use for optional auth scenarios.
    """
    if not credentials:
        return None

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )

        user_id = payload.get("sub")
        if not user_id:
            return None

        return UserContext(
            user_id=user_id,
            role=payload.get("role", "authenticated"),
            email=payload.get("email"),
        )
    except JWTError:
        return None
