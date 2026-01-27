"""FastAPI dependency injection utilities."""

from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.core.auth import UserContext, get_current_user, get_optional_user

# Type aliases for cleaner dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
CurrentUser = Annotated[UserContext, Depends(get_current_user)]
OptionalUser = Annotated[UserContext | None, Depends(get_optional_user)]
