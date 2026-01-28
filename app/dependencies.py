"""FastAPI dependency injection utilities."""

from typing import Annotated

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.core.auth import UserContext, get_current_user, get_optional_user
from app.services.llm.client import LLMClient

# Valid LLM providers for header validation
VALID_LLM_PROVIDERS = ["openrouter", "anthropic"]


def get_llm_client(
    x_llm_provider: Annotated[str | None, Header(alias="X-LLM-Provider")] = None,
) -> LLMClient:
    """
    Get LLM client with optional provider override via header.

    The X-LLM-Provider header allows clients to select which LLM provider
    to use for a specific request. Valid values are 'openrouter' or 'anthropic'.
    Invalid values are ignored and the default provider is used.

    Args:
        x_llm_provider: Optional header value for provider selection

    Returns:
        LLMClient configured with the appropriate provider
    """
    provider = None

    if x_llm_provider:
        normalized = x_llm_provider.lower().strip()
        if normalized in VALID_LLM_PROVIDERS:
            provider = normalized

    return LLMClient(provider=provider)


# Type aliases for cleaner dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
CurrentUser = Annotated[UserContext, Depends(get_current_user)]
OptionalUser = Annotated[UserContext | None, Depends(get_optional_user)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
