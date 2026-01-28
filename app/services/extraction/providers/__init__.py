"""Document extraction providers with runtime selection."""

from app.config import get_settings
from app.core.logging import get_logger

from .base import DocumentContent, DocumentProvider
from .docling import DoclingProvider
from .reducto import ReductoProvider

__all__ = [
    "DocumentContent",
    "DocumentProvider",
    "DoclingProvider",
    "ReductoProvider",
    "get_available_providers",
    "get_provider",
    "init_providers",
]

logger = get_logger(__name__)

# Global provider registry
_providers: dict[str, DocumentProvider] = {}
_initialized = False


def init_providers() -> dict[str, DocumentProvider]:
    """
    Initialize available document providers based on configuration.

    Providers are initialized once and cached. Docling is always available
    as it's a local library. Reducto requires an API key.

    Returns:
        Dictionary mapping provider names to provider instances.
    """
    global _providers, _initialized

    if _initialized:
        return _providers

    settings = get_settings()

    # Docling is always available (local processing)
    docling = DoclingProvider()
    if docling.is_available():
        _providers["docling"] = docling
        logger.info("Initialized Docling provider")
    else:
        logger.warning("Docling provider not available (docling package not installed)")

    # Reducto requires API key
    if settings.reducto_api_key:
        reducto = ReductoProvider(api_key=settings.reducto_api_key)
        if reducto.is_available():
            _providers["reducto"] = reducto
            logger.info("Initialized Reducto provider")
    else:
        logger.debug("Reducto provider not configured (no API key)")

    _initialized = True
    logger.info("Document providers initialized", available=list(_providers.keys()))

    return _providers


def get_provider(name: str | None = None) -> DocumentProvider:
    """
    Get a document provider by name, with fallback logic.

    If no name is provided, uses the default provider from settings.
    If the requested provider is not available, falls back to Docling.

    Args:
        name: Provider name ("docling", "reducto") or None for default.

    Returns:
        The requested or fallback DocumentProvider.

    Raises:
        RuntimeError: If no providers are available.
    """
    providers = init_providers()
    settings = get_settings()

    # Use default if no name specified
    provider_name = name or settings.extraction_default_provider

    # Normalize provider name
    if provider_name:
        provider_name = provider_name.lower().strip()

    # Return requested provider if available
    if provider_name in providers:
        logger.debug("Using provider", provider=provider_name)
        return providers[provider_name]

    # Log fallback
    if provider_name:
        logger.warning(
            "Requested provider not available, falling back",
            requested=provider_name,
            fallback="docling",
        )

    # Fallback to Docling
    if "docling" in providers:
        return providers["docling"]

    # Last resort: return any available provider
    if providers:
        fallback = next(iter(providers.values()))
        logger.warning("Using fallback provider", provider=fallback.name)
        return fallback

    raise RuntimeError("No document providers available")


def get_available_providers() -> list[str]:
    """
    Get list of available provider names.

    Returns:
        List of provider names that are currently available.
    """
    return list(init_providers().keys())


def reset_providers() -> None:
    """
    Reset the provider registry.

    Primarily used for testing to reinitialize providers with
    different configurations.
    """
    global _providers, _initialized
    _providers = {}
    _initialized = False
