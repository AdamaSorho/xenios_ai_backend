"""Abstract base class for document extraction providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentContent:
    """Normalized document extraction result from any provider."""

    text: str
    """Full text content extracted from the document."""

    markdown: str
    """Markdown formatted content."""

    tables: list[dict] = field(default_factory=list)
    """Extracted tables as list of dictionaries."""

    metadata: dict = field(default_factory=dict)
    """Document metadata (source path, format, provider-specific info)."""

    pages: int = 1
    """Page count of the document."""

    provider: str = ""
    """Name of the provider that extracted this content."""


class DocumentProvider(ABC):
    """
    Abstract base class for document extraction providers.

    Providers are responsible for extracting text, tables, and metadata
    from documents. They must normalize their output to DocumentContent
    regardless of the underlying extraction mechanism.
    """

    name: str = ""
    """Unique provider identifier."""

    @abstractmethod
    async def extract(self, file_path: Path) -> DocumentContent:
        """
        Extract content from a document file.

        Args:
            file_path: Path to the document file.

        Returns:
            DocumentContent with extracted text, tables, and metadata.

        Raises:
            Exception: If extraction fails.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is configured and available.

        Returns:
            True if the provider can be used, False otherwise.
        """
        pass

    @abstractmethod
    def supports_file_type(self, extension: str) -> bool:
        """
        Check if the provider supports the given file type.

        Args:
            extension: File extension including the dot (e.g., ".pdf").

        Returns:
            True if the provider can process this file type.
        """
        pass
