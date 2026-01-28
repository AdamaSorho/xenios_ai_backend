"""Docling-based document extraction provider (local)."""

import asyncio
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

from .base import DocumentContent, DocumentProvider

logger = get_logger(__name__)


class DoclingProvider(DocumentProvider):
    """
    Local document extraction using IBM Docling.

    Docling provides local, free PDF and image extraction with support
    for tables, text, and document structure. It runs synchronously
    so extraction is wrapped in a thread pool executor.
    """

    name = "docling"

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(self) -> None:
        self._converter: Any = None

    @property
    def converter(self) -> Any:
        """
        Lazy-load the DocumentConverter.

        Docling is a heavy import, so we defer loading until first use.
        """
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
            logger.info("Docling DocumentConverter initialized")
        return self._converter

    async def extract(self, file_path: Path) -> DocumentContent:
        """
        Extract document content using Docling.

        Runs Docling synchronously in a thread pool executor to avoid
        blocking the async event loop.

        Args:
            file_path: Path to the document file.

        Returns:
            DocumentContent with extracted text, markdown, and tables.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_sync, file_path)

    def _extract_sync(self, file_path: Path) -> DocumentContent:
        """
        Synchronous extraction implementation.

        Args:
            file_path: Path to the document file.

        Returns:
            DocumentContent with extracted data.
        """
        logger.debug("Starting Docling extraction", file_path=str(file_path))

        result = self.converter.convert(str(file_path))
        doc = result.document

        # Extract plain text from document
        text_parts: list[str] = []
        if hasattr(doc, "texts"):
            for text_block in doc.texts:
                if hasattr(text_block, "text") and text_block.text:
                    text_parts.append(text_block.text)

        # Extract tables
        tables: list[dict] = []
        if hasattr(doc, "tables"):
            for table in doc.tables:
                try:
                    if hasattr(table, "export_to_dataframe"):
                        df = table.export_to_dataframe()
                        tables.append(df.to_dict(orient="records"))
                    elif hasattr(table, "cells"):
                        # Fallback: extract cells directly
                        table_data: list[list[str]] = []
                        for row in table.cells:
                            row_data = []
                            for cell in row:
                                if hasattr(cell, "text"):
                                    row_data.append(cell.text)
                            table_data.append(row_data)
                        tables.append({"rows": table_data})
                except Exception as e:
                    logger.warning("Failed to extract table", error=str(e))

        # Get markdown export
        markdown = ""
        try:
            markdown = doc.export_to_markdown()
        except Exception as e:
            logger.warning("Failed to export markdown", error=str(e))
            # Fallback: use text parts
            markdown = "\n\n".join(text_parts)

        # If text_parts is empty but we have markdown, extract text from markdown
        if not text_parts and markdown:
            text_parts = [markdown]

        # Get page count
        pages = 1
        if hasattr(doc, "pages"):
            pages = len(doc.pages)

        logger.debug(
            "Docling extraction complete",
            file_path=str(file_path),
            text_length=len("\n".join(text_parts)),
            table_count=len(tables),
            page_count=pages,
        )

        return DocumentContent(
            text="\n".join(text_parts),
            markdown=markdown,
            tables=tables,
            metadata={
                "source": str(file_path),
                "format": file_path.suffix.lower(),
            },
            pages=pages,
            provider=self.name,
        )

    def is_available(self) -> bool:
        """
        Check if Docling is available.

        Docling is a local library, so it's always available if installed.
        """
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_file_type(self, extension: str) -> bool:
        """
        Check if Docling supports the file type.

        Args:
            extension: File extension including the dot.

        Returns:
            True if Docling can process this file type.
        """
        return extension.lower() in self.SUPPORTED_EXTENSIONS
