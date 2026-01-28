"""Reducto.ai cloud-based document extraction provider."""

from pathlib import Path

import httpx

from app.core.logging import get_logger

from .base import DocumentContent, DocumentProvider

logger = get_logger(__name__)


class ReductoProvider(DocumentProvider):
    """
    Cloud-based document extraction using Reducto.ai API.

    Reducto.ai provides high-accuracy ML-based document extraction
    with excellent support for complex layouts, tables, and forms.
    Requires an API key and incurs per-page costs.
    """

    name = "reducto"
    BASE_URL = "https://api.reducto.ai"

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, api_key: str, timeout: float = 120.0) -> None:
        """
        Initialize the Reducto provider.

        Args:
            api_key: Reducto.ai API key.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    async def extract(self, file_path: Path) -> DocumentContent:
        """
        Extract document content using Reducto.ai API.

        Args:
            file_path: Path to the document file.

        Returns:
            DocumentContent with extracted text, markdown, and tables.

        Raises:
            httpx.HTTPStatusError: If the API returns an error.
            Exception: If extraction fails.
        """
        logger.debug("Starting Reducto extraction", file_path=str(file_path))

        # Determine content type
        content_type = "application/pdf"
        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            content_type = f"image/{file_path.suffix.lower().lstrip('.')}"

        # Upload file and extract
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, content_type)}
            response = await self.client.post(
                "/v1/parse",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files=files,
                data={
                    "output_format": "markdown",
                    "extract_tables": "true",
                },
            )
            response.raise_for_status()
            data = response.json()

        # Extract content from response
        text = data.get("text", "")
        markdown = data.get("markdown", data.get("text", ""))
        tables = data.get("tables", [])
        page_count = data.get("page_count", data.get("pages", 1))

        logger.debug(
            "Reducto extraction complete",
            file_path=str(file_path),
            text_length=len(text),
            table_count=len(tables),
            page_count=page_count,
            job_id=data.get("job_id"),
        )

        return DocumentContent(
            text=text,
            markdown=markdown,
            tables=tables,
            metadata={
                "source": str(file_path),
                "format": file_path.suffix.lower(),
                "reducto_job_id": data.get("job_id"),
                "reducto_status": data.get("status"),
            },
            pages=page_count,
            provider=self.name,
        )

    def is_available(self) -> bool:
        """
        Check if Reducto is available.

        Returns:
            True if API key is configured.
        """
        return bool(self._api_key)

    def supports_file_type(self, extension: str) -> bool:
        """
        Check if Reducto supports the file type.

        Currently Reducto primarily supports PDFs.

        Args:
            extension: File extension including the dot.

        Returns:
            True if Reducto can process this file type.
        """
        return extension.lower() in self.SUPPORTED_EXTENSIONS

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
