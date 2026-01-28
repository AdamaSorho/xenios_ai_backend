"""Tests for document extraction providers."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.extraction.providers.base import DocumentContent, DocumentProvider
from app.services.extraction.providers.docling import DoclingProvider
from app.services.extraction.providers.reducto import ReductoProvider
from app.services.extraction.providers import (
    get_available_providers,
    get_provider,
    init_providers,
    reset_providers,
)


class TestDocumentContent:
    """Tests for DocumentContent dataclass."""

    def test_create_document_content(self):
        """Test creating a DocumentContent instance."""
        content = DocumentContent(
            text="Sample text content",
            markdown="# Sample markdown",
            tables=[{"col1": ["a", "b"], "col2": [1, 2]}],
            metadata={"source": "/path/to/file.pdf"},
            pages=3,
            provider="docling",
        )

        assert content.text == "Sample text content"
        assert content.markdown == "# Sample markdown"
        assert len(content.tables) == 1
        assert content.pages == 3
        assert content.provider == "docling"

    def test_document_content_defaults(self):
        """Test DocumentContent default values."""
        content = DocumentContent(text="text", markdown="markdown")

        assert content.tables == []
        assert content.metadata == {}
        assert content.pages == 1
        assert content.provider == ""


class TestDoclingProvider:
    """Tests for DoclingProvider."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        provider = DoclingProvider()
        assert provider.name == "docling"

    def test_supported_file_types(self):
        """Test supported file type detection."""
        provider = DoclingProvider()

        assert provider.supports_file_type(".pdf")
        assert provider.supports_file_type(".PDF")
        assert provider.supports_file_type(".png")
        assert provider.supports_file_type(".jpg")
        assert provider.supports_file_type(".jpeg")
        assert provider.supports_file_type(".tiff")

        assert not provider.supports_file_type(".csv")
        assert not provider.supports_file_type(".txt")
        assert not provider.supports_file_type(".docx")

    def test_is_available_with_docling_installed(self):
        """Test availability when docling is installed."""
        provider = DoclingProvider()

        # Mock the import to succeed
        with patch.dict("sys.modules", {"docling.document_converter": MagicMock()}):
            assert provider.is_available()

    def test_is_available_without_docling(self):
        """Test availability when docling is not installed."""
        provider = DoclingProvider()

        # Mock the import to fail
        with patch.dict("sys.modules", {"docling.document_converter": None}):
            with patch("builtins.__import__", side_effect=ImportError("No docling")):
                assert not provider.is_available()

    @pytest.mark.asyncio
    async def test_extract_returns_document_content(self):
        """Test that extract returns DocumentContent."""
        provider = DoclingProvider()

        # Create mock document converter result
        mock_doc = MagicMock()
        mock_doc.texts = []
        mock_doc.tables = []
        mock_doc.export_to_markdown.return_value = "# Test Document\n\nContent here."

        mock_result = MagicMock()
        mock_result.document = mock_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        provider._converter = mock_converter

        # Test extraction
        content = await provider.extract(Path("/fake/path/test.pdf"))

        assert isinstance(content, DocumentContent)
        assert content.provider == "docling"
        assert "Test Document" in content.markdown

    @pytest.mark.asyncio
    async def test_extract_handles_tables(self):
        """Test that extract properly handles tables."""
        provider = DoclingProvider()

        # Create mock with table
        mock_table = MagicMock()
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [{"col1": "a", "col2": 1}]
        mock_table.export_to_dataframe.return_value = mock_df

        mock_doc = MagicMock()
        mock_doc.texts = []
        mock_doc.tables = [mock_table]
        mock_doc.export_to_markdown.return_value = "| col1 | col2 |"

        mock_result = MagicMock()
        mock_result.document = mock_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        provider._converter = mock_converter

        content = await provider.extract(Path("/fake/test.pdf"))

        assert len(content.tables) == 1


class TestReductoProvider:
    """Tests for ReductoProvider."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        provider = ReductoProvider(api_key="test-key")
        assert provider.name == "reducto"

    def test_supported_file_types(self):
        """Test supported file type detection."""
        provider = ReductoProvider(api_key="test-key")

        assert provider.supports_file_type(".pdf")
        assert provider.supports_file_type(".PDF")

        assert not provider.supports_file_type(".csv")
        assert not provider.supports_file_type(".png")

    def test_is_available_with_api_key(self):
        """Test availability with API key configured."""
        provider = ReductoProvider(api_key="rsk_test123")
        assert provider.is_available()

    def test_is_not_available_without_api_key(self):
        """Test availability without API key."""
        provider = ReductoProvider(api_key="")
        assert not provider.is_available()

    @pytest.mark.asyncio
    async def test_extract_calls_api(self):
        """Test that extract makes the correct API call."""
        provider = ReductoProvider(api_key="test-key")

        # Create a mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Extracted text content",
            "markdown": "# Extracted markdown",
            "tables": [{"header": ["A", "B"], "data": [[1, 2]]}],
            "page_count": 2,
            "job_id": "job-123",
        }
        mock_response.raise_for_status = MagicMock()

        # Mock the HTTP client
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        provider._client = mock_client

        # Create a temporary file path
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf content")
            temp_path = Path(f.name)

        try:
            content = await provider.extract(temp_path)

            # Verify API was called
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args

            assert call_args[0][0] == "/v1/parse"
            assert "Authorization" in call_args[1]["headers"]

            # Verify response parsing
            assert isinstance(content, DocumentContent)
            assert content.text == "Extracted text content"
            assert content.markdown == "# Extracted markdown"
            assert content.pages == 2
            assert content.provider == "reducto"
            assert content.metadata["reducto_job_id"] == "job-123"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test that close properly closes the HTTP client."""
        provider = ReductoProvider(api_key="test-key")

        mock_client = AsyncMock()
        mock_client.is_closed = False
        provider._client = mock_client

        await provider.close()

        mock_client.aclose.assert_called_once()


class TestProviderManager:
    """Tests for provider initialization and selection."""

    def setup_method(self):
        """Reset providers before each test."""
        reset_providers()

    def teardown_method(self):
        """Reset providers after each test."""
        reset_providers()

    def test_init_providers_with_no_reducto_key(self):
        """Test initialization when no Reducto API key is configured."""
        # Set up test environment without Reducto key
        with patch("app.services.extraction.providers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                reducto_api_key="",
                extraction_default_provider="docling",
            )

            # Need to also mock Docling availability
            with patch.object(DoclingProvider, "is_available", return_value=True):
                providers = init_providers()

                assert "docling" in providers
                assert "reducto" not in providers

    def test_init_providers_with_reducto_key(self):
        """Test initialization with Reducto API key configured."""
        with patch("app.services.extraction.providers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                reducto_api_key="rsk_test123",
                extraction_default_provider="docling",
            )

            with patch.object(DoclingProvider, "is_available", return_value=True):
                providers = init_providers()

                assert "docling" in providers
                assert "reducto" in providers

    def test_get_provider_default(self):
        """Test getting the default provider."""
        with patch("app.services.extraction.providers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                reducto_api_key="",
                extraction_default_provider="docling",
            )

            with patch.object(DoclingProvider, "is_available", return_value=True):
                provider = get_provider()

                assert provider.name == "docling"

    def test_get_provider_by_name(self):
        """Test getting a specific provider by name."""
        with patch("app.services.extraction.providers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                reducto_api_key="rsk_test123",
                extraction_default_provider="docling",
            )

            with patch.object(DoclingProvider, "is_available", return_value=True):
                provider = get_provider("reducto")

                assert provider.name == "reducto"

    def test_get_provider_fallback_to_docling(self):
        """Test fallback to docling when requested provider unavailable."""
        with patch("app.services.extraction.providers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                reducto_api_key="",  # No Reducto key
                extraction_default_provider="docling",
            )

            with patch.object(DoclingProvider, "is_available", return_value=True):
                # Request reducto but it's not available
                provider = get_provider("reducto")

                # Should fallback to docling
                assert provider.name == "docling"

    def test_get_provider_case_insensitive(self):
        """Test that provider name is case insensitive."""
        with patch("app.services.extraction.providers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                reducto_api_key="",
                extraction_default_provider="docling",
            )

            with patch.object(DoclingProvider, "is_available", return_value=True):
                provider = get_provider("DOCLING")
                assert provider.name == "docling"

                provider = get_provider("Docling")
                assert provider.name == "docling"

    def test_get_available_providers(self):
        """Test listing available providers."""
        with patch("app.services.extraction.providers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                reducto_api_key="rsk_test",
                extraction_default_provider="docling",
            )

            with patch.object(DoclingProvider, "is_available", return_value=True):
                available = get_available_providers()

                assert "docling" in available
                assert "reducto" in available

    def test_get_provider_no_providers_raises(self):
        """Test that RuntimeError is raised when no providers available."""
        with patch("app.services.extraction.providers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                reducto_api_key="",
                extraction_default_provider="docling",
            )

            # Make docling unavailable too
            with patch.object(DoclingProvider, "is_available", return_value=False):
                with pytest.raises(RuntimeError, match="No document providers available"):
                    get_provider()


class TestExtractionResultMetadata:
    """Tests for ExtractionResult metadata field."""

    def test_extraction_result_with_metadata(self):
        """Test ExtractionResult includes metadata."""
        from app.services.extraction.base import ExtractionResult

        result = ExtractionResult(
            success=True,
            data={"weight_kg": 75.5},
            confidence=0.95,
            metadata={"provider": "docling", "custom_field": "value"},
        )

        assert result.metadata["provider"] == "docling"
        assert result.metadata["custom_field"] == "value"

    def test_extraction_result_default_metadata(self):
        """Test ExtractionResult has empty dict as default metadata."""
        from app.services.extraction.base import ExtractionResult

        result = ExtractionResult(success=True, confidence=0.9)

        assert result.metadata == {}
