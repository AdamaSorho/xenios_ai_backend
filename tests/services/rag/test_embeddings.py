"""Tests for embedding service."""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.schemas.rag import EmbeddingSourceType


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        return db

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI embedding client."""
        with patch("app.services.rag.embeddings.OpenAIEmbeddingClient") as mock:
            client = AsyncMock()
            client.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_content_hashing_prevents_duplicate(self, mock_db, mock_openai_client):
        """Test that same content doesn't regenerate embedding."""
        from app.services.rag.embeddings import EmbeddingService

        # Setup: existing embedding with same hash
        text = "test content"
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = content_hash
        mock_db.execute.return_value = mock_result

        service = EmbeddingService(mock_db)

        # Test _should_update returns False for same hash
        should_update = await service._should_update(
            client_id=uuid4(),
            source_type="test",
            source_id="test_id",
            text=text,
            force=False,
        )

        assert should_update is False

    @pytest.mark.asyncio
    async def test_force_updates_even_if_unchanged(self, mock_db, mock_openai_client):
        """Test that force=True bypasses hash check."""
        from app.services.rag.embeddings import EmbeddingService

        service = EmbeddingService(mock_db)

        # Test _should_update returns True when force=True
        should_update = await service._should_update(
            client_id=uuid4(),
            source_type="test",
            source_id="test_id",
            text="test content",
            force=True,
        )

        assert should_update is True

    @pytest.mark.asyncio
    async def test_new_content_triggers_update(self, mock_db, mock_openai_client):
        """Test that new content (different hash) triggers update."""
        from app.services.rag.embeddings import EmbeddingService

        # Setup: no existing embedding
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = EmbeddingService(mock_db)

        should_update = await service._should_update(
            client_id=uuid4(),
            source_type="test",
            source_id="test_id",
            text="new content",
            force=False,
        )

        assert should_update is True

    @pytest.mark.asyncio
    async def test_store_embedding_creates_new(self, mock_db, mock_openai_client):
        """Test that _store_embedding creates new embedding when none exists."""
        from app.services.rag.embeddings import EmbeddingService

        # Setup: no existing embedding
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = EmbeddingService(mock_db)

        await service._store_embedding(
            client_id=uuid4(),
            source_type="test",
            source_id="test_id",
            text="test content",
        )

        # Verify embedding was added
        assert mock_db.add.called
        assert mock_db.flush.called

    @pytest.mark.asyncio
    async def test_update_client_embeddings_processes_all_types(self, mock_db, mock_openai_client):
        """Test that update_client_embeddings processes all source types."""
        from app.services.rag.embeddings import EmbeddingService

        # Setup: mock all the internal methods
        service = EmbeddingService(mock_db)

        with patch.object(service, "_update_health_profile_embedding", return_value=(1, 0)):
            with patch.object(service, "_update_metric_summary_embeddings", return_value=(2, 0)):
                with patch.object(service, "_update_session_summary_embeddings", return_value=(1, 0)):
                    with patch.object(service, "_update_checkin_summary_embeddings", return_value=(1, 0)):
                        with patch.object(service, "_update_lab_result_embeddings", return_value=(1, 0)):
                            with patch.object(service, "_update_health_goal_embeddings", return_value=(1, 0)):
                                with patch.object(service, "_update_message_thread_embeddings", return_value=(1, 0)):
                                    result = await service.update_client_embeddings(
                                        client_id=uuid4(),
                                        source_types=None,  # All types
                                        force=False,
                                    )

        assert result.updated_count == 8  # Sum of all updated
        assert result.skipped_count == 0
