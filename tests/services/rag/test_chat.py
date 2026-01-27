"""Tests for chat service."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.schemas.rag import SearchResult


class TestChatService:
    """Tests for ChatService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        return db

    @pytest.fixture
    def mock_retrieval_service(self):
        """Create a mock retrieval service."""
        with patch("app.services.rag.chat.RetrievalService") as mock:
            service = AsyncMock()
            service.retrieve_context = AsyncMock(return_value=[])
            mock.return_value = service
            yield service

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        with patch("app.services.rag.chat.LLMClient") as mock:
            client = MagicMock()
            client.complete = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Test response"}}],
                    "usage": {"total_tokens": 100},
                }
            )
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_no_context_response_has_flag(
        self, mock_db, mock_retrieval_service, mock_llm_client
    ):
        """Test that has_context=false when no embeddings found."""
        from app.services.rag.chat import ChatService

        # Setup: no context retrieved
        mock_retrieval_service.retrieve_context.return_value = []

        # Mock conversation history query
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        service = ChatService(mock_db)

        response = await service.generate_response(
            client_id=uuid4(),
            coach_id=uuid4(),
            message="test message",
        )

        assert response.has_context is False
        assert response.confidence == 0.0
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_conversation_history_loaded(
        self, mock_db, mock_retrieval_service, mock_llm_client
    ):
        """Test that previous messages are included in context."""
        from app.services.rag.chat import ChatService

        conversation_id = uuid4()

        # Setup: mock conversation history
        mock_history = MagicMock()
        mock_history.role = "user"
        mock_history.content = "previous message"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_history]
        mock_db.execute.return_value = mock_result

        service = ChatService(mock_db)

        await service.generate_response(
            client_id=uuid4(),
            coach_id=uuid4(),
            message="test message",
            conversation_id=conversation_id,
        )

        # Verify LLM was called with conversation history
        assert mock_llm_client.complete.called

    @pytest.mark.asyncio
    async def test_context_policy_limits_items(
        self, mock_db, mock_retrieval_service, mock_llm_client
    ):
        """Test that context policy limits number of items."""
        from app.services.rag.chat import ChatService

        # Create many context items
        contexts = [
            SearchResult(
                source_type="health_profile",
                source_id=f"test_{i}",
                content=f"content {i}",
                relevance_score=0.9 - i * 0.05,
                metadata={},
            )
            for i in range(15)
        ]

        mock_retrieval_service.retrieve_context.return_value = contexts

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        service = ChatService(mock_db)

        # Apply context policy
        limited = service._apply_context_policy(contexts, max_items=10, max_tokens=4000)

        assert len(limited) <= 10

    @pytest.mark.asyncio
    async def test_context_policy_priority_ordering(
        self, mock_db, mock_retrieval_service, mock_llm_client
    ):
        """Test that context is ordered by priority."""
        from app.services.rag.chat import ChatService

        # Create contexts of different types
        contexts = [
            SearchResult(
                source_type="message_thread",  # Lower priority
                source_id="msg",
                content="message",
                relevance_score=0.95,
                metadata={},
            ),
            SearchResult(
                source_type="health_profile",  # Higher priority
                source_id="profile",
                content="profile",
                relevance_score=0.8,
                metadata={},
            ),
        ]

        service = ChatService(mock_db)

        ordered = service._apply_context_policy(contexts, max_items=10, max_tokens=4000)

        # Health profile should come first despite lower relevance score
        assert ordered[0].source_type == "health_profile"

    @pytest.mark.asyncio
    async def test_sources_included_when_requested(
        self, mock_db, mock_retrieval_service, mock_llm_client
    ):
        """Test that sources are included when include_sources=True."""
        from app.services.rag.chat import ChatService

        contexts = [
            SearchResult(
                source_type="health_profile",
                source_id="test",
                content="test content",
                relevance_score=0.9,
                metadata={},
            )
        ]
        mock_retrieval_service.retrieve_context.return_value = contexts

        # Mock LLM to include source marker
        mock_llm_client.complete.return_value = {
            "choices": [{"message": {"content": "Based on [Source 1]: test content..."}}],
            "usage": {"total_tokens": 100},
        }

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        service = ChatService(mock_db)

        response = await service.generate_response(
            client_id=uuid4(),
            coach_id=uuid4(),
            message="test message",
            include_sources=True,
        )

        assert response.has_context is True
        assert len(response.sources) > 0

    @pytest.mark.asyncio
    async def test_confidence_calculated_from_relevance(
        self, mock_db, mock_retrieval_service, mock_llm_client
    ):
        """Test that confidence is based on average relevance score."""
        from app.services.rag.chat import ChatService

        contexts = [
            SearchResult(source_type="test", source_id="1", content="c1", relevance_score=0.9, metadata={}),
            SearchResult(source_type="test", source_id="2", content="c2", relevance_score=0.8, metadata={}),
        ]

        service = ChatService(mock_db)
        confidence = service._calculate_confidence(contexts)

        assert confidence == 0.85  # Average of 0.9 and 0.8

    @pytest.mark.asyncio
    async def test_no_context_confidence_zero(
        self, mock_db, mock_retrieval_service, mock_llm_client
    ):
        """Test that confidence is 0 when no context."""
        from app.services.rag.chat import ChatService

        service = ChatService(mock_db)
        confidence = service._calculate_confidence([])

        assert confidence == 0.0
