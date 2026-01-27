"""Tests for chat API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException


class TestChatEndpoints:
    """Tests for /api/v1/chat endpoints."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock authenticated user."""
        user = MagicMock()
        user.user_id = str(uuid4())
        user.role = "authenticated"
        user.email = "coach@test.com"
        return user

    @pytest.mark.asyncio
    async def test_coach_cannot_access_other_clients(self, mock_user):
        """Test authorization - coach can only access their clients."""
        from app.core.auth import verify_coach_client_relationship

        with patch("app.core.auth.text") as mock_text:
            # Setup: coach-client relationship doesn't exist
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_db.execute.return_value = mock_result

            with pytest.raises(HTTPException) as exc_info:
                await verify_coach_client_relationship(
                    mock_db,
                    coach_id=mock_user.user_id,
                    client_id=str(uuid4()),
                    raise_404=True,
                )

            # Should return 404, not 403
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_client_returns_404(self, mock_user):
        """Test that invalid client_id returns 404 not 403."""
        from app.core.auth import verify_coach_client_relationship

        with patch("app.core.auth.text") as mock_text:
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_db.execute.return_value = mock_result

            with pytest.raises(HTTPException) as exc_info:
                await verify_coach_client_relationship(
                    mock_db,
                    coach_id=mock_user.user_id,
                    client_id=str(uuid4()),
                    raise_404=True,
                )

            assert exc_info.value.status_code == 404
            assert "not found" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_valid_relationship_passes(self, mock_user):
        """Test that valid coach-client relationship allows access."""
        from app.core.auth import verify_coach_client_relationship

        with patch("app.core.auth.text") as mock_text:
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = 1  # Relationship exists
            mock_db.execute.return_value = mock_result

            result = await verify_coach_client_relationship(
                mock_db,
                coach_id=mock_user.user_id,
                client_id=str(uuid4()),
                raise_404=True,
            )

            assert result is True


class TestChatResponseFormats:
    """Tests for chat response format compliance."""

    def test_chat_response_has_required_fields(self):
        """Test ChatResponse has all spec-required fields."""
        from app.schemas.rag import ChatResponse

        response = ChatResponse(
            response="Test response",
            sources=[],
            confidence=0.8,
            has_context=True,
            conversation_id=uuid4(),
            tokens_used=100,
        )

        assert response.response == "Test response"
        assert response.sources == []
        assert response.confidence == 0.8
        assert response.has_context is True
        assert response.conversation_id is not None
        assert response.tokens_used == 100

    def test_source_citation_format(self):
        """Test SourceCitation format matches spec."""
        from app.schemas.rag import SourceCitation

        citation = SourceCitation(
            source_type="health_profile",
            source_id="test-123",
            relevance_score=0.85,
            snippet="Test snippet content",
            date=None,
        )

        assert citation.source_type == "health_profile"
        assert citation.relevance_score == 0.85

    def test_no_context_response_format(self):
        """Test response format when no context found."""
        from app.schemas.rag import ChatResponse

        response = ChatResponse(
            response="I don't have specific data about this.",
            sources=[],
            confidence=0.0,
            has_context=False,
            conversation_id=uuid4(),
            tokens_used=50,
        )

        assert response.has_context is False
        assert response.confidence == 0.0
        assert response.sources == []
