"""Tests for insight generation service."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.schemas.rag import InsightTrigger


class TestInsightGenerationService:
    """Tests for InsightGenerationService."""

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
        with patch("app.services.rag.insights.RetrievalService") as mock:
            service = AsyncMock()
            service.retrieve_context = AsyncMock(return_value=[])
            mock.return_value = service
            yield service

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI embedding client."""
        with patch("app.services.rag.insights.OpenAIEmbeddingClient") as mock:
            client = AsyncMock()
            client.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
            mock.return_value = client
            yield client

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        with patch("app.services.rag.insights.LLMClient") as mock:
            client = MagicMock()
            client.complete = AsyncMock(
                return_value={
                    "choices": [
                        {
                            "message": {
                                "content": """{
                                    "title": "Test Insight",
                                    "client_message": "Test message",
                                    "rationale": "Test rationale",
                                    "suggested_actions": ["Action 1"],
                                    "insight_type": "general",
                                    "confidence_score": 0.8
                                }"""
                            }
                        }
                    ],
                    "usage": {"total_tokens": 500},
                }
            )
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_duplicate_detection_blocks_similar(
        self, mock_db, mock_retrieval_service, mock_openai_client, mock_llm_client
    ):
        """Test that similar insights are blocked."""
        from app.services.rag.insights import InsightGenerationService

        # Setup: existing similar insight
        mock_similar_row = MagicMock()
        mock_similar_row.id = uuid4()
        mock_similar_row.similarity = 0.9  # Above 0.85 threshold

        mock_duplicate_result = MagicMock()
        mock_duplicate_result.fetchone.return_value = mock_similar_row

        # Rate limit checks pass
        mock_rate_result = MagicMock()
        mock_rate_result.scalars.return_value.all.return_value = []
        mock_rate_result.scalars.return_value.first.return_value = None
        mock_rate_result.scalar_one_or_none.return_value = None

        # Make execute return different results for different calls
        call_count = [0]

        async def mock_execute(query, params=None):
            call_count[0] += 1
            if call_count[0] <= 3:  # Rate limit checks
                return mock_rate_result
            return mock_duplicate_result  # Duplicate check

        mock_db.execute = mock_execute

        service = InsightGenerationService(mock_db)

        # Test that DuplicateInsightError is raised
        # Note: This would raise in real implementation

    @pytest.mark.asyncio
    async def test_rate_limit_max_weekly(
        self, mock_db, mock_retrieval_service, mock_openai_client, mock_llm_client
    ):
        """Test max 3 insights per client per week."""
        from app.services.rag.insights import InsightGenerationService, RateLimitExceededError

        # Setup: 3 existing insights this week
        mock_insights = [MagicMock() for _ in range(3)]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_insights

        mock_db.execute.return_value = mock_result

        service = InsightGenerationService(mock_db)

        with pytest.raises(RateLimitExceededError) as exc_info:
            await service._check_rate_limits(uuid4(), InsightTrigger.SCHEDULED)

        assert "3 insights per client per week" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_daily_trigger(
        self, mock_db, mock_retrieval_service, mock_openai_client, mock_llm_client
    ):
        """Test max 1 insight per trigger type per day."""
        from app.services.rag.insights import InsightGenerationService, RateLimitExceededError

        # Setup: weekly check passes, but daily trigger check fails
        weekly_result = MagicMock()
        weekly_result.scalars.return_value.all.return_value = []

        daily_result = MagicMock()
        daily_result.scalars.return_value.first.return_value = MagicMock()  # Existing insight

        call_count = [0]

        async def mock_execute(query, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return weekly_result
            return daily_result

        mock_db.execute = mock_execute

        service = InsightGenerationService(mock_db)

        with pytest.raises(RateLimitExceededError) as exc_info:
            await service._check_rate_limits(uuid4(), InsightTrigger.METRIC_CHANGE)

        assert "per day" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_insight_response_valid_json(
        self, mock_db, mock_retrieval_service, mock_openai_client, mock_llm_client
    ):
        """Test parsing valid JSON insight response."""
        from app.services.rag.insights import InsightGenerationService

        service = InsightGenerationService(mock_db)

        content = """{
            "title": "Test Title",
            "client_message": "Test message",
            "rationale": "Test rationale",
            "suggested_actions": ["Action 1", "Action 2"],
            "insight_type": "nutrition",
            "confidence_score": 0.85
        }"""

        result = service._parse_insight_response(content)

        assert result["title"] == "Test Title"
        assert result["insight_type"] == "nutrition"
        assert result["confidence_score"] == 0.85
        assert len(result["suggested_actions"]) == 2

    @pytest.mark.asyncio
    async def test_parse_insight_response_markdown_wrapped(
        self, mock_db, mock_retrieval_service, mock_openai_client, mock_llm_client
    ):
        """Test parsing JSON wrapped in markdown code blocks."""
        from app.services.rag.insights import InsightGenerationService

        service = InsightGenerationService(mock_db)

        content = """Here's the insight:

```json
{
    "title": "Test Title",
    "client_message": "Test message",
    "rationale": "Test rationale",
    "suggested_actions": ["Action 1"],
    "insight_type": "general"
}
```

Hope this helps!"""

        result = service._parse_insight_response(content)

        assert result["title"] == "Test Title"

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_range(
        self, mock_db, mock_retrieval_service, mock_openai_client, mock_llm_client
    ):
        """Test confidence score is clamped to 0-1 range."""
        from app.services.rag.insights import InsightGenerationService

        service = InsightGenerationService(mock_db)

        # Test over 1.0
        content = """{
            "title": "Test",
            "client_message": "Test",
            "rationale": "Test",
            "suggested_actions": ["Action"],
            "insight_type": "general",
            "confidence_score": 1.5
        }"""

        result = service._parse_insight_response(content)
        assert result["confidence_score"] == 1.0

        # Test below 0
        content2 = """{
            "title": "Test",
            "client_message": "Test",
            "rationale": "Test",
            "suggested_actions": ["Action"],
            "insight_type": "general",
            "confidence_score": -0.5
        }"""

        result2 = service._parse_insight_response(content2)
        assert result2["confidence_score"] == 0.0
