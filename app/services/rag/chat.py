"""Chat service for RAG-grounded responses."""

from collections.abc import AsyncIterator
from datetime import datetime
from uuid import UUID, uuid4

import tiktoken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.logging import get_logger
from app.models.rag import ChatHistory
from app.schemas.rag import ChatResponse, SearchResult, SourceCitation, StreamChunk
from app.services.llm.client import LLMClient
from app.services.rag.prompts import (
    GROUNDED_CHAT_SYSTEM_PROMPT,
    NO_CONTEXT_SYSTEM_PROMPT,
    build_context_string,
)
from app.services.rag.retrieval import RetrievalService

logger = get_logger(__name__)


class ChatService:
    """
    Service for generating grounded chat responses.

    Retrieves relevant context from embeddings and generates responses
    that cite sources and stay grounded in the provided data.
    """

    # Context window policy
    CONTEXT_PRIORITY = [
        "health_profile",
        "session_summary",
        "health_metric_summary",
        "checkin_summary",
        "lab_result",
        "health_goal",
        "message_thread",
    ]

    def __init__(
        self,
        db: AsyncSession,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.db = db
        self.retrieval_service = RetrievalService(db)
        self.llm_client = llm_client or LLMClient()
        self.settings = get_settings()

        # Token counting for context window management
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def generate_response(
        self,
        client_id: UUID,
        coach_id: UUID,
        message: str,
        conversation_id: UUID | None = None,
        max_context_items: int = 10,
        include_sources: bool = True,
    ) -> ChatResponse:
        """
        Generate a grounded chat response.

        Args:
            client_id: Client being discussed
            coach_id: Coach making the request
            message: User's message
            conversation_id: Optional conversation ID for continuity
            max_context_items: Maximum context items to include
            include_sources: Whether to include source citations

        Returns:
            ChatResponse with response text, sources, and metadata
        """
        # Load conversation history if continuing
        conversation_history = []
        if conversation_id:
            conversation_history = await self._load_conversation_history(
                conversation_id=conversation_id,
                limit=10,
            )

        # Retrieve relevant context (get extra for filtering)
        raw_contexts = await self.retrieval_service.retrieve_context(
            client_id=client_id,
            query=message,
            max_items=max_context_items * 2,
        )

        # Apply context policy (priority ordering, token limits)
        contexts = self._apply_context_policy(
            raw_contexts,
            max_items=max_context_items,
            max_tokens=self.settings.rag_max_context_tokens,
        )

        # Build context string and select prompt
        has_context = len(contexts) > 0
        if has_context:
            context_str = build_context_string(contexts, include_markers=True)
            system_prompt = GROUNDED_CHAT_SYSTEM_PROMPT.format(context=context_str)
        else:
            system_prompt = NO_CONTEXT_SYSTEM_PROMPT

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        for hist_msg in conversation_history:
            messages.append({"role": hist_msg.role, "content": hist_msg.content})
        messages.append({"role": "user", "content": message})

        # Generate response
        result = await self.llm_client.complete(task="chat", messages=messages)

        response_content = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)

        # Extract citations
        sources: list[SourceCitation] = []
        if include_sources and has_context:
            sources = self._extract_citations(response_content, contexts)

        # Generate or use conversation ID
        final_conversation_id = conversation_id or uuid4()

        # Persist chat history
        await self._persist_chat_history(
            conversation_id=final_conversation_id,
            client_id=client_id,
            coach_id=coach_id,
            user_message=message,
            assistant_response=response_content,
            sources_used=sources,
            tokens_used=tokens_used,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(contexts) if has_context else 0.0

        return ChatResponse(
            response=response_content,
            sources=sources,
            confidence=confidence,
            has_context=has_context,
            conversation_id=final_conversation_id,
            tokens_used=tokens_used,
        )

    async def generate_response_stream(
        self,
        client_id: UUID,
        coach_id: UUID,
        message: str,
        conversation_id: UUID | None = None,
        max_context_items: int = 10,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a grounded chat response via SSE.

        Yields:
            StreamChunk objects for each part of the response
        """
        import json

        # Load conversation history
        conversation_history = []
        if conversation_id:
            conversation_history = await self._load_conversation_history(
                conversation_id=conversation_id,
                limit=10,
            )

        # Retrieve context
        raw_contexts = await self.retrieval_service.retrieve_context(
            client_id=client_id,
            query=message,
            max_items=max_context_items * 2,
        )

        contexts = self._apply_context_policy(
            raw_contexts,
            max_items=max_context_items,
            max_tokens=self.settings.rag_max_context_tokens,
        )

        has_context = len(contexts) > 0
        if has_context:
            context_str = build_context_string(contexts, include_markers=True)
            system_prompt = GROUNDED_CHAT_SYSTEM_PROMPT.format(context=context_str)
        else:
            system_prompt = NO_CONTEXT_SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]
        for hist_msg in conversation_history:
            messages.append({"role": hist_msg.role, "content": hist_msg.content})
        messages.append({"role": "user", "content": message})

        # Stream response
        full_response = ""
        tokens_used = 0

        try:
            async for chunk_data in self.llm_client.stream(task="chat", messages=messages):
                try:
                    chunk_json = json.loads(chunk_data)
                    delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        full_response += content
                        yield StreamChunk(type="chunk", content=content)

                    # Check for usage in final chunk
                    if "usage" in chunk_json:
                        tokens_used = chunk_json["usage"].get("total_tokens", 0)

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error("Stream generation failed", error=str(e))
            yield StreamChunk(
                type="error",
                code="GENERATION_FAILED",
                message="Failed to generate response",
            )
            return

        # Generate conversation ID
        final_conversation_id = conversation_id or uuid4()

        # Extract citations
        sources = self._extract_citations(full_response, contexts) if has_context else []

        # Persist to history
        await self._persist_chat_history(
            conversation_id=final_conversation_id,
            client_id=client_id,
            coach_id=coach_id,
            user_message=message,
            assistant_response=full_response,
            sources_used=sources,
            tokens_used=tokens_used,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(contexts) if has_context else 0.0

        # Yield final done chunk
        yield StreamChunk(
            type="done",
            sources=sources,
            confidence=confidence,
            has_context=has_context,
            conversation_id=final_conversation_id,
            tokens_used=tokens_used,
        )

    def _apply_context_policy(
        self,
        contexts: list[SearchResult],
        max_items: int,
        max_tokens: int = 4000,
    ) -> list[SearchResult]:
        """
        Apply context window policy: priority ordering and token limits.

        Per spec:
        - Priority: health_profile > session > metrics > checkins > labs > goals > messages
        - Max 10 items, 4K tokens total
        - Truncate oldest/lowest-relevance items if exceeds limit
        """
        if not contexts:
            return []

        # Sort by priority, then by relevance
        def sort_key(ctx: SearchResult) -> tuple[int, float]:
            type_priority = (
                self.CONTEXT_PRIORITY.index(ctx.source_type)
                if ctx.source_type in self.CONTEXT_PRIORITY
                else 99
            )
            return (type_priority, -ctx.relevance_score)

        sorted_contexts = sorted(contexts, key=sort_key)

        # Limit by items
        limited = sorted_contexts[:max_items]

        # Limit by tokens
        total_tokens = 0
        result: list[SearchResult] = []

        for ctx in limited:
            ctx_tokens = len(self.tokenizer.encode(ctx.content))
            if total_tokens + ctx_tokens > max_tokens:
                break
            total_tokens += ctx_tokens
            result.append(ctx)

        return result

    def _extract_citations(
        self,
        response: str,
        contexts: list[SearchResult],
    ) -> list[SourceCitation]:
        """
        Extract which sources were used in the response.

        Looks for [Source N] markers or content overlap.
        """
        citations: list[SourceCitation] = []

        for i, ctx in enumerate(contexts):
            marker = f"[Source {i + 1}]"

            # Check if source marker is in response or content is referenced
            if marker in response or self._content_referenced(response, ctx.content):
                citations.append(
                    SourceCitation(
                        source_type=ctx.source_type,
                        source_id=ctx.source_id,
                        relevance_score=ctx.relevance_score,
                        snippet=ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
                        date=ctx.metadata.get("date") if ctx.metadata else None,
                    )
                )

        return citations

    def _content_referenced(self, response: str, content: str) -> bool:
        """Check if content appears to be referenced in response."""
        # Check for significant overlap (first 50 chars of content)
        content_start = content[:50].lower()
        return content_start in response.lower()

    def _calculate_confidence(self, contexts: list[SearchResult]) -> float:
        """Calculate confidence based on context relevance scores."""
        if not contexts:
            return 0.0
        avg_relevance = sum(c.relevance_score for c in contexts) / len(contexts)
        return min(avg_relevance, 1.0)

    async def _load_conversation_history(
        self,
        conversation_id: UUID,
        limit: int = 10,
    ) -> list[ChatHistory]:
        """Load previous messages from conversation."""
        result = await self.db.execute(
            select(ChatHistory)
            .where(ChatHistory.conversation_id == conversation_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        return list(reversed(messages))  # Oldest first

    async def _persist_chat_history(
        self,
        conversation_id: UUID,
        client_id: UUID,
        coach_id: UUID,
        user_message: str,
        assistant_response: str,
        sources_used: list[SourceCitation],
        tokens_used: int,
    ) -> None:
        """Persist both user and assistant messages to history."""
        # Store user message
        user_entry = ChatHistory(
            conversation_id=conversation_id,
            client_id=client_id,
            coach_id=coach_id,
            role="user",
            content=user_message,
        )
        self.db.add(user_entry)

        # Store assistant message
        assistant_entry = ChatHistory(
            conversation_id=conversation_id,
            client_id=client_id,
            coach_id=coach_id,
            role="assistant",
            content=assistant_response,
            sources_used=[s.model_dump() for s in sources_used],
            tokens_used=tokens_used,
        )
        self.db.add(assistant_entry)

        await self.db.flush()
