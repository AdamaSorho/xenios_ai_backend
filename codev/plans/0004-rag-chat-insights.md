# Plan 0004: RAG Chat & Insights

**Spec**: [codev/specs/0004-rag-chat-insights.md](../specs/0004-rag-chat-insights.md)
**Status**: Ready for implementation
**Estimated Phases**: 6

---

## Implementation Strategy

Build the RAG system incrementally, starting with pgvector setup, then embedding generation, retrieval, chat, and finally insight generation. Each phase produces working, testable functionality.

**Key Principles:**
- pgvector and database schema first (foundation)
- Reuse LLM client patterns from Spec 0001
- OpenAI client for embeddings (separate from Anthropic)
- Test with sample data at each phase
- Security and authorization from the start
- Rate limiting via Redis sliding window middleware
- Structured logging with PHI redaction

---

## Phase 1: Database Schema & pgvector Setup

**Goal**: Set up pgvector extension and create all required tables.

### 1.1 Create migration file

**File**: `scripts/migrations/0004_rag_tables.sql`

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Embeddings table
CREATE TABLE ai_backend.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,

    -- Source reference
    source_type VARCHAR(50) NOT NULL,
    source_id TEXT NOT NULL,
    source_table VARCHAR(100),

    -- Content
    content_text TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,

    -- Embedding vector (1536 dimensions for ada-002)
    embedding vector(1536) NOT NULL,

    -- Metadata for filtering
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE(client_id, source_type, source_id)
);

-- Indexes for fast retrieval
CREATE INDEX idx_embeddings_client ON ai_backend.embeddings(client_id);
CREATE INDEX idx_embeddings_source_type ON ai_backend.embeddings(client_id, source_type);
CREATE INDEX idx_embeddings_hash ON ai_backend.embeddings(client_id, content_hash);

-- Vector similarity index (IVFFlat for approximate nearest neighbor)
-- Note: IVFFlat requires training data; use HNSW for better cold-start performance
CREATE INDEX idx_embeddings_vector ON ai_backend.embeddings
    USING hnsw (embedding vector_cosine_ops);

-- Chat history for context continuity
CREATE TABLE ai_backend.chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,

    -- RAG metadata
    sources_used JSONB,
    tokens_used INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_history_conversation ON ai_backend.chat_history(conversation_id, created_at);
CREATE INDEX idx_chat_history_client ON ai_backend.chat_history(client_id, created_at DESC);

-- Insight generation log
CREATE TABLE ai_backend.insight_generation_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    trigger VARCHAR(50) NOT NULL,
    triggering_data JSONB NOT NULL,

    -- Result
    insight_id UUID,
    insight_type VARCHAR(50),
    title TEXT,
    title_embedding vector(1536),
    status VARCHAR(20) NOT NULL,
    error_message TEXT,

    -- Metrics
    context_items_used INTEGER,
    tokens_used INTEGER,
    generation_time_ms INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_insight_gen_log_client ON ai_backend.insight_generation_log(client_id, created_at DESC);
CREATE INDEX idx_insight_gen_log_dedup ON ai_backend.insight_generation_log(client_id, insight_type, created_at DESC)
    WHERE status = 'generated';
```

### 1.2 Add pgvector to dependencies

**File**: `pyproject.toml`

```toml
dependencies = [
    # ... existing deps
    "pgvector>=0.2.0",
    "openai>=1.0.0",  # For embeddings
]
```

### 1.3 Create SQLAlchemy models

**File**: `app/models/rag.py`

```python
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from app.database import Base
import uuid

class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False)

    source_type = Column(String(50), nullable=False)
    source_id = Column(Text, nullable=False)
    source_table = Column(String(100))

    content_text = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)

    embedding = Column(Vector(1536), nullable=False)

    metadata = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), server_default="now()")
    updated_at = Column(DateTime(timezone=True), server_default="now()")


class ChatHistory(Base):
    __tablename__ = "chat_history"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), nullable=False)
    client_id = Column(UUID(as_uuid=True), nullable=False)
    coach_id = Column(UUID(as_uuid=True), nullable=False)

    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)

    sources_used = Column(JSONB)
    tokens_used = Column(Integer)

    created_at = Column(DateTime(timezone=True), server_default="now()")


class InsightGenerationLog(Base):
    __tablename__ = "insight_generation_log"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False)
    coach_id = Column(UUID(as_uuid=True), nullable=False)

    trigger = Column(String(50), nullable=False)
    triggering_data = Column(JSONB, nullable=False)

    insight_id = Column(UUID(as_uuid=True))
    insight_type = Column(String(50))
    title = Column(Text)
    title_embedding = Column(Vector(1536))
    status = Column(String(20), nullable=False)
    error_message = Column(Text)

    context_items_used = Column(Integer)
    tokens_used = Column(Integer)
    generation_time_ms = Column(Integer)

    created_at = Column(DateTime(timezone=True), server_default="now()")
```

### 1.4 Verification

- Run migration
- Verify pgvector extension is enabled
- Test inserting and querying a sample vector

---

## Phase 2: Embedding Service

**Goal**: Generate and store embeddings using OpenAI ada-002.

### 2.1 Create OpenAI client wrapper

**File**: `app/services/rag/__init__.py`

```python
# Package init
```

**File**: `app/services/rag/openai_client.py`

```python
import openai
from app.core.config import settings

class OpenAIEmbeddingClient:
    """Wrapper for OpenAI embeddings API."""

    MODEL = "text-embedding-ada-002"
    DIMENSIONS = 1536

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = await self.client.embeddings.create(
            model=self.MODEL,
            input=text,
        )
        return response.data[0].embedding

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        response = await self.client.embeddings.create(
            model=self.MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]
```

### 2.2 Create embedding service

**File**: `app/services/rag/embeddings.py`

```python
import hashlib
from uuid import UUID
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.rag import Embedding
from app.services.rag.openai_client import OpenAIEmbeddingClient
from app.schemas.rag import EmbeddingUpdateResult, EmbeddingSourceType

class EmbeddingService:
    """Generate and store embeddings."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.openai_client = OpenAIEmbeddingClient()

    async def update_client_embeddings(
        self,
        client_id: UUID,
        source_types: list[str] | None = None,
        force: bool = False,
    ) -> EmbeddingUpdateResult:
        """Update embeddings for a client's data."""
        updated = 0
        skipped = 0

        # Health profile
        if not source_types or EmbeddingSourceType.HEALTH_PROFILE in source_types:
            result = await self._update_health_profile_embedding(client_id, force)
            if result:
                updated += 1
            else:
                skipped += 1

        # Metric summaries
        if not source_types or EmbeddingSourceType.HEALTH_METRIC_SUMMARY in source_types:
            u, s = await self._update_metric_summary_embeddings(client_id, force)
            updated += u
            skipped += s

        # Session summaries
        if not source_types or EmbeddingSourceType.SESSION_SUMMARY in source_types:
            u, s = await self._update_session_summary_embeddings(client_id, force)
            updated += u
            skipped += s

        # Check-in summaries
        if not source_types or EmbeddingSourceType.CHECKIN_SUMMARY in source_types:
            u, s = await self._update_checkin_summary_embeddings(client_id, force)
            updated += u
            skipped += s

        # Lab results
        if not source_types or EmbeddingSourceType.LAB_RESULT in source_types:
            u, s = await self._update_lab_result_embeddings(client_id, force)
            updated += u
            skipped += s

        # Health goals
        if not source_types or EmbeddingSourceType.HEALTH_GOAL in source_types:
            u, s = await self._update_health_goal_embeddings(client_id, force)
            updated += u
            skipped += s

        # Message threads (daily aggregates)
        if not source_types or EmbeddingSourceType.MESSAGE_THREAD in source_types:
            u, s = await self._update_message_thread_embeddings(client_id, force)
            updated += u
            skipped += s

        return EmbeddingUpdateResult(updated_count=updated, skipped_count=skipped)

    async def _should_update(
        self,
        client_id: UUID,
        source_type: str,
        source_id: str,
        text: str,
        force: bool,
    ) -> bool:
        """Check if embedding needs update based on content hash."""
        if force:
            return True

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        result = await self.db.execute(
            select(Embedding.content_hash).where(
                and_(
                    Embedding.client_id == client_id,
                    Embedding.source_type == source_type,
                    Embedding.source_id == source_id,
                )
            )
        )
        existing_hash = result.scalar_one_or_none()

        return existing_hash != content_hash

    async def _store_embedding(
        self,
        client_id: UUID,
        source_type: str,
        source_id: str,
        text: str,
        source_table: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Generate and store embedding."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        embedding_vector = await self.openai_client.generate_embedding(text)

        # Upsert
        existing = await self.db.execute(
            select(Embedding).where(
                and_(
                    Embedding.client_id == client_id,
                    Embedding.source_type == source_type,
                    Embedding.source_id == source_id,
                )
            )
        )
        embedding = existing.scalar_one_or_none()

        if embedding:
            embedding.content_text = text
            embedding.content_hash = content_hash
            embedding.embedding = embedding_vector
            embedding.metadata = metadata or {}
        else:
            embedding = Embedding(
                client_id=client_id,
                source_type=source_type,
                source_id=source_id,
                source_table=source_table,
                content_text=text,
                content_hash=content_hash,
                embedding=embedding_vector,
                metadata=metadata or {},
            )
            self.db.add(embedding)

        await self.db.flush()

    # Source-specific builders (implement each)
    async def _update_health_profile_embedding(self, client_id: UUID, force: bool) -> bool:
        """Build and store health profile embedding."""
        # Query client_health_profiles table
        # Build text summary
        # Check if update needed
        # Store embedding
        pass

    async def _update_metric_summary_embeddings(self, client_id: UUID, force: bool) -> tuple[int, int]:
        """Build and store metric summary embeddings (weekly aggregates)."""
        pass

    async def _update_session_summary_embeddings(self, client_id: UUID, force: bool) -> tuple[int, int]:
        """Build and store session summary embeddings."""
        pass

    async def _update_checkin_summary_embeddings(self, client_id: UUID, force: bool) -> tuple[int, int]:
        """Build and store check-in summary embeddings."""
        pass

    async def _update_lab_result_embeddings(self, client_id: UUID, force: bool) -> tuple[int, int]:
        """Build and store lab result embeddings."""
        pass
```

### 2.3 Create schemas

**File**: `app/schemas/rag.py`

```python
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime, date
from enum import Enum
from typing import Optional

class EmbeddingSourceType(str, Enum):
    HEALTH_PROFILE = "health_profile"
    HEALTH_METRIC_SUMMARY = "health_metric_summary"
    HEALTH_GOAL = "health_goal"
    LAB_RESULT = "lab_result"
    SESSION_SUMMARY = "session_summary"
    CHECKIN_SUMMARY = "checkin_summary"
    MESSAGE_THREAD = "message_thread"

class EmbeddingUpdateRequest(BaseModel):
    client_id: UUID
    source_type: str | None = None
    force: bool = False

class EmbeddingUpdateResult(BaseModel):
    updated_count: int
    skipped_count: int

class EmbeddingSearchRequest(BaseModel):
    client_id: UUID
    query: str
    limit: int = 10
    source_types: list[str] | None = None

class SearchResult(BaseModel):
    source_type: str
    source_id: str
    content: str
    relevance_score: float
    metadata: dict
```

### 2.4 Add OPENAI_API_KEY to settings

**File**: `app/core/config.py` (add)

```python
OPENAI_API_KEY: str = ""
```

### 2.5 Create embeddings API endpoints

**File**: `app/api/v1/embeddings.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_db
from app.core.auth import get_current_user, verify_coach_client_relationship
from app.services.rag.embeddings import EmbeddingService
from app.services.rag.retrieval import RetrievalService
from app.schemas.rag import EmbeddingUpdateRequest, EmbeddingUpdateResult, EmbeddingSearchRequest

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

@router.post("/update", response_model=EmbeddingUpdateResult)
async def update_embeddings(
    request: EmbeddingUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Update embeddings for a client."""
    await verify_coach_client_relationship(
        db, coach_id=current_user.id, client_id=request.client_id
    )

    service = EmbeddingService(db)
    result = await service.update_client_embeddings(
        client_id=request.client_id,
        source_types=[request.source_type] if request.source_type else None,
        force=request.force,
    )
    return result

@router.post("/search", response_model=list[SearchResult])
async def search_embeddings(
    request: EmbeddingSearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Search embeddings for a client."""
    await verify_coach_client_relationship(
        db, coach_id=current_user.id, client_id=request.client_id
    )

    service = RetrievalService(db)
    results = await service.retrieve_context(
        client_id=request.client_id,
        query=request.query,
        max_items=request.limit,
        source_types=request.source_types,
    )
    return results
```

### 2.6 Create batch embedding endpoint for onboarding

```python
@router.post("/batch-update")
async def batch_update_embeddings(
    client_ids: list[UUID],
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Batch update embeddings for multiple clients (onboarding)."""
    from app.workers.tasks.rag import update_client_embeddings

    for client_id in client_ids:
        await verify_coach_client_relationship(
            db, coach_id=current_user.id, client_id=client_id
        )
        # Queue Celery task
        update_client_embeddings.delay(str(client_id), force=True)

    return {"queued": len(client_ids)}
```

---

## Phase 3: Retrieval Service

**Goal**: Implement semantic search using pgvector.

### 3.1 Create retrieval service

**File**: `app/services/rag/retrieval.py`

```python
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.services.rag.openai_client import OpenAIEmbeddingClient
from app.schemas.rag import SearchResult

class RetrievalService:
    """Retrieve relevant context using semantic search."""

    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, db: AsyncSession):
        self.db = db
        self.openai_client = OpenAIEmbeddingClient()

    async def retrieve_context(
        self,
        client_id: UUID,
        query: str,
        max_items: int = 10,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant context for RAG."""
        # Generate query embedding
        query_embedding = await self.openai_client.generate_embedding(query)

        # Execute vector search
        results = await self._vector_search(
            client_id=client_id,
            query_embedding=query_embedding,
            limit=max_items,
            source_types=source_types,
        )

        # Filter by threshold and fetch content
        contexts = []
        for result in results:
            if result["similarity"] < self.SIMILARITY_THRESHOLD:
                continue

            content = await self._fetch_source_content(result)
            if content is None:
                continue

            contexts.append(SearchResult(
                source_type=result["source_type"],
                source_id=result["source_id"],
                content=content,
                relevance_score=result["similarity"],
                metadata=result["metadata"],
            ))

        return contexts

    async def _vector_search(
        self,
        client_id: UUID,
        query_embedding: list[float],
        limit: int,
        source_types: list[str] | None,
    ) -> list[dict]:
        """Execute pgvector similarity search."""
        source_filter = ""
        params = {
            "embedding": query_embedding,
            "client_id": str(client_id),
            "limit": limit,
        }

        if source_types:
            source_filter = "AND source_type = ANY(:source_types)"
            params["source_types"] = source_types

        query = text(f"""
            SELECT
                id, source_type, source_id, content_text, metadata,
                1 - (embedding <=> :embedding::vector) as similarity
            FROM ai_backend.embeddings
            WHERE client_id = :client_id
            {source_filter}
            AND 1 - (embedding <=> :embedding::vector) >= :threshold
            ORDER BY embedding <=> :embedding::vector
            LIMIT :limit
        """)

        params["threshold"] = self.SIMILARITY_THRESHOLD

        result = await self.db.execute(query, params)
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "source_type": row.source_type,
                "source_id": row.source_id,
                "content_text": row.content_text,
                "metadata": row.metadata,
                "similarity": row.similarity,
            }
            for row in rows
        ]

    async def _fetch_source_content(self, result: dict) -> str | None:
        """Fetch full source content based on type."""
        source_type = result["source_type"]

        # For aggregated types, use stored content_text
        if source_type in ["health_profile", "health_metric_summary", "message_thread"]:
            return result["content_text"]

        # For single-record types, fetch from source table
        # Implement joins based on source_type
        # Return None if source is deleted
        return result["content_text"]  # Fallback to stored text
```

---

## Phase 4: Chat Endpoints

**Goal**: Implement grounded chat with source citations.

### 4.1 Create chat service with context policy

**File**: `app/services/rag/chat.py`

Context policy from spec:
- 10 items maximum, 4,000 tokens total context limit
- Priority: health profile > recent session > relevant metrics > check-ins > lab results
- Truncate oldest/lowest-relevance items if exceeds 4K tokens
- Use `include_sources` flag from request

```python
from uuid import UUID, uuid4
from sqlalchemy.ext.asyncio import AsyncSession
import tiktoken

from app.services.rag.retrieval import RetrievalService
from app.services.llm.client import LLMClient
from app.models.rag import ChatHistory
from app.schemas.rag import ChatRequest, ChatResponse, SourceCitation

class ChatService:
    """Generate grounded chat responses."""

    SYSTEM_PROMPT = """You are a knowledgeable health coaching assistant helping a coach
communicate with their client. Use ONLY the provided context to answer questions.

IMPORTANT RULES:
1. Only state facts that are supported by the provided context
2. If you don't have enough context to answer, say so
3. Reference specific data points with dates when available
4. Be encouraging but factual
5. If asked about something not in the context, explain what information you do have

CONTEXT:
{context}

The coach is asking about their client. Respond helpfully using the context above."""

    NO_CONTEXT_PROMPT = """You are a health coaching assistant. The coach is asking about their client,
but I don't have specific health data available for this query.

Please acknowledge that you don't have enough specific information and suggest what data might help."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.retrieval_service = RetrievalService(db)
        self.llm_client = LLMClient()

    async def generate_response(
        self,
        client_id: UUID,
        coach_id: UUID,
        message: str,
        conversation_id: UUID | None = None,
    ) -> ChatResponse:
        """Generate grounded response with citations."""
        # Load conversation history
        conversation_history = []
        if conversation_id:
            conversation_history = await self._load_conversation_history(
                conversation_id=conversation_id,
                limit=10,
            )

        # Retrieve relevant context
        contexts = await self.retrieval_service.retrieve_context(
            client_id=client_id,
            query=message,
            max_items=10,
        )

        # Build context string
        has_context = len(contexts) > 0
        if has_context:
            context_str = self._build_context_string(contexts)
            system_prompt = self.SYSTEM_PROMPT.format(context=context_str)
        else:
            system_prompt = self.NO_CONTEXT_PROMPT

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        for hist_msg in conversation_history:
            messages.append({"role": hist_msg.role, "content": hist_msg.content})
        messages.append({"role": "user", "content": message})

        # Generate response
        response = await self.llm_client.complete(
            task="chat",
            messages=messages,
        )

        # Extract citations
        sources = self._extract_citations(response.content, contexts) if has_context else []

        # Persist chat history
        final_conversation_id = conversation_id or uuid4()
        await self._persist_chat_history(
            conversation_id=final_conversation_id,
            client_id=client_id,
            coach_id=coach_id,
            user_message=message,
            assistant_response=response.content,
            sources_used=sources,
            tokens_used=response.usage.total_tokens,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(contexts) if has_context else 0.0

        return ChatResponse(
            response=response.content,
            sources=sources,
            confidence=confidence,
            has_context=has_context,
            conversation_id=final_conversation_id,
            tokens_used=response.usage.total_tokens,
        )

    def _build_context_string(self, contexts: list) -> str:
        """Build context string with source markers."""
        parts = []
        for i, ctx in enumerate(contexts):
            marker = f"[Source {i+1}]"
            date_str = f" ({ctx.metadata.get('date')})" if ctx.metadata.get('date') else ""
            parts.append(f"{marker} {ctx.source_type}{date_str}:\n{ctx.content}\n")
        return "\n".join(parts)

    def _extract_citations(self, response: str, contexts: list) -> list[SourceCitation]:
        """Extract which sources were used in the response."""
        citations = []
        for i, ctx in enumerate(contexts):
            marker = f"[Source {i+1}]"
            if marker in response or ctx.content[:50] in response:
                citations.append(SourceCitation(
                    source_type=ctx.source_type,
                    source_id=ctx.source_id,
                    relevance_score=ctx.relevance_score,
                    snippet=ctx.content[:200],
                    date=ctx.metadata.get("date"),
                ))
        return citations

    def _calculate_confidence(self, contexts: list) -> float:
        """Calculate confidence based on context relevance."""
        if not contexts:
            return 0.0
        avg_relevance = sum(c.relevance_score for c in contexts) / len(contexts)
        return min(avg_relevance, 1.0)

    def _apply_context_policy(
        self,
        contexts: list,
        max_items: int,
        max_tokens: int = 4000,
    ) -> list:
        """Apply context window policy: priority ordering and token limits."""
        # Priority order
        priority_order = [
            "health_profile",
            "session_summary",
            "health_metric_summary",
            "checkin_summary",
            "lab_result",
            "health_goal",
            "message_thread",
        ]

        # Sort by priority, then by relevance
        def sort_key(ctx):
            type_priority = priority_order.index(ctx.source_type) if ctx.source_type in priority_order else 99
            return (type_priority, -ctx.relevance_score)

        sorted_contexts = sorted(contexts, key=sort_key)

        # Limit by items
        limited = sorted_contexts[:max_items]

        # Limit by tokens
        encoder = tiktoken.encoding_for_model("gpt-4")
        total_tokens = 0
        result = []

        for ctx in limited:
            ctx_tokens = len(encoder.encode(ctx.content))
            if total_tokens + ctx_tokens > max_tokens:
                break
            total_tokens += ctx_tokens
            result.append(ctx)

        return result

    async def _load_conversation_history(self, conversation_id: UUID, limit: int) -> list:
        """Load previous messages from conversation."""
        from sqlalchemy import select
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
        sources_used: list,
        tokens_used: int,
    ) -> None:
        """Persist both user and assistant messages."""
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
            sources_used=[s.dict() for s in sources_used],
            tokens_used=tokens_used,
        )
        self.db.add(assistant_entry)

        await self.db.flush()
```

### 4.2 Create chat API endpoints

**File**: `app/api/v1/chat.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_db
from app.core.auth import get_current_user, verify_coach_client_relationship
from app.services.rag.chat import ChatService
from app.schemas.rag import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/complete", response_model=ChatResponse)
async def chat_complete(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Generate grounded chat response."""
    await verify_coach_client_relationship(
        db, coach_id=current_user.id, client_id=request.client_id
    )

    chat_service = ChatService(db)
    response = await chat_service.generate_response(
        client_id=request.client_id,
        coach_id=current_user.id,
        message=request.message,
        conversation_id=request.conversation_id,
        max_context_items=request.max_context_items,
        include_sources=request.include_sources,
    )

    return response


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Stream grounded chat response via SSE."""
    from fastapi.responses import StreamingResponse
    import json

    await verify_coach_client_relationship(
        db, coach_id=current_user.id, client_id=request.client_id
    )

    chat_service = ChatService(db)

    async def generate():
        try:
            async for chunk in chat_service.generate_response_stream(
                client_id=request.client_id,
                coach_id=current_user.id,
                message=request.message,
                conversation_id=request.conversation_id,
            ):
                if chunk.type == "chunk":
                    yield f"event: chunk\ndata: {json.dumps({'type': 'chunk', 'content': chunk.content})}\n\n"
                elif chunk.type == "done":
                    yield f"event: done\ndata: {json.dumps(chunk.dict())}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'code': 'GENERATION_FAILED', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 4.3 Add chat schemas

**File**: `app/schemas/rag.py` (add)

```python
class ChatRequest(BaseModel):
    client_id: UUID
    message: str
    conversation_id: UUID | None = None
    include_sources: bool = True
    max_context_items: int = 10

class SourceCitation(BaseModel):
    source_type: str
    source_id: str
    relevance_score: float
    snippet: str
    date: date | None = None

class ChatResponse(BaseModel):
    response: str
    sources: list[SourceCitation]
    confidence: float
    has_context: bool
    conversation_id: UUID
    tokens_used: int
```

---

## Phase 5: Insight Generation

**Goal**: Generate proactive insights and write to MVP insights table.

### 5.1 Create insight service

**File**: `app/services/rag/insights.py`

```python
from uuid import UUID
from datetime import datetime, timedelta
import json
import time
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, text

from app.services.rag.retrieval import RetrievalService
from app.services.rag.openai_client import OpenAIEmbeddingClient
from app.services.llm.client import LLMClient
from app.models.rag import InsightGenerationLog
from app.schemas.rag import InsightGenerationRequest, InsightTrigger, GeneratedInsight

class DuplicateInsightError(Exception):
    pass

class RateLimitExceededError(Exception):
    pass

class InsightGenerationService:
    """Generate proactive insights from client health data."""

    INSIGHT_PROMPT = """Analyze the following client health data and generate an actionable insight.

CLIENT CONTEXT:
{context}

RECENT CHANGES:
{changes}

Generate an insight that:
1. Identifies a meaningful trend, achievement, or concern
2. Provides specific, actionable advice
3. Is encouraging and supportive in tone
4. References specific data points

Respond in JSON format:
{{
    "title": "Short, attention-grabbing title (max 50 chars)",
    "client_message": "Message to show the client (2-3 sentences, encouraging)",
    "rationale": "Why this insight matters (for coach review)",
    "suggested_actions": ["Action 1", "Action 2"],
    "insight_type": "nutrition|training|recovery|motivation|general",
    "confidence_score": 0.0-1.0
}}"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.retrieval_service = RetrievalService(db)
        self.openai_client = OpenAIEmbeddingClient()
        self.llm_client = LLMClient()

    async def generate_insight(
        self,
        client_id: UUID,
        coach_id: UUID,
        trigger: InsightTrigger,
        context: dict | None = None,
    ) -> GeneratedInsight:
        """Generate insight and write to MVP insights table."""
        start_time = time.time()

        # Check rate limits
        await self._check_rate_limits(client_id)

        # Gather client context
        client_context = await self._gather_client_context(client_id)

        # Identify recent changes
        changes = await self._identify_changes(client_id, trigger, context)

        # Generate insight via LLM
        prompt = self.INSIGHT_PROMPT.format(
            context=client_context,
            changes=json.dumps(changes),
        )

        response = await self.llm_client.complete(
            task="insight_generation",
            messages=[{"role": "user", "content": prompt}],
        )

        insight_data = self._parse_insight_response(response.content)

        # Check for duplicates using title embedding
        title_embedding = await self.openai_client.generate_embedding(insight_data["title"])
        if await self._is_duplicate_insight(client_id, insight_data["insight_type"], title_embedding):
            # Log as duplicate
            await self._log_generation(
                client_id=client_id,
                coach_id=coach_id,
                trigger=trigger,
                triggering_data=context or {},
                status="duplicate",
                insight_type=insight_data["insight_type"],
                title=insight_data["title"],
                title_embedding=title_embedding,
            )
            raise DuplicateInsightError("Similar insight already pending")

        # Write to MVP insights table
        insight_id = await self._write_to_mvp_insights(
            client_id=client_id,
            coach_id=coach_id,
            insight_data=insight_data,
            triggering_data={
                "trigger": trigger.value,
                "changes": changes,
                "context_summary": client_context[:500],
            },
        )

        generation_time_ms = int((time.time() - start_time) * 1000)

        # Log successful generation
        await self._log_generation(
            client_id=client_id,
            coach_id=coach_id,
            trigger=trigger,
            triggering_data=context or {},
            status="generated",
            insight_id=insight_id,
            insight_type=insight_data["insight_type"],
            title=insight_data["title"],
            title_embedding=title_embedding,
            tokens_used=response.usage.total_tokens,
            generation_time_ms=generation_time_ms,
        )

        return GeneratedInsight(
            id=insight_id,
            client_id=client_id,
            coach_id=coach_id,
            **insight_data,
        )

    async def _check_rate_limits(self, client_id: UUID, trigger: InsightTrigger) -> None:
        """Check all rate limit rules from spec."""
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        day_ago = now - timedelta(days=1)

        # Rule 1: Max 3 per client per week
        result = await self.db.execute(
            select(InsightGenerationLog).where(
                and_(
                    InsightGenerationLog.client_id == client_id,
                    InsightGenerationLog.status == "generated",
                    InsightGenerationLog.created_at >= week_ago,
                )
            )
        )
        if len(result.scalars().all()) >= 3:
            raise RateLimitExceededError("Max 3 insights per client per week")

        # Rule 2: Max 1 insight per trigger type per day
        result = await self.db.execute(
            select(InsightGenerationLog).where(
                and_(
                    InsightGenerationLog.client_id == client_id,
                    InsightGenerationLog.trigger == trigger.value,
                    InsightGenerationLog.status == "generated",
                    InsightGenerationLog.created_at >= day_ago,
                )
            )
        )
        if result.scalars().first():
            raise RateLimitExceededError(f"Max 1 {trigger.value} insight per day")

        # Rule 3: 48-hour cooldown after approval/rejection
        # Check if last insight of same type was approved/rejected in last 48 hours
        cooldown_ago = now - timedelta(hours=48)
        # Note: This requires checking public.insights table status
        # Query: SELECT FROM public.insights WHERE client_id = $1
        #        AND insight_type = $2 AND status IN ('approved', 'rejected')
        #        AND updated_at >= $3

    async def _is_duplicate_insight(
        self,
        client_id: UUID,
        insight_type: str,
        title_embedding: list[float],
    ) -> bool:
        """Check for duplicate insights using type and title similarity."""
        week_ago = datetime.utcnow() - timedelta(days=7)

        # Check same insight_type in last 7 days
        query = text("""
            SELECT id, 1 - (title_embedding <=> :embedding::vector) as similarity
            FROM ai_backend.insight_generation_log
            WHERE client_id = :client_id
            AND insight_type = :insight_type
            AND status = 'generated'
            AND created_at >= :week_ago
            AND 1 - (title_embedding <=> :embedding::vector) > 0.85
            LIMIT 1
        """)

        result = await self.db.execute(query, {
            "client_id": str(client_id),
            "insight_type": insight_type,
            "embedding": title_embedding,
            "week_ago": week_ago,
        })

        return result.fetchone() is not None

    async def _write_to_mvp_insights(
        self,
        client_id: UUID,
        coach_id: UUID,
        insight_data: dict,
        triggering_data: dict,
    ) -> UUID:
        """Write insight to MVP's public.insights table."""
        query = text("""
            INSERT INTO public.insights (
                coach_id, client_id, title, client_message, rationale,
                suggested_actions, confidence_score, triggering_data,
                insight_type, status, expires_at
            ) VALUES (
                :coach_id, :client_id, :title, :client_message, :rationale,
                :suggested_actions, :confidence_score, :triggering_data,
                :insight_type, 'pending', NOW() + INTERVAL '7 days'
            ) RETURNING id
        """)

        result = await self.db.execute(query, {
            "coach_id": str(coach_id),
            "client_id": str(client_id),
            "title": insight_data["title"],
            "client_message": insight_data["client_message"],
            "rationale": insight_data["rationale"],
            "suggested_actions": json.dumps(insight_data["suggested_actions"]),
            "confidence_score": insight_data["confidence_score"],
            "triggering_data": json.dumps(triggering_data),
            "insight_type": insight_data["insight_type"],
        })

        return result.fetchone()[0]

    def _parse_insight_response(self, content: str) -> dict:
        """Parse LLM JSON response."""
        # Extract JSON from response (may be wrapped in markdown)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())

    async def _gather_client_context(self, client_id: UUID) -> str:
        """Gather relevant context for insight generation."""
        # Use retrieval service to get recent data
        contexts = await self.retrieval_service.retrieve_context(
            client_id=client_id,
            query="recent health progress and changes",
            max_items=5,
        )
        return "\n".join([c.content for c in contexts])

    async def _identify_changes(
        self,
        client_id: UUID,
        trigger: InsightTrigger,
        context: dict | None,
    ) -> dict:
        """Identify relevant changes based on trigger."""
        if context:
            return context
        # Default: return empty changes
        return {"trigger": trigger.value}

    async def _log_generation(self, **kwargs) -> None:
        """Log insight generation attempt."""
        log_entry = InsightGenerationLog(
            client_id=kwargs["client_id"],
            coach_id=kwargs["coach_id"],
            trigger=kwargs["trigger"].value if hasattr(kwargs["trigger"], "value") else kwargs["trigger"],
            triggering_data=kwargs.get("triggering_data", {}),
            insight_id=kwargs.get("insight_id"),
            insight_type=kwargs.get("insight_type"),
            title=kwargs.get("title"),
            title_embedding=kwargs.get("title_embedding"),
            status=kwargs["status"],
            context_items_used=kwargs.get("context_items_used"),
            tokens_used=kwargs.get("tokens_used"),
            generation_time_ms=kwargs.get("generation_time_ms"),
        )
        self.db.add(log_entry)
        await self.db.flush()
```

### 5.2 Create insights API endpoints

**File**: `app/api/v1/insights.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_db
from app.core.auth import get_current_user, verify_coach_client_relationship
from app.services.rag.insights import InsightGenerationService, DuplicateInsightError, RateLimitExceededError
from app.schemas.rag import InsightGenerationRequest, InsightGenerationResponse

router = APIRouter(prefix="/insights", tags=["insights"])

@router.post("/generate", response_model=InsightGenerationResponse)
async def generate_insight(
    request: InsightGenerationRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Generate an insight for a client."""
    # Verify coach-client relationship
    await verify_coach_client_relationship(
        db, coach_id=current_user.id, client_id=request.client_id
    )

    insight_service = InsightGenerationService(db)

    try:
        result = await insight_service.generate_insight(
            client_id=request.client_id,
            coach_id=current_user.id,
            trigger=request.trigger,
            context=request.context,
        )
        return InsightGenerationResponse(
            insight_id=result.id,
            title=result.title,
            confidence_score=result.confidence_score,
            status="generated",
        )
    except DuplicateInsightError:
        return InsightGenerationResponse(
            insight_id=None,
            title=None,
            confidence_score=None,
            status="duplicate",
        )
    except RateLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))


@router.get("/pending", response_model=list[GeneratedInsight])
async def get_pending_insights(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Get pending insights for the coach to review."""
    from sqlalchemy import text

    # Query MVP insights table for pending insights belonging to this coach
    query = text("""
        SELECT id, client_id, coach_id, title, client_message, rationale,
               suggested_actions, confidence_score, triggering_data,
               insight_type, expires_at, created_at
        FROM public.insights
        WHERE coach_id = :coach_id
        AND status = 'pending'
        AND expires_at > NOW()
        ORDER BY created_at DESC
        LIMIT :limit
    """)

    result = await db.execute(query, {"coach_id": str(current_user.id), "limit": limit})
    rows = result.fetchall()

    return [
        GeneratedInsight(
            id=row.id,
            client_id=row.client_id,
            coach_id=row.coach_id,
            title=row.title,
            client_message=row.client_message,
            rationale=row.rationale,
            suggested_actions=row.suggested_actions,
            confidence_score=row.confidence_score,
            triggering_data=row.triggering_data,
            insight_type=row.insight_type,
            expires_at=row.expires_at,
        )
        for row in rows
    ]
```

---

## Phase 5.5: Rate Limiting & Logging Middleware

**Goal**: Implement rate limiting and audit logging per spec AC8.

### 5.5.1 Create rate limiting middleware

**File**: `app/middleware/rate_limit.py`

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import time

from app.core.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis sliding window rate limiting."""

    LIMITS = {
        "/api/v1/chat/": {"requests": 100, "window": 3600},      # 100/hour
        "/api/v1/embeddings/": {"requests": 10, "window": 3600}, # 10/hour
        "/api/v1/insights/": {"requests": 50, "window": 86400},  # 50/day
    }

    def __init__(self, app):
        super().__init__(app)
        self.redis = redis.from_url(settings.REDIS_URL)

    async def dispatch(self, request: Request, call_next):
        # Skip for system/service keys
        api_key = request.headers.get("X-API-Key")
        if api_key == settings.SYSTEM_API_KEY:
            return await call_next(request)

        # Find matching limit
        path = request.url.path
        limit_config = None
        for prefix, config in self.LIMITS.items():
            if path.startswith(prefix):
                limit_config = config
                break

        if not limit_config:
            return await call_next(request)

        # Get coach_id from JWT
        coach_id = getattr(request.state, "user_id", "anonymous")

        # Redis sliding window
        key = f"ratelimit:{path.split('/')[3]}:{coach_id}:{int(time.time()) // limit_config['window']}"
        current = await self.redis.incr(key)
        await self.redis.expire(key, limit_config["window"])

        if current > limit_config["requests"]:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(limit_config["window"])}
            )

        return await call_next(request)
```

### 5.5.2 Create audit logging middleware

**File**: `app/middleware/audit_log.py`

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import time

logger = structlog.get_logger("audit")

class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log all operations for compliance."""

    # Fields allowed in logs (PHI redaction)
    ALLOWED_FIELDS = ["client_id", "coach_id", "source_type", "operation", "status", "tokens_used"]

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        # Log audit entry
        duration_ms = int((time.time() - start_time) * 1000)

        await logger.ainfo(
            "api_request",
            path=request.url.path,
            method=request.method,
            coach_id=getattr(request.state, "user_id", None),
            client_id=request.query_params.get("client_id"),
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response
```

### 5.5.3 Configure structured logging

**File**: `app/core/logging.py`

```python
import structlog

def configure_logging():
    """Configure structured logging with PHI redaction."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_phi_processor,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

def _redact_phi_processor(logger, method_name, event_dict):
    """Redact PHI from log events."""
    phi_fields = ["content", "message", "response", "embedding", "content_text"]
    for field in phi_fields:
        if field in event_dict:
            event_dict[field] = "[REDACTED]"
    return event_dict
```

---

## Phase 6: Celery Tasks & Testing

**Goal**: Create Celery tasks and comprehensive tests.

### 6.1 Create Celery tasks

**File**: `app/workers/tasks/rag.py`

```python
import asyncio
from uuid import UUID

from app.workers.celery import celery_app
from app.database import get_async_session
from app.services.rag.embeddings import EmbeddingService
from app.services.rag.insights import InsightGenerationService, DuplicateInsightError
from app.schemas.rag import InsightTrigger

@celery_app.task(
    bind=True,
    queue="llm",
    max_retries=2,
    time_limit=300,  # 5 minutes
)
def update_client_embeddings(
    self,
    client_id: str,
    source_types: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Update embeddings for a client."""
    async def _run():
        async with get_async_session() as db:
            service = EmbeddingService(db)
            result = await service.update_client_embeddings(
                client_id=UUID(client_id),
                source_types=source_types,
                force=force,
            )
            await db.commit()
            return result.dict()

    return asyncio.run(_run())


@celery_app.task(
    bind=True,
    queue="llm",
    max_retries=1,
    time_limit=60,
)
def generate_client_insight(
    self,
    client_id: str,
    coach_id: str,
    trigger: str,
    context: dict | None = None,
) -> dict:
    """Generate insight for a client."""
    async def _run():
        async with get_async_session() as db:
            service = InsightGenerationService(db)
            try:
                result = await service.generate_insight(
                    client_id=UUID(client_id),
                    coach_id=UUID(coach_id),
                    trigger=InsightTrigger(trigger),
                    context=context,
                )
                await db.commit()
                return {"status": "generated", "insight_id": str(result.id)}
            except DuplicateInsightError:
                return {"status": "duplicate"}

    return asyncio.run(_run())


@celery_app.task(queue="llm")
def batch_update_embeddings():
    """Nightly batch update for all active clients."""
    # Query all active clients
    # Queue individual embedding update tasks
    pass
```

### 6.2 Register routers

**File**: `app/api/v1/__init__.py` (add)

```python
from app.api.v1.chat import router as chat_router
from app.api.v1.insights import router as insights_router

# Add to router includes
api_router.include_router(chat_router)
api_router.include_router(insights_router)
```

### 6.3 Create tests

**File**: `tests/services/rag/test_embeddings.py`

```python
import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.services.rag.embeddings import EmbeddingService

@pytest.mark.asyncio
async def test_content_hashing_prevents_duplicate():
    """Test that same content doesn't regenerate embedding."""
    # Mock DB and OpenAI client
    pass

@pytest.mark.asyncio
async def test_force_updates_even_if_unchanged():
    """Test that force=True bypasses hash check."""
    pass
```

**File**: `tests/services/rag/test_retrieval.py`

```python
import pytest
from uuid import uuid4

from app.services.rag.retrieval import RetrievalService

@pytest.mark.asyncio
async def test_similarity_threshold_filters_results():
    """Test that results below threshold are excluded."""
    pass

@pytest.mark.asyncio
async def test_no_context_returns_empty_list():
    """Test that client with no embeddings returns empty."""
    pass
```

**File**: `tests/services/rag/test_chat.py`

```python
import pytest
from uuid import uuid4

from app.services.rag.chat import ChatService

@pytest.mark.asyncio
async def test_no_context_response_has_flag():
    """Test that has_context=false when no embeddings found."""
    pass

@pytest.mark.asyncio
async def test_conversation_history_loaded():
    """Test that previous messages are included in context."""
    pass
```

**File**: `tests/services/rag/test_insights.py`

```python
import pytest
from uuid import uuid4

from app.services.rag.insights import InsightGenerationService, DuplicateInsightError

@pytest.mark.asyncio
async def test_duplicate_detection_blocks_similar():
    """Test that similar insights are blocked."""
    pass

@pytest.mark.asyncio
async def test_rate_limit_enforced():
    """Test max 3 insights per client per week."""
    pass
```

**File**: `tests/api/v1/test_chat.py`

```python
import pytest
from httpx import AsyncClient
from uuid import uuid4

@pytest.mark.asyncio
async def test_coach_cannot_access_other_clients():
    """Test authorization - coach can only access their clients."""
    pass

@pytest.mark.asyncio
async def test_invalid_client_returns_404():
    """Test that invalid client_id returns 404 not 403."""
    pass
```

---

## Files to Create (Summary)

### Phase 1
- `scripts/migrations/0004_rag_tables.sql`
- `app/models/rag.py`

### Phase 2
- `app/services/rag/__init__.py`
- `app/services/rag/openai_client.py`
- `app/services/rag/embeddings.py`
- `app/schemas/rag.py`
- `app/api/v1/embeddings.py`

### Phase 3
- `app/services/rag/retrieval.py`

### Phase 4
- `app/services/rag/chat.py`
- `app/api/v1/chat.py`

### Phase 5
- `app/services/rag/insights.py`
- `app/api/v1/insights.py`

### Phase 5.5
- `app/middleware/rate_limit.py`
- `app/middleware/audit_log.py`
- `app/core/logging.py`

### Phase 6
- `app/workers/tasks/rag.py`
- `tests/services/rag/test_embeddings.py`
- `tests/services/rag/test_retrieval.py`
- `tests/services/rag/test_chat.py`
- `tests/services/rag/test_insights.py`
- `tests/api/v1/test_chat.py`
- `tests/api/v1/test_rate_limiting.py`
- `tests/api/v1/test_auth.py`

---

## Execution Flow

```
Phase 1: Database Schema ────────────────► pgvector, tables, indexes
    ▼
Phase 2: Embedding Service ──────────────► OpenAI client, content hashing, storage, API endpoints
    ▼
Phase 3: Retrieval Service ──────────────► Vector search, threshold filtering
    ▼
Phase 4: Chat Endpoints ─────────────────► Grounded chat, streaming, conversation history
    ▼
Phase 5: Insight Generation ─────────────► MVP integration, deduplication, pending endpoint
    ▼
Phase 5.5: Middleware ───────────────────► Rate limiting, audit logging, PHI redaction
    ▼
Phase 6: Celery & Testing ───────────────► Background tasks, comprehensive tests
```

---

## Verification Checklist

After each phase, verify:

### Phase 1
- [ ] pgvector extension enabled
- [ ] All tables created with correct schema
- [ ] Indexes created (including HNSW vector index)
- [ ] Can insert and query sample vector

### Phase 2
- [ ] OpenAI API key configured
- [ ] Can generate embedding for sample text
- [ ] Content hash deduplication works
- [ ] Embeddings stored correctly
- [ ] All source types implemented (profile, metrics, sessions, checkins, labs, goals, messages)
- [ ] /embeddings/update endpoint works
- [ ] /embeddings/search endpoint works
- [ ] Batch update for onboarding works

### Phase 3
- [ ] Vector search returns ranked results
- [ ] Similarity threshold filtering works
- [ ] No-context case handled

### Phase 4
- [ ] /chat/complete endpoint returns grounded response
- [ ] /chat/stream endpoint returns SSE stream
- [ ] Sources included when include_sources=true
- [ ] Conversation history persisted
- [ ] has_context flag correct
- [ ] Context policy applied (priority, 4K token limit)
- [ ] max_context_items honored

### Phase 5
- [ ] /insights/generate creates insight in MVP table
- [ ] /insights/pending returns coach's pending insights
- [ ] Duplicate detection: same type in 7 days blocked
- [ ] Duplicate detection: title similarity > 0.85 blocked
- [ ] Rate limit: max 3 per client per week
- [ ] Rate limit: max 1 per trigger type per day
- [ ] Rate limit: 48-hour cooldown after approval/rejection
- [ ] Generation log populated with all fields

### Phase 5.5
- [ ] Redis rate limiting middleware active
- [ ] Rate limits: 100 chat/hr, 10 embed/hr, 50 insight/day
- [ ] 429 response with Retry-After header
- [ ] System API key bypasses rate limits
- [ ] Audit logging captures all operations
- [ ] PHI redacted from logs

### Phase 6
- [ ] Celery tasks execute correctly
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Auth tests verify coach-client relationship
- [ ] Rate limit tests verify enforcement
- [ ] Error scenario tests pass

---

**Plan Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-27
