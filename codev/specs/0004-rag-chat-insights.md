# Spec 0004: RAG Chat & Insights

## Overview

**What**: Build a Retrieval-Augmented Generation (RAG) system that grounds AI chat responses in client-specific health data and generates proactive insight drafts for coach review.

**Why**: Coaches need AI assistance that understands each client's unique context:
- Generic AI responses lack relevance without client health history
- Coaches spend time manually reviewing data before responding
- Proactive insights could identify trends coaches might miss
- Current message suggestions don't leverage client health context

RAG-grounded AI enables:
- Chat responses citing specific client metrics and trends
- AI-generated insight drafts based on health data changes
- Semantic search across client history
- Personalized recommendations grounded in actual data

**Who**:
- Coaches chatting with clients (grounded responses)
- Coaches reviewing AI-generated insights
- System generating proactive insights from data changes

## Goals

### Must Have
1. pgvector extension for embedding storage
2. Embedding generation for client health data (metrics, notes, session summaries)
3. Semantic retrieval of relevant context for chat
4. Grounded chat endpoint that cites sources
5. Insight generation from health data changes
6. Integration with MVP's existing insights approval workflow
7. Celery tasks for embedding updates and insight generation

### Should Have
- Embedding updates triggered by data changes (webhooks)
- Confidence scoring for retrieved context relevance
- Source attribution in chat responses
- Batch embedding generation for onboarding
- Insight deduplication (don't regenerate similar insights)

### Won't Have (MVP)
- Fine-tuned embedding models (use OpenAI ada-002)
- Real-time streaming RAG (batch retrieval first)
- Multi-modal embeddings (text only)
- Client-facing chat (coach-only for MVP)

## Technical Context

### MVP Data Available for RAG

| Data Source | Table | Key Fields | Embedding Strategy |
|-------------|-------|------------|-------------------|
| Health Profile | `client_health_profiles` | goals, conditions, preferences | Embed full profile |
| Health Metrics | `health_metrics` | type, value, recorded_at | Embed as time-series summaries |
| Health Goals | `health_goals` | metric_type, target, status | Embed goal + progress |
| Lab Results | `lab_values` | biomarker, value, status | Embed with reference context |
| Session Transcripts | `ai_backend.transcripts` | full_text, summary | Embed summaries |
| Session Summaries | `ai_backend.session_summaries` | topics, concerns, action_items | Embed structured summary |
| Check-ins | `checkins` | responses, ai_summary | Embed AI summaries |
| Messages | `messages` | content | Embed recent conversation |

### Existing Insights Workflow (MVP)

The MVP already has an insights approval system:

```
AI generates insight → Coach reviews → Approve/Reject → Client sees (if approved)
```

**Insights table fields:**
- `title`, `client_message`, `rationale`
- `suggested_actions[]`
- `confidence_score`
- `triggering_data` (JSONB with metrics context)
- `insight_type` (nutrition, training, recovery, motivation, general)
- `status` (pending, approved, rejected, expired)

This spec generates insight drafts that feed into this existing workflow.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Xenios MVP (Next.js)                        │
│                                                                 │
│  POST /api/ai-assistant/chat ───────────────────────────────┼───┐
│       → Grounded chat with client context                      │
│                                                                 │
│  Existing: GET /api/insights ───────────────────────────────┼───┤
│       → Coach reviews AI-generated insights                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Xenios AI Backend (This Spec)                 │
│                                                                 │
│  POST /api/v1/chat/complete                                    │
│       → Retrieve relevant context via embeddings               │
│       → Generate grounded response with citations              │
│       → Return response with sources                           │
│                                                                 │
│  POST /api/v1/insights/generate                                │
│       → Analyze client health data changes                     │
│       → Generate insight drafts                                │
│       → Write to MVP insights table                            │
│                                                                 │
│  Celery Workers:                                               │
│       → update_embeddings (on data change)                     │
│       → generate_insights (scheduled + triggered)              │
│                                                                 │
│  pgvector:                                                     │
│       → ai_backend.embeddings table                            │
│       → Semantic similarity search                             │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Implementation

### Data Models

#### Embedding Record

```python
class EmbeddingRecord(BaseModel):
    id: UUID
    client_id: UUID

    # Source reference
    source_type: str  # health_metric, session_summary, checkin, message, etc.
    source_id: UUID   # ID in source table
    source_table: str # Full table name for joins

    # Content
    content_text: str        # Original text that was embedded
    content_hash: str        # SHA256 for deduplication

    # Embedding
    embedding: list[float]   # 1536 dimensions for ada-002

    # Metadata for filtering
    metadata: dict           # {metric_type, date, insight_type, etc.}

    # Timestamps
    created_at: datetime
    updated_at: datetime

class EmbeddingSourceType(str, Enum):
    HEALTH_PROFILE = "health_profile"
    HEALTH_METRIC_SUMMARY = "health_metric_summary"  # Aggregated metrics
    HEALTH_GOAL = "health_goal"
    LAB_RESULT = "lab_result"
    SESSION_SUMMARY = "session_summary"
    CHECKIN_SUMMARY = "checkin_summary"
    MESSAGE_THREAD = "message_thread"  # Recent conversation context
```

#### Chat Request/Response

```python
class ChatRequest(BaseModel):
    client_id: UUID
    message: str
    conversation_id: UUID | None = None  # For context continuity
    include_sources: bool = True
    max_context_items: int = 10

class ChatResponse(BaseModel):
    response: str
    sources: list[SourceCitation]
    confidence: float
    tokens_used: int

class SourceCitation(BaseModel):
    source_type: str
    source_id: UUID
    relevance_score: float
    snippet: str  # Relevant excerpt
    date: date | None
```

#### Insight Generation

```python
class InsightGenerationRequest(BaseModel):
    client_id: UUID
    trigger: InsightTrigger  # scheduled, metric_change, goal_progress, etc.
    context: dict | None = None  # Additional context for generation

class InsightTrigger(str, Enum):
    SCHEDULED = "scheduled"           # Daily/weekly batch
    METRIC_CHANGE = "metric_change"   # Significant change detected
    GOAL_PROGRESS = "goal_progress"   # Goal milestone reached
    CHECKIN_SUBMITTED = "checkin_submitted"
    SESSION_COMPLETED = "session_completed"

class GeneratedInsight(BaseModel):
    """Maps to MVP insights table structure."""
    client_id: UUID
    coach_id: UUID
    title: str
    client_message: str
    rationale: str
    suggested_actions: list[str]
    confidence_score: float
    triggering_data: dict
    insight_type: str
    expires_at: datetime
```

### Project Structure (Additions)

```
app/
├── services/
│   └── rag/
│       ├── __init__.py
│       ├── embeddings.py       # Embedding generation and storage
│       ├── retrieval.py        # Semantic search and context retrieval
│       ├── chat.py             # Grounded chat generation
│       ├── insights.py         # Insight generation logic
│       └── prompts.py          # RAG prompt templates
│
├── workers/
│   └── tasks/
│       └── rag.py              # Embedding and insight Celery tasks
│
├── api/
│   └── v1/
│       ├── chat.py             # Chat endpoints
│       └── insights.py         # Insight generation endpoints
│
└── schemas/
    └── rag.py                  # RAG request/response schemas
```

### API Endpoints

```
POST /api/v1/chat/complete
  Request:
    - client_id: UUID (required)
    - message: string (required)
    - conversation_id: UUID (optional)
    - include_sources: bool (default true)
  Response:
    - response: string
    - sources: list[SourceCitation]
    - confidence: float

POST /api/v1/chat/stream
  Request: Same as /complete
  Response: SSE stream with chunks and final sources

POST /api/v1/embeddings/update
  Request:
    - client_id: UUID
    - source_type: string (optional - update specific type)
    - force: bool (default false - skip if unchanged)
  Response:
    - updated_count: int
    - skipped_count: int

POST /api/v1/embeddings/search
  Request:
    - client_id: UUID
    - query: string
    - limit: int (default 10)
    - source_types: list[string] (optional filter)
  Response:
    - results: list[SearchResult]

POST /api/v1/insights/generate
  Request:
    - client_id: UUID
    - trigger: InsightTrigger
    - context: dict (optional)
  Response:
    - insight_id: UUID (in MVP insights table)
    - title: string
    - confidence_score: float

GET /api/v1/insights/pending
  Query:
    - coach_id: UUID (from auth)
    - limit: int
  Response:
    - insights: list[GeneratedInsight]
```

### Embedding Strategy

#### What Gets Embedded

1. **Health Profile Summary** (1 embedding per client)
   ```
   "Client profile: 35-year-old male, primary goal weight loss,
   target weight 85kg from 95kg. Medical conditions: none.
   Dietary restrictions: lactose intolerant. Activity level: moderate.
   Preferred workouts: strength training, cycling."
   ```

2. **Health Metric Summaries** (1 per metric type per week)
   ```
   "Weight progress (Jan 15-21, 2026): Started 93.2kg, ended 92.8kg,
   trend: -0.4kg (-0.4%). 7-day average: 93.0kg.
   On track for goal of 85kg by March 2026."
   ```

3. **Session Summaries** (1 per session)
   ```
   "Session Jan 20, 2026: Discussed nutrition struggles with meal prep.
   Client concerns: time constraints, snacking at work.
   Action items: prepare weekly meal containers, keep healthy snacks at desk.
   Coach recommendations: try batch cooking on Sundays."
   ```

4. **Lab Result Summaries** (1 per lab panel)
   ```
   "Lab results Jan 15, 2026 (Quest Diagnostics):
   LDL cholesterol 142 mg/dL (HIGH, ref <100),
   HDL 45 mg/dL (normal),
   Triglycerides 180 mg/dL (borderline high).
   Overall lipid panel shows elevated cardiovascular risk markers."
   ```

5. **Check-in Summaries** (1 per check-in)
   ```
   "Check-in Jan 18, 2026: Adherence score 75%.
   Client reported: 'Struggled with cravings this week'.
   AI highlights: missed 2 workout days, water intake below target.
   Concerns flagged: declining motivation."
   ```

#### Embedding Generation

```python
class EmbeddingService:
    """Generate and store embeddings using OpenAI ada-002."""

    MODEL = "text-embedding-ada-002"
    DIMENSIONS = 1536

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = await self.openai_client.embeddings.create(
            model=self.MODEL,
            input=text,
        )
        return response.data[0].embedding

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
        if not source_types or "health_profile" in source_types:
            profile_text = await self._build_health_profile_text(client_id)
            if await self._should_update(client_id, "health_profile", profile_text, force):
                await self._store_embedding(client_id, "health_profile", profile_text)
                updated += 1
            else:
                skipped += 1

        # Metric summaries (by type, by week)
        if not source_types or "health_metric_summary" in source_types:
            metric_summaries = await self._build_metric_summaries(client_id)
            for summary in metric_summaries:
                if await self._should_update(client_id, "health_metric_summary", summary.text, force):
                    await self._store_embedding(
                        client_id, "health_metric_summary", summary.text,
                        metadata={"metric_type": summary.metric_type, "week": summary.week}
                    )
                    updated += 1
                else:
                    skipped += 1

        # Session summaries
        # Check-in summaries
        # Lab results
        # ... etc

        return EmbeddingUpdateResult(updated_count=updated, skipped_count=skipped)

    def _should_update(
        self,
        client_id: UUID,
        source_type: str,
        text: str,
        force: bool,
    ) -> bool:
        """Check if embedding needs update based on content hash."""
        if force:
            return True

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        existing = await self._get_existing_embedding(client_id, source_type)

        if existing and existing.content_hash == content_hash:
            return False
        return True
```

### Retrieval Strategy

**Similarity Threshold:** 0.7 (cosine similarity)
- Results below this threshold are excluded from context
- Prevents hallucination from irrelevant matches

**No-Context Handling:**
When no relevant context is found (empty embeddings, all below threshold, or missing data):
1. Return response with `has_context: false` flag
2. Use fallback prompt that acknowledges limited data
3. Response should clearly state: "I don't have enough information about [topic] to provide specific guidance"
4. Include `sources: []` (empty array)
5. Set `confidence: 0.0`

```python
class RetrievalService:
    """Retrieve relevant context for RAG."""

    SIMILARITY_THRESHOLD = 0.7  # Minimum cosine similarity for relevance

    async def retrieve_context(
        self,
        client_id: UUID,
        query: str,
        max_items: int = 10,
        source_types: list[str] | None = None,
    ) -> list[RetrievedContext]:
        """
        Retrieve relevant context using semantic search.

        1. Generate embedding for query
        2. Find similar embeddings for this client
        3. Fetch source content
        4. Return ranked results
        """
        query_embedding = await self.embedding_service.generate_embedding(query)

        # pgvector similarity search
        results = await self._vector_search(
            client_id=client_id,
            query_embedding=query_embedding,
            limit=max_items,
            source_types=source_types,
        )

        # Fetch full content for each result
        contexts = []
        for result in results:
            content = await self._fetch_source_content(result)
            contexts.append(RetrievedContext(
                source_type=result.source_type,
                source_id=result.source_id,
                content=content,
                relevance_score=result.similarity,
                date=result.metadata.get("date"),
            ))

        return contexts

    async def _vector_search(
        self,
        client_id: UUID,
        query_embedding: list[float],
        limit: int,
        source_types: list[str] | None,
    ) -> list[SearchResult]:
        """Execute pgvector similarity search."""

        query = """
            SELECT
                id, source_type, source_id, content_text, metadata,
                1 - (embedding <=> $1::vector) as similarity
            FROM ai_backend.embeddings
            WHERE client_id = $2
            {source_filter}
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """

        source_filter = ""
        if source_types:
            source_filter = f"AND source_type = ANY($4)"

        # Execute query
        # ...
```

### Grounded Chat Generation

```python
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

    async def generate_response(
        self,
        client_id: UUID,
        message: str,
        conversation_id: UUID | None = None,
    ) -> ChatResponse:
        """Generate grounded response with citations."""

        # 1. Retrieve relevant context
        contexts = await self.retrieval_service.retrieve_context(
            client_id=client_id,
            query=message,
            max_items=10,
        )

        # 2. Build context string with source markers
        context_str = self._build_context_string(contexts)

        # 3. Generate response
        response = await self.llm_client.complete(
            task="chat",  # Uses Opus 4.5
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT.format(context=context_str)},
                {"role": "user", "content": message},
            ],
        )

        # 4. Extract citations from response
        sources = self._extract_citations(response.content, contexts)

        return ChatResponse(
            response=response.content,
            sources=sources,
            confidence=self._calculate_confidence(contexts),
            tokens_used=response.usage.total_tokens,
        )

    def _build_context_string(self, contexts: list[RetrievedContext]) -> str:
        """Build context string with source markers for citation."""
        parts = []
        for i, ctx in enumerate(contexts):
            marker = f"[Source {i+1}]"
            date_str = f" ({ctx.date})" if ctx.date else ""
            parts.append(f"{marker} {ctx.source_type}{date_str}:\n{ctx.content}\n")
        return "\n".join(parts)
```

### Insight Generation

```python
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

    async def generate_insight(
        self,
        client_id: UUID,
        coach_id: UUID,
        trigger: InsightTrigger,
        context: dict | None = None,
    ) -> GeneratedInsight:
        """Generate insight and write to MVP insights table."""

        # 1. Gather client context
        client_context = await self._gather_client_context(client_id)

        # 2. Identify recent changes based on trigger
        changes = await self._identify_changes(client_id, trigger, context)

        # 3. Check for duplicate/similar recent insights
        # Deduplication criteria:
        # - Same insight_type within 7-day window
        # - Same triggering metric type (if metric-based trigger)
        # - Title embedding similarity > 0.85 to pending/approved insights
        if await self._is_duplicate_insight(client_id, changes):
            raise DuplicateInsightError("Similar insight already pending")

        # 4. Generate insight
        prompt = self.INSIGHT_PROMPT.format(
            context=client_context,
            changes=changes,
        )

        response = await self.llm_client.complete(
            task="insight_generation",  # Uses Opus 4.5
            messages=[{"role": "user", "content": prompt}],
        )

        insight_data = self._parse_insight_response(response.content)

        # 5. Write to MVP insights table
        insight_id = await self._write_to_mvp_insights(
            client_id=client_id,
            coach_id=coach_id,
            insight_data=insight_data,
            triggering_data={
                "trigger": trigger,
                "changes": changes,
                "context_summary": client_context[:500],
            },
        )

        return GeneratedInsight(
            id=insight_id,
            **insight_data,
        )

    async def _write_to_mvp_insights(
        self,
        client_id: UUID,
        coach_id: UUID,
        insight_data: dict,
        triggering_data: dict,
    ) -> UUID:
        """Write insight to MVP's public.insights table."""

        query = """
            INSERT INTO public.insights (
                coach_id, client_id, title, client_message, rationale,
                suggested_actions, confidence_score, triggering_data,
                insight_type, status, expires_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending', NOW() + INTERVAL '7 days'
            ) RETURNING id
        """
        # Execute and return insight_id
```

### Database Schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Embeddings table
CREATE TABLE ai_backend.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,

    -- Source reference
    source_type VARCHAR(50) NOT NULL,
    source_id UUID,
    source_table VARCHAR(100),

    -- Content
    content_text TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,  -- SHA256 for deduplication

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
CREATE INDEX idx_embeddings_vector ON ai_backend.embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Chat history for context continuity
CREATE TABLE ai_backend.chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    role VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,

    -- RAG metadata
    sources_used JSONB,  -- List of source citations
    tokens_used INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_history_conversation ON ai_backend.chat_history(conversation_id, created_at);
CREATE INDEX idx_chat_history_client ON ai_backend.chat_history(client_id, created_at DESC);

-- Insight generation log (for deduplication and analytics)
CREATE TABLE ai_backend.insight_generation_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,

    trigger VARCHAR(50) NOT NULL,
    triggering_data JSONB NOT NULL,

    -- Result
    insight_id UUID,  -- FK to public.insights if generated
    status VARCHAR(20) NOT NULL,  -- generated, duplicate, failed
    error_message TEXT,

    -- Metrics
    context_items_used INTEGER,
    tokens_used INTEGER,
    generation_time_ms INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_insight_gen_log_client ON ai_backend.insight_generation_log(client_id, created_at DESC);
```

### Celery Tasks

```python
@celery_app.task(
    bind=True,
    queue="llm",
    max_retries=2,
)
def update_client_embeddings(
    self,
    client_id: str,
    source_types: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Update embeddings for a client."""
    embedding_service = EmbeddingService()
    result = asyncio.run(
        embedding_service.update_client_embeddings(
            client_id=UUID(client_id),
            source_types=source_types,
            force=force,
        )
    )
    return result.dict()

@celery_app.task(
    bind=True,
    queue="llm",
    max_retries=1,
)
def generate_client_insight(
    self,
    client_id: str,
    coach_id: str,
    trigger: str,
    context: dict | None = None,
) -> dict:
    """Generate insight for a client."""
    insight_service = InsightGenerationService()
    try:
        result = asyncio.run(
            insight_service.generate_insight(
                client_id=UUID(client_id),
                coach_id=UUID(coach_id),
                trigger=InsightTrigger(trigger),
                context=context,
            )
        )
        return {"status": "generated", "insight_id": str(result.id)}
    except DuplicateInsightError:
        return {"status": "duplicate"}

@celery_app.task(queue="llm")
def batch_generate_insights():
    """Scheduled task to generate insights for all active clients."""
    # Get all active client-coach pairs
    # For each, check if insights should be generated
    # Queue individual insight generation tasks
    pass
```

## Security & Authorization

### Authentication
All endpoints require:
1. **X-API-Key header**: Backend-to-backend authentication
2. **Authorization header**: Bearer JWT from Supabase Auth

### Authorization Rules

| Endpoint | Who Can Access |
|----------|----------------|
| POST /chat/complete | Coach (for their clients only) |
| POST /embeddings/update | Coach (for their clients) or System |
| POST /insights/generate | Coach (for their clients) or System |
| GET /insights/pending | Coach (sees only their insights) |

### Data Privacy

- Embeddings stored per-client, isolated by client_id
- No cross-client data access in vector search
- Chat history tied to coach-client relationship
- Insights written to coach's queue only

### PHI/PII Protection

**Logging Redaction:**
- NEVER log full embedding content or client health data
- Log only: client_id, source_type, operation type, success/failure
- Redact any PII from error messages before logging
- Use structured logging with explicit allow-list of loggable fields

**Data Retention:**
- Embeddings: Retained while client is active, deleted on client deletion
- Chat history: 90-day retention, then archived/deleted
- Insight generation logs: 1-year retention for analytics
- Follow same retention as MVP's health data policies

**Access Auditing:**
- Log all embedding reads/writes with coach_id, client_id, timestamp
- Log all chat completions with coach_id, client_id, tokens_used
- Log insight generation attempts (success/duplicate/failure)
- Audit logs retained 2 years for compliance

**Error Handling:**
- Never expose client health data in error messages
- Return generic errors to API; log details server-side
- Use error codes that don't leak PHI (e.g., "RETRIEVAL_FAILED" not "No weight data found")

## Acceptance Criteria

### AC1: pgvector Setup
- [ ] pgvector extension enabled in database
- [ ] Embeddings table created with vector column
- [ ] IVFFlat index for approximate nearest neighbor search
- [ ] Can store and query 1536-dimension vectors

### AC2: Embedding Generation
- [ ] Health profile embedded for each client
- [ ] Metric summaries embedded by type and time period
- [ ] Session summaries embedded
- [ ] Check-in summaries embedded
- [ ] Content hashing prevents duplicate embeddings
- [ ] Batch embedding update endpoint works

### AC3: Semantic Retrieval
- [ ] Query embedding generated from user message
- [ ] Top-K similar embeddings retrieved per client
- [ ] Results below 0.7 similarity threshold excluded
- [ ] Source content fetched for retrieved embeddings
- [ ] Relevance scores calculated and returned
- [ ] Filtering by source type works
- [ ] No-context case returns empty sources and confidence 0.0

### AC4: Grounded Chat
- [ ] Chat endpoint accepts client_id and message
- [ ] Relevant context retrieved via embeddings
- [ ] Response generated using context
- [ ] Source citations included in response
- [ ] Response stays grounded in provided context
- [ ] No-context: response acknowledges limited data, has_context=false
- [ ] No-context: response does not hallucinate facts

### AC5: Insight Generation
- [ ] Insights generated from health data changes
- [ ] Insights written to MVP insights table
- [ ] Duplicate detection: same insight_type within 7 days blocked
- [ ] Duplicate detection: title similarity > 0.85 blocked
- [ ] Rate limiting: max 3 insights per client per week enforced
- [ ] Confidence scores assigned
- [ ] Triggering data recorded for audit

### AC6: Celery Integration
- [ ] Embedding update task works
- [ ] Insight generation task works
- [ ] Batch insight generation scheduled task works

### AC7: Performance
- [ ] Embedding generation < 500ms per item
- [ ] Vector search < 100ms for top-10 results
- [ ] Chat response < 5 seconds (including retrieval + generation)
- [ ] Insight generation < 10 seconds

### AC8: Security & Authorization
- [ ] Coach can only access their own clients' data
- [ ] Invalid client_id returns 404 (not 403)
- [ ] All operations logged with coach_id, client_id, timestamp
- [ ] No PHI in logs or error messages
- [ ] Rate limiting enforced per coach

## Test Plan

### Unit Tests
- Embedding generation
- Content hashing and deduplication
- Vector similarity calculation
- Context building for prompts
- Insight JSON parsing
- Similarity threshold filtering
- No-context response generation
- Insight deduplication logic

### Integration Tests
- Full embedding update flow
- Vector search with filters
- Chat endpoint with context retrieval
- Insight generation to MVP table
- Celery task execution

### Auth/Authorization Tests
- Coach can only access their clients' embeddings
- Coach cannot query other coaches' clients
- Invalid client_id returns 404 (not 403, to avoid enumeration)
- Missing X-API-Key returns 401
- Invalid JWT returns 401
- Expired JWT returns 401
- System tasks bypass coach-client check with service key

### Error Scenario Tests
- **OpenAI API failure:** Graceful degradation, retry logic, error response
- **Vector search timeout:** Returns partial results or error within SLA
- **No embeddings for client:** Returns no-context response (not error)
- **Empty query:** Returns 400 with validation error
- **Malformed client_id:** Returns 400 with validation error
- **Database connection failure:** Returns 503, logs error, alerts
- **Rate limit exceeded:** Returns 429 with retry-after header
- **Insight generation fails mid-process:** Rolls back, logs, doesn't create partial insight

### Test Data
- Sample client with health profile
- Health metrics over 30 days
- 2-3 session summaries
- 1 lab result
- Sample check-ins
- Client with no data (for no-context testing)
- Multiple coaches with separate clients (for auth testing)

## Dependencies

- **Spec 0001**: AI Backend Foundation (LLM client, Celery, database)
- **Spec 0002**: Document Extraction (lab results, health data)
- **Spec 0003**: Transcription (session summaries for embedding)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| pgvector performance at scale | Medium | Medium | IVFFlat index, partition by client |
| Embedding costs (OpenAI) | Low | Low | Hash-based deduplication, batch updates |
| Hallucination despite RAG | Medium | High | Strict prompting, confidence thresholds |
| Insight spam | Medium | Medium | Deduplication, rate limiting per client |
| Stale embeddings | Low | Medium | Trigger updates on data changes |

## Cost Considerations

### OpenAI Embeddings (ada-002)
- $0.0001 per 1K tokens
- Average client: ~50 embeddings, ~500 tokens each = 25K tokens = $0.0025/client
- Monthly updates: ~$0.01/client

### LLM (Chat + Insights)
- Opus 4.5: ~$0.015 per 1K input, $0.075 per 1K output
- Average chat: ~3K input (context), ~500 output = ~$0.08
- Average insight: ~2K input, ~300 output = ~$0.05

**Estimated: ~$5-10/month per active client**

## Design Decisions (Resolved)

### Embedding Update Triggers
**Decision:** Hybrid approach - immediate for high-value data, nightly batch for the rest.

**Immediate updates (via Celery task on data change):**
- Lab results (high impact, infrequent)
- Session summaries (immediate relevance for follow-up)
- Health profile changes (affects all context)

**Nightly batch updates:**
- Health metrics (aggregated weekly summaries)
- Check-in summaries (daily aggregation)
- Message threads (end-of-day summary)

**Rationale:** Balances freshness with cost. High-impact, infrequent events update immediately. High-frequency, lower-impact data batches for efficiency.

### Insight Generation Frequency
**Decision:** Maximum 3 insights per client per week, triggered by events not schedule.

**Event triggers:**
- Metric milestone reached (goal progress > 80% or significant regression)
- Lab result uploaded (always generates insight)
- Session completed (if notable changes discussed)
- Check-in streak milestone (7, 14, 30 days)

**Rate limiting:**
- Max 3 insights per client per week (regardless of triggers)
- Max 1 insight per trigger type per day
- Cooldown: 48 hours after insight is approved/rejected before same type

**Rationale:** Prevents insight fatigue. Quality over quantity. Coaches can manually trigger if needed.

### Context Window for Chat
**Decision:** 10 items maximum, 4,000 tokens total context limit.

**Selection priority:**
1. Health profile (always included, ~500 tokens)
2. Most recent session summary (~800 tokens)
3. Relevant metric summaries (top 3 by similarity, ~600 tokens each)
4. Recent check-ins (last 7 days, ~400 tokens)
5. Lab results if medically relevant to query (~500 tokens)

**Truncation:** If total exceeds 4,000 tokens, trim oldest/lowest-relevance items first.

**Rationale:** Opus 4.5 context is expensive. 4K tokens provides good grounding while controlling costs. Most queries don't need full history.

## Future Considerations

- Fine-tuned embedding model for health/fitness domain
- Streaming RAG responses
- Multi-modal embeddings (images from check-ins)
- Client-facing grounded chat
- Insight effectiveness tracking (which insights lead to action)

---

**Spec Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-27
