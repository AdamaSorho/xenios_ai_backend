# Plan 0003: Transcription & Session Processing

**Spec**: [codev/specs/0003-transcription-session-processing.md](../specs/0003-transcription-session-processing.md)
**Status**: Ready for implementation
**Estimated Phases**: 6

---

## Implementation Strategy

Build the transcription pipeline incrementally, starting with Deepgram integration, then adding speaker diarization, and finally LLM summarization. Each phase produces working, testable functionality.

**Key Principles:**
- Deepgram integration first (core dependency)
- Reuse S3 storage patterns from Spec 0002
- Test with sample audio files at each phase
- Security and authorization from the start
- File-based processing for large audio (avoid loading 500MB into memory)
- Intent classification as part of diarization phase

**Status Enum (per spec):**
```
pending → uploading → transcribing → diarizing → summarizing → completed
                                                             → partial (if summary fails)
                                                             → failed
```

---

## Phase 1: Infrastructure & Deepgram Client

**Goal**: Set up transcription infrastructure, Deepgram client wrapper, and database schema.

### Tasks

1.1 **Add transcription dependencies to `pyproject.toml`**
```toml
dependencies = [
    # ... existing ...
    "deepgram-sdk>=3.0.0",
    "ffmpeg-python>=0.2.0",  # Audio duration/format detection
]
```

1.2 **Create transcription service structure**
```
app/services/transcription/
├── __init__.py
├── deepgram.py         # Deepgram client wrapper
├── audio.py            # Audio file utilities (duration, format)
└── storage.py          # Audio file S3 operations (reuse from extraction)
```

1.3 **Implement Deepgram client wrapper** (`app/services/transcription/deepgram.py`)
```python
from deepgram import DeepgramClient, PrerecordedOptions
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class DeepgramService:
    """Wrapper for Deepgram transcription API."""

    def __init__(self):
        settings = get_settings()
        self.client = DeepgramClient(settings.deepgram_api_key)

    async def transcribe_file(
        self,
        file_path: str,
        options: dict | None = None,
    ) -> TranscriptionResponse:
        """
        Transcribe audio bytes using Deepgram.

        Default options:
        - model: nova-2
        - diarize: true
        - punctuate: true
        - utterances: true
        - paragraphs: true
        """
        default_options = PrerecordedOptions(
            model="nova-2",
            language="en",
            punctuate=True,
            diarize=True,
            utterances=True,
            smart_format=True,
            paragraphs=True,
        )

        if options:
            for key, value in options.items():
                setattr(default_options, key, value)

        try:
            # Stream from file to avoid loading into memory
            with open(file_path, "rb") as audio_file:
                response = await self.client.listen.asyncrest.v1.transcribe_file(
                    {"buffer": audio_file},
                    default_options,
                )
            return self._parse_response(response)
        except Exception as e:
            logger.error("Deepgram transcription failed", error=str(e))
            raise

    def _parse_response(self, response) -> TranscriptionResponse:
        """Parse Deepgram response into our data model."""
        # Extract utterances, speakers, confidence, etc.
        pass
```

1.4 **Implement audio utilities** (`app/services/transcription/audio.py`)
```python
import ffmpeg
from pathlib import Path

class AudioService:
    """Audio file utilities."""

    SUPPORTED_FORMATS = ["mp3", "m4a", "wav", "mp4", "aac", "ogg", "webm"]
    MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
    MAX_DURATION_SECONDS = 2 * 60 * 60  # 2 hours
    MIN_DURATION_SECONDS = 10

    def get_audio_info(self, file_path: str) -> AudioInfo:
        """Get audio duration and format using ffprobe."""
        try:
            probe = ffmpeg.probe(file_path)
            audio_stream = next(
                s for s in probe["streams"] if s["codec_type"] == "audio"
            )
            return AudioInfo(
                duration_seconds=float(probe["format"]["duration"]),
                format=probe["format"]["format_name"],
                codec=audio_stream["codec_name"],
                sample_rate=int(audio_stream.get("sample_rate", 0)),
                channels=int(audio_stream.get("channels", 0)),
                size_bytes=int(probe["format"]["size"]),
            )
        except Exception as e:
            raise AudioProcessingError(f"Failed to probe audio: {e}")

    def validate_audio(self, file_path: str) -> tuple[bool, str | None]:
        """Validate audio file meets requirements."""
        info = self.get_audio_info(file_path)

        if info.size_bytes > self.MAX_SIZE_BYTES:
            return False, f"File too large: {info.size_bytes} bytes (max {self.MAX_SIZE_BYTES})"

        if info.duration_seconds > self.MAX_DURATION_SECONDS:
            return False, f"Duration too long: {info.duration_seconds}s (max {self.MAX_DURATION_SECONDS}s)"

        if info.duration_seconds < self.MIN_DURATION_SECONDS:
            return False, f"Duration too short: {info.duration_seconds}s (min {self.MIN_DURATION_SECONDS}s)"

        return True, None
```

1.5 **Create database migration** (`scripts/migrations/0003_transcription_tables.sql`)
```sql
-- Transcription jobs
CREATE TABLE ai_backend.transcription_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,
    conversation_id UUID,

    -- Audio file
    audio_url TEXT NOT NULL,
    audio_filename VARCHAR(255) NOT NULL,
    audio_format VARCHAR(20) NOT NULL,
    audio_size_bytes INTEGER NOT NULL,
    audio_duration_seconds DECIMAL(10,2),

    -- Processing
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    transcription_completed_at TIMESTAMPTZ,
    summary_completed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Session metadata
    session_date DATE,
    session_type VARCHAR(50),
    session_title VARCHAR(255),

    -- Webhook
    webhook_url TEXT,
    webhook_secret TEXT,
    webhook_sent_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_transcription_jobs_client ON ai_backend.transcription_jobs(client_id);
CREATE INDEX idx_transcription_jobs_coach ON ai_backend.transcription_jobs(coach_id);
CREATE INDEX idx_transcription_jobs_status ON ai_backend.transcription_jobs(status);
CREATE INDEX idx_transcription_jobs_created ON ai_backend.transcription_jobs(created_at DESC);

-- Transcripts
CREATE TABLE ai_backend.transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ai_backend.transcription_jobs(id) ON DELETE CASCADE,

    full_text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    duration_seconds DECIMAL(10,2) NOT NULL,

    deepgram_request_id VARCHAR(100),
    model_used VARCHAR(50),
    confidence_score DECIMAL(4,3),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_transcripts_job ON ai_backend.transcripts(job_id);
CREATE INDEX idx_transcripts_fulltext ON ai_backend.transcripts
    USING gin(to_tsvector('english', full_text));

-- Utterances
CREATE TABLE ai_backend.utterances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_id UUID NOT NULL REFERENCES ai_backend.transcripts(id) ON DELETE CASCADE,

    speaker_number INTEGER NOT NULL,
    speaker_label VARCHAR(20),
    speaker_confidence DECIMAL(4,3),

    text TEXT NOT NULL,
    start_time DECIMAL(10,3) NOT NULL,
    end_time DECIMAL(10,3) NOT NULL,
    confidence DECIMAL(4,3),

    intent VARCHAR(50),
    sentiment VARCHAR(20),

    sequence_number INTEGER NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_utterances_transcript ON ai_backend.utterances(transcript_id);
CREATE INDEX idx_utterances_speaker ON ai_backend.utterances(transcript_id, speaker_label);

-- Session summaries
CREATE TABLE ai_backend.session_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ai_backend.transcription_jobs(id) ON DELETE CASCADE,
    transcript_id UUID NOT NULL REFERENCES ai_backend.transcripts(id) ON DELETE CASCADE,

    executive_summary TEXT NOT NULL,
    key_topics JSONB NOT NULL DEFAULT '[]',
    client_concerns JSONB NOT NULL DEFAULT '[]',
    coach_recommendations JSONB NOT NULL DEFAULT '[]',
    action_items JSONB NOT NULL DEFAULT '[]',
    goals_discussed JSONB NOT NULL DEFAULT '[]',
    coaching_moments JSONB NOT NULL DEFAULT '[]',

    session_type_detected VARCHAR(50),
    client_sentiment VARCHAR(20),
    engagement_score DECIMAL(3,2),

    llm_model VARCHAR(100),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_session_summaries_job ON ai_backend.session_summaries(job_id);
```

1.6 **Create transcription schemas** (`app/schemas/transcription.py`)
- `TranscriptionUploadRequest`
- `TranscriptionJobResponse`
- `TranscriptionStatusResponse`
- `TranscriptResponse`
- `UtteranceResponse`
- `SessionSummaryResponse`

### Acceptance Criteria Coverage
- AC1: Deepgram Integration (partial - client setup)

### Verification
```bash
# Run migration
psql $DATABASE_URL -f scripts/migrations/0003_transcription_tables.sql

# Test Deepgram connection
uv run python -c "
from app.services.transcription.deepgram import DeepgramService
import asyncio
svc = DeepgramService()
print('Deepgram client initialized')
"
```

---

## Phase 2: API Endpoints & Celery Task

**Goal**: Implement upload endpoint, status tracking, and transcription Celery task.

### Tasks

2.1 **Implement transcription API router** (`app/api/v1/transcription.py`)
```python
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.core.auth import get_current_user

router = APIRouter(prefix="/transcription", tags=["transcription"])

@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    client_id: UUID = Form(...),
    session_date: date | None = Form(None),
    session_type: str | None = Form(None),
    session_title: str | None = Form(None),
    conversation_id: UUID | None = Form(None),
    webhook_url: str | None = Form(None),
    user: UserContext = Depends(get_current_user),
) -> TranscriptionJobResponse:
    """
    Upload audio for transcription.

    Returns job_id immediately. Poll /status/{job_id} for results.
    """
    # 1. Validate file format
    # 2. Save to temp file, validate duration
    # 3. Upload to S3
    # 4. Generate webhook_secret if webhook_url provided
    # 5. Create transcription_job record (owned by user.coach_id)
    # 6. Queue Celery task
    # 7. Return job_id and webhook_secret (one-time display)
    pass

def generate_webhook_secret() -> str:
    """Generate secure webhook signing secret."""
    import secrets
    return secrets.token_urlsafe(32)

@router.get("/status/{job_id}")
async def get_transcription_status(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
) -> TranscriptionStatusResponse:
    """Get transcription job status."""
    # Only return if job.coach_id == user.coach_id
    # Return 404 if not found or not owned
    pass

@router.get("/{job_id}/transcript")
async def get_transcript(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
) -> TranscriptResponse:
    """Get full transcript with utterances."""
    pass

@router.get("/{job_id}/summary")
async def get_summary(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
) -> SessionSummaryResponse:
    """Get AI-generated session summary."""
    pass

@router.get("/sessions")
async def list_sessions(
    client_id: UUID | None = None,
    status: str | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    user: UserContext = Depends(get_current_user),
) -> PaginatedResponse[TranscriptionJobResponse]:
    """List transcription jobs for coach."""
    pass

@router.post("/{job_id}/reprocess")
async def reprocess_transcription(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
) -> TranscriptionJobResponse:
    """Retry failed transcription."""
    pass

@router.delete("/{job_id}")
async def delete_transcription(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
) -> dict:
    """Delete transcription job and all associated data."""
    # Delete from S3
    # Delete from database (CASCADE handles related records)
    pass
```

2.2 **Implement authorization helper**
```python
async def get_job_or_404(
    job_id: UUID,
    coach_id: UUID,
    db: AsyncSession,
) -> TranscriptionJob:
    """Get job if owned by coach, else 404."""
    job = await db.get(TranscriptionJob, job_id)
    if not job or job.coach_id != coach_id:
        raise HTTPException(status_code=404, detail="Transcription job not found")
    return job
```

2.3 **Implement webhook validation**
```python
import httpx

async def validate_webhook_url(url: str) -> bool:
    """Validate webhook URL is reachable and HTTPS."""
    if not url.startswith("https://") and "localhost" not in url:
        return False

    try:
        async with httpx.AsyncClient() as client:
            response = await client.head(url, timeout=5.0)
            return response.status_code < 400
    except Exception:
        return False
```

2.4 **Implement transcription Celery task** (`app/workers/tasks/transcription.py`)
```python
from app.workers.celery_app import celery_app
from app.workers.tasks.base import BaseTask

@celery_app.task(
    bind=True,
    base=BaseTask,
    queue="transcription",
    max_retries=2,
    soft_time_limit=600,  # 10 minutes
    time_limit=900,       # 15 minutes hard limit
)
def process_transcription(self, job_id: str) -> dict:
    """
    Process audio transcription.

    1. Update status to 'transcribing'
    2. Download audio from S3
    3. Validate audio duration
    4. Send to Deepgram
    5. Store transcript and utterances
    6. Update status to 'summarizing'
    7. Generate summary with LLM
    8. Store summary
    9. Update status to 'completed'
    10. Send webhook
    """
    pass
```

2.5 **Add transcription router to v1** (`app/api/v1/router.py`)
```python
from app.api.v1.transcription import router as transcription_router
router.include_router(transcription_router)
```

### Acceptance Criteria Coverage
- AC2: Audio Upload
- AC5: Transcript Storage (partial)
- AC7: API Completeness

### Verification
```bash
# Test upload endpoint
curl -X POST http://localhost:8000/api/v1/transcription/upload \
  -H "X-API-Key: $API_KEY" \
  -H "Authorization: Bearer $JWT" \
  -F "file=@test_audio.mp3" \
  -F "client_id=$CLIENT_ID"

# Check status
curl http://localhost:8000/api/v1/transcription/status/{job_id} \
  -H "X-API-Key: $API_KEY" \
  -H "Authorization: Bearer $JWT"
```

---

## Phase 3: Transcription Processing

**Goal**: Complete Deepgram transcription flow with transcript storage.

### Tasks

3.1 **Implement full transcription task**
```python
async def _process_transcription(self, job_id: str):
    """Core transcription logic."""
    job = await self._get_job(job_id)

    try:
        # Update status
        await self._update_status(job, "transcribing")

        # Download audio
        audio_bytes = await storage_service.download_file(job.audio_url)

        # Transcribe with Deepgram
        deepgram = DeepgramService()
        result = await deepgram.transcribe(audio_bytes)

        # Store transcript
        transcript = await self._store_transcript(job, result)

        # Store utterances
        await self._store_utterances(transcript, result.utterances)

        await self._update_status(job, "transcription_completed")

        return transcript.id

    except DeepgramError as e:
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=30 * (self.request.retries + 1))
        await self._mark_failed(job, str(e))
        raise
```

3.2 **Implement transcript storage**
```python
async def _store_transcript(
    self,
    job: TranscriptionJob,
    result: TranscriptionResponse,
) -> Transcript:
    """Store transcript in database."""
    transcript = Transcript(
        job_id=job.id,
        full_text=result.full_text,
        word_count=result.word_count,
        duration_seconds=result.duration_seconds,
        deepgram_request_id=result.request_id,
        model_used=result.model,
        confidence_score=result.confidence,
    )
    # Save to database
    return transcript

async def _store_utterances(
    self,
    transcript: Transcript,
    utterances: list[DeepgramUtterance],
) -> None:
    """Store individual utterances."""
    for i, utt in enumerate(utterances):
        utterance = Utterance(
            transcript_id=transcript.id,
            speaker_number=utt.speaker,
            text=utt.text,
            start_time=utt.start,
            end_time=utt.end,
            confidence=utt.confidence,
            sequence_number=i,
        )
        # Batch insert
```

3.3 **Implement Deepgram response parsing**
```python
@dataclass
class TranscriptionResponse:
    full_text: str
    word_count: int
    duration_seconds: float
    request_id: str
    model: str
    confidence: float
    utterances: list[DeepgramUtterance]
    words: list[DeepgramWord]

@dataclass
class DeepgramUtterance:
    speaker: int
    text: str
    start: float
    end: float
    confidence: float
    words: list[DeepgramWord]
```

3.4 **Add retry logic with exponential backoff**
```python
@celery_app.task(
    bind=True,
    autoretry_for=(DeepgramTimeoutError, DeepgramRateLimitError),
    retry_backoff=30,  # 30s, 60s, 120s
    retry_backoff_max=300,
    retry_jitter=True,
)
def process_transcription(self, job_id: str):
    pass
```

### Acceptance Criteria Coverage
- AC1: Deepgram Integration
- AC3: Transcription Processing
- AC5: Transcript Storage

### Verification
```bash
# Test with sample audio
uv run pytest tests/test_transcription_task.py -v

# End-to-end test
curl -X POST http://localhost:8000/api/v1/transcription/upload \
  -F "file=@tests/fixtures/sample_session.mp3" \
  -F "client_id=$CLIENT_ID"
# Poll status until completed
# Retrieve transcript
```

---

## Phase 4: Speaker Diarization

**Goal**: Implement speaker identification and labeling (coach vs client).

### Tasks

4.1 **Create diarization service** (`app/services/transcription/diarization.py`)
```python
class DiarizationService:
    """Assign speaker labels based on conversation patterns."""

    COACHING_TERMS = [
        "goal", "progress", "plan", "tracking", "consistency",
        "nutrition", "training", "recovery", "sleep", "stress",
        "habit", "routine", "accountability", "check-in", "review",
        "how", "what", "why", "tell me", "describe",
    ]

    SPEAKER_CONFIDENCE_THRESHOLD = 0.7

    def assign_speaker_labels(
        self,
        utterances: list[Utterance],
    ) -> list[Utterance]:
        """
        Assign coach/client labels to speakers.

        Heuristics:
        1. Question ratio - coaches ask more questions
        2. Coaching terms - coaches use specific vocabulary
        3. First speaker - often the coach
        4. Directive language - coaches give instructions
        """
        # Count speakers
        speakers = set(u.speaker_number for u in utterances)

        if len(speakers) == 1:
            # Monologue - label as "speaker"
            for u in utterances:
                u.speaker_label = "speaker"
                u.speaker_confidence = 1.0
            return utterances

        if len(speakers) > 2:
            # Multi-party - label coach and participants
            return self._handle_multi_speaker(utterances, speakers)

        # Two speakers - identify coach vs client
        speaker_scores = self._calculate_speaker_scores(utterances)

        # Highest score is likely coach
        coach_speaker = max(speaker_scores, key=speaker_scores.get)
        coach_confidence = speaker_scores[coach_speaker]

        for u in utterances:
            if coach_confidence >= self.SPEAKER_CONFIDENCE_THRESHOLD:
                u.speaker_label = "coach" if u.speaker_number == coach_speaker else "client"
                u.speaker_confidence = coach_confidence
            else:
                u.speaker_label = f"speaker_{u.speaker_number}"
                u.speaker_confidence = coach_confidence

        return utterances

    def _calculate_speaker_scores(
        self,
        utterances: list[Utterance],
    ) -> dict[int, float]:
        """Calculate coach likelihood score for each speaker."""
        speaker_stats = {}

        for speaker in set(u.speaker_number for u in utterances):
            speaker_utterances = [u for u in utterances if u.speaker_number == speaker]
            total_text = " ".join(u.text for u in speaker_utterances)

            stats = {
                "question_ratio": self._count_questions(speaker_utterances) / len(speaker_utterances),
                "coaching_terms": self._count_coaching_terms(total_text) / len(total_text.split()),
                "first_speaker": 1.0 if utterances[0].speaker_number == speaker else 0.0,
                "directive_ratio": self._count_directives(speaker_utterances) / len(speaker_utterances),
            }

            # Weighted score
            score = (
                stats["question_ratio"] * 0.30 +
                stats["coaching_terms"] * 0.30 +
                stats["first_speaker"] * 0.15 +
                stats["directive_ratio"] * 0.25
            )
            speaker_stats[speaker] = score

        return speaker_stats

    def _count_questions(self, utterances: list[Utterance]) -> int:
        return sum(1 for u in utterances if "?" in u.text)

    def _count_coaching_terms(self, text: str) -> int:
        text_lower = text.lower()
        return sum(1 for term in self.COACHING_TERMS if term in text_lower)

    def _count_directives(self, utterances: list[Utterance]) -> int:
        directive_starts = ["try", "make sure", "remember", "focus", "let's", "you should", "i want you"]
        count = 0
        for u in utterances:
            text_lower = u.text.lower()
            if any(text_lower.startswith(d) for d in directive_starts):
                count += 1
        return count

    def _handle_multi_speaker(
        self,
        utterances: list[Utterance],
        speakers: set[int],
    ) -> list[Utterance]:
        """Handle 3+ speakers."""
        # Still identify likely coach
        speaker_scores = self._calculate_speaker_scores(utterances)
        coach_speaker = max(speaker_scores, key=speaker_scores.get)

        participant_num = 1
        speaker_labels = {}
        for speaker in sorted(speakers):
            if speaker == coach_speaker:
                speaker_labels[speaker] = "coach"
            else:
                speaker_labels[speaker] = f"participant_{participant_num}"
                participant_num += 1

        for u in utterances:
            u.speaker_label = speaker_labels[u.speaker_number]
            u.speaker_confidence = speaker_scores.get(u.speaker_number, 0.5)

        return utterances
```

4.2 **Integrate diarization into transcription task**
```python
# In process_transcription task, after storing utterances:
diarization_service = DiarizationService()
labeled_utterances = diarization_service.assign_speaker_labels(utterances)
await self._update_utterance_labels(labeled_utterances)
```

4.3 **Add manual label override endpoint**
```python
@router.patch("/{job_id}/speakers")
async def update_speaker_labels(
    job_id: UUID,
    speaker_updates: list[SpeakerLabelUpdate],
    user: UserContext = Depends(get_current_user),
) -> dict:
    """
    Manually correct speaker labels.

    Body: [{"speaker_number": 0, "label": "coach"}, ...]
    """
    pass
```

4.4 **Create intent classification service** (`app/services/transcription/intent.py`)
```python
from app.services.llm.client import LLMClient

class IntentClassificationService:
    """Classify utterance intents using LLM."""

    INTENTS = [
        "question_open", "question_closed", "reflection", "advice",
        "encouragement", "challenge", "instruction", "acknowledgment",
        "concern", "commitment", "resistance", "update",
    ]

    INTENT_PROMPT = '''Classify the intent of this coaching utterance.

UTTERANCE: {text}
SPEAKER: {speaker_label}

Classify as one of: {intents}

Respond with just the classification word.'''

    def __init__(self):
        self.llm_client = LLMClient()

    async def classify_utterances(
        self,
        utterances: list[Utterance],
        batch_size: int = 10,
    ) -> list[Utterance]:
        """Classify intents for utterances in batches."""
        # Batch classification to reduce API calls
        # Use simple task model (Sonnet 4) for speed
        for batch in self._batch(utterances, batch_size):
            intents = await self._classify_batch(batch)
            for utt, intent in zip(batch, intents):
                utt.intent = intent
        return utterances

    async def _classify_batch(self, utterances: list[Utterance]) -> list[str]:
        """Classify a batch of utterances."""
        batch_prompt = self._build_batch_prompt(utterances)
        response = await self.llm_client.complete(
            task="intent_classification",  # Uses Sonnet 4 per model config
            messages=[{"role": "user", "content": batch_prompt}],
        )
        return self._parse_batch_response(response.content, len(utterances))
```

4.5 **Integrate intent classification into pipeline**
```python
# In process_transcription task, after diarization:
intent_service = IntentClassificationService()
# Only classify a sample if transcript is very long (cost control)
if len(labeled_utterances) > 100:
    sample = labeled_utterances[:50] + labeled_utterances[-50:]
    await intent_service.classify_utterances(sample)
else:
    await intent_service.classify_utterances(labeled_utterances)
await self._update_utterance_intents(labeled_utterances)
```

### Acceptance Criteria Coverage
- AC4: Speaker Diarization
- AC11: Error Handling (multi-speaker)
- Intent Classification (from spec Must Have #7)

### Verification
```bash
# Test diarization with 2-speaker audio
uv run pytest tests/test_diarization.py -v

# Test with 3-speaker audio
uv run pytest tests/test_diarization.py::test_multi_speaker -v
```

---

## Phase 5: Session Summarization

**Goal**: Implement LLM-based session summarization with structured extraction.

### Tasks

5.1 **Create summarization service** (`app/services/transcription/summarization.py`)
```python
from app.services.llm.client import LLMClient

class SummarizationService:
    """Generate session summaries using LLM (Opus 4.5 per spec)."""

    # Uses task="session_summary" which routes to Opus 4.5 per MODEL_CONFIG in Spec 0001

    SESSION_SUMMARY_PROMPT = '''You are analyzing a coaching session transcript.

TRANSCRIPT:
{transcript}

Generate a structured analysis in JSON format:

{{
  "executive_summary": "2-3 sentence overview of the session",
  "key_topics": ["topic1", "topic2", ...],
  "client_concerns": ["concern1", "concern2", ...],
  "coach_recommendations": ["rec1", "rec2", ...],
  "action_items": [
    {{"description": "...", "owner": "coach|client", "priority": "high|medium|low"}}
  ],
  "goals_discussed": ["goal1", "goal2", ...],
  "coaching_moments": [
    {{
      "type": "breakthrough|concern|goal_set|commitment|resistance",
      "timestamp_seconds": 123.4,
      "description": "...",
      "significance": "..."
    }}
  ],
  "session_type_detected": "nutrition|training|mindset|accountability|general",
  "client_sentiment": "positive|neutral|negative|mixed",
  "engagement_score": 0.85
}}

Focus on actionable insights. Be specific about action items and coaching moments.'''

    def __init__(self):
        self.llm_client = LLMClient()

    async def generate_summary(
        self,
        transcript: Transcript,
        utterances: list[Utterance],
    ) -> SessionSummary:
        """Generate session summary from transcript."""

        # Format transcript with speaker labels
        formatted_transcript = self._format_transcript(utterances)

        # Truncate if too long (keep first and last portions)
        if len(formatted_transcript) > 100000:
            formatted_transcript = self._truncate_transcript(formatted_transcript)

        prompt = self.SESSION_SUMMARY_PROMPT.format(
            transcript=formatted_transcript
        )

        response = await self.llm_client.complete(
            task="session_summary",
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse JSON response
        summary_data = self._parse_summary_response(response.content)

        return SessionSummary(
            transcript_id=transcript.id,
            **summary_data,
            llm_model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    def _format_transcript(self, utterances: list[Utterance]) -> str:
        """Format transcript with timestamps and speakers."""
        lines = []
        for u in utterances:
            timestamp = self._format_timestamp(u.start_time)
            lines.append(f"[{timestamp}] {u.speaker_label}: {u.text}")
        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _truncate_transcript(self, text: str, max_chars: int = 80000) -> str:
        """Keep first and last portions if too long."""
        if len(text) <= max_chars:
            return text

        half = max_chars // 2
        return (
            text[:half] +
            "\n\n[... transcript truncated for length ...]\n\n" +
            text[-half:]
        )

    def _parse_summary_response(self, content: str) -> dict:
        """Parse LLM JSON response with fallbacks."""
        import json

        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)
        except json.JSONDecodeError:
            # Return minimal summary on parse failure
            return {
                "executive_summary": "Summary generation failed - manual review needed",
                "key_topics": [],
                "client_concerns": [],
                "coach_recommendations": [],
                "action_items": [],
                "goals_discussed": [],
                "coaching_moments": [],
                "session_type_detected": "general",
                "client_sentiment": "neutral",
                "engagement_score": 0.5,
            }
```

5.2 **Integrate summarization into transcription task**
```python
# After diarization:
await self._update_status(job, "summarizing")

summarization_service = SummarizationService()
try:
    summary = await summarization_service.generate_summary(
        transcript=transcript,
        utterances=labeled_utterances,
    )
    summary.job_id = job.id
    await self._store_summary(summary)
    await self._update_status(job, "completed")
except Exception as e:
    logger.error("Summarization failed", job_id=str(job.id), error=str(e))
    # Mark as partial - transcript still available
    job.status = "partial"
    job.error_message = f"Summarization failed: {e}"
```

5.3 **Handle partial results**
```python
@router.get("/{job_id}/summary")
async def get_summary(
    job_id: UUID,
    user: UserContext = Depends(get_current_user),
) -> SessionSummaryResponse:
    """Get session summary. Returns 404 if summarization failed."""
    job = await get_job_or_404(job_id, user.coach_id, db)

    if job.status == "partial":
        raise HTTPException(
            status_code=404,
            detail="Summary not available - summarization failed. Transcript is still accessible."
        )

    summary = await db.query(SessionSummary).filter_by(job_id=job_id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    return SessionSummaryResponse.from_orm(summary)
```

### Acceptance Criteria Coverage
- AC6: Session Summarization

### Verification
```bash
# Test summarization
uv run pytest tests/test_summarization.py -v

# Test full flow
curl http://localhost:8000/api/v1/transcription/{job_id}/summary
```

---

## Phase 6: Webhooks, Testing & Documentation

**Goal**: Implement webhook notifications, comprehensive tests, and documentation.

### Tasks

6.1 **Implement webhook service** (`app/services/transcription/webhook.py`)
```python
import hmac
import hashlib
import httpx
from datetime import datetime
from uuid import uuid4

class WebhookService:
    """Send signed webhook notifications."""

    MAX_RETRIES = 5
    RETRY_DELAYS = [60, 300, 900, 1800, 3600]  # 1m, 5m, 15m, 30m, 1h

    def sign_payload(self, payload: dict, secret: str) -> str:
        """Generate HMAC-SHA256 signature."""
        message = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    async def send_webhook(
        self,
        url: str,
        payload: dict,
        secret: str,
    ) -> bool:
        """Send webhook with signature."""
        signature = self.sign_payload(payload, secret)
        timestamp = str(int(datetime.utcnow().timestamp()))
        webhook_id = str(uuid4())

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-ID": webhook_id,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=10.0,
            )
            return response.status_code < 400

    def build_completion_payload(self, job: TranscriptionJob) -> dict:
        """Build webhook payload for completion."""
        return {
            "event": "transcription.completed",
            "job_id": str(job.id),
            "client_id": str(job.client_id),
            "status": job.status,
            "transcript_id": str(job.transcript_id) if job.transcript_id else None,
            "summary_status": "available" if job.status == "completed" else "failed",
            "duration_seconds": float(job.audio_duration_seconds) if job.audio_duration_seconds else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }

    def build_failure_payload(self, job: TranscriptionJob) -> dict:
        """Build webhook payload for failure."""
        return {
            "event": "transcription.failed",
            "job_id": str(job.id),
            "client_id": str(job.client_id),
            "status": "failed",
            "error_message": job.error_message,
            "failed_at": datetime.utcnow().isoformat(),
        }
```

6.2 **Implement webhook Celery task**
```python
@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=60,
)
def send_webhook_notification(
    self,
    job_id: str,
    event_type: str,  # "completed" or "failed"
):
    """Send webhook with retries."""
    job = get_job_sync(job_id)
    if not job.webhook_url:
        return

    webhook_service = WebhookService()

    if event_type == "completed":
        payload = webhook_service.build_completion_payload(job)
    else:
        payload = webhook_service.build_failure_payload(job)

    try:
        success = asyncio.run(
            webhook_service.send_webhook(
                job.webhook_url,
                payload,
                job.webhook_secret,
            )
        )
        if success:
            job.webhook_sent_at = datetime.utcnow()
            save_job(job)
        else:
            raise WebhookDeliveryError("Webhook returned error status")
    except Exception as e:
        retry_delay = WebhookService.RETRY_DELAYS[min(self.request.retries, 4)]
        raise self.retry(countdown=retry_delay, exc=e)
```

6.3 **Create test fixtures**
- `tests/fixtures/sample_short.mp3` (5 min, 2 speakers)
- `tests/fixtures/sample_medium.mp3` (15 min, 2 speakers)
- `tests/fixtures/sample_mono.mp3` (5 min, 1 speaker)

6.4 **Write comprehensive tests**
- `tests/test_transcription_api.py` - API endpoint tests
- `tests/test_deepgram_client.py` - Deepgram client tests
- `tests/test_diarization.py` - Speaker labeling tests
- `tests/test_summarization.py` - LLM summarization tests
- `tests/test_webhook.py` - Webhook delivery tests
- `tests/test_transcription_security.py` - Auth/authorization tests

6.5 **Security tests**
```python
class TestTranscriptionSecurity:
    async def test_unauthorized_access_returns_401(self, client):
        response = await client.get("/api/v1/transcription/status/xxx")
        assert response.status_code == 401

    async def test_cross_coach_access_returns_404(self, client, other_coach_job):
        # Job belongs to coach B, request from coach A
        response = await client.get(
            f"/api/v1/transcription/status/{other_coach_job.id}",
            headers={"Authorization": f"Bearer {coach_a_token}"},
        )
        assert response.status_code == 404

    async def test_file_size_limit_enforced(self, client):
        large_file = b"x" * (501 * 1024 * 1024)  # 501 MB
        response = await client.post(
            "/api/v1/transcription/upload",
            files={"file": ("large.mp3", large_file)},
        )
        assert response.status_code == 413
```

6.6 **Update Makefile**
```makefile
# Transcription-specific commands
test-transcription:
	uv run pytest tests/test_*transcription*.py tests/test_*diarization*.py tests/test_*summarization*.py tests/test_*intent*.py -v

worker-transcription:
	celery -A app.workers.celery_app worker -l info -Q transcription -c 3
```

6.7 **Configure S3 lifecycle policy for audio deletion**
```json
// S3 bucket lifecycle rule (per spec AC12: 90-day retention)
{
  "Rules": [
    {
      "ID": "Delete transcription audio after 90 days",
      "Filter": {"Prefix": "transcriptions/"},
      "Status": "Enabled",
      "Expiration": {"Days": 90}
    }
  ]
}
```
Document S3 lifecycle setup in deployment docs.

6.7 **Update Docker Compose** for transcription worker
- Ensure DEEPGRAM_API_KEY in transcription worker environment
- Add ffmpeg to worker Docker image

### Acceptance Criteria Coverage
- AC8: Webhook Notifications
- AC9: Performance
- AC10: Security & Authorization
- AC11: Error Handling
- AC12: Data Retention

### Verification
```bash
# Run full test suite
make test-transcription

# Test webhook delivery
curl -X POST http://localhost:8000/api/v1/transcription/upload \
  -F "file=@tests/fixtures/sample_short.mp3" \
  -F "client_id=$CLIENT_ID" \
  -F "webhook_url=https://webhook.site/xxx"
# Verify webhook received at webhook.site
```

---

## Implementation Order Summary

```
Phase 1: Infrastructure ──────────────────► Deepgram client, audio utils, DB schema
    │
    ▼
Phase 2: API & Celery ────────────────────► Upload endpoint, status tracking, task
    │
    ▼
Phase 3: Transcription ───────────────────► Full Deepgram flow, transcript storage
    │
    ▼
Phase 4: Diarization ─────────────────────► Speaker identification, labeling
    │
    ▼
Phase 5: Summarization ───────────────────► LLM summary, coaching moments
    │
    ▼
Phase 6: Webhooks & Testing ──────────────► Notifications, tests, documentation
```

---

## Files to Create (Summary)

### Phase 1
- `app/services/transcription/__init__.py`
- `app/services/transcription/deepgram.py`
- `app/services/transcription/audio.py`
- `app/schemas/transcription.py`
- `scripts/migrations/0003_transcription_tables.sql`

### Phase 2
- `app/api/v1/transcription.py`
- `app/workers/tasks/transcription.py`

### Phase 3
- Updates to `deepgram.py` and `transcription.py` task

### Phase 4
- `app/services/transcription/diarization.py`
- `app/services/transcription/intent.py`
- `tests/test_intent_classification.py`

### Phase 5
- `app/services/transcription/summarization.py`

### Phase 6
- `app/services/transcription/webhook.py`
- `tests/test_transcription_api.py`
- `tests/test_deepgram_client.py`
- `tests/test_diarization.py`
- `tests/test_summarization.py`
- `tests/test_webhook.py`
- `tests/test_transcription_security.py`
- `tests/fixtures/sample_short.mp3`
- `tests/fixtures/sample_medium.mp3`
- `tests/fixtures/sample_mono.mp3`

---

## Test Audio Files Required

| File | Duration | Speakers | Purpose |
|------|----------|----------|---------|
| sample_short.mp3 | 5 min | 2 | Quick integration tests |
| sample_medium.mp3 | 15 min | 2 | Standard session test |
| sample_mono.mp3 | 5 min | 1 | Single speaker handling |
| sample_multi.mp3 | 10 min | 3+ | Multi-party handling |
| sample_poor_quality.mp3 | 5 min | 2 | Low quality audio |

**Note**: Builder should create synthetic test audio or request samples.

---

## Estimated Effort

| Phase | Complexity | Notes |
|-------|------------|-------|
| Phase 1 | Medium | Deepgram SDK, ffmpeg integration |
| Phase 2 | Medium | API + Celery task wiring |
| Phase 3 | Medium | Deepgram response handling |
| Phase 4 | High | Speaker identification heuristics |
| Phase 5 | High | LLM prompting, JSON parsing |
| Phase 6 | Medium | Testing, webhook delivery |

---

**Plan Status**: Ready for approval
**Author**: Architect
**Created**: 2025-01-27
