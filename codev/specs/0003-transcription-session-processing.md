# Spec 0003: Transcription & Session Processing

## Overview

**What**: Build an audio transcription and session processing pipeline that converts coaching session recordings into structured, searchable transcripts with speaker identification, intent classification, and AI-generated summaries.

**Why**: Coaches conduct live sessions (in-person, video calls, phone calls) with clients. Currently:
- Session notes are manual and inconsistent
- Key coaching moments are lost or forgotten
- No searchable history of client conversations
- Time-consuming to review past sessions

Automated transcription enables:
- Instant session transcripts with speaker labels
- AI-identified coaching moments and client concerns
- Searchable session history for better continuity
- Time savings: 30-60 min session → 2 min review

**Who**:
- Coaches recording client sessions
- Coaches reviewing session history
- System generating insights from session content

## Goals

### Must Have
1. Deepgram integration for audio transcription
2. Speaker diarization (distinguish coach vs client voices)
3. Audio file upload and storage (S3/R2)
4. Async transcription via Celery worker
5. Transcript storage with timestamps and speaker labels
6. Session summarization using LLM (Opus 4.5)
7. Intent classification for coaching moments
8. API endpoints for upload, status, and retrieval
9. Webhook notification on completion

### Should Have
- Real-time transcription via WebSocket (for live sessions)
- Key moment extraction (goals, concerns, action items)
- Sentiment analysis per speaker turn
- Session tagging (nutrition, training, mindset, etc.)
- Search across transcripts

### Won't Have (MVP)
- Live session recording (handled by external tools like Zoom)
- Video transcription (audio only)
- Multi-language transcription
- Custom vocabulary training
- Real-time coaching prompts during sessions

## Technical Context

### Audio Input Formats

| Source | Format | Typical Size | Duration |
|--------|--------|--------------|----------|
| Zoom recording | M4A, MP4 | 50-200 MB | 30-60 min |
| Phone recording | MP3, WAV | 20-100 MB | 15-45 min |
| Voice memo | M4A, AAC | 10-50 MB | 5-30 min |
| Uploaded file | Any audio | Up to 500 MB | Up to 2 hours |

### Deepgram Integration

Deepgram provides:
- Speech-to-text with 95%+ accuracy
- Speaker diarization (identify different speakers)
- Punctuation and formatting
- Word-level timestamps
- Multiple audio format support

```python
from deepgram import DeepgramClient, PrerecordedOptions

client = DeepgramClient(api_key)

options = PrerecordedOptions(
    model="nova-2",           # Latest model
    language="en",
    punctuate=True,
    diarize=True,             # Speaker separation
    utterances=True,          # Logical segments
    smart_format=True,        # Numbers, dates formatting
    paragraphs=True,          # Paragraph breaks
)

response = await client.listen.asyncrest.v1.transcribe_file(
    {"buffer": audio_bytes},
    options
)
```

### Integration with MVP

```
┌─────────────────────────────────────────────────────────────────┐
│                     Xenios MVP (Next.js)                        │
│                                                                 │
│  New endpoints to add:                                          │
│  POST /api/sessions/upload ──────────────────────────────────┼──┐
│  GET  /api/sessions/{id} ────────────────────────────────────┼──┤
│  GET  /api/sessions/{id}/transcript ─────────────────────────┼──┤
│                                                                 │
│  Existing tables to integrate:                                  │
│  - conversations (link sessions to conversations)               │
│  - clients (session belongs to client)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Xenios AI Backend (This Spec)                 │
│                                                                 │
│  POST /api/v1/transcription/upload                              │
│       → Upload audio to S3/R2                                   │
│       → Queue transcription job                                 │
│       → Return job_id                                           │
│                                                                 │
│  GET /api/v1/transcription/status/{job_id}                      │
│       → Return transcription status                             │
│                                                                 │
│  GET /api/v1/transcription/{job_id}/transcript                  │
│       → Return full transcript with speakers                    │
│                                                                 │
│  GET /api/v1/transcription/{job_id}/summary                     │
│       → Return AI-generated summary                             │
│                                                                 │
│  Celery Worker (transcription queue):                           │
│       → Download audio from S3                                  │
│       → Send to Deepgram                                        │
│       → Process diarization                                     │
│       → Generate summary with LLM                               │
│       → Store results                                           │
│       → Notify via webhook                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Implementation

### Data Models

#### Transcription Job

```python
class TranscriptionJob(BaseModel):
    id: UUID
    client_id: UUID
    coach_id: UUID
    conversation_id: UUID | None  # Link to existing conversation

    # Source audio
    audio_url: str
    audio_filename: str
    audio_format: str  # mp3, m4a, wav, etc.
    audio_size_bytes: int
    audio_duration_seconds: float | None

    # Processing status
    status: TranscriptionStatus  # pending, transcribing, summarizing, completed, failed
    started_at: datetime | None
    transcription_completed_at: datetime | None
    summary_completed_at: datetime | None
    completed_at: datetime | None

    # Results
    transcript_id: UUID | None  # Reference to transcript
    summary_id: UUID | None     # Reference to summary
    error_message: str | None

    # Metadata
    session_date: date | None   # When session occurred
    session_type: str | None    # in_person, video_call, phone_call
    session_title: str | None   # Optional title
    created_at: datetime
    updated_at: datetime

class TranscriptionStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"
```

#### Transcript

```python
class Transcript(BaseModel):
    id: UUID
    job_id: UUID

    # Full transcript
    full_text: str              # Complete transcript text
    word_count: int
    duration_seconds: float

    # Speaker-separated utterances
    utterances: list[Utterance]

    # Deepgram metadata
    deepgram_request_id: str
    model_used: str
    confidence_score: float

    created_at: datetime

class Utterance(BaseModel):
    """A single speaker turn in the conversation."""
    id: UUID
    transcript_id: UUID

    speaker: int                # Speaker number (0, 1, etc.)
    speaker_label: str | None   # "coach" or "client" (assigned after processing)

    text: str
    start_time: float           # Seconds from start
    end_time: float
    confidence: float

    # Word-level detail (optional, for highlighting)
    words: list[Word] | None

class Word(BaseModel):
    word: str
    start: float
    end: float
    confidence: float
    speaker: int
```

#### Session Summary

```python
class SessionSummary(BaseModel):
    id: UUID
    job_id: UUID
    transcript_id: UUID

    # AI-generated content
    executive_summary: str      # 2-3 sentence overview
    key_topics: list[str]       # Main topics discussed

    # Structured extraction
    client_concerns: list[str]  # Issues client raised
    coach_recommendations: list[str]  # Advice given
    action_items: list[ActionItem]
    goals_discussed: list[str]

    # Session analysis
    session_type_detected: str  # nutrition, training, mindset, general
    client_sentiment: str       # positive, neutral, negative, mixed
    engagement_score: float     # 0-1 based on client participation

    # Coaching moments
    coaching_moments: list[CoachingMoment]

    # Metadata
    llm_model: str
    prompt_tokens: int
    completion_tokens: int
    created_at: datetime

class ActionItem(BaseModel):
    description: str
    owner: str  # "coach" or "client"
    due_date: str | None
    priority: str  # high, medium, low

class CoachingMoment(BaseModel):
    """A significant moment in the coaching session."""
    type: str  # breakthrough, concern, goal_set, resistance, commitment
    timestamp: float  # Seconds into session
    description: str
    utterance_id: UUID  # Reference to specific utterance
    significance: str  # Why this moment matters
```

### Project Structure (Additions)

```
app/
├── services/
│   └── transcription/
│       ├── __init__.py
│       ├── deepgram.py         # Deepgram client wrapper
│       ├── diarization.py      # Speaker identification
│       ├── summarization.py    # LLM summarization
│       ├── intent.py           # Intent/moment classification
│       └── storage.py          # Audio file handling
│
├── workers/
│   └── tasks/
│       └── transcription.py    # Transcription Celery tasks
│
├── api/
│   └── v1/
│       └── transcription.py    # Transcription API endpoints
│
└── schemas/
    └── transcription.py        # Request/response schemas
```

### API Endpoints

```
POST /api/v1/transcription/upload
  Request:
    - file: audio file (multipart/form-data)
    - client_id: UUID (required)
    - session_date: date (optional)
    - session_type: string (optional)
    - session_title: string (optional)
    - conversation_id: UUID (optional, link to existing conversation)
    - webhook_url: string (optional)
  Response:
    - job_id: UUID
    - status: "pending"
    - estimated_duration: int (seconds)

GET /api/v1/transcription/status/{job_id}
  Response:
    - job_id: UUID
    - status: TranscriptionStatus
    - progress: float (0-100)
    - audio_duration: float (if known)
    - started_at: datetime
    - estimated_completion: datetime (if processing)
    - error_message: string (if failed)

GET /api/v1/transcription/{job_id}/transcript
  Response:
    - transcript_id: UUID
    - full_text: string
    - utterances: list[Utterance]
    - word_count: int
    - duration_seconds: float
    - confidence_score: float

GET /api/v1/transcription/{job_id}/summary
  Response:
    - summary_id: UUID
    - executive_summary: string
    - key_topics: list[string]
    - client_concerns: list[string]
    - coach_recommendations: list[string]
    - action_items: list[ActionItem]
    - coaching_moments: list[CoachingMoment]
    - client_sentiment: string
    - engagement_score: float

GET /api/v1/transcription/sessions
  Query params:
    - client_id: UUID (optional)
    - status: string (optional)
    - from_date: date (optional)
    - to_date: date (optional)
    - limit: int (default 20)
    - offset: int (default 0)
  Response:
    - sessions: list[TranscriptionJobSummary]
    - total: int

POST /api/v1/transcription/{job_id}/reprocess
  - Retry failed transcription or regenerate summary

DELETE /api/v1/transcription/{job_id}
  - Delete job and associated data
```

### Processing Pipeline

```
1. Upload Request
   │
   ├── Validate audio file (format, size)
   ├── Upload to S3/R2: sessions/{client_id}/{year}/{month}/{job_id}.{ext}
   ├── Create TranscriptionJob record (status: pending)
   ├── Queue Celery task
   └── Return job_id immediately

2. Celery Worker: process_transcription
   │
   ├── Update status: transcribing
   ├── Download audio from S3
   ├── Get audio duration (ffprobe)
   │
   ├── Send to Deepgram
   │   ├── model: nova-2
   │   ├── diarize: true
   │   ├── punctuate: true
   │   ├── utterances: true
   │   └── paragraphs: true
   │
   ├── Update status: diarizing
   ├── Process speaker diarization
   │   ├── Identify speakers (speaker 0, 1, etc.)
   │   ├── Assign labels based on heuristics:
   │   │   - First speaker often is coach (initiated session)
   │   │   - Speaking patterns (coach asks questions)
   │   │   - Word usage patterns
   │   └── Allow manual correction later
   │
   ├── Store Transcript record
   │
   ├── Update status: summarizing
   ├── Generate summary with LLM
   │   ├── Extract key topics
   │   ├── Identify client concerns
   │   ├── Extract action items
   │   ├── Identify coaching moments
   │   └── Generate executive summary
   │
   ├── Store SessionSummary record
   │
   ├── Update status: completed
   └── Send webhook notification

3. Webhook Notification
   │
   POST {webhook_url}
   Body:
   {
     "event": "transcription.completed",
     "job_id": "...",
     "client_id": "...",
     "status": "completed",
     "transcript_id": "...",
     "summary_id": "...",
     "duration_seconds": 1847,
     "word_count": 5432
   }
```

### Speaker Diarization Strategy

Deepgram returns speaker numbers (0, 1, 2...) but doesn't know who is coach vs client.

```python
class DiarizationService:
    """Assign speaker labels based on conversation patterns."""

    def assign_speaker_labels(
        self,
        utterances: list[Utterance],
        coach_id: UUID,
    ) -> list[Utterance]:
        """
        Heuristics for speaker identification:

        1. Question ratio: Coaches ask more questions
        2. Speaking time: Often roughly equal, but coach may lead
        3. First speaker: Usually coach (opens session)
        4. Vocabulary: Coach uses coaching terms
        5. Directive language: Coach gives instructions
        """

        speaker_stats = self._calculate_speaker_stats(utterances)

        # Score each speaker
        scores = {}
        for speaker_id, stats in speaker_stats.items():
            scores[speaker_id] = (
                stats['question_ratio'] * 0.3 +
                stats['directive_ratio'] * 0.2 +
                stats['first_speaker_bonus'] * 0.2 +
                stats['coaching_terms_ratio'] * 0.3
            )

        # Highest score is likely coach
        coach_speaker = max(scores, key=scores.get)

        # Assign labels
        for utterance in utterances:
            utterance.speaker_label = (
                "coach" if utterance.speaker == coach_speaker
                else "client"
            )

        return utterances

    COACHING_TERMS = [
        "goal", "progress", "plan", "tracking", "consistency",
        "nutrition", "training", "recovery", "sleep", "stress",
        "habit", "routine", "accountability", "check-in", "review"
    ]
```

### Summarization Prompts

```python
SESSION_SUMMARY_PROMPT = """You are analyzing a coaching session transcript between a health/fitness coach and their client.

TRANSCRIPT:
{transcript}

Generate a structured analysis with the following sections:

1. EXECUTIVE SUMMARY (2-3 sentences)
What was this session about? What was accomplished?

2. KEY TOPICS (list of 3-5 topics)
Main subjects discussed in the session.

3. CLIENT CONCERNS (list)
Issues, challenges, or worries the client expressed.

4. COACH RECOMMENDATIONS (list)
Specific advice or guidance the coach provided.

5. ACTION ITEMS (list with owner and priority)
Concrete next steps discussed. Format: "[owner] - [action] - [priority: high/medium/low]"

6. COACHING MOMENTS (list with timestamp)
Significant moments such as:
- Breakthroughs or realizations
- Concerns requiring follow-up
- Goals set or commitments made
- Resistance or pushback
Format: "[timestamp] - [type] - [description]"

7. SESSION ANALYSIS
- Session type: nutrition / training / mindset / accountability / general
- Client sentiment: positive / neutral / negative / mixed
- Client engagement: 0-100 score based on participation level

Respond in JSON format."""

INTENT_CLASSIFICATION_PROMPT = """Classify the intent of this coaching utterance.

UTTERANCE: {text}
SPEAKER: {speaker_label}
CONTEXT: {previous_utterances}

Classify as one of:
- question_open: Open-ended question seeking information
- question_closed: Yes/no or specific answer question
- reflection: Summarizing or reflecting back
- advice: Giving specific recommendation
- encouragement: Positive reinforcement
- challenge: Pushing client to think differently
- instruction: Direct guidance or direction
- acknowledgment: Confirming understanding
- concern: Expressing worry or issue
- commitment: Making a promise or commitment
- resistance: Pushing back or expressing doubt
- update: Sharing information or status

Respond with just the classification."""
```

### Database Schema

```sql
-- Transcription jobs
CREATE TABLE ai_backend.transcription_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL,
    coach_id UUID NOT NULL,
    conversation_id UUID,  -- Link to MVP conversations table

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

    -- Full transcript
    full_text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    duration_seconds DECIMAL(10,2) NOT NULL,

    -- Deepgram metadata
    deepgram_request_id VARCHAR(100),
    model_used VARCHAR(50),
    confidence_score DECIMAL(4,3),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_transcripts_job ON ai_backend.transcripts(job_id);

-- Utterances (speaker turns)
CREATE TABLE ai_backend.utterances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_id UUID NOT NULL REFERENCES ai_backend.transcripts(id) ON DELETE CASCADE,

    -- Speaker
    speaker_number INTEGER NOT NULL,
    speaker_label VARCHAR(20),  -- 'coach' or 'client'

    -- Content
    text TEXT NOT NULL,
    start_time DECIMAL(10,3) NOT NULL,
    end_time DECIMAL(10,3) NOT NULL,
    confidence DECIMAL(4,3),

    -- Classification
    intent VARCHAR(50),
    sentiment VARCHAR(20),

    -- Sequence
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

    -- Summary content
    executive_summary TEXT NOT NULL,
    key_topics JSONB NOT NULL DEFAULT '[]',
    client_concerns JSONB NOT NULL DEFAULT '[]',
    coach_recommendations JSONB NOT NULL DEFAULT '[]',
    action_items JSONB NOT NULL DEFAULT '[]',
    goals_discussed JSONB NOT NULL DEFAULT '[]',
    coaching_moments JSONB NOT NULL DEFAULT '[]',

    -- Analysis
    session_type_detected VARCHAR(50),
    client_sentiment VARCHAR(20),
    engagement_score DECIMAL(3,2),

    -- LLM metadata
    llm_model VARCHAR(100),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_session_summaries_job ON ai_backend.session_summaries(job_id);

-- Full-text search on transcripts
CREATE INDEX idx_transcripts_fulltext ON ai_backend.transcripts
    USING gin(to_tsvector('english', full_text));
```

## Security & Authorization

### Authentication
All endpoints require:
1. **X-API-Key header**: Backend-to-backend authentication (from MVP)
2. **Authorization header**: Bearer JWT from Supabase Auth

### Authorization Rules

| Endpoint | Who Can Access |
|----------|----------------|
| POST /upload | Coach (creates job for their client) |
| GET /status/{job_id} | Coach who owns the job |
| GET /{job_id}/transcript | Coach who owns the job |
| GET /{job_id}/summary | Coach who owns the job |
| GET /sessions | Coach (sees only their jobs) |
| DELETE /{job_id} | Coach who owns the job |

**Ownership enforcement**:
- `coach_id` extracted from JWT claims
- All queries filter by `coach_id = current_user.id`
- 404 returned for jobs owned by other coaches (not 403, to prevent enumeration)

### Webhook Security

Webhooks are signed to prevent spoofing:

```python
# Webhook signature generation
import hmac
import hashlib

def sign_webhook(payload: dict, secret: str) -> str:
    message = json.dumps(payload, sort_keys=True)
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"sha256={signature}"

# Headers sent with webhook
headers = {
    "X-Webhook-Signature": sign_webhook(payload, webhook_secret),
    "X-Webhook-Timestamp": str(int(time.time())),
    "X-Webhook-ID": str(uuid4()),  # For idempotency
}
```

**Webhook validation requirements**:
- URL must be HTTPS (except localhost for development)
- URL must respond to HEAD request with 2xx
- Signature verification documented for webhook consumers
- Timestamp checked (reject if > 5 minutes old)

## Data Retention & Privacy

### Retention Policy

| Data Type | Retention | Deletion Trigger |
|-----------|-----------|------------------|
| Audio files (S3) | 90 days | Auto-delete via S3 lifecycle |
| Transcripts | Indefinite | Manual deletion or client deletion |
| Summaries | Indefinite | Cascade from transcript |
| Job metadata | Indefinite | Cascade from transcript |

### Deletion Cascade

When a transcription job is deleted:
1. Audio file deleted from S3
2. Transcript record deleted (CASCADE)
3. Utterances deleted (CASCADE)
4. Summary deleted (CASCADE)
5. Job record deleted

### Privacy Considerations

- Audio files stored with randomized UUIDs (not client names)
- No PII in S3 object keys
- Transcripts contain PII (client health discussions) - encrypted at rest
- Logs sanitized: no transcript content in logs
- Deepgram data processing agreement required (HIPAA compliance)

## Error Handling

### File Size & Duration Limits

| Limit | Value | Behavior |
|-------|-------|----------|
| Max file size | 500 MB | Reject at upload with 413 |
| Max duration | 2 hours | Reject after duration check with 400 |
| Min duration | 10 seconds | Reject with 400 |

### Failure Modes

| Failure | Retry | Behavior |
|---------|-------|----------|
| S3 upload fails | 3x | Mark job failed, return error |
| Deepgram timeout | 2x | Exponential backoff (30s, 60s) |
| Deepgram API error | 2x | Log error, mark failed |
| LLM summarization fails | 2x | Mark summary_failed, transcript still available |
| Webhook delivery fails | 5x | Exponential backoff (1m, 5m, 15m, 30m, 1h) |

### Partial Results

If summarization fails but transcription succeeds:
- Job status: `partial`
- Transcript available via API
- Summary endpoint returns 404
- Webhook indicates `"summary_status": "failed"`

### Multi-Speaker Handling

| Scenario | Behavior |
|----------|----------|
| 2 speakers | Label as coach/client based on heuristics |
| 3+ speakers | Label speaker 0 as coach, others as "participant_N" |
| 1 speaker | Label as "speaker" (no coach/client distinction) |
| Unclear speakers | Low confidence score, "unknown" labels |

Speaker confidence threshold: 0.7
- Above 0.7: Assign coach/client label
- Below 0.7: Label as "speaker_N" with note for manual review

## Acceptance Criteria

### AC1: Deepgram Integration
- [ ] Deepgram client configured with API key
- [ ] Can transcribe audio files (MP3, M4A, WAV, MP4)
- [ ] Receives word-level timestamps
- [ ] Receives speaker diarization data
- [ ] Handles Deepgram API errors gracefully
- [ ] Respects Deepgram rate limits

### AC2: Audio Upload
- [ ] POST /upload accepts audio files up to 500MB
- [ ] Supports MP3, M4A, WAV, MP4, AAC, OGG formats
- [ ] Uploads to S3/R2 with structured path
- [ ] Returns job_id within 3 seconds
- [ ] Validates file format before processing

### AC3: Transcription Processing
- [ ] Celery task processes audio asynchronously
- [ ] Status updates visible via API
- [ ] Transcription completes within 2x audio duration
- [ ] Word-level timestamps available
- [ ] Confidence scores recorded

### AC4: Speaker Diarization
- [ ] Distinguishes multiple speakers
- [ ] Assigns "coach" and "client" labels
- [ ] Labels based on conversation heuristics
- [ ] Handles 2-speaker conversations accurately
- [ ] Gracefully handles multi-party calls

### AC5: Transcript Storage
- [ ] Full transcript stored in database
- [ ] Individual utterances stored with timestamps
- [ ] Speaker labels stored per utterance
- [ ] Full-text search enabled on transcripts
- [ ] Can retrieve transcript by job_id

### AC6: Session Summarization
- [ ] LLM generates executive summary
- [ ] Extracts key topics (3-5 per session)
- [ ] Identifies client concerns
- [ ] Extracts action items with owners
- [ ] Identifies significant coaching moments
- [ ] Detects session type
- [ ] Assesses client sentiment

### AC7: API Completeness
- [ ] All endpoints documented with OpenAPI
- [ ] Proper authentication required
- [ ] Pagination on list endpoints
- [ ] Filtering by client, date, status
- [ ] Appropriate error responses

### AC8: Webhook Notifications
- [ ] Webhook sent on completion
- [ ] Webhook sent on failure
- [ ] Webhook payload includes job details
- [ ] Retry logic for failed webhooks
- [ ] Webhook URL validated on job creation

### AC9: Performance
- [ ] 30-minute audio transcribed in < 5 minutes
- [ ] Summary generated in < 30 seconds
- [ ] Status endpoint responds in < 100ms
- [ ] Transcript retrieval in < 500ms

### AC10: Security & Authorization
- [ ] Endpoints require valid API key and JWT
- [ ] Coach can only access their own jobs
- [ ] 404 returned for other coaches' jobs (not 403)
- [ ] Webhook signatures validated
- [ ] HTTPS required for webhook URLs (except localhost)

### AC11: Error Handling
- [ ] Files > 500MB rejected with 413
- [ ] Audio > 2 hours rejected with 400
- [ ] Deepgram failures retried with backoff
- [ ] Partial results available if summary fails
- [ ] 3+ speakers labeled correctly

### AC12: Data Retention
- [ ] Audio files deleted after 90 days (S3 lifecycle)
- [ ] DELETE endpoint removes all associated data
- [ ] No PII in logs or S3 keys

## Test Plan

### Unit Tests
- Deepgram client wrapper
- Speaker diarization logic
- Summarization prompt formatting
- Audio format validation
- Utterance parsing

### Integration Tests
- Full upload → transcription → summary flow
- API endpoint tests
- Celery task execution
- S3 upload/download
- Database storage

### Test Audio Files
- Short session (5 min, 2 speakers)
- Medium session (30 min, 2 speakers)
- Long session (60 min, 2 speakers)
- Poor audio quality sample
- Single speaker (monologue)

### Manual Testing
- Upload real coaching session recording
- Verify speaker labels accuracy
- Review summary quality
- Test webhook delivery

### Security Tests
- Unauthorized access returns 401
- Cross-coach access returns 404
- Invalid webhook signature rejected
- File size limits enforced
- Audio duration limits enforced
- Webhook HTTPS requirement enforced

## Dependencies

- **Spec 0001**: AI Backend Foundation (Celery, Redis, LLM client, S3 config)
- **Spec 0002**: Document Extraction (S3 storage service reuse)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Deepgram cost overruns | Medium | Medium | Monitor usage; implement quotas per coach |
| Poor diarization accuracy | Medium | Medium | Manual correction UI; confidence thresholds |
| Long audio processing time | Low | Low | Progress updates; background processing |
| LLM hallucination in summary | Medium | Medium | Ground in transcript; confidence scoring |
| Audio quality issues | Medium | Medium | Pre-processing; quality warnings |
| Speaker misidentification | Medium | High | Confidence scores; manual override |

## Performance Requirements

| Operation | Target | Maximum |
|-----------|--------|---------|
| Audio upload (100MB) | 10 seconds | 30 seconds |
| Transcription (30 min audio) | 3 minutes | 5 minutes |
| Summarization | 15 seconds | 30 seconds |
| End-to-end (30 min session) | 4 minutes | 6 minutes |
| Status check | 50ms | 100ms |
| Transcript retrieval | 200ms | 500ms |

## Cost Considerations

### Deepgram Pricing (Nova-2)
- $0.0043 per minute of audio
- 30-minute session = ~$0.13
- 100 sessions/month = ~$13

### LLM Summarization (Opus 4.5)
- ~5000 tokens per summary
- ~$0.075 per summary
- 100 sessions/month = ~$7.50

### Storage (S3/R2)
- 100MB average per session
- 100 sessions = 10GB
- R2: Free (10GB included)

**Total estimated: ~$20/month for 100 sessions**

## Open Questions

1. **Speaker confirmation UI**: Should coaches confirm/correct speaker labels?
2. **Transcript editing**: Allow coaches to edit transcripts for errors?
3. **Real-time option**: Priority for live transcription in future?

*Note: Retention policy resolved - 90 days for audio, indefinite for transcripts with manual deletion option.*

## Future Considerations

- Real-time transcription via WebSocket
- Custom vocabulary for coaching terms
- Speaker voice profiles (recognize returning speakers)
- Integration with Zoom/Google Meet APIs
- Sentiment trends across sessions
- Coaching effectiveness metrics
- Multi-language support

---

**Spec Status**: Ready for review
**Author**: Architect
**Created**: 2025-01-27
