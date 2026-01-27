"""Session analytics service for computing per-session metrics."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.analytics import SessionAnalytics
from app.services.analytics.coaching_style import CoachingStyleAnalyzer
from app.services.analytics.engagement import (
    calculate_engagement_score,
    calculate_response_elaboration_score,
    count_words_in_utterances,
)
from app.services.analytics.sentiment import (
    aggregate_sentiment,
    calculate_sentiment_variance,
    extract_sentiment_scores,
)
from app.services.analytics.talk_time import TalkTimeAnalyzer

logger = get_logger(__name__)


class SessionAnalyticsError(Exception):
    """Raised when session analytics computation fails."""

    pass


class SessionAnalyticsService:
    """
    Compute analytics for a coaching session.

    Analyzes:
    - Talk-time metrics (per-speaker speaking time)
    - Coaching style (questions vs statements)
    - Engagement score (composite metric)
    - Sentiment analysis

    Note: Cue detection is handled separately in Phase 3.
    """

    MODEL_VERSION = "v1.0.0"

    def __init__(self, db: AsyncSession):
        """Initialize the session analytics service."""
        self.db = db
        self.talk_time_analyzer = TalkTimeAnalyzer()
        self.coaching_style_analyzer = CoachingStyleAnalyzer()

    async def compute_for_job(
        self,
        job_id: UUID,
    ) -> tuple[SessionAnalytics, list[dict[str, Any]]]:
        """
        Compute all session analytics for a completed transcription job.

        Args:
            job_id: UUID of the transcription job

        Returns:
            Tuple of (SessionAnalytics model, utterances list)

        Raises:
            SessionAnalyticsError: If computation fails
        """
        logger.info("Computing session analytics", job_id=str(job_id))

        try:
            # 1. Fetch job details
            job = await self._fetch_job(job_id)
            if not job:
                raise SessionAnalyticsError(f"Job not found: {job_id}")

            # 2. Fetch transcript and utterances
            transcript, utterances = await self._fetch_transcript_and_utterances(job_id)
            if not transcript:
                raise SessionAnalyticsError(f"Transcript not found for job: {job_id}")

            # 3. Convert utterances to dicts for processing
            utterance_dicts = [self._utterance_to_dict(u) for u in utterances]

            # 4. Compute metrics
            analytics = await self._compute_metrics(
                job_id=job_id,
                job=job,
                transcript=transcript,
                utterances=utterance_dicts,
            )

            # 5. Store analytics
            self.db.add(analytics)
            await self.db.flush()

            logger.info(
                "Session analytics computed",
                job_id=str(job_id),
                engagement_score=float(analytics.engagement_score) if analytics.engagement_score else None,
                total_turns=analytics.total_turns,
            )

            return analytics, utterance_dicts

        except SessionAnalyticsError:
            raise
        except Exception as e:
            logger.error("Session analytics computation failed", job_id=str(job_id), error=str(e))
            raise SessionAnalyticsError(f"Analytics computation failed: {e}") from e

    async def compute_basic_metrics(
        self,
        job_id: UUID,
    ) -> tuple[SessionAnalytics, list[dict[str, Any]]]:
        """
        Compute basic metrics without cues (for use before cue detection).

        This is called first, then cue detection updates the cue counts.

        Returns:
            Tuple of (SessionAnalytics model not yet committed, utterances list)
        """
        return await self.compute_for_job(job_id)

    async def _fetch_job(self, job_id: UUID) -> dict[str, Any] | None:
        """Fetch transcription job details."""
        result = await self.db.execute(
            text("""
                SELECT id, client_id, coach_id, session_date, status
                FROM ai_backend.transcription_jobs
                WHERE id = :job_id
            """),
            {"job_id": str(job_id)},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def _fetch_transcript_and_utterances(
        self,
        job_id: UUID,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Fetch transcript and utterances for a job."""
        # Fetch transcript
        transcript_result = await self.db.execute(
            text("""
                SELECT id, full_text, word_count, duration_seconds, confidence_score
                FROM ai_backend.transcripts
                WHERE job_id = :job_id
            """),
            {"job_id": str(job_id)},
        )
        transcript_row = transcript_result.mappings().first()
        transcript = dict(transcript_row) if transcript_row else None

        if not transcript:
            return None, []

        # Fetch utterances
        utterances_result = await self.db.execute(
            text("""
                SELECT id, speaker_label, speaker_confidence, text,
                       start_time, end_time, confidence, intent, sentiment, sequence_number
                FROM ai_backend.utterances
                WHERE transcript_id = :transcript_id
                ORDER BY sequence_number
            """),
            {"transcript_id": str(transcript["id"])},
        )
        utterances = [dict(row) for row in utterances_result.mappings()]

        return transcript, utterances

    def _utterance_to_dict(self, utterance: dict[str, Any]) -> dict[str, Any]:
        """Ensure utterance is a dict with expected fields."""
        return {
            "id": utterance.get("id"),
            "speaker_label": utterance.get("speaker_label", ""),
            "speaker_confidence": utterance.get("speaker_confidence"),
            "text": utterance.get("text", ""),
            "start_time": float(utterance.get("start_time", 0)),
            "end_time": float(utterance.get("end_time", 0)),
            "confidence": utterance.get("confidence"),
            "intent": utterance.get("intent"),
            "sentiment": utterance.get("sentiment"),
            "sequence_number": utterance.get("sequence_number", 0),
        }

    async def _compute_metrics(
        self,
        job_id: UUID,
        job: dict[str, Any],
        transcript: dict[str, Any],
        utterances: list[dict[str, Any]],
    ) -> SessionAnalytics:
        """Compute all metrics and create SessionAnalytics model."""
        # Separate by speaker
        coach_utterances = [
            u for u in utterances
            if u.get("speaker_label", "").lower() == "coach"
        ]
        client_utterances = [
            u for u in utterances
            if u.get("speaker_label", "").lower() == "client"
        ]

        # 1. Talk-time analysis
        talk_time = self.talk_time_analyzer.compute(utterances)

        # 2. Coaching style analysis
        coaching_style = self.coaching_style_analyzer.compute(coach_utterances)

        # 3. Sentiment analysis
        client_sentiment = aggregate_sentiment(utterances, "client")
        coach_sentiment = aggregate_sentiment(utterances, "coach")
        client_sentiment_scores = extract_sentiment_scores(utterances, "client")
        sentiment_variance = calculate_sentiment_variance(client_sentiment_scores)

        # 4. Response elaboration
        response_elaboration = calculate_response_elaboration_score(client_utterances)

        # 5. Engagement score
        # Note: cue counts are 0 initially, will be updated after cue detection
        total_client_words = count_words_in_utterances(client_utterances)
        duration_minutes = talk_time.total_duration_seconds / 60

        engagement = calculate_engagement_score(
            client_talk_percentage=talk_time.client_talk_percentage,
            duration_minutes=duration_minutes,
            client_turns=talk_time.client_turns,
            total_client_words=total_client_words,
            sentiment_score=client_sentiment,
            commitment_cue_count=0,  # Updated after cue detection
            resistance_cue_count=0,
        )

        # 6. Compute quality warnings
        quality_warnings = self._compute_quality_warnings(
            utterances=utterances,
            session_duration=talk_time.total_duration_seconds,
            transcript=transcript,
        )

        # Create analytics model
        session_date = job.get("session_date")
        if session_date is None:
            # Fallback to today if session_date not set
            session_date = datetime.now(timezone.utc).date()

        return SessionAnalytics(
            job_id=job_id,
            client_id=job["client_id"],
            coach_id=job["coach_id"],
            session_date=session_date,
            # Talk-time
            total_duration_seconds=talk_time.total_duration_seconds,
            coach_talk_time_seconds=talk_time.coach_talk_time_seconds,
            client_talk_time_seconds=talk_time.client_talk_time_seconds,
            silence_time_seconds=talk_time.silence_time_seconds,
            coach_talk_percentage=talk_time.coach_talk_percentage,
            client_talk_percentage=talk_time.client_talk_percentage,
            # Turn-taking
            total_turns=talk_time.total_turns,
            coach_turns=talk_time.coach_turns,
            client_turns=talk_time.client_turns,
            average_turn_duration_coach=talk_time.average_turn_duration_coach,
            average_turn_duration_client=talk_time.average_turn_duration_client,
            interruption_count=talk_time.interruption_count,
            # Coaching style
            coach_question_count=coaching_style.coach_question_count,
            coach_statement_count=coaching_style.coach_statement_count,
            question_to_statement_ratio=coaching_style.question_to_statement_ratio,
            open_question_count=coaching_style.open_question_count,
            closed_question_count=coaching_style.closed_question_count,
            # Cue counts (initialized to 0, updated after cue detection)
            cue_resistance_count=0,
            cue_commitment_count=0,
            cue_breakthrough_count=0,
            cue_concern_count=0,
            cue_deflection_count=0,
            cue_enthusiasm_count=0,
            cue_doubt_count=0,
            cue_agreement_count=0,
            cue_goal_setting_count=0,
            # Sentiment
            client_sentiment_score=client_sentiment,
            coach_sentiment_score=coach_sentiment,
            sentiment_variance=sentiment_variance,
            # Engagement
            engagement_score=engagement,
            response_elaboration_score=response_elaboration,
            # Quality
            quality_warning=len(quality_warnings) > 0,
            quality_warnings=quality_warnings,
            # Metadata
            computed_at=datetime.now(timezone.utc),
            model_version=self.MODEL_VERSION,
        )

    def _compute_quality_warnings(
        self,
        utterances: list[dict[str, Any]],
        session_duration: float,
        transcript: dict[str, Any],
    ) -> list[str]:
        """Compute quality warnings for the session."""
        warnings = []

        # Short session (< 2 minutes)
        if session_duration < 120:
            warnings.append("short_session")

        # Low confidence diarization (average speaker_confidence < 0.7)
        confidences = [
            u.get("speaker_confidence")
            for u in utterances
            if u.get("speaker_confidence") is not None
        ]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence < 0.7:
                warnings.append("low_confidence_diarization")

        # Missing sentiment (< 50% of client utterances have sentiment)
        client_utterances = [
            u for u in utterances
            if u.get("speaker_label", "").lower() == "client"
        ]
        if client_utterances:
            with_sentiment = [u for u in client_utterances if u.get("sentiment")]
            if len(with_sentiment) < len(client_utterances) * 0.5:
                warnings.append("missing_sentiment")

        # Low transcript confidence
        transcript_confidence = transcript.get("confidence_score")
        if transcript_confidence is not None and float(transcript_confidence) < 0.7:
            warnings.append("low_transcript_confidence")

        return warnings

    async def get_by_job_id(self, job_id: UUID) -> SessionAnalytics | None:
        """Fetch existing session analytics by job ID."""
        result = await self.db.execute(
            select(SessionAnalytics).where(SessionAnalytics.job_id == job_id)
        )
        return result.scalar_one_or_none()

    async def update_cue_counts(
        self,
        analytics: SessionAnalytics,
        cue_counts: dict[str, int],
    ) -> SessionAnalytics:
        """
        Update cue counts after cue detection.

        Args:
            analytics: SessionAnalytics model to update
            cue_counts: Dict mapping cue_type to count

        Returns:
            Updated SessionAnalytics
        """
        analytics.cue_resistance_count = cue_counts.get("resistance", 0)
        analytics.cue_commitment_count = cue_counts.get("commitment", 0)
        analytics.cue_breakthrough_count = cue_counts.get("breakthrough", 0)
        analytics.cue_concern_count = cue_counts.get("concern", 0)
        analytics.cue_deflection_count = cue_counts.get("deflection", 0)
        analytics.cue_enthusiasm_count = cue_counts.get("enthusiasm", 0)
        analytics.cue_doubt_count = cue_counts.get("doubt", 0)
        analytics.cue_agreement_count = cue_counts.get("agreement", 0)
        analytics.cue_goal_setting_count = cue_counts.get("goal_setting", 0)

        # Recalculate engagement with cue data
        commitment = analytics.cue_commitment_count
        resistance = analytics.cue_resistance_count

        total_client_words = 0
        # Fetch client word count from utterances (simplified - use cached value in production)
        result = await self.db.execute(
            text("""
                SELECT SUM(array_length(string_to_array(u.text, ' '), 1)) as word_count
                FROM ai_backend.utterances u
                JOIN ai_backend.transcripts t ON u.transcript_id = t.id
                WHERE t.job_id = :job_id AND u.speaker_label = 'client'
            """),
            {"job_id": str(analytics.job_id)},
        )
        row = result.first()
        if row and row[0]:
            total_client_words = int(row[0])

        duration_minutes = float(analytics.total_duration_seconds) / 60

        analytics.engagement_score = calculate_engagement_score(
            client_talk_percentage=float(analytics.client_talk_percentage),
            duration_minutes=duration_minutes,
            client_turns=analytics.client_turns,
            total_client_words=total_client_words,
            sentiment_score=float(analytics.client_sentiment_score or 0),
            commitment_cue_count=commitment,
            resistance_cue_count=resistance,
        )

        await self.db.flush()
        return analytics


# Global service instance
_session_analytics_service: SessionAnalyticsService | None = None


def get_session_analytics_service(db: AsyncSession) -> SessionAnalyticsService:
    """Get a session analytics service instance."""
    return SessionAnalyticsService(db)
