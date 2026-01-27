"""Deepgram client wrapper for audio transcription."""

from dataclasses import dataclass, field
from typing import Any

from deepgram import DeepgramClient, PrerecordedOptions
from deepgram.clients.prerecorded import PrerecordedResponse

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DeepgramError(Exception):
    """Base exception for Deepgram errors."""

    pass


class DeepgramTimeoutError(DeepgramError):
    """Raised when Deepgram request times out."""

    pass


class DeepgramRateLimitError(DeepgramError):
    """Raised when Deepgram rate limit is hit."""

    pass


@dataclass
class DeepgramWord:
    """A single word with timing information."""

    word: str
    start: float
    end: float
    confidence: float
    speaker: int | None = None


@dataclass
class DeepgramUtterance:
    """A single speaker utterance (turn) in the conversation."""

    speaker: int
    text: str
    start: float
    end: float
    confidence: float
    words: list[DeepgramWord] = field(default_factory=list)


@dataclass
class TranscriptionResponse:
    """Parsed response from Deepgram transcription."""

    full_text: str
    word_count: int
    duration_seconds: float
    request_id: str
    model: str
    confidence: float
    utterances: list[DeepgramUtterance]
    words: list[DeepgramWord]


class DeepgramService:
    """
    Wrapper for Deepgram transcription API.

    Provides async transcription with diarization and word-level timing.
    """

    def __init__(self) -> None:
        """Initialize the Deepgram service."""
        self._settings = get_settings()
        self._client: DeepgramClient | None = None

    @property
    def client(self) -> DeepgramClient:
        """Get or create the Deepgram client."""
        if self._client is None:
            if not self._settings.deepgram_api_key:
                raise DeepgramError("Deepgram API key not configured")
            self._client = DeepgramClient(self._settings.deepgram_api_key)
        return self._client

    async def transcribe_file(
        self,
        file_path: str,
        language: str = "en",
        diarize: bool = True,
        custom_options: dict[str, Any] | None = None,
    ) -> TranscriptionResponse:
        """
        Transcribe an audio file using Deepgram.

        Args:
            file_path: Path to the audio file
            language: Language code (default: "en")
            diarize: Enable speaker diarization (default: True)
            custom_options: Additional Deepgram options to override defaults

        Returns:
            TranscriptionResponse with full transcript and utterances

        Raises:
            DeepgramError: On API errors
            DeepgramTimeoutError: On timeout
            DeepgramRateLimitError: On rate limit
        """
        options = PrerecordedOptions(
            model="nova-2",
            language=language,
            punctuate=True,
            diarize=diarize,
            utterances=True,
            smart_format=True,
            paragraphs=True,
        )

        # Apply custom options
        if custom_options:
            for key, value in custom_options.items():
                if hasattr(options, key):
                    setattr(options, key, value)

        logger.info(
            "Starting Deepgram transcription",
            file_path=file_path,
            language=language,
            diarize=diarize,
        )

        try:
            # Read file and send to Deepgram
            with open(file_path, "rb") as audio_file:
                source = {"buffer": audio_file.read()}
                response = await self.client.listen.asyncrest.v1.transcribe_file(
                    source,
                    options,
                )

            return self._parse_response(response)

        except TimeoutError as e:
            logger.error("Deepgram request timed out", error=str(e))
            raise DeepgramTimeoutError(f"Transcription timed out: {e}") from e
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning("Deepgram rate limit hit", error=str(e))
                raise DeepgramRateLimitError(f"Rate limit exceeded: {e}") from e
            logger.error("Deepgram transcription failed", error=str(e))
            raise DeepgramError(f"Transcription failed: {e}") from e

    async def transcribe_url(
        self,
        audio_url: str,
        language: str = "en",
        diarize: bool = True,
        custom_options: dict[str, Any] | None = None,
    ) -> TranscriptionResponse:
        """
        Transcribe audio from a URL using Deepgram.

        Args:
            audio_url: URL to the audio file
            language: Language code (default: "en")
            diarize: Enable speaker diarization (default: True)
            custom_options: Additional Deepgram options

        Returns:
            TranscriptionResponse with full transcript and utterances
        """
        options = PrerecordedOptions(
            model="nova-2",
            language=language,
            punctuate=True,
            diarize=diarize,
            utterances=True,
            smart_format=True,
            paragraphs=True,
        )

        if custom_options:
            for key, value in custom_options.items():
                if hasattr(options, key):
                    setattr(options, key, value)

        logger.info(
            "Starting Deepgram transcription from URL",
            audio_url=audio_url[:50] + "..." if len(audio_url) > 50 else audio_url,
            language=language,
        )

        try:
            source = {"url": audio_url}
            response = await self.client.listen.asyncrest.v1.transcribe_url(
                source,
                options,
            )

            return self._parse_response(response)

        except TimeoutError as e:
            logger.error("Deepgram request timed out", error=str(e))
            raise DeepgramTimeoutError(f"Transcription timed out: {e}") from e
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning("Deepgram rate limit hit", error=str(e))
                raise DeepgramRateLimitError(f"Rate limit exceeded: {e}") from e
            logger.error("Deepgram transcription failed", error=str(e))
            raise DeepgramError(f"Transcription failed: {e}") from e

    def _parse_response(self, response: PrerecordedResponse) -> TranscriptionResponse:
        """Parse Deepgram response into our data model."""
        results = response.results

        # Get channels - typically just one for most audio
        channels = results.channels
        if not channels:
            raise DeepgramError("No channels in transcription response")

        # Use first channel
        channel = channels[0]
        alternatives = channel.alternatives
        if not alternatives:
            raise DeepgramError("No alternatives in transcription response")

        # Use first alternative (highest confidence)
        alt = alternatives[0]

        # Extract full transcript
        full_text = alt.transcript or ""
        word_count = len(full_text.split())

        # Calculate overall confidence
        confidence = alt.confidence if hasattr(alt, "confidence") else 0.0

        # Parse words
        words: list[DeepgramWord] = []
        if hasattr(alt, "words") and alt.words:
            for w in alt.words:
                words.append(
                    DeepgramWord(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=w.confidence if hasattr(w, "confidence") else 0.0,
                        speaker=w.speaker if hasattr(w, "speaker") else None,
                    )
                )

        # Parse utterances (for diarization)
        utterances: list[DeepgramUtterance] = []
        if hasattr(results, "utterances") and results.utterances:
            for utt in results.utterances:
                # Get words for this utterance
                utt_words = [
                    w for w in words if utt.start <= w.start <= utt.end
                ]
                utterances.append(
                    DeepgramUtterance(
                        speaker=utt.speaker if hasattr(utt, "speaker") else 0,
                        text=utt.transcript,
                        start=utt.start,
                        end=utt.end,
                        confidence=utt.confidence if hasattr(utt, "confidence") else 0.0,
                        words=utt_words,
                    )
                )

        # Get duration from metadata
        duration = 0.0
        if hasattr(response, "metadata") and response.metadata:
            duration = response.metadata.duration or 0.0

        # Get request ID
        request_id = ""
        if hasattr(response, "metadata") and response.metadata:
            request_id = response.metadata.request_id or ""

        # Get model info
        model = "nova-2"
        if hasattr(response, "metadata") and response.metadata:
            if hasattr(response.metadata, "model_info"):
                model = str(response.metadata.model_info)

        logger.info(
            "Transcription parsed",
            duration_seconds=duration,
            word_count=word_count,
            utterance_count=len(utterances),
            confidence=confidence,
        )

        return TranscriptionResponse(
            full_text=full_text,
            word_count=word_count,
            duration_seconds=duration,
            request_id=request_id,
            model=model,
            confidence=confidence,
            utterances=utterances,
            words=words,
        )


# Global service instance
_deepgram_service: DeepgramService | None = None


def get_deepgram_service() -> DeepgramService:
    """Get the global Deepgram service instance."""
    global _deepgram_service
    if _deepgram_service is None:
        _deepgram_service = DeepgramService()
    return _deepgram_service
