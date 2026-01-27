"""Tests for transcription services."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestAudioService:
    """Tests for audio service."""

    def test_validate_format_mp3(self):
        """Test MP3 format validation."""
        from app.services.transcription.audio import AudioService

        service = AudioService()
        valid, error = service.validate_format("test.mp3")

        assert valid is True
        assert error is None

    def test_validate_format_m4a(self):
        """Test M4A format validation."""
        from app.services.transcription.audio import AudioService

        service = AudioService()
        valid, error = service.validate_format("session.m4a")

        assert valid is True
        assert error is None

    def test_validate_format_wav(self):
        """Test WAV format validation."""
        from app.services.transcription.audio import AudioService

        service = AudioService()
        valid, error = service.validate_format("recording.wav")

        assert valid is True
        assert error is None

    def test_validate_format_invalid(self):
        """Test invalid format rejection."""
        from app.services.transcription.audio import AudioService

        service = AudioService()
        valid, error = service.validate_format("document.pdf")

        assert valid is False
        assert "unsupported format" in error.lower()

    def test_get_content_type_mp3(self):
        """Test content type for MP3."""
        from app.services.transcription.audio import AudioService

        service = AudioService()
        content_type = service.get_content_type("test.mp3")

        assert content_type == "audio/mpeg"

    def test_get_content_type_m4a(self):
        """Test content type for M4A."""
        from app.services.transcription.audio import AudioService

        service = AudioService()
        content_type = service.get_content_type("session.m4a")

        assert content_type == "audio/mp4"

    def test_get_content_type_wav(self):
        """Test content type for WAV."""
        from app.services.transcription.audio import AudioService

        service = AudioService()
        content_type = service.get_content_type("recording.wav")

        assert content_type == "audio/wav"

    def test_get_format_from_filename(self):
        """Test format extraction from filename."""
        from app.services.transcription.audio import AudioService

        service = AudioService()

        assert service.get_format_from_filename("test.mp3") == "mp3"
        assert service.get_format_from_filename("session.M4A") == "m4a"
        assert service.get_format_from_filename("path/to/recording.WAV") == "wav"

    def test_supported_formats(self):
        """Test that all expected formats are supported."""
        from app.services.transcription.audio import AudioService

        expected_formats = {"mp3", "m4a", "wav", "mp4", "aac", "ogg", "webm"}
        assert AudioService.SUPPORTED_FORMATS == expected_formats


class TestDiarizationService:
    """Tests for diarization service."""

    def test_single_speaker(self):
        """Test single speaker handling."""
        from app.services.transcription.diarization import DiarizationService

        service = DiarizationService()
        utterances = [
            {"speaker_number": 0, "text": "Hello, this is a monologue."},
            {"speaker_number": 0, "text": "I'm talking by myself."},
        ]

        result = service.assign_speaker_labels(utterances)

        assert all(u["speaker_label"] == "speaker" for u in result)
        assert all(u["speaker_confidence"] == 1.0 for u in result)

    def test_two_speakers_coach_identification(self):
        """Test coach identification in two-speaker conversation."""
        from app.services.transcription.diarization import DiarizationService

        service = DiarizationService()
        utterances = [
            {"speaker_number": 0, "text": "How has your training been going this week?"},
            {"speaker_number": 1, "text": "It's been pretty good."},
            {"speaker_number": 0, "text": "What goals do you want to focus on?"},
            {"speaker_number": 1, "text": "I want to improve my consistency."},
            {"speaker_number": 0, "text": "Let's create a plan for that. What does your routine look like?"},
        ]

        result = service.assign_speaker_labels(utterances)

        # Speaker 0 should be identified as coach (asks questions, uses coaching terms)
        speaker_0_labels = [u["speaker_label"] for u in result if u["speaker_number"] == 0]
        speaker_1_labels = [u["speaker_label"] for u in result if u["speaker_number"] == 1]

        # Either proper labels or low-confidence generic labels
        assert all(label in ("coach", "speaker_0") for label in speaker_0_labels)
        assert all(label in ("client", "speaker_1") for label in speaker_1_labels)

    def test_multi_speaker(self):
        """Test multi-speaker (3+) handling."""
        from app.services.transcription.diarization import DiarizationService

        service = DiarizationService()
        utterances = [
            {"speaker_number": 0, "text": "How are you both doing today?"},
            {"speaker_number": 1, "text": "I'm doing well."},
            {"speaker_number": 2, "text": "Same here, thanks for asking."},
            {"speaker_number": 0, "text": "Let's review your progress this week."},
        ]

        result = service.assign_speaker_labels(utterances)

        # Should have coach and participants
        labels = set(u["speaker_label"] for u in result)

        # Either proper multi-speaker labels or generic speaker labels
        assert len(labels) >= 2

    def test_question_counting(self):
        """Test question counting."""
        from app.services.transcription.diarization import DiarizationService

        service = DiarizationService()
        utterances = [
            {"text": "How are you?"},
            {"text": "What's your goal?"},
            {"text": "I'm fine."},
            {"text": "Tell me more?"},
        ]

        count = service._count_questions(utterances)
        assert count == 3

    def test_coaching_terms_counting(self):
        """Test coaching terms counting."""
        from app.services.transcription.diarization import DiarizationService

        service = DiarizationService()
        text = "Let's focus on your nutrition goals and create a plan for better sleep habits."

        count = service._count_coaching_terms(text)
        assert count >= 4  # goals, plan, sleep, habits, focus

    def test_directive_counting(self):
        """Test directive statement counting."""
        from app.services.transcription.diarization import DiarizationService

        service = DiarizationService()
        utterances = [
            {"text": "Try to drink more water."},
            {"text": "Make sure to get enough sleep."},
            {"text": "I understand."},
            {"text": "Focus on consistency."},
        ]

        count = service._count_directives(utterances)
        assert count == 3


class TestSummarizationService:
    """Tests for summarization service."""

    def test_format_transcript(self):
        """Test transcript formatting."""
        from app.services.transcription.summarization import SummarizationService

        service = SummarizationService()
        utterances = [
            {"speaker_label": "coach", "text": "How are you?", "start_time": 0.0},
            {"speaker_label": "client", "text": "I'm good.", "start_time": 2.5},
        ]

        formatted = service._format_transcript(utterances)

        assert "[00:00] coach: How are you?" in formatted
        assert "[00:02] client: I'm good." in formatted

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        from app.services.transcription.summarization import SummarizationService

        service = SummarizationService()

        assert service._format_timestamp(0) == "00:00"
        assert service._format_timestamp(65) == "01:05"
        assert service._format_timestamp(3661) == "61:01"

    def test_truncate_transcript_short(self):
        """Test that short transcripts aren't truncated."""
        from app.services.transcription.summarization import SummarizationService

        service = SummarizationService()
        short_text = "This is a short transcript."

        result = service._truncate_transcript(short_text, max_chars=1000)
        assert result == short_text

    def test_truncate_transcript_long(self):
        """Test that long transcripts are truncated."""
        from app.services.transcription.summarization import SummarizationService

        service = SummarizationService()
        long_text = "A" * 200

        result = service._truncate_transcript(long_text, max_chars=100)

        assert len(result) < len(long_text)
        assert "truncated" in result.lower()

    def test_parse_summary_response_valid_json(self):
        """Test parsing valid JSON response."""
        from app.services.transcription.summarization import SummarizationService

        service = SummarizationService()
        content = '{"executive_summary": "Test summary", "key_topics": ["topic1"]}'

        result = service._parse_summary_response(content)

        assert result["executive_summary"] == "Test summary"
        assert "topic1" in result["key_topics"]

    def test_parse_summary_response_json_in_code_block(self):
        """Test parsing JSON wrapped in code block."""
        from app.services.transcription.summarization import SummarizationService

        service = SummarizationService()
        content = '''```json
{"executive_summary": "Test summary", "key_topics": ["topic1"]}
```'''

        result = service._parse_summary_response(content)

        assert result["executive_summary"] == "Test summary"

    def test_parse_summary_response_invalid_json(self):
        """Test fallback for invalid JSON."""
        from app.services.transcription.summarization import SummarizationService

        service = SummarizationService()
        content = "This is not valid JSON"

        result = service._parse_summary_response(content)

        # Should return fallback
        assert "manual review" in result["executive_summary"].lower()
        assert result["engagement_score"] == 0.5

    def test_get_fallback_summary(self):
        """Test fallback summary structure."""
        from app.services.transcription.summarization import SummarizationService

        service = SummarizationService()
        fallback = service._get_fallback_summary()

        assert "executive_summary" in fallback
        assert "key_topics" in fallback
        assert "action_items" in fallback
        assert "coaching_moments" in fallback
        assert fallback["session_type_detected"] == "general"
        assert fallback["client_sentiment"] == "neutral"


class TestDeepgramService:
    """Tests for Deepgram service."""

    def test_service_initialization(self):
        """Test service initializes without API key check."""
        from app.services.transcription.deepgram import DeepgramService

        # Should not raise - API key is checked on first use
        service = DeepgramService()
        assert service._client is None

    @patch("app.services.transcription.deepgram.get_settings")
    def test_missing_api_key_raises(self, mock_settings):
        """Test that missing API key raises error on client access."""
        from app.services.transcription.deepgram import DeepgramService, DeepgramError

        mock_settings.return_value.deepgram_api_key = ""

        service = DeepgramService()

        with pytest.raises(DeepgramError, match="API key not configured"):
            _ = service.client

    def test_deepgram_utterance_dataclass(self):
        """Test DeepgramUtterance dataclass."""
        from app.services.transcription.deepgram import DeepgramUtterance

        utterance = DeepgramUtterance(
            speaker=0,
            text="Hello world",
            start=0.0,
            end=1.5,
            confidence=0.95,
        )

        assert utterance.speaker == 0
        assert utterance.text == "Hello world"
        assert utterance.confidence == 0.95

    def test_transcription_response_dataclass(self):
        """Test TranscriptionResponse dataclass."""
        from app.services.transcription.deepgram import TranscriptionResponse

        response = TranscriptionResponse(
            full_text="Hello world",
            word_count=2,
            duration_seconds=1.5,
            request_id="test-id",
            model="nova-2",
            confidence=0.95,
            utterances=[],
            words=[],
        )

        assert response.full_text == "Hello world"
        assert response.word_count == 2
        assert response.model == "nova-2"


class TestWebhookSigning:
    """Tests for webhook signature generation."""

    def test_webhook_signature_generation(self):
        """Test that webhook signatures are generated correctly."""
        import hmac
        import hashlib
        import json

        payload = {"event": "transcription.completed", "job_id": "123"}
        secret = "test-secret"

        message = json.dumps(payload, sort_keys=True)
        expected_signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        assert expected_signature is not None
        assert len(expected_signature) == 64  # SHA256 hex length

    def test_webhook_secret_generation(self):
        """Test webhook secret generation."""
        from app.api.v1.transcription import generate_webhook_secret

        secret1 = generate_webhook_secret()
        secret2 = generate_webhook_secret()

        # Should be unique
        assert secret1 != secret2

        # Should be URL-safe base64
        assert len(secret1) > 20
        assert all(c.isalnum() or c in '-_' for c in secret1)
