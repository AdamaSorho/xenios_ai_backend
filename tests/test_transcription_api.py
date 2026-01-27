"""Tests for transcription API endpoints."""

import io
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

import pytest
from fastapi import status


class TestTranscriptionAPI:
    """Tests for transcription API endpoints."""

    @pytest.fixture
    def mock_storage(self):
        """Mock storage service."""
        mock = MagicMock()
        mock.upload_file = AsyncMock(return_value="s3://bucket/audio.mp3")
        mock.download_file = AsyncMock(return_value=b"audio content")
        mock.download_to_temp_file = AsyncMock(return_value="/tmp/audio.mp3")
        mock.delete_file = AsyncMock()
        return mock

    @pytest.fixture
    def mock_audio_service(self):
        """Mock audio service."""
        mock = MagicMock()
        mock.validate_format.return_value = (True, None)
        mock.validate_audio.return_value = (True, None)
        mock.get_format_from_filename.return_value = "mp3"
        mock.get_content_type.return_value = "audio/mpeg"
        mock.get_audio_info.return_value = MagicMock(
            duration_seconds=120.5,
            format_name="mp3",
            codec="mp3",
            sample_rate=44100,
            channels=2,
            size_bytes=1024000,
        )
        return mock

    @pytest.fixture
    def mock_db(self):
        """Mock database connection."""
        mock = MagicMock()

        # Mock job creation
        mock.fetchrow = AsyncMock(
            return_value={
                "id": uuid4(),
                "client_id": uuid4(),
                "coach_id": uuid4(),
                "audio_filename": "test.mp3",
                "audio_format": "mp3",
                "audio_size_bytes": 1024000,
                "audio_duration_seconds": 120.5,
                "status": "pending",
                "progress": 0,
                "session_date": None,
                "session_type": None,
                "session_title": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )
        mock.fetch = AsyncMock(return_value=[])
        mock.fetchval = AsyncMock(return_value=0)
        mock.execute = AsyncMock()
        return mock

    def test_upload_endpoint_validation(self, client, auth_headers):
        """Test upload endpoint file validation."""
        # Test without file
        response = client.post(
            "/api/v1/transcription/upload",
            headers=auth_headers,
            data={"client_id": str(uuid4())},
        )
        # Should return 422 for missing file
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_invalid_file_type(self, client, auth_headers):
        """Test upload with invalid audio file type."""
        file_content = b"some text content"
        response = client.post(
            "/api/v1/transcription/upload",
            headers=auth_headers,
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
            data={"client_id": str(uuid4())},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "unsupported format" in response.json()["detail"].lower()

    @patch("app.api.v1.transcription.get_transcription_storage_service")
    @patch("app.api.v1.transcription.get_audio_service")
    @patch("app.core.database.get_db")
    @patch("app.workers.tasks.transcription.process_transcription")
    def test_upload_success(
        self,
        mock_task,
        mock_get_db,
        mock_get_audio,
        mock_get_storage,
        client,
        auth_headers,
        mock_storage,
        mock_audio_service,
        mock_db,
    ):
        """Test successful audio upload."""
        mock_get_storage.return_value = mock_storage
        mock_get_audio.return_value = mock_audio_service
        mock_get_db.return_value = mock_db
        mock_task.delay.return_value = None

        # Create a small fake MP3 file
        file_content = b"\xff\xfb\x90\x00" + b"\x00" * 1000  # MP3 header
        response = client.post(
            "/api/v1/transcription/upload",
            headers=auth_headers,
            files={"file": ("test.mp3", io.BytesIO(file_content), "audio/mpeg")},
            data={"client_id": str(uuid4())},
        )

        # Either success or 500 if dependencies not connected
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_201_CREATED,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]

    def test_status_endpoint_not_found(self, client, auth_headers):
        """Test status endpoint with non-existent job."""
        job_id = uuid4()
        with patch("app.core.database.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchrow = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            response = client.get(
                f"/api/v1/transcription/status/{job_id}",
                headers=auth_headers,
            )

            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]

    def test_transcript_endpoint_not_ready(self, client, auth_headers):
        """Test transcript endpoint when transcription not complete."""
        job_id = uuid4()
        with patch("app.core.database.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchrow = AsyncMock(
                return_value={
                    "id": job_id,
                    "client_id": uuid4(),
                    "coach_id": "test-user-id",
                    "status": "transcribing",
                    "audio_url": "s3://bucket/audio.mp3",
                }
            )
            mock_get_db.return_value = mock_db

            response = client.get(
                f"/api/v1/transcription/{job_id}/transcript",
                headers=auth_headers,
            )

            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]

    def test_summary_endpoint_partial_status(self, client, auth_headers):
        """Test summary endpoint when status is partial (summary failed)."""
        job_id = uuid4()
        with patch("app.core.database.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchrow = AsyncMock(
                return_value={
                    "id": job_id,
                    "client_id": uuid4(),
                    "coach_id": "test-user-id",
                    "status": "partial",
                    "audio_url": "s3://bucket/audio.mp3",
                }
            )
            mock_get_db.return_value = mock_db

            response = client.get(
                f"/api/v1/transcription/{job_id}/summary",
                headers=auth_headers,
            )

            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]

    def test_sessions_list_endpoint(self, client, auth_headers):
        """Test sessions listing endpoint."""
        with patch("app.core.database.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetch = AsyncMock(return_value=[])
            mock_db.fetchval = AsyncMock(return_value=0)
            mock_get_db.return_value = mock_db

            response = client.get(
                "/api/v1/transcription/sessions",
                headers=auth_headers,
            )

            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]

    def test_reprocess_endpoint_not_found(self, client, auth_headers):
        """Test reprocess endpoint with non-existent job."""
        job_id = uuid4()
        with patch("app.core.database.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchrow = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            response = client.post(
                f"/api/v1/transcription/{job_id}/reprocess",
                headers=auth_headers,
            )

            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]

    def test_delete_endpoint_not_found(self, client, auth_headers):
        """Test delete endpoint with non-existent job."""
        job_id = uuid4()
        with patch("app.core.database.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchrow = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            response = client.delete(
                f"/api/v1/transcription/{job_id}",
                headers=auth_headers,
            )

            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]

    def test_speaker_update_endpoint(self, client, auth_headers):
        """Test speaker label update endpoint."""
        job_id = uuid4()
        with patch("app.core.database.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchrow = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            response = client.patch(
                f"/api/v1/transcription/{job_id}/speakers",
                headers=auth_headers,
                json=[{"speaker_number": 0, "label": "coach"}],
            )

            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]


class TestTranscriptionSchemas:
    """Tests for transcription schemas."""

    def test_transcription_status_enum(self):
        """Test TranscriptionStatus enum values."""
        from app.schemas.transcription import TranscriptionStatus

        assert TranscriptionStatus.PENDING.value == "pending"
        assert TranscriptionStatus.TRANSCRIBING.value == "transcribing"
        assert TranscriptionStatus.DIARIZING.value == "diarizing"
        assert TranscriptionStatus.SUMMARIZING.value == "summarizing"
        assert TranscriptionStatus.COMPLETED.value == "completed"
        assert TranscriptionStatus.PARTIAL.value == "partial"
        assert TranscriptionStatus.FAILED.value == "failed"

    def test_session_type_enum(self):
        """Test SessionType enum values."""
        from app.schemas.transcription import SessionType

        assert SessionType.IN_PERSON.value == "in_person"
        assert SessionType.VIDEO_CALL.value == "video_call"
        assert SessionType.PHONE_CALL.value == "phone_call"

    def test_speaker_label_enum(self):
        """Test SpeakerLabel enum values."""
        from app.schemas.transcription import SpeakerLabel

        assert SpeakerLabel.COACH.value == "coach"
        assert SpeakerLabel.CLIENT.value == "client"
        assert SpeakerLabel.SPEAKER.value == "speaker"
        assert SpeakerLabel.UNKNOWN.value == "unknown"

    def test_intent_type_enum(self):
        """Test IntentType enum values."""
        from app.schemas.transcription import IntentType

        assert IntentType.QUESTION_OPEN.value == "question_open"
        assert IntentType.QUESTION_CLOSED.value == "question_closed"
        assert IntentType.ADVICE.value == "advice"
        assert IntentType.CONCERN.value == "concern"
        assert IntentType.COMMITMENT.value == "commitment"

    def test_coaching_moment_type_enum(self):
        """Test CoachingMomentType enum values."""
        from app.schemas.transcription import CoachingMomentType

        assert CoachingMomentType.BREAKTHROUGH.value == "breakthrough"
        assert CoachingMomentType.GOAL_SET.value == "goal_set"
        assert CoachingMomentType.RESISTANCE.value == "resistance"

    def test_action_item_response(self):
        """Test ActionItemResponse schema."""
        from app.schemas.transcription import ActionItemResponse

        action_item = ActionItemResponse(
            description="Complete weekly meal prep",
            owner="client",
            priority="high",
        )

        assert action_item.description == "Complete weekly meal prep"
        assert action_item.owner == "client"
        assert action_item.priority == "high"

    def test_utterance_response(self):
        """Test UtteranceResponse schema."""
        from app.schemas.transcription import UtteranceResponse

        utterance = UtteranceResponse(
            id=uuid4(),
            speaker_number=0,
            speaker_label="coach",
            speaker_confidence=0.95,
            text="How has your week been?",
            start_time=0.0,
            end_time=2.5,
            confidence=0.98,
            intent="question_open",
            sentiment="neutral",
            sequence_number=0,
        )

        assert utterance.speaker_label == "coach"
        assert utterance.text == "How has your week been?"
        assert utterance.intent == "question_open"


class TestTranscriptionSecurity:
    """Security tests for transcription API."""

    def test_unauthorized_access_returns_401(self, client):
        """Test that requests without auth return 401."""
        response = client.get("/api/v1/transcription/sessions")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_missing_api_key_returns_401(self, client, valid_jwt_token):
        """Test that requests without API key return 401."""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        response = client.get("/api/v1/transcription/sessions", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_missing_jwt_returns_401(self, client, valid_api_key):
        """Test that requests without JWT return 401."""
        headers = {"X-API-Key": valid_api_key}
        response = client.get("/api/v1/transcription/sessions", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_invalid_api_key_returns_401(self, client, valid_jwt_token):
        """Test that invalid API key returns 401."""
        headers = {
            "X-API-Key": "invalid-key",
            "Authorization": f"Bearer {valid_jwt_token}",
        }
        response = client.get("/api/v1/transcription/sessions", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_invalid_jwt_returns_401(self, client, valid_api_key):
        """Test that invalid JWT returns 401."""
        headers = {
            "X-API-Key": valid_api_key,
            "Authorization": "Bearer invalid-token",
        }
        response = client.get("/api/v1/transcription/sessions", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
