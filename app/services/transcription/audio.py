"""Audio file utilities for transcription."""

import subprocess
import json
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""

    pass


class AudioValidationError(Exception):
    """Raised when audio validation fails."""

    pass


@dataclass
class AudioInfo:
    """Information about an audio file."""

    duration_seconds: float
    format_name: str
    codec: str
    sample_rate: int
    channels: int
    size_bytes: int
    bit_rate: int | None = None


class AudioService:
    """
    Audio file utilities using ffprobe.

    Provides duration detection, format validation, and metadata extraction.
    """

    # Supported audio formats (per spec)
    SUPPORTED_FORMATS = {"mp3", "m4a", "wav", "mp4", "aac", "ogg", "webm"}

    # Content type mapping
    CONTENT_TYPE_MAP = {
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "wav": "audio/wav",
        "mp4": "video/mp4",
        "aac": "audio/aac",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
    }

    # Limits per spec
    MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
    MAX_DURATION_SECONDS = 2 * 60 * 60  # 2 hours
    MIN_DURATION_SECONDS = 10  # 10 seconds

    def get_audio_info(self, file_path: str) -> AudioInfo:
        """
        Get audio file information using ffprobe.

        Args:
            file_path: Path to the audio file

        Returns:
            AudioInfo with duration, format, and codec details

        Raises:
            AudioProcessingError: If ffprobe fails
        """
        if not Path(file_path).exists():
            raise AudioProcessingError(f"File not found: {file_path}")

        try:
            # Run ffprobe to get file info
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise AudioProcessingError(
                    f"ffprobe failed: {result.stderr}"
                )

            probe_data = json.loads(result.stdout)

            # Get format info
            format_info = probe_data.get("format", {})

            # Find the audio stream
            streams = probe_data.get("streams", [])
            audio_stream = next(
                (s for s in streams if s.get("codec_type") == "audio"),
                None,
            )

            if not audio_stream:
                raise AudioProcessingError("No audio stream found in file")

            return AudioInfo(
                duration_seconds=float(format_info.get("duration", 0)),
                format_name=format_info.get("format_name", "unknown"),
                codec=audio_stream.get("codec_name", "unknown"),
                sample_rate=int(audio_stream.get("sample_rate", 0)),
                channels=int(audio_stream.get("channels", 0)),
                size_bytes=int(format_info.get("size", 0)),
                bit_rate=int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
            )

        except subprocess.TimeoutExpired as e:
            raise AudioProcessingError(f"ffprobe timed out: {e}") from e
        except json.JSONDecodeError as e:
            raise AudioProcessingError(f"Failed to parse ffprobe output: {e}") from e
        except FileNotFoundError:
            raise AudioProcessingError(
                "ffprobe not found. Please install ffmpeg."
            ) from None
        except Exception as e:
            raise AudioProcessingError(f"Failed to probe audio: {e}") from e

    def validate_audio(
        self,
        file_path: str,
        check_duration: bool = True,
    ) -> tuple[bool, str | None]:
        """
        Validate audio file meets requirements.

        Args:
            file_path: Path to the audio file
            check_duration: Whether to check duration limits

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            info = self.get_audio_info(file_path)
        except AudioProcessingError as e:
            return False, str(e)

        # Check size
        if info.size_bytes > self.MAX_SIZE_BYTES:
            return False, (
                f"File too large: {info.size_bytes / (1024 * 1024):.1f} MB "
                f"(max {self.MAX_SIZE_BYTES / (1024 * 1024):.0f} MB)"
            )

        # Check duration if requested
        if check_duration:
            if info.duration_seconds > self.MAX_DURATION_SECONDS:
                return False, (
                    f"Duration too long: {info.duration_seconds / 60:.1f} minutes "
                    f"(max {self.MAX_DURATION_SECONDS / 60:.0f} minutes)"
                )

            if info.duration_seconds < self.MIN_DURATION_SECONDS:
                return False, (
                    f"Duration too short: {info.duration_seconds:.1f} seconds "
                    f"(min {self.MIN_DURATION_SECONDS} seconds)"
                )

        return True, None

    def validate_format(self, filename: str) -> tuple[bool, str | None]:
        """
        Validate file format by extension.

        Args:
            filename: Original filename

        Returns:
            Tuple of (is_valid, error_message)
        """
        ext = Path(filename).suffix.lower().lstrip(".")

        if ext not in self.SUPPORTED_FORMATS:
            return False, (
                f"Unsupported format: {ext}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        return True, None

    def get_content_type(self, filename: str) -> str:
        """
        Get the MIME content type for a file.

        Args:
            filename: Original filename

        Returns:
            MIME content type string
        """
        ext = Path(filename).suffix.lower().lstrip(".")
        return self.CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

    def get_format_from_filename(self, filename: str) -> str:
        """
        Get the format extension from a filename.

        Args:
            filename: Original filename

        Returns:
            Format extension without dot
        """
        return Path(filename).suffix.lower().lstrip(".")


# Global service instance
_audio_service: AudioService | None = None


def get_audio_service() -> AudioService:
    """Get the global audio service instance."""
    global _audio_service
    if _audio_service is None:
        _audio_service = AudioService()
    return _audio_service
