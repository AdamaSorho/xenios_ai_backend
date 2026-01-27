"""Language cue detection service using LLM analysis."""

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.core.logging import get_logger
from app.models.analytics import LanguageCue
from app.schemas.analytics import CueType
from app.services.analytics.prompts import CUE_DETECTION_PROMPT
from app.services.llm.client import LLMClient, LLMError

logger = get_logger(__name__)


class CueDetectionError(Exception):
    """Raised when cue detection fails."""

    pass


@dataclass
class DetectedCue:
    """A cue detected by the LLM."""

    cue_type: str
    confidence: float
    interpretation: str


class CueDetectionService:
    """
    Detect language cues in client utterances using LLM analysis.

    Detects:
    - Resistance, commitment, breakthrough
    - Concern, deflection, enthusiasm
    - Doubt, agreement, goal_setting
    """

    # LLM Configuration
    MODEL_TASK = "cue_detection"  # Maps to gpt-4o-mini per config
    MAX_RETRIES = 2
    TIMEOUT_SECONDS = 30
    RATE_LIMIT_DELAY = 0.5  # Seconds between requests
    MIN_CONFIDENCE = 0.7  # Only keep high-confidence cues

    # PII patterns
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize the cue detection service."""
        self._llm_client = llm_client

    @property
    def llm_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    async def detect_cues(
        self,
        utterances: list[dict[str, Any]],
        session_analytics_id: UUID,
    ) -> tuple[list[LanguageCue], dict[str, int]]:
        """
        Detect language cues in client utterances.

        Args:
            utterances: List of utterance dicts with id, text, speaker_label, start_time
            session_analytics_id: UUID of the session analytics record

        Returns:
            Tuple of (list of LanguageCue models, cue count dict)

        Raises:
            CueDetectionError: If all cue detection attempts fail
        """
        cues = []
        errors = []

        # Only analyze client utterances (coach utterances less relevant for risk)
        client_utterances = [
            u for u in utterances
            if u.get("speaker_label", "").lower() == "client"
        ]

        logger.info(
            "Starting cue detection",
            session_analytics_id=str(session_analytics_id),
            total_utterances=len(utterances),
            client_utterances=len(client_utterances),
        )

        for i, utterance in enumerate(client_utterances):
            # Get context (previous 2 utterances from any speaker)
            context = self._get_context(utterances, utterance)

            try:
                detected = await self._analyze_utterance_with_retry(utterance, context)

                for cue in detected:
                    if cue.confidence >= self.MIN_CONFIDENCE:
                        cues.append(LanguageCue(
                            session_analytics_id=session_analytics_id,
                            utterance_id=utterance["id"],
                            cue_type=cue.cue_type,
                            confidence=cue.confidence,
                            text_excerpt=self._redact_pii(utterance.get("text", "")[:200]),
                            timestamp_seconds=float(utterance.get("start_time", 0)),
                            preceding_context=self._redact_pii(context) if context else None,
                            interpretation=cue.interpretation,
                        ))

            except CueDetectionError as e:
                # Log error but continue - cue detection is non-critical
                logger.warning(
                    "cue_detection_failed_for_utterance",
                    utterance_id=str(utterance.get("id")),
                    error=str(e),
                )
                errors.append(str(e))
                continue

            # Rate limiting between LLM calls
            if i < len(client_utterances) - 1:
                await asyncio.sleep(self.RATE_LIMIT_DELAY)

        # Calculate cue counts
        cue_counts = self._count_cues(cues)

        logger.info(
            "Cue detection complete",
            session_analytics_id=str(session_analytics_id),
            total_cues=len(cues),
            cue_counts=cue_counts,
            errors=len(errors),
        )

        # If all utterances failed, raise error
        if errors and len(errors) == len(client_utterances):
            raise CueDetectionError(f"All cue detection failed: {errors[0]}")

        return cues, cue_counts

    async def _analyze_utterance_with_retry(
        self,
        utterance: dict[str, Any],
        context: str,
    ) -> list[DetectedCue]:
        """Analyze a single utterance with retries."""
        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self._call_llm(utterance, context),
                    timeout=self.TIMEOUT_SECONDS,
                )

            except asyncio.TimeoutError:
                last_error = CueDetectionError("LLM timeout")
            except json.JSONDecodeError as e:
                last_error = CueDetectionError(f"Malformed JSON from LLM: {e}")
            except LLMError as e:
                if "rate" in str(e).lower():
                    # Exponential backoff on rate limit
                    await asyncio.sleep(2 ** attempt)
                    last_error = CueDetectionError("Rate limit exceeded")
                else:
                    last_error = CueDetectionError(f"LLM error: {e}")
            except Exception as e:
                last_error = CueDetectionError(f"Unexpected error: {e}")

            if attempt < self.MAX_RETRIES:
                await asyncio.sleep(1)  # Brief pause before retry

        raise last_error

    async def _call_llm(
        self,
        utterance: dict[str, Any],
        context: str,
    ) -> list[DetectedCue]:
        """Call LLM and parse response."""
        prompt = CUE_DETECTION_PROMPT.format(
            text=utterance.get("text", ""),
            speaker_label=utterance.get("speaker_label", "client"),
            context=context or "No previous context",
        )

        response = await self.llm_client.complete(
            task=self.MODEL_TASK,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response["choices"][0]["message"]["content"]
        return self._parse_llm_response(content)

    def _parse_llm_response(self, response: str) -> list[DetectedCue]:
        """Parse and validate LLM JSON response."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                # Try finding JSON object directly
                match = re.search(r'\{[^{}]*"cues"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise

        cues_data = data.get("cues", [])
        validated = []

        # Valid cue types from enum
        valid_cue_types = {ct.value for ct in CueType}

        for cue in cues_data:
            # Validate required fields
            if not all(k in cue for k in ["cue_type", "confidence"]):
                continue

            # Validate cue_type is known
            cue_type = cue["cue_type"]
            if cue_type not in valid_cue_types:
                continue

            # Validate confidence range
            try:
                conf = float(cue["confidence"])
                if not (0 <= conf <= 1):
                    continue
            except (ValueError, TypeError):
                continue

            validated.append(DetectedCue(
                cue_type=cue_type,
                confidence=conf,
                interpretation=cue.get("interpretation", ""),
            ))

        return validated

    def _get_context(
        self,
        utterances: list[dict[str, Any]],
        current: dict[str, Any],
    ) -> str:
        """Get preceding utterances for context."""
        current_seq = current.get("sequence_number", 0)

        # Find previous 2 utterances
        previous = [
            u for u in utterances
            if u.get("sequence_number", 0) < current_seq
        ]
        previous = sorted(previous, key=lambda x: x.get("sequence_number", 0))[-2:]

        if not previous:
            return ""

        context_lines = []
        for u in previous:
            speaker = u.get("speaker_label", "unknown")
            text = u.get("text", "")[:150]  # Truncate for context
            context_lines.append(f"{speaker}: {text}")

        return "\n".join(context_lines)

    def _redact_pii(self, text: str) -> str:
        """
        Redact potential PII from text excerpts.

        Redacts (pattern-based):
        - Phone numbers (US format)
        - Email addresses
        - SSN patterns

        NOT redacted (out of scope for MVP):
        - Names (would require NER model)
        - Addresses (would require NER model)
        """
        text = self.PHONE_PATTERN.sub('[PHONE]', text)
        text = self.EMAIL_PATTERN.sub('[EMAIL]', text)
        text = self.SSN_PATTERN.sub('[SSN]', text)
        return text

    def _count_cues(self, cues: list[LanguageCue]) -> dict[str, int]:
        """Count cues by type."""
        counts = Counter(cue.cue_type for cue in cues)
        return {
            "resistance": counts.get("resistance", 0),
            "commitment": counts.get("commitment", 0),
            "breakthrough": counts.get("breakthrough", 0),
            "concern": counts.get("concern", 0),
            "deflection": counts.get("deflection", 0),
            "enthusiasm": counts.get("enthusiasm", 0),
            "doubt": counts.get("doubt", 0),
            "agreement": counts.get("agreement", 0),
            "goal_setting": counts.get("goal_setting", 0),
        }


# Global service instance
_cue_detection_service: CueDetectionService | None = None


def get_cue_detection_service() -> CueDetectionService:
    """Get the global cue detection service instance."""
    global _cue_detection_service
    if _cue_detection_service is None:
        _cue_detection_service = CueDetectionService()
    return _cue_detection_service
