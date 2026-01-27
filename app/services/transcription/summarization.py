"""Session summarization service using LLM."""

import json
from typing import Any

from app.core.logging import get_logger
from app.services.llm.client import LLMClient, LLMError

logger = get_logger(__name__)


class SummarizationError(Exception):
    """Raised when summarization fails."""

    pass


class SummarizationService:
    """
    Generate session summaries using LLM (Opus 4.5 per spec).

    Extracts key topics, concerns, action items, and coaching moments.
    """

    SESSION_SUMMARY_PROMPT = '''You are analyzing a coaching session transcript between a health/fitness coach and their client.

TRANSCRIPT:
{transcript}

Generate a structured analysis in JSON format:

{{
  "executive_summary": "2-3 sentence overview of what was discussed and accomplished",
  "key_topics": ["topic1", "topic2", "topic3"],
  "client_concerns": ["concern1", "concern2"],
  "coach_recommendations": ["recommendation1", "recommendation2"],
  "action_items": [
    {{"description": "specific action", "owner": "coach or client", "priority": "high, medium, or low"}}
  ],
  "goals_discussed": ["goal1", "goal2"],
  "coaching_moments": [
    {{
      "type": "breakthrough, concern, goal_set, commitment, or resistance",
      "timestamp_seconds": 123.4,
      "description": "what happened",
      "significance": "why this matters"
    }}
  ],
  "session_type_detected": "nutrition, training, mindset, accountability, or general",
  "client_sentiment": "positive, neutral, negative, or mixed",
  "engagement_score": 0.85
}}

Important:
- Be specific and actionable in your analysis
- For coaching_moments, include the timestamp_seconds from the transcript
- engagement_score should be 0.0 to 1.0 based on client participation level
- Focus on insights that help the coach track client progress'''

    def __init__(self) -> None:
        """Initialize the summarization service."""
        self._llm_client: LLMClient | None = None

    @property
    def llm_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    async def generate_summary(
        self,
        full_text: str,
        utterances: list[dict],
    ) -> dict[str, Any]:
        """
        Generate session summary from transcript.

        Args:
            full_text: Full transcript text
            utterances: List of utterance dicts with speaker_label, text, start_time

        Returns:
            Summary dict with all extracted information

        Raises:
            SummarizationError: If summarization fails
        """
        # Format transcript with speaker labels and timestamps
        formatted_transcript = self._format_transcript(utterances)

        # Truncate if too long (Opus 4.5 can handle large contexts but be reasonable)
        if len(formatted_transcript) > 100000:
            formatted_transcript = self._truncate_transcript(formatted_transcript)

        prompt = self.SESSION_SUMMARY_PROMPT.format(
            transcript=formatted_transcript
        )

        logger.info(
            "Generating session summary",
            transcript_length=len(formatted_transcript),
            utterance_count=len(utterances),
        )

        try:
            response = await self.llm_client.complete(
                task="session_summary",  # Routes to Opus 4.5 per model config
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse JSON response
            content = response["choices"][0]["message"]["content"]
            summary_data = self._parse_summary_response(content)

            # Add LLM metadata
            summary_data["llm_model"] = response.get("model", "unknown")
            usage = response.get("usage", {})
            summary_data["prompt_tokens"] = usage.get("prompt_tokens", 0)
            summary_data["completion_tokens"] = usage.get("completion_tokens", 0)

            logger.info(
                "Summary generated successfully",
                topics=len(summary_data.get("key_topics", [])),
                action_items=len(summary_data.get("action_items", [])),
                moments=len(summary_data.get("coaching_moments", [])),
            )

            return summary_data

        except LLMError as e:
            logger.error("LLM summarization failed", error=str(e))
            raise SummarizationError(f"Failed to generate summary: {e}") from e
        except Exception as e:
            logger.error("Summarization failed", error=str(e))
            raise SummarizationError(f"Summarization error: {e}") from e

    def _format_transcript(self, utterances: list[dict]) -> str:
        """Format transcript with timestamps and speakers."""
        lines = []
        for u in utterances:
            timestamp = self._format_timestamp(u.get("start_time", 0))
            speaker = u.get("speaker_label", "unknown")
            text = u.get("text", "")
            lines.append(f"[{timestamp}] {speaker}: {text}")
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

    def _parse_summary_response(self, content: str) -> dict[str, Any]:
        """Parse LLM JSON response with fallbacks."""
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]

            return json.loads(content.strip())

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse summary JSON, using fallback",
                error=str(e),
            )
            # Return minimal summary on parse failure
            return self._get_fallback_summary()

    def _get_fallback_summary(self) -> dict[str, Any]:
        """Return a fallback summary when parsing fails."""
        return {
            "executive_summary": "Summary generation encountered an issue - manual review recommended",
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


# Global service instance
_summarization_service: SummarizationService | None = None


def get_summarization_service() -> SummarizationService:
    """Get the global summarization service instance."""
    global _summarization_service
    if _summarization_service is None:
        _summarization_service = SummarizationService()
    return _summarization_service
