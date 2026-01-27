"""Prompt templates for analytics LLM calls."""

CUE_DETECTION_PROMPT = """Analyze this coaching session utterance for language cues.

UTTERANCE: {text}
SPEAKER: {speaker_label}
CONTEXT (previous 2 utterances): {context}

Identify any of these cue types present:
- resistance: Client pushing back, expressing doubt, or avoiding ("I don't think I can...", "That won't work")
- commitment: Client making promises or expressing determination ("I will...", "I'm going to...")
- breakthrough: Moments of insight or realization ("I just realized...", "That makes sense now")
- concern: Client expressing worry or anxiety ("I'm worried about...", "What if...")
- deflection: Changing subject, giving vague answers, avoiding specifics
- enthusiasm: Expressing excitement or positive energy
- doubt: Hesitation or uncertainty in responses
- goal_setting: Setting concrete, measurable objectives

For each cue found, provide:
1. cue_type: The type from the list above
2. confidence: 0.0-1.0 how confident you are
3. interpretation: Brief explanation of why this is significant

If no significant cues are present, return empty array.

Respond in JSON format:
{{
  "cues": [
    {{"cue_type": "...", "confidence": 0.X, "interpretation": "..."}}
  ]
}}
"""
