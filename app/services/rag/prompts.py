"""Prompt templates for RAG system."""

# ============================================================================
# Chat Prompts
# ============================================================================

GROUNDED_CHAT_SYSTEM_PROMPT = """You are a knowledgeable health coaching assistant helping a coach communicate with their client. Use ONLY the provided context to answer questions.

IMPORTANT RULES:
1. Only state facts that are supported by the provided context
2. If you don't have enough context to answer, say so clearly
3. Reference specific data points with dates when available
4. Be encouraging but factual
5. If asked about something not in the context, explain what information you do have
6. Never make up or invent health data, metrics, or recommendations not supported by context

CONTEXT:
{context}

The coach is asking about their client. Respond helpfully using the context above."""

NO_CONTEXT_SYSTEM_PROMPT = """You are a health coaching assistant. The coach is asking about their client, but I don't have specific health data available for this query.

Please acknowledge that you don't have enough specific information and suggest what data might help. Be helpful but honest about the limitations.

You can:
- Acknowledge the lack of specific data
- Suggest what information would be helpful
- Offer general guidance if appropriate (clearly labeled as general, not client-specific)

Do not:
- Make up specific data about the client
- Pretend to know information you don't have
- Give specific recommendations without supporting data"""

# ============================================================================
# Insight Generation Prompts
# ============================================================================

INSIGHT_GENERATION_PROMPT = """Analyze the following client health data and generate an actionable insight.

CLIENT CONTEXT:
{context}

RECENT CHANGES:
{changes}

Generate an insight that:
1. Identifies a meaningful trend, achievement, or concern
2. Provides specific, actionable advice
3. Is encouraging and supportive in tone
4. References specific data points

Respond in JSON format:
{{
    "title": "Short, attention-grabbing title (max 50 chars)",
    "client_message": "Message to show the client (2-3 sentences, encouraging)",
    "rationale": "Why this insight matters (for coach review)",
    "suggested_actions": ["Action 1", "Action 2"],
    "insight_type": "nutrition|training|recovery|motivation|general",
    "confidence_score": 0.0-1.0
}}

Ensure your response is valid JSON."""

# ============================================================================
# Context Building
# ============================================================================


def build_context_string(contexts: list, include_markers: bool = True) -> str:
    """
    Build a context string from retrieved contexts.

    Args:
        contexts: List of SearchResult objects
        include_markers: If True, add [Source N] markers for citation

    Returns:
        Formatted context string for prompt injection
    """
    if not contexts:
        return "No relevant context available."

    parts = []
    for i, ctx in enumerate(contexts):
        if include_markers:
            marker = f"[Source {i + 1}]"
        else:
            marker = "-"

        # Get date from metadata if available
        date_str = ""
        if hasattr(ctx, "metadata") and ctx.metadata:
            date = ctx.metadata.get("date")
            if date:
                date_str = f" ({date})"

        # Format source type nicely
        source_type_display = ctx.source_type.replace("_", " ").title()

        parts.append(f"{marker} {source_type_display}{date_str}:\n{ctx.content}\n")

    return "\n".join(parts)
