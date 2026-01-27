"""Prompt templates for LLM tasks."""



def build_chat_messages(
    user_message: str,
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Build messages list for a chat completion.

    Args:
        user_message: The current user message
        system_prompt: Optional system prompt
        history: Optional conversation history

    Returns:
        List of message dicts ready for LLM API
    """
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    return messages


def build_intent_classification_messages(
    user_message: str,
    intents: list[str],
) -> list[dict[str, str]]:
    """
    Build messages for intent classification task.

    Args:
        user_message: The user message to classify
        intents: List of possible intent labels

    Returns:
        Messages list for classification
    """
    intent_list = ", ".join(f'"{i}"' for i in intents)

    system_prompt = f"""You are an intent classifier. Classify the user's message into one of these intents: {intent_list}.

Respond with ONLY a JSON object in this format:
{{"intent": "<intent>", "confidence": <0.0-1.0>}}

Do not include any other text."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def build_entity_extraction_messages(
    text: str,
    entity_types: list[str],
) -> list[dict[str, str]]:
    """
    Build messages for entity extraction task.

    Args:
        text: The text to extract entities from
        entity_types: List of entity types to extract

    Returns:
        Messages list for extraction
    """
    entity_list = ", ".join(entity_types)

    system_prompt = f"""You are an entity extractor. Extract the following entity types from the text: {entity_list}.

Respond with ONLY a JSON object where keys are entity types and values are arrays of extracted entities.
Example: {{"person": ["John", "Mary"], "date": ["2024-01-15"]}}

If no entities of a type are found, use an empty array."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def build_summary_messages(
    content: str,
    context: str | None = None,
    max_length: int | None = None,
) -> list[dict[str, str]]:
    """
    Build messages for summarization task.

    Args:
        content: The content to summarize
        context: Optional context about the content
        max_length: Optional maximum summary length in words

    Returns:
        Messages list for summarization
    """
    length_instruction = ""
    if max_length:
        length_instruction = f" Keep the summary under {max_length} words."

    context_section = ""
    if context:
        context_section = f"\n\nContext: {context}"

    system_prompt = f"""You are a professional summarizer. Create a clear, concise summary of the provided content.{length_instruction}

Focus on:
- Key points and main ideas
- Important details and conclusions
- Actionable insights{context_section}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please summarize the following:\n\n{content}"},
    ]


# Default system prompts for different contexts
DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    "coaching_assistant": """You are an AI assistant for Xenios, a coaching platform. You help coaches and clients with:
- Understanding session notes and insights
- Tracking goals and progress
- Providing evidence-based coaching suggestions

Be professional, empathetic, and focused on supporting the coaching relationship.""",
    "general": """You are a helpful AI assistant. Provide clear, accurate, and helpful responses.""",
}


def get_system_prompt(prompt_type: str) -> str:
    """Get a predefined system prompt by type."""
    return DEFAULT_SYSTEM_PROMPTS.get(prompt_type, DEFAULT_SYSTEM_PROMPTS["general"])
