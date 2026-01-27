"""Schemas for LLM API endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMCompleteRequest(BaseModel):
    """Request body for LLM completion endpoint."""

    task: str = Field(
        ...,
        description="Task type for model selection (e.g., 'chat', 'session_summary')",
    )
    messages: list[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task": "chat",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, how are you?"},
                    ],
                }
            ]
        }
    }


class LLMStreamRequest(BaseModel):
    """Request body for LLM streaming endpoint."""

    task: str = Field(
        ...,
        description="Task type for model selection",
    )
    messages: list[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )


class LLMUsage(BaseModel):
    """Token usage information from LLM response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: Message
    finish_reason: str | None = None


class LLMCompleteResponse(BaseModel):
    """Response from LLM completion endpoint."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[LLMChoice]
    usage: LLMUsage | None = None


class IntentClassificationRequest(BaseModel):
    """Request for intent classification."""

    message: str = Field(..., description="The message to classify")
    intents: list[str] = Field(
        ...,
        description="List of possible intent labels",
        min_length=2,
    )


class IntentClassificationResponse(BaseModel):
    """Response from intent classification."""

    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class EntityExtractionRequest(BaseModel):
    """Request for entity extraction."""

    text: str = Field(..., description="Text to extract entities from")
    entity_types: list[str] = Field(
        ...,
        description="Types of entities to extract",
        min_length=1,
    )


class EntityExtractionResponse(BaseModel):
    """Response from entity extraction."""

    entities: dict[str, list[str]] = Field(
        ...,
        description="Extracted entities by type",
    )


class AvailableTasksResponse(BaseModel):
    """Response listing available LLM tasks."""

    tasks: list[str]
