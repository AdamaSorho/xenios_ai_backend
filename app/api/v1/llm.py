"""LLM API endpoints for completions and streaming."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from app.core.auth import UserContext, get_current_user
from app.core.logging import get_logger
from app.schemas.llm import (
    AvailableTasksResponse,
    EntityExtractionRequest,
    EntityExtractionResponse,
    IntentClassificationRequest,
    IntentClassificationResponse,
    LLMCompleteRequest,
    LLMCompleteResponse,
    LLMStreamRequest,
)
from app.services.llm import LLMClient, LLMError, list_available_tasks
from app.services.llm.prompts import (
    build_entity_extraction_messages,
    build_intent_classification_messages,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/tasks", response_model=AvailableTasksResponse)
async def get_available_tasks() -> AvailableTasksResponse:
    """
    List available LLM task types.

    Returns the configured task types that can be used with the completion endpoints.
    """
    return AvailableTasksResponse(tasks=list_available_tasks())


@router.post("/complete", response_model=LLMCompleteResponse)
async def llm_complete(
    request: LLMCompleteRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
) -> LLMCompleteResponse:
    """
    Send a completion request to the LLM.

    Requires authentication with both API key and JWT.
    The model is automatically selected based on the task type.
    """
    logger.info(
        "LLM complete request",
        user_id=user.user_id,
        task=request.task,
    )

    try:
        client = LLMClient()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result = await client.complete(request.task, messages)
        return LLMCompleteResponse(**result)

    except ValueError as e:
        # Invalid task type
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from None
    except LLMError as e:
        logger.error("LLM completion failed", error=str(e), user_id=user.user_id)
        raise HTTPException(
            status_code=e.status_code or status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service error: {e.message}",
        ) from None


@router.post("/stream")
async def llm_stream(
    request: LLMStreamRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
) -> EventSourceResponse:
    """
    Stream a completion response from the LLM using Server-Sent Events.

    Requires authentication with both API key and JWT.
    Returns an SSE stream of completion chunks.
    """
    logger.info(
        "LLM stream request",
        user_id=user.user_id,
        task=request.task,
    )

    async def event_generator() -> AsyncIterator[dict]:
        try:
            client = LLMClient()
            messages = [{"role": m.role, "content": m.content} for m in request.messages]

            async for chunk in client.stream(request.task, messages):
                yield {"event": "message", "data": chunk}

            yield {"event": "done", "data": ""}

        except ValueError as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        except LLMError as e:
            yield {"event": "error", "data": json.dumps({"error": e.message})}

    return EventSourceResponse(event_generator())


@router.post("/classify-intent", response_model=IntentClassificationResponse)
async def classify_intent(
    request: IntentClassificationRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
) -> IntentClassificationResponse:
    """
    Classify the intent of a message.

    Uses the intent_classification task type with deterministic temperature.
    """
    logger.info(
        "Intent classification request",
        user_id=user.user_id,
        intent_count=len(request.intents),
    )

    try:
        client = LLMClient()
        messages = build_intent_classification_messages(request.message, request.intents)
        result = await client.complete_with_json("intent_classification", messages)

        return IntentClassificationResponse(
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.0),
        )

    except LLMError as e:
        logger.error("Intent classification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Classification failed: {e.message}",
        ) from None


@router.post("/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities(
    request: EntityExtractionRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
) -> EntityExtractionResponse:
    """
    Extract entities from text.

    Uses the entity_extraction task type with deterministic temperature.
    """
    logger.info(
        "Entity extraction request",
        user_id=user.user_id,
        entity_types=request.entity_types,
    )

    try:
        client = LLMClient()
        messages = build_entity_extraction_messages(request.text, request.entity_types)
        result = await client.complete_with_json("entity_extraction", messages)

        return EntityExtractionResponse(entities=result)

    except LLMError as e:
        logger.error("Entity extraction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Extraction failed: {e.message}",
        ) from None
