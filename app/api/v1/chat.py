"""Chat API endpoints for RAG-grounded responses (Spec 0004)."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.auth import UserContext, get_current_user, verify_coach_client_relationship
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.schemas.rag import ChatRequest, ChatResponse
from app.services.rag.chat import ChatService

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/complete", response_model=ChatResponse)
async def chat_complete(
    request: ChatRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    http_request: Request,
):
    """
    Generate a grounded chat response.

    Uses RAG to retrieve relevant client health data and generate a response
    that cites sources. Returns 404 if client not found or coach doesn't have access.

    Response includes:
    - response: The generated text
    - sources: Citations for data used
    - confidence: Score based on context relevance
    - has_context: Whether relevant context was found
    - conversation_id: ID for continuing the conversation
    """
    async with get_db_session() as db:
        # Verify coach has access (returns 404 per spec to prevent enumeration)
        await verify_coach_client_relationship(
            db,
            coach_id=user.user_id,
            client_id=str(request.client_id),
            raise_404=True,
        )

        # Set client_id on request state for audit logging
        http_request.state.client_id = str(request.client_id)

        logger.info(
            "Chat completion request",
            coach_id=user.user_id,
            client_id=str(request.client_id),
            has_conversation_id=request.conversation_id is not None,
        )

        chat_service = ChatService(db)
        response = await chat_service.generate_response(
            client_id=request.client_id,
            coach_id=user.user_id,
            message=request.message,
            conversation_id=request.conversation_id,
            max_context_items=request.max_context_items,
            include_sources=request.include_sources,
        )

        return response


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    http_request: Request,
):
    """
    Stream a grounded chat response via Server-Sent Events.

    Uses RAG to retrieve relevant client health data and streams the response.

    SSE Event Format:
    - event: chunk, data: {"type": "chunk", "content": "text..."}
    - event: done, data: {"type": "done", "sources": [...], "confidence": 0.85, ...}
    - event: error, data: {"type": "error", "code": "...", "message": "..."}
    """
    async def generate():
        async with get_db_session() as db:
            # Verify coach has access
            try:
                await verify_coach_client_relationship(
                    db,
                    coach_id=user.user_id,
                    client_id=str(request.client_id),
                    raise_404=True,
                )
            except Exception as e:
                error_data = json.dumps({
                    "type": "error",
                    "code": "UNAUTHORIZED",
                    "message": "Client not found",
                })
                yield f"event: error\ndata: {error_data}\n\n"
                return

            # Set client_id for audit logging
            http_request.state.client_id = str(request.client_id)

            logger.info(
                "Chat stream request",
                coach_id=user.user_id,
                client_id=str(request.client_id),
            )

            chat_service = ChatService(db)

            try:
                async for chunk in chat_service.generate_response_stream(
                    client_id=request.client_id,
                    coach_id=user.user_id,
                    message=request.message,
                    conversation_id=request.conversation_id,
                    max_context_items=request.max_context_items,
                ):
                    if chunk.type == "chunk":
                        data = json.dumps({"type": "chunk", "content": chunk.content})
                        yield f"event: chunk\ndata: {data}\n\n"

                    elif chunk.type == "done":
                        # Convert sources to dicts for JSON serialization
                        sources_data = [s.model_dump() for s in (chunk.sources or [])]
                        data = json.dumps({
                            "type": "done",
                            "sources": sources_data,
                            "confidence": chunk.confidence,
                            "has_context": chunk.has_context,
                            "conversation_id": str(chunk.conversation_id) if chunk.conversation_id else None,
                            "tokens_used": chunk.tokens_used,
                        })
                        yield f"event: done\ndata: {data}\n\n"

                    elif chunk.type == "error":
                        data = json.dumps({
                            "type": "error",
                            "code": chunk.code,
                            "message": chunk.message,
                        })
                        yield f"event: error\ndata: {data}\n\n"

            except Exception as e:
                logger.error("Chat stream failed", error=str(e))
                error_data = json.dumps({
                    "type": "error",
                    "code": "GENERATION_FAILED",
                    "message": "Failed to generate response",
                })
                yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
