import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from src.chat.llm import BedrockClient
from src.chat.retriever import QdrantRetriever
from src.chat.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationListItem,
    ConversationListResponse,
    ConversationRenameRequest,
    DocumentChatRequest,
    PersonalizationUpdate,
    ProfileMemoryCreate,
)
from src.chat.service import ChatService
from src.core.auth import get_current_user
from src.core.config import Settings, get_settings
from src.core.database import get_db
from src.core.exceptions import APIException
from src.limiter import limiter

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache
def get_llm_client_singleton() -> BedrockClient:
    settings = get_settings()
    return BedrockClient(settings)


@lru_cache
def get_retriever_singleton() -> QdrantRetriever:
    settings = get_settings()
    return QdrantRetriever(settings)


def get_chat_service(
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> ChatService:
    """
    Dependency to get ChatService instance.

    Args:
        settings (Settings): Application settings.
        db (Session): Database session.
        current_user (dict): Current authenticated user claims.

    Returns:
        ChatService: Initialized chat service.
    """
    llm_client = get_llm_client_singleton()
    retriever = get_retriever_singleton()
    return ChatService(llm_client, retriever, db, current_user, settings)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("5/minute")
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    _current_user: dict = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    """
    Endpoint to handle authenticated chat requests.
    Supports both streaming and non-streaming responses.

    Args:
        request (Request): FastAPI request object (for rate limiting).
        chat_request (ChatRequest): Chat request with messages and optional conversation_id.
        service (ChatService): Injected chat service with user context.

    Returns:
        ChatResponse or EventSourceResponse: Response based on stream flag.

    Raises:
        HTTPException: For validation or service errors.
    """
    if not chat_request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Get the last user message
    last_message = chat_request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    try:
        # Streaming response
        if chat_request.stream:
            logger.info("Streaming request initiated for user: %s", service.user.email)

            async def event_generator():
                try:
                    stream_gen, conversation_id = service.generate_response_stream(
                        chat_request.messages,
                        chat_request.conversation_id,
                    )

                    # Send conversation ID first
                    yield {"event": "conversation_id", "data": str(conversation_id)}

                    # Stream response chunks
                    for event in stream_gen:
                        yield event

                except APIException as e:
                    logger.error("APIException during streaming: %s", e.message)
                    yield {"event": "error", "data": e.message}
                except Exception as e:
                    logger.error("Unexpected error during streaming: %s", e)
                    yield {"event": "error", "data": "Internal server error"}

            return EventSourceResponse(event_generator())

        # Non-streaming response
        logger.info(
            "Non-streaming request initiated for user: %s",
            service.user.email,
        )
        response_text, conversation_id, message_id, metadata = (
            service.generate_response(
                chat_request.messages,
                chat_request.conversation_id,
            )
        )
        return ChatResponse(
            conversation_id=conversation_id,
            message_id=message_id,
            response=response_text,
            metadata=metadata,
        )

    except APIException as e:
        logger.error("APIException: %s", e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chat/document", response_model=ChatResponse)
@limiter.limit("5/minute")
async def chat_document_endpoint(
    request: Request,
    chat_request: DocumentChatRequest,
    _current_user: dict = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    """
    Chat endpoint that scopes RAG retrieval to a single pinned document.
    Provide document_id and/or title in the body to bypass similarity search
    and use ONLY the chunks belonging to that document.
    """
    if not chat_request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = chat_request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    try:
        if chat_request.stream:
            logger.info(
                "Streaming pinned-document request for user: %s",
                service.user.email,
            )

            async def event_generator():
                try:
                    stream_gen, conversation_id = (
                        service.generate_response_stream_for_document(
                            chat_request.messages,
                            document_id=chat_request.document_id,
                            title=chat_request.title,
                            conversation_id=chat_request.conversation_id,
                        )
                    )
                    yield {"event": "conversation_id", "data": str(conversation_id)}
                    for event in stream_gen:
                        yield event
                except APIException as e:
                    logger.error("APIException during streaming: %s", e.message)
                    yield {"event": "error", "data": e.message}
                except Exception as e:
                    logger.error("Unexpected error during streaming: %s", e)
                    yield {"event": "error", "data": "Internal server error"}

            return EventSourceResponse(event_generator())

        logger.info(
            "Pinned-document request for user: %s",
            service.user.email,
        )
        response_text, conversation_id, message_id, metadata = (
            service.generate_response_for_document(
                chat_request.messages,
                document_id=chat_request.document_id,
                title=chat_request.title,
                conversation_id=chat_request.conversation_id,
            )
        )
        return ChatResponse(
            conversation_id=conversation_id,
            message_id=message_id,
            response=response_text,
            metadata=metadata,
        )

    except APIException as e:
        logger.error("APIException: %s", e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/memory")
async def get_profile_memory(service: ChatService = Depends(get_chat_service)):
    try:
        return {
            "personalization_enabled": service.is_personalization_enabled(),
            "items": service.list_profile_memory(),
        }
    except APIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/memory")
async def add_profile_memory(
    payload: ProfileMemoryCreate,
    service: ChatService = Depends(get_chat_service),
):
    try:
        service.add_profile_memory(payload.content, confidence=payload.confidence)
        return {"status": "ok"}
    except APIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/memory")
async def clear_profile_memory(service: ChatService = Depends(get_chat_service)):
    try:
        deleted = service.clear_profile_memory()
        return {"status": "ok", "deleted": deleted}
    except APIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.put("/memory/personalization")
async def set_personalization(
    payload: PersonalizationUpdate,
    service: ChatService = Depends(get_chat_service),
):
    try:
        service.set_personalization(payload.enabled)
        return {"status": "ok", "enabled": payload.enabled}
    except APIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    service: ChatService = Depends(get_chat_service),
):
    try:
        items, total = service.list_conversations(limit=limit, offset=offset)
        return ConversationListResponse(items=items, total=total)
    except APIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    service: ChatService = Depends(get_chat_service),
):
    from uuid import UUID

    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

    try:
        detail = service.get_conversation_detail(conv_uuid)
        return ConversationDetail(**detail)
    except APIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    service: ChatService = Depends(get_chat_service),
):
    from uuid import UUID

    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

    try:
        service.delete_conversation(conv_uuid)
    except APIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.patch("/conversations/{conversation_id}", response_model=ConversationListItem)
async def rename_conversation(
    conversation_id: str,
    payload: ConversationRenameRequest,
    _current_user: dict = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    from uuid import UUID

    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

    try:
        result = service.rename_conversation(conv_uuid, payload.title)
        return ConversationListItem(**result)
    except APIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
