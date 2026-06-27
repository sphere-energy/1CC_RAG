import asyncio
import logging
import threading
from functools import lru_cache
from typing import Any
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Header,
    HTTPException,
    Request,
)
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from src.chat.ingest import DocumentIngestService
from src.chat.llm import BedrockClient
from src.chat.models import DocumentRecord
from src.chat.retriever import QdrantRetriever
from src.chat.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationListItem,
    ConversationListResponse,
    ConversationRenameRequest,
    DocumentChatRequest,
    DocumentIngestRequest,
    IngestResponse,
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

_SENTINEL: Any = object()


async def _iter_sync_gen(sync_gen: Any) -> Any:
    """
    Consume a synchronous generator in a **single dedicated background thread**
    and yield its items asynchronously via an asyncio.Queue.

    Why a single thread matters
    ---------------------------
    * boto3 EventStream reads from an underlying urllib3 HTTP socket.  That
      socket must be read from the same OS thread throughout the stream.
    * SQLAlchemy sessions are not thread-safe across different threads.

    Using ``loop.run_in_executor`` dispatches each ``next()`` call to an
    arbitrary worker thread, breaking both invariants.  This helper avoids
    that by keeping the generator lifecycle confined to one thread while the
    asyncio event loop remains free between yields.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)

    def _producer() -> None:
        try:
            for item in sync_gen:
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
        except BaseException as exc:
            try:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            except Exception:
                pass
        finally:
            try:
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()
            except Exception:
                pass

    threading.Thread(target=_producer, daemon=True).start()

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


def _collaborative_stream_error_message() -> str:
    return (
        "I hit a temporary issue while preparing your answer. "
        "Please try again, and if possible include the document title or a short excerpt so I can help more precisely."
    )


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

                    # Stream response chunks — run each next() in a thread pool
                    # so blocking boto3/DB calls don't freeze the event loop.
                    async for event in _iter_sync_gen(stream_gen):
                        yield event

                except APIException as e:
                    logger.error("APIException during streaming: %s", e.message)
                    yield {
                        "event": "error",
                        "data": _collaborative_stream_error_message(),
                    }
                except Exception as e:
                    logger.error("Unexpected error during streaming: %s", e)
                    yield {
                        "event": "error",
                        "data": _collaborative_stream_error_message(),
                    }

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
                            legislation_id=chat_request.resolved_legislation_id,
                            domain=chat_request.domain,
                            title=chat_request.title,
                            conversation_id=chat_request.conversation_id,
                        )
                    )
                    yield {"event": "conversation_id", "data": str(conversation_id)}
                    async for event in _iter_sync_gen(stream_gen):
                        yield event
                except APIException as e:
                    logger.error(
                        "APIException during pinned-document streaming: %s",
                        e.message,
                        exc_info=True,
                    )
                    yield {
                        "event": "error",
                        "data": _collaborative_stream_error_message(),
                    }
                except Exception as e:
                    logger.error(
                        "Unexpected error during pinned-document streaming: %s",
                        e,
                        exc_info=True,
                    )
                    yield {
                        "event": "error",
                        "data": _collaborative_stream_error_message(),
                    }

            return EventSourceResponse(event_generator())

        logger.info(
            "Pinned-document request for user: %s",
            service.user.email,
        )
        response_text, conversation_id, message_id, metadata = (
            service.generate_response_for_document(
                chat_request.messages,
                legislation_id=chat_request.resolved_legislation_id,
                domain=chat_request.domain,
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


def _verify_internal_key(
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    Dependency that validates the shared secret sent by KMS.

    If ``ingest_internal_api_key`` is empty (e.g. local dev), the check is
    skipped so the endpoint can be called without authentication.
    """
    expected = settings.ingest_internal_api_key
    if not expected:
        logger.warning(
            "INGEST_INTERNAL_API_KEY is not set — ingest endpoint is unprotected",
        )
        return
    if not x_internal_api_key or x_internal_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing internal API key",
        )


@lru_cache
def get_ingest_service_singleton() -> DocumentIngestService:
    settings = get_settings()
    llm_client = get_llm_client_singleton()
    return DocumentIngestService(llm_client, settings)


@router.post(
    "/documents/ingest",
    response_model=IngestResponse,
    status_code=202,
    summary="Trigger asynchronous document ingestion into Qdrant",
)
async def ingest_document(
    ingest_request: DocumentIngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
) -> IngestResponse:
    """
    Called by the KMS Go backend after a successful document upload.

    The heavy work (download → parse → chunk → embed → index) is dispatched
    as a background task so this endpoint returns **202 Accepted** immediately.
    """
    # Persist (upsert) document metadata so retrigger calls can replay ingestion.
    record = db.get(DocumentRecord, ingest_request.document_id)
    if record is None:
        record = DocumentRecord(
            document_id=ingest_request.document_id,
            legislation_id=ingest_request.legislation_id,
            file_url=ingest_request.file_url,
            title=ingest_request.title,
            publication_date=ingest_request.publication_date,
        )
        db.add(record)
    else:
        record.legislation_id = ingest_request.legislation_id
        record.file_url = ingest_request.file_url
        record.title = ingest_request.title
        record.publication_date = ingest_request.publication_date
    db.commit()

    ingest_service = get_ingest_service_singleton()

    def _run() -> None:
        try:
            result = ingest_service.ingest(ingest_request)
            logger.info(
                "Background ingestion finished: legislation_id=%s document_id=%s result=%s",
                ingest_request.legislation_id,
                ingest_request.document_id,
                result,
            )
        except Exception as exc:
            logger.error(
                "Background ingestion failed: legislation_id=%s document_id=%s error=%s",
                ingest_request.legislation_id,
                ingest_request.document_id,
                exc,
                exc_info=True,
            )
            ingest_service._notify_kms(str(ingest_request.document_id), "failed")

    background_tasks.add_task(_run)

    return IngestResponse(
        status="accepted",
        message="Document queued for ingestion",
        legislation_id=ingest_request.legislation_id,
        document_id=ingest_request.document_id,
    )


@router.post(
    "/upload/retrigger/{document_id}",
    response_model=IngestResponse,
    status_code=202,
    summary="Delete existing Qdrant chunks for a document and re-trigger ingestion",
)
async def retrigger_ingest(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
    body: DocumentIngestRequest | None = Body(None),
) -> IngestResponse:
    """
    Called by the KMS Go backend to re-index a document (e.g. after a failed
    ingestion or a document replacement).

    The request body is optional:
    - If provided, the document metadata is saved/updated and used for ingestion.
    - If absent, the metadata stored from the original /documents/ingest call is used.

    Existing Qdrant points are deleted synchronously before the new ingestion
    job is queued, preventing duplicate chunks.
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid document_id UUID") from exc

    if body is not None:
        # Body provided: upsert the record so we always have up-to-date metadata.
        if str(body.document_id) != document_id:
            raise HTTPException(
                status_code=422,
                detail="Path document_id does not match body document_id",
            )
        record = db.get(DocumentRecord, doc_uuid)
        if record is None:
            record = DocumentRecord(
                document_id=body.document_id,
                legislation_id=body.legislation_id,
                file_url=body.file_url,
                title=body.title,
                publication_date=body.publication_date,
            )
            db.add(record)
        else:
            record.legislation_id = body.legislation_id
            record.file_url = body.file_url
            record.title = body.title
            record.publication_date = body.publication_date
        db.commit()
        ingest_request = body
    else:
        # No body: look up previously stored metadata.
        record = db.get(DocumentRecord, doc_uuid)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No ingest record found for document_id={document_id}. "
                    "Send the full document metadata in the request body to retrigger ingestion."
                ),
            )
        ingest_request = DocumentIngestRequest(
            document_id=record.document_id,
            legislation_id=record.legislation_id,
            file_url=record.file_url,
            title=record.title,
            publication_date=record.publication_date,
        )

    ingest_service = get_ingest_service_singleton()

    # Remove stale chunks immediately so searches don't see old content while
    # the background job is running.
    ingest_service.delete_document_chunks(document_id)

    def _run() -> None:
        try:
            result = ingest_service.ingest(ingest_request)
            logger.info(
                "Retrigger ingestion finished: legislation_id=%s document_id=%s result=%s",
                ingest_request.legislation_id,
                ingest_request.document_id,
                result,
            )
        except Exception as exc:
            logger.error(
                "Retrigger ingestion failed: legislation_id=%s document_id=%s error=%s",
                ingest_request.legislation_id,
                ingest_request.document_id,
                exc,
                exc_info=True,
            )
            ingest_service._notify_kms(str(ingest_request.document_id), "failed")

    background_tasks.add_task(_run)

    return IngestResponse(
        status="accepted",
        message="Document re-queued for ingestion",
        legislation_id=ingest_request.legislation_id,
        document_id=ingest_request.document_id,
    )
