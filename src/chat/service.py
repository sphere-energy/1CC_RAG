import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from src.chat.llm import BedrockClient
from src.chat.models import Conversation, User
from src.chat.models import Message as DBMessage
from src.chat.retriever import QdrantRetriever
from src.chat.schemas import Message
from src.core.config import Settings
from src.core.exceptions import APIException, QdrantException

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service class for handling chat interactions using RAG.
    Orchestrates the flow between LLM, Retriever, and Database.
    """

    def __init__(
        self,
        llm_client: BedrockClient,
        retriever: QdrantRetriever,
        db: Session,
        user_claims: dict,
        settings: Settings,
    ):
        """
        Initialize the ChatService.

        Args:
            llm_client (BedrockClient): Client for LLM operations.
            retriever (QdrantRetriever): Client for retrieval operations.
            db (Session): Database session.
            user_claims (dict): User claims from Cognito JWT.
        """
        self.llm_client = llm_client
        self.retriever = retriever
        self.db = db
        self.user_claims = user_claims
        self.settings = settings
        self.user = self._get_or_create_user()

    def _get_or_create_control_conversation(self) -> Conversation:
        existing = (
            self.db.query(Conversation)
            .filter(
                Conversation.user_id == self.user.id,
                Conversation.title == "__memory_controls__",
            )
            .first()
        )
        if existing:
            return existing

        conv = Conversation(user_id=self.user.id, title="__memory_controls__")
        self.db.add(conv)
        self.db.commit()
        self.db.refresh(conv)
        return conv

    def is_personalization_enabled(self) -> bool:
        control = self._get_or_create_control_conversation()
        latest_policy = (
            self.db.query(DBMessage)
            .filter(
                DBMessage.conversation_id == control.id,
                DBMessage.role == "system",
            )
            .order_by(DBMessage.created_at.desc())
            .first()
        )
        if not latest_policy or not latest_policy.message_metadata:
            return True
        return bool(latest_policy.message_metadata.get("personalization_enabled", True))

    def set_personalization(self, enabled: bool) -> None:
        control = self._get_or_create_control_conversation()
        message = DBMessage(
            conversation_id=control.id,
            role="system",
            content="Personalization setting updated",
            message_metadata={
                "memory_type": "policy",
                "personalization_enabled": enabled,
            },
        )
        self.db.add(message)
        self.db.commit()

    def list_profile_memory(self) -> list[dict[str, Any]]:
        conversations = (
            self.db.query(Conversation.id)
            .filter(Conversation.user_id == self.user.id)
            .all()
        )
        conversation_ids = [row[0] for row in conversations]
        if not conversation_ids:
            return []

        records = (
            self.db.query(DBMessage)
            .filter(
                DBMessage.conversation_id.in_(conversation_ids),
                DBMessage.role == "system",
            )
            .order_by(DBMessage.created_at.desc())
            .all()
        )
        output: list[dict[str, Any]] = []
        for record in records:
            metadata = record.message_metadata or {}
            if metadata.get("memory_type") != "profile":
                continue
            output.append(
                {
                    "content": record.content,
                    "confidence": metadata.get("confidence", 0.7),
                    "source": metadata.get("source", "user"),
                    "created_at": str(record.created_at),
                    "expires_at": metadata.get("expires_at"),
                },
            )
        return output

    def add_profile_memory(self, content: str, confidence: float = 0.7) -> None:
        control = self._get_or_create_control_conversation()
        record = DBMessage(
            conversation_id=control.id,
            role="system",
            content=content.strip()[:240],
            message_metadata={
                "memory_type": "profile",
                "source": "user",
                "confidence": max(0.0, min(1.0, confidence)),
                "created_at": "now",
                "expires_at": None,
            },
        )
        self.db.add(record)
        self.db.commit()

    def clear_profile_memory(self) -> int:
        control = self._get_or_create_control_conversation()
        records = (
            self.db.query(DBMessage)
            .filter(DBMessage.conversation_id == control.id, DBMessage.role == "system")
            .all()
        )
        deleted = 0
        for record in records:
            metadata = record.message_metadata or {}
            if metadata.get("memory_type") == "profile":
                self.db.delete(record)
                deleted += 1
        self.db.commit()
        return deleted

    def _get_or_create_user(self) -> User:
        """Get or create user from Cognito claims."""
        cognito_sub = self.user_claims.get("sub")
        if not cognito_sub:
            raise APIException(
                message="Missing required subject claim",
                status_code=401,
                error_type="authentication_error",
            )

        user = self.db.query(User).filter(User.cognito_sub == cognito_sub).first()

        if not user:
            user = User(
                cognito_sub=cognito_sub,
                email=self.user_claims.get("email", ""),
                username=self.user_claims.get("cognito:username"),
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            logger.info("Created new user: %s", user.email)

        return user

    def _resolve_conversation(self, conversation_id: UUID | None) -> Conversation:
        """Resolve an existing conversation with ownership validation or create a new one."""
        if conversation_id is None:
            conv = Conversation(user_id=self.user.id)
            self.db.add(conv)
            self.db.commit()
            self.db.refresh(conv)
            logger.info("Created new conversation: %s", conv.id)
            return conv

        conv = (
            self.db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )
        if conv is None:
            raise APIException(
                message="Conversation not found",
                status_code=404,
                error_type="not_found",
            )

        if conv.user_id != self.user.id:
            raise APIException(
                message="You are not authorized to access this conversation",
                status_code=403,
                error_type="authorization_error",
            )

        return conv

    def delete_conversation(self, conversation_id: UUID) -> None:
        """Hard-delete a conversation and all its messages (cascade)."""
        conv = (
            self.db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )
        if conv is None:
            raise APIException(
                message="Conversation not found",
                status_code=404,
                error_type="not_found",
            )
        if conv.user_id != self.user.id:
            raise APIException(
                message="You are not authorized to delete this conversation",
                status_code=403,
                error_type="authorization_error",
            )
        self.db.delete(conv)
        self.db.commit()
        logger.info(
            "Deleted conversation %s for user %s",
            conversation_id,
            self.user.id,
        )

    def rename_conversation(self, conversation_id: UUID, title: str) -> dict:
        """Rename a conversation. Returns updated conversation data."""
        conv = (
            self.db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )
        if conv is None:
            raise APIException(
                message="Conversation not found",
                status_code=404,
                error_type="not_found",
            )
        if conv.user_id != self.user.id:
            raise APIException(
                message="You are not authorized to rename this conversation",
                status_code=403,
                error_type="authorization_error",
            )
        conv.title = title.strip()[:120]
        self.db.commit()
        self.db.refresh(conv)
        logger.info("Renamed conversation %s to '%s'", conversation_id, conv.title)
        return {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
        }

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List conversations for the current user, newest first."""
        from sqlalchemy import func

        base_query = self.db.query(Conversation).filter(
            Conversation.user_id == self.user.id,
        )
        total = base_query.count()

        conversations = (
            base_query.order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = []
        for conv in conversations:
            msg_count = (
                self.db.query(func.count(DBMessage.id))
                .filter(DBMessage.conversation_id == conv.id)
                .scalar()
            )
            items.append(
                {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "message_count": msg_count or 0,
                },
            )

        return items, total

    def get_conversation_detail(self, conversation_id: UUID) -> dict:
        """Get a single conversation with all its messages."""
        conv = self._resolve_conversation(conversation_id)
        return {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "messages": conv.messages,  # loaded via relationship, ordered by created_at
        }

    def _extract_profile_memories(self, messages: list[Message]) -> list[str]:
        profile_memories: list[str] = []
        pattern = re.compile(
            r"\b(i prefer|my preference|please always|remember that)\b",
            re.IGNORECASE,
        )
        for msg in messages:
            if msg.role == "user" and pattern.search(msg.content):
                profile_memories.append(msg.content.strip()[:240])
        return profile_memories

    def _build_history_summary(self, messages: list[Message]) -> str:
        if len(messages) <= self.settings.summary_trigger_messages:
            return ""

        salient_keywords = {
            "must",
            "required",
            "deadline",
            "fine",
            "obligation",
            "article",
            "compliance",
            "prohibited",
            "penalty",
            "exemption",
            "procedure",
            "policy",
            "process",
            "guideline",
            "responsible",
            "approval",
            "documentation",
            "standard",
        }
        selected: list[str] = []
        for msg in messages[:-1]:
            lowered = msg.content.lower()
            if any(keyword in lowered for keyword in salient_keywords):
                selected.append(f"{msg.role}: {msg.content.strip()[:180]}")
            if len(selected) >= 8:
                break

        return "\n".join(selected)

    def _classify_intent(self, user_query: str) -> str:
        query = user_query.lower()
        if any(token in query for token in ["compare", "difference between", "vs "]):
            return "procedural_guidance"
        if any(
            token in query for token in ["follow up", "as above", "continue", "clarify"]
        ):
            return "follow_up_clarification"
        if any(
            token in query
            for token in [
                "recipe",
                "movie",
                "football",
                "travel plan",
                "song lyrics",
                "sports score",
                "weather",
                "stock price",
            ]
        ):
            return "out_of_domain"
        if any(
            token in query
            for token in [
                "legal",
                "regulation",
                "compliance",
                "legislation",
                "directive",
                "law",
                "obligation",
                "penalty",
                "fine",
                "deadline",
                "prohibited",
                "exemption",
                "article",
                "statutory",
                "decree",
            ]
        ):
            return "legal_lookup"
        return "document_lookup"

    def _sanitize_user_query(self, user_query: str) -> str:
        blocked_patterns = [
            r"ignore\s+previous\s+instructions",
            r"reveal\s+system\s+prompt",
            r"you\s+are\s+no\s+longer",
            r"bypass\s+safety",
        ]
        sanitized = user_query
        for pattern in blocked_patterns:
            sanitized = re.sub(pattern, "[blocked]", sanitized, flags=re.IGNORECASE)
        return sanitized

    def _validate_output(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return (
                "Here is the best guidance I can provide right now based on the available context. "
                "If you share the exact document title or a short excerpt, I can refine this further."
            )
        return cleaned

    @staticmethod
    def _not_indexed_message() -> str:
        """Human-friendly message when a pinned document has no Qdrant chunks."""
        return (
            "This document has **not been indexed** into the knowledge base yet, "
            "so I cannot answer questions about its content.\n\n"
            "**To enable document chat:**\n"
            "1. Open the document in the Library\n"
            "2. Click **Edit** and re-save it with **Include in Vector DB** enabled, "
            "or ask an administrator to re-trigger indexing\n\n"
            "If the document was uploaded recently, indexing may still be in progress — "
            "check the indexing status badge on the library card and try again in a moment."
        )

    def _append_uncertainty_if_needed(
        self,
        response_text: str,
        has_sources: bool,
    ) -> str:
        if has_sources:
            return response_text
        return (
            f"{response_text}\n\n"
            "To improve precision, share the exact document title, section, or a short excerpt and I will refine the answer."
        )

    def _resolve_no_context_reason(
        self,
        has_sources: bool,
        retrieval_error: Exception | None,
        pinned_document: bool = False,
    ) -> str | None:
        if has_sources:
            return None
        if retrieval_error is not None:
            return "retrieval_degraded"
        if pinned_document:
            return "pinned_filter_no_hits"
        return "no_relevant_hits"

    def _truncate_prompt(self, prompt: str) -> str:
        if len(prompt) <= self.settings.max_prompt_characters:
            return prompt
        return prompt[: self.settings.max_prompt_characters]

    def _build_workflow_state(self, user_query: str, intent: str) -> dict[str, Any]:
        if intent != "procedural_guidance":
            return {"workflow_enabled": False, "steps": []}

        steps = [
            {"step": "classify_request", "status": "completed"},
            {"step": "retrieve_sources", "status": "completed"},
            {"step": "build_actor_checklist", "status": "completed"},
            {"step": "compose_cited_response", "status": "completed"},
        ]
        return {
            "workflow_enabled": True,
            "workflow_type": "deterministic_procedural_guidance",
            "input": user_query[:200],
            "steps": steps,
            "tool_contract_version": "1.0",
        }

    def _infer_document_profile(
        self,
        context_docs: list[dict[str, Any]],
        intent: str,
    ) -> str:
        """Infer dominant document profile from retrieved context and intent."""
        if intent in ("legal_lookup", "procedural_guidance"):
            return "legal_regulatory"

        if not context_docs:
            return "employee_general"

        internal_keywords = {
            "policy",
            "procedure",
            "handbook",
            "hr",
            "finance",
            "payroll",
            "expense",
            "vacation",
            "onboarding",
            "internal",
        }
        legal_keywords = {
            "law",
            "directive",
            "regulation",
            "article",
            "decree",
            "compliance",
            "legislation",
        }

        internal_votes = 0
        legal_votes = 0

        for doc in context_docs[:12]:
            title = str(doc.get("title") or "").lower()
            document_id = str(doc.get("document_id") or "").lower()
            source_kind = str(doc.get("source_kind") or "").lower()
            source_origin = str(doc.get("source_origin") or "").lower()
            bag = " ".join([title, document_id, source_kind, source_origin])

            if any(token in bag for token in legal_keywords):
                legal_votes += 1
            if any(token in bag for token in internal_keywords):
                internal_votes += 1

            if source_kind == "company":
                internal_votes += 1
            if source_kind == "user":
                internal_votes += 0

        if legal_votes >= 2 and legal_votes >= internal_votes:
            return "legal_regulatory"
        if internal_votes >= 2:
            return "internal_company"
        return "employee_general"

    def _build_document_profile_guidance(self, document_profile: str) -> str:
        if document_profile == "legal_regulatory":
            return (
                "Treat sources as legal or regulatory material. Prioritize legal precision, "
                "article-level grounding, temporal validity, and actor-specific obligations."
            )
        if document_profile == "internal_company":
            return (
                "Treat sources as internal company documentation (for example HR, finance, "
                "operations, and internal policies). Prioritize procedure clarity, ownership, "
                "required steps, deadlines, and exceptions."
            )
        return (
            "Treat sources as employee-provided or mixed-format documentation. Infer structure "
            "from the content, summarize clearly, and answer directly even when style or format "
            "is non-standard. Ask one concise clarification only when it materially improves accuracy."
        )

    def generate_response(
        self,
        messages: list[Message],
        conversation_id: UUID = None,
    ) -> tuple[str, UUID, UUID, dict[str, Any]]:
        """
        Generate a response using RAG with conversation history and save to database.

        Args:
            messages (List[Message]): Conversation history.
            conversation_id (UUID): Optional conversation ID.

        Returns:
            tuple[str, UUID, UUID]: (response_text, conversation_id, message_id)

        Raises:
            BedrockException: If LLM generation fails.
            QdrantException: If retrieval fails.
        """
        logger.info("Processing chat request with %d messages", len(messages))

        conversation = self._resolve_conversation(conversation_id)

        # Save user message
        user_message_obj = DBMessage(
            conversation_id=conversation.id,
            role="user",
            content=messages[-1].content,
        )
        self.db.add(user_message_obj)
        self.db.commit()

        # Extract query from last user message
        user_query = messages[-1].content

        intent = self._classify_intent(user_query)

        retrieval_error = None
        context_docs: list[dict[str, Any]] = []
        retrieval_diagnostics: dict[str, Any] = {
            "retrieved_k": 0,
            "rerank_scores": [],
            "citation_coverage": 0.0,
        }

        try:
            query_embedding = self.llm_client.generate_embedding(user_query)
            context_docs, retrieval_diagnostics = self.retriever.retrieve(
                query_embedding,
                user_query=user_query,
            )
        except QdrantException as exc:
            retrieval_error = exc
            logger.warning("Retrieval degraded for conversation %s", conversation.id)

        # 3. Format context and prompt
        formatted_context = self._format_context(context_docs)
        document_profile = self._infer_document_profile(context_docs, intent)
        prompt = self._construct_prompt(
            messages,
            formatted_context,
            intent,
            document_profile,
        )
        prompt = self._truncate_prompt(prompt)

        # 4. Generate answer
        response_text = self.llm_client.generate_text(prompt)
        response_text = self._validate_output(response_text)
        response_text = self._append_uncertainty_if_needed(
            response_text,
            has_sources=bool(context_docs),
        )

        sources = [
            {
                "title": doc.get("title"),
                "document_id": doc.get("document_id"),
                "score": doc.get("score"),
                "chunk_id": doc.get("chunk_id"),
                "source_kind": doc.get("source_kind"),
                "source_origin": doc.get("source_origin"),
            }
            for doc in context_docs[:5]
        ]
        retrieval_diagnostics["citation_coverage"] = 1.0 if sources else 0.0

        profile_memories = self._extract_profile_memories(messages)
        if self.is_personalization_enabled():
            for memory in profile_memories:
                self.add_profile_memory(memory)
            persisted_profile_memory = [
                item["content"] for item in self.list_profile_memory()[:8]
            ]
        else:
            persisted_profile_memory = []
        history_summary = self._build_history_summary(messages)

        metadata = {
            "sources": sources,
            "model": self.llm_client.text_model_id,
            "intent": intent,
            "document_profile": document_profile,
            "degraded_mode": retrieval_error is not None,
            "collaboration_mode_applied": "balanced_answer_first",
            "no_context_reason": self._resolve_no_context_reason(
                has_sources=bool(sources),
                retrieval_error=retrieval_error,
            ),
            "retrieval": retrieval_diagnostics,
            "memory": {
                "session_memory_messages": min(
                    len(messages),
                    self.settings.max_history_messages,
                ),
                "episodic_summary": history_summary,
                "profile_memory": profile_memories,
                "persisted_profile_memory": persisted_profile_memory,
            },
            "workflow": self._build_workflow_state(user_query, intent),
        }

        # 5. Save assistant message with metadata
        assistant_message_obj = DBMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=response_text,
            message_metadata=metadata,
        )
        self.db.add(assistant_message_obj)

        # Update conversation title from first message if not set
        if not conversation.title and len(messages) == 1:
            conversation.title = user_query[:100]  # First 100 chars

        # Always update timestamp so conversations sort by last activity
        conversation.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(assistant_message_obj)

        logger.info("Response generated and saved to database")
        return response_text, conversation.id, assistant_message_obj.id, metadata

    def generate_response_stream(
        self,
        messages: list[Message],
        conversation_id: UUID = None,
    ) -> tuple[Iterator[dict[str, str]], UUID]:
        """
        Generate a streaming response using RAG.

        Args:
            messages (List[Message]): Conversation history.
            conversation_id (UUID): Optional conversation ID.

        Yields:
            str: Text chunks as they are generated.

        Returns:
            tuple[Iterator[str], UUID]: (text_chunks, conversation_id)

        Raises:
            BedrockException: If LLM generation fails.
            QdrantException: If retrieval fails.
        """
        logger.info("Processing streaming chat request with %d messages", len(messages))

        conversation = self._resolve_conversation(conversation_id)

        # Save user message
        user_message_obj = DBMessage(
            conversation_id=conversation.id,
            role="user",
            content=messages[-1].content,
        )
        self.db.add(user_message_obj)
        self.db.commit()

        # Extract query from last user message
        user_query = messages[-1].content

        intent = self._classify_intent(user_query)

        retrieval_error = None
        context_docs: list[dict[str, Any]] = []
        retrieval_diagnostics: dict[str, Any] = {
            "retrieved_k": 0,
            "rerank_scores": [],
            "citation_coverage": 0.0,
        }

        try:
            query_embedding = self.llm_client.generate_embedding(user_query)
            context_docs, retrieval_diagnostics = self.retriever.retrieve(
                query_embedding,
                user_query=user_query,
            )
        except QdrantException as exc:
            retrieval_error = exc
            logger.warning(
                "Retrieval degraded for streaming conversation %s",
                conversation.id,
            )

        # 3. Format prompt
        formatted_context = self._format_context(context_docs)
        document_profile = self._infer_document_profile(context_docs, intent)
        prompt = self._construct_prompt(
            messages,
            formatted_context,
            intent,
            document_profile,
        )
        prompt = self._truncate_prompt(prompt)

        # 4. Generate streaming answer and accumulate
        accumulated_response = []

        sources = [
            {
                "title": doc.get("title"),
                "document_id": doc.get("document_id"),
                "score": doc.get("score"),
                "chunk_id": doc.get("chunk_id"),
                "source_kind": doc.get("source_kind"),
                "source_origin": doc.get("source_origin"),
            }
            for doc in context_docs[:5]
        ]
        retrieval_diagnostics["citation_coverage"] = 1.0 if sources else 0.0

        profile_memories = self._extract_profile_memories(messages)
        if self.is_personalization_enabled():
            for memory in profile_memories:
                self.add_profile_memory(memory)
            persisted_profile_memory = [
                item["content"] for item in self.list_profile_memory()[:8]
            ]
        else:
            persisted_profile_memory = []
        history_summary = self._build_history_summary(messages)

        def stream_and_save() -> Iterator[dict[str, str]]:
            yield {"event": "progress", "data": "retrieval_complete"}
            for chunk in self.llm_client.generate_text_stream(prompt):
                accumulated_response.append(chunk)
                yield {"event": "data", "data": chunk}

            # After streaming completes, save to database
            full_response = "".join(accumulated_response)
            full_response = self._validate_output(full_response)
            full_response = self._append_uncertainty_if_needed(
                full_response,
                has_sources=bool(context_docs),
            )

            metadata = {
                "sources": sources,
                "model": self.llm_client.text_model_id,
                "streaming": True,
                "intent": intent,
                "document_profile": document_profile,
                "degraded_mode": retrieval_error is not None,
                "collaboration_mode_applied": "balanced_answer_first",
                "no_context_reason": self._resolve_no_context_reason(
                    has_sources=bool(sources),
                    retrieval_error=retrieval_error,
                ),
                "retrieval": retrieval_diagnostics,
                "memory": {
                    "session_memory_messages": min(
                        len(messages),
                        self.settings.max_history_messages,
                    ),
                    "episodic_summary": history_summary,
                    "profile_memory": profile_memories,
                    "persisted_profile_memory": persisted_profile_memory,
                },
                "workflow": self._build_workflow_state(user_query, intent),
            }

            assistant_message_obj = DBMessage(
                conversation_id=conversation.id,
                role="assistant",
                content=full_response,
                message_metadata=metadata,
            )
            self.db.add(assistant_message_obj)

            # Update title if needed
            if not conversation.title and len(messages) == 1:
                conversation.title = user_query[:100]

            # Always update timestamp so conversations sort by last activity
            conversation.updated_at = datetime.utcnow()
            self.db.commit()
            yield {
                "event": "metadata",
                "data": json.dumps(
                    {
                        "intent": intent,
                        "document_profile": document_profile,
                        "degraded_mode": retrieval_error is not None,
                        "no_context_reason": self._resolve_no_context_reason(
                            has_sources=bool(sources),
                            retrieval_error=retrieval_error,
                        ),
                        "sources": sources,
                    },
                ),
            }
            logger.info("Streaming response saved to database")

        return stream_and_save(), conversation.id

    def generate_response_for_document(
        self,
        messages: list[Message],
        document_id: str | None = None,
        title: str | None = None,
        conversation_id: UUID | None = None,
    ) -> tuple[str, UUID, UUID, dict[str, Any]]:
        """
        Generate a RAG response using ONLY the chunks from the pinned document.
        Skips vector similarity search entirely — retrieval is filter-based.

        Args:
            messages: Conversation history.
            document_id: Exact document_id to filter chunks by.
            title: Exact title to filter chunks by.
            conversation_id: Optional conversation ID to continue.

        Returns:
            tuple of (response_text, conversation_id, message_id, metadata).
        """
        logger.info(
            "Processing pinned-document chat request (document_id=%s, title=%s)",
            document_id,
            title,
        )

        conversation = self._resolve_conversation(conversation_id)

        user_message_obj = DBMessage(
            conversation_id=conversation.id,
            role="user",
            content=messages[-1].content,
        )
        self.db.add(user_message_obj)
        self.db.commit()

        user_query = messages[-1].content
        intent = self._classify_intent(user_query)
        # Never short-circuit for out_of_domain — user explicitly pinned a document

        context_docs: list[dict[str, Any]] = []
        retrieval_diagnostics: dict[str, Any] = {
            "retrieved_k": 0,
            "rerank_scores": [],
            "citation_coverage": 0.0,
            "pinned_document": True,
        }
        retrieval_error = None

        try:
            context_docs, retrieval_diagnostics = self.retriever.retrieve_by_document(
                document_id=document_id,
                title=title,
            )
        except QdrantException as exc:
            retrieval_error = exc
            logger.warning(
                "Pinned-document retrieval degraded for conversation %s",
                conversation.id,
            )

        if not context_docs and retrieval_error is None:
            logger.warning(
                "Pinned-document: no Qdrant chunks found (document_id=%s title=%s) — "
                "returning not-indexed message without calling LLM",
                document_id,
                title,
            )
            response_text = self._not_indexed_message()
            not_indexed_metadata: dict[str, Any] = {
                "sources": [],
                "model": self.llm_client.text_model_id,
                "intent": "not_indexed",
                "document_profile": "unknown",
                "degraded_mode": False,
                "no_context_reason": "pinned_filter_no_hits",
                "retrieval": retrieval_diagnostics,
            }
            assistant_message_obj = DBMessage(
                conversation_id=conversation.id,
                role="assistant",
                content=response_text,
                message_metadata=not_indexed_metadata,
            )
            self.db.add(assistant_message_obj)
            if not conversation.title and len(messages) == 1:
                conversation.title = user_query[:100]
            conversation.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(assistant_message_obj)
            return (
                response_text,
                conversation.id,
                assistant_message_obj.id,
                not_indexed_metadata,
            )

        formatted_context = self._format_context(context_docs)
        document_profile = self._infer_document_profile(context_docs, intent)
        prompt = self._construct_prompt(
            messages,
            formatted_context,
            intent,
            document_profile,
        )
        prompt = self._truncate_prompt(prompt)

        response_text = self.llm_client.generate_text(prompt)
        response_text = self._validate_output(response_text)
        response_text = self._append_uncertainty_if_needed(
            response_text,
            has_sources=bool(context_docs),
        )

        sources = [
            {
                "title": doc.get("title"),
                "document_id": doc.get("document_id"),
                "score": doc.get("score"),
                "chunk_id": doc.get("chunk_id"),
                "source_kind": doc.get("source_kind"),
                "source_origin": doc.get("source_origin"),
            }
            for doc in context_docs[:5]
        ]
        retrieval_diagnostics["citation_coverage"] = 1.0 if sources else 0.0

        profile_memories = self._extract_profile_memories(messages)
        if self.is_personalization_enabled():
            for memory in profile_memories:
                self.add_profile_memory(memory)
            persisted_profile_memory = [
                item["content"] for item in self.list_profile_memory()[:8]
            ]
        else:
            persisted_profile_memory = []
        history_summary = self._build_history_summary(messages)

        metadata = {
            "sources": sources,
            "model": self.llm_client.text_model_id,
            "intent": intent,
            "document_profile": document_profile,
            "degraded_mode": retrieval_error is not None,
            "collaboration_mode_applied": "balanced_answer_first",
            "no_context_reason": self._resolve_no_context_reason(
                has_sources=bool(sources),
                retrieval_error=retrieval_error,
                pinned_document=True,
            ),
            "retrieval": retrieval_diagnostics,
            "memory": {
                "session_memory_messages": min(
                    len(messages),
                    self.settings.max_history_messages,
                ),
                "episodic_summary": history_summary,
                "profile_memory": profile_memories,
                "persisted_profile_memory": persisted_profile_memory,
            },
            "workflow": self._build_workflow_state(user_query, intent),
        }

        assistant_message_obj = DBMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=response_text,
            message_metadata=metadata,
        )
        self.db.add(assistant_message_obj)

        if not conversation.title and len(messages) == 1:
            conversation.title = user_query[:100]

        conversation.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(assistant_message_obj)

        logger.info("Pinned-document response generated and saved to database")
        return response_text, conversation.id, assistant_message_obj.id, metadata

    def generate_response_stream_for_document(
        self,
        messages: list[Message],
        document_id: str | None = None,
        title: str | None = None,
        conversation_id: UUID | None = None,
    ) -> tuple[Iterator[dict[str, str]], UUID]:
        """
        Streaming variant of generate_response_for_document.
        Uses ONLY chunks from the pinned document — no vector similarity search.
        """
        logger.info(
            "Processing streaming pinned-document chat request (document_id=%s, title=%s)",
            document_id,
            title,
        )

        conversation = self._resolve_conversation(conversation_id)

        user_message_obj = DBMessage(
            conversation_id=conversation.id,
            role="user",
            content=messages[-1].content,
        )
        self.db.add(user_message_obj)
        self.db.commit()

        user_query = messages[-1].content
        intent = self._classify_intent(user_query)

        context_docs: list[dict[str, Any]] = []
        retrieval_diagnostics: dict[str, Any] = {
            "retrieved_k": 0,
            "rerank_scores": [],
            "citation_coverage": 0.0,
            "pinned_document": True,
        }
        retrieval_error = None

        try:
            context_docs, retrieval_diagnostics = self.retriever.retrieve_by_document(
                document_id=document_id,
                title=title,
            )
        except QdrantException as exc:
            retrieval_error = exc
            logger.warning(
                "Pinned-document retrieval degraded for streaming conversation %s",
                conversation.id,
            )

        if not context_docs and retrieval_error is None:
            logger.warning(
                "Streaming pinned-document: no Qdrant chunks (document_id=%s title=%s) — "
                "returning not-indexed message without calling LLM",
                document_id,
                title,
            )
            not_indexed_msg = self._not_indexed_message()
            not_indexed_meta: dict[str, Any] = {
                "sources": [],
                "model": self.llm_client.text_model_id,
                "intent": "not_indexed",
                "document_profile": "unknown",
                "degraded_mode": False,
                "no_context_reason": "pinned_filter_no_hits",
                "retrieval": retrieval_diagnostics,
            }

            def _not_indexed_stream() -> Iterator[dict[str, str]]:
                yield {"event": "progress", "data": "retrieval_complete"}
                yield {"event": "data", "data": not_indexed_msg}
                assistant_message_obj = DBMessage(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=not_indexed_msg,
                    message_metadata=not_indexed_meta,
                )
                self.db.add(assistant_message_obj)
                if not conversation.title and len(messages) == 1:
                    conversation.title = user_query[:100]
                conversation.updated_at = datetime.utcnow()
                self.db.commit()
                yield {
                    "event": "metadata",
                    "data": json.dumps(
                        {
                            "intent": "not_indexed",
                            "document_profile": "unknown",
                            "degraded_mode": False,
                            "no_context_reason": "pinned_filter_no_hits",
                            "sources": [],
                        }
                    ),
                }

            return _not_indexed_stream(), conversation.id

        formatted_context = self._format_context(context_docs)
        document_profile = self._infer_document_profile(context_docs, intent)
        prompt = self._construct_prompt(
            messages,
            formatted_context,
            intent,
            document_profile,
        )
        prompt = self._truncate_prompt(prompt)

        accumulated_response: list[str] = []

        sources = [
            {
                "title": doc.get("title"),
                "document_id": doc.get("document_id"),
                "score": doc.get("score"),
                "chunk_id": doc.get("chunk_id"),
                "source_kind": doc.get("source_kind"),
                "source_origin": doc.get("source_origin"),
            }
            for doc in context_docs[:5]
        ]
        retrieval_diagnostics["citation_coverage"] = 1.0 if sources else 0.0

        profile_memories = self._extract_profile_memories(messages)
        if self.is_personalization_enabled():
            for memory in profile_memories:
                self.add_profile_memory(memory)
            persisted_profile_memory = [
                item["content"] for item in self.list_profile_memory()[:8]
            ]
        else:
            persisted_profile_memory = []
        history_summary = self._build_history_summary(messages)

        def stream_and_save() -> Iterator[dict[str, str]]:
            yield {"event": "progress", "data": "retrieval_complete"}
            for chunk in self.llm_client.generate_text_stream(prompt):
                accumulated_response.append(chunk)
                yield {"event": "data", "data": chunk}

            full_response = "".join(accumulated_response)
            full_response = self._validate_output(full_response)
            full_response = self._append_uncertainty_if_needed(
                full_response,
                has_sources=bool(context_docs),
            )

            metadata = {
                "sources": sources,
                "model": self.llm_client.text_model_id,
                "streaming": True,
                "intent": intent,
                "document_profile": document_profile,
                "degraded_mode": retrieval_error is not None,
                "collaboration_mode_applied": "balanced_answer_first",
                "no_context_reason": self._resolve_no_context_reason(
                    has_sources=bool(sources),
                    retrieval_error=retrieval_error,
                    pinned_document=True,
                ),
                "retrieval": retrieval_diagnostics,
                "memory": {
                    "session_memory_messages": min(
                        len(messages),
                        self.settings.max_history_messages,
                    ),
                    "episodic_summary": history_summary,
                    "profile_memory": profile_memories,
                    "persisted_profile_memory": persisted_profile_memory,
                },
                "workflow": self._build_workflow_state(user_query, intent),
            }

            assistant_message_obj = DBMessage(
                conversation_id=conversation.id,
                role="assistant",
                content=full_response,
                message_metadata=metadata,
            )
            self.db.add(assistant_message_obj)

            if not conversation.title and len(messages) == 1:
                conversation.title = user_query[:100]

            conversation.updated_at = datetime.utcnow()
            self.db.commit()
            yield {
                "event": "metadata",
                "data": json.dumps(
                    {
                        "intent": intent,
                        "document_profile": document_profile,
                        "degraded_mode": retrieval_error is not None,
                        "no_context_reason": self._resolve_no_context_reason(
                            has_sources=bool(sources),
                            retrieval_error=retrieval_error,
                            pinned_document=True,
                        ),
                        "sources": sources,
                    },
                ),
            }
            logger.info("Streaming pinned-document response saved to database")

        return stream_and_save(), conversation.id

    def _format_context(self, docs: list[dict]) -> str:
        """Format retrieved documents into a string."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(
                f"Source {i}:\nTitle: {doc.get('title', 'Unknown')}\nContent: {doc.get('text', '')}\n",
            )
        return "\n".join(formatted)

    def _construct_prompt(
        self,
        messages: list[Message],
        context: str,
        intent: str,
        document_profile: str,
    ) -> str:
        """Construct the prompt for the LLM with conversation history."""
        bounded_messages = messages[-self.settings.max_history_messages :]
        history_lines = []
        for msg in bounded_messages[:-1]:
            history_lines.append(f"{msg.role.capitalize()}: {msg.content}")
        history_text = "\n".join(history_lines)

        # Get current query
        current_query = self._sanitize_user_query(messages[-1].content)
        summary = self._build_history_summary(bounded_messages)

        is_legal_intent = intent in ("legal_lookup", "procedural_guidance")

        intent_instructions: dict[str, str] = {
            "legal_lookup": "Prioritize precise legal grounding and source-backed obligations.",
            "follow_up_clarification": "Resolve ambiguity from prior turns and explicitly state assumptions.",
            "procedural_guidance": "Provide step-by-step compliance actions and clearly name responsible actors.",
            "document_lookup": "Provide a clear, informative answer based on the company documentation sources.",
            "out_of_domain": "Answer with the best available guidance from indexed sources first, including user-uploaded documents, and then ask one focused follow-up only if it improves precision.",
        }
        intent_instruction = intent_instructions.get(
            intent,
            intent_instructions["document_lookup"],
        )
        document_profile_guidance = self._build_document_profile_guidance(
            document_profile,
        )

        if is_legal_intent:
            response_structure = """
# RESPONSE STRUCTURE (MANDATORY)

For each question, structure your response as follows:

1. **One-Sentence Answer**: Provide a direct, actionable answer upfront
2. **Detailed Explanation**: Break down the legal basis and implications
3. **Actor-Specific Obligations**: List requirements by stakeholder (producer, PRO, authority, distributor, etc.)
4. **Practical Implications**: Explain what this means in practice"""

            citation_and_accuracy = """
# LEGAL CITATION REQUIREMENTS (CRITICAL)

Every legal reference MUST include a metadata block in this format:

**[Legal Source Metadata]**
- Official Citation: [e.g., BEK nr 986 af 20/06/2025]
- Document Title: [Full official name]
- Relevant Section: [Article/Paragraph/Annex number]
- Jurisdiction: [Country/EU]
- Status: [In force since DD/MM/YYYY | Repealed by X | Superseded by Y]
- Transitional Period: [If applicable: dates and conditions]

# ACCURACY & PRECISION STANDARDS

1. **Source-Based Answers**: Base ALL legal interpretations exclusively on the provided sources below
2. **Answer-First Collaboration**: Always provide the best available answer first. If precision is limited, ask one specific follow-up request to refine the answer.
3. **Chronological Accuracy**:
   - Always verify if legislation is current or superseded
   - Explicitly note when laws have been repealed or amended
   - Identify applicable transitional periods
4. **Internal Consistency**: Never contradict yourself about the same paragraph/article within a response
5. **Completeness**: Explain concepts thoroughly without assuming prior legal knowledge"""
        else:
            response_structure = """
# RESPONSE STRUCTURE (MANDATORY)

For each question, structure your response as follows:

1. **Direct Answer**: Provide a clear, direct answer upfront
2. **Context & Details**: Expand with relevant context from the documentation
3. **Practical Takeaways**: Summarise the key points for the reader"""

            citation_and_accuracy = """
# ACCURACY & PRECISION STANDARDS

1. **Source-Based Answers**: Base ALL answers exclusively on the provided documentation sources below
2. **Answer-First Collaboration**: Always provide the best available answer first. If precision is limited, ask one specific follow-up request to refine the answer.
3. **Internal Consistency**: Never contradict yourself within a response
4. **Completeness**: Explain concepts clearly without assuming prior knowledge"""

        return f"""
You are the 1CC & Techprotect knowledge assistant. You help employees and consultants work with multiple document classes: legal and legislation sources, internal company documentation, and employee-provided documents in any format.

# SOURCE SCOPE POLICY (CRITICAL)

    - Treat the currently indexed and uploaded documents as the active working corpus for this conversation.
    - User-uploaded documents are first-class sources and must be handled as valid reference material, regardless of format or origin.
    - External references or research papers are valid sources when present in the working corpus.
    - Never critique the corpus composition and never classify documents as "wrong" for the conversation.
    - Never start with rejection-style wording. Start with a useful answer, then optionally ask one targeted follow-up to improve precision.
    - Keep the tone formal, practical, and company-ready.

    # PROHIBITED RESPONSE STYLE (MANDATORY)

    - Do not say that the provided sources are "not company documentation".
    - Do not say that the documents are mismatched, irrelevant by category, or unusable because they are academic.
    - Do not block the user before attempting an answer from the available sources.
    - If context is weak, provide the best answer possible first, then ask one precise follow-up (for example a section, title, or excerpt) to improve accuracy.

# DOCUMENT PROFILE ADAPTATION (MANDATORY)

- Active document profile: {document_profile}
- Adaptation guidance: {document_profile_guidance}
- If profile signals legal/regulatory content, follow strict legal structure and citation metadata requirements.
- If profile signals internal company documentation, focus on operational clarity and actionable next steps.
- If profile signals employee/general documentation, focus on understanding intent and explaining content clearly regardless of format.

Intent route: {intent}
Instruction: {intent_instruction}
{response_structure}
{citation_and_accuracy}

# FORMATTING GUIDELINES (IMPORTANT FOR READABILITY)

- **Always use proper markdown with line breaks** - never output inline lists
- Use clear headings for each sub-question (## Sub-question 1, ## Sub-question 2, etc.)
- Use bullet points for lists of obligations, requirements, or multiple actors
- For numbered lists, put each item on its own line:
  1. First item
  2. Second item
  3. Third item
- Bold key terms and important concepts
- Use tables when comparing requirements across actors or jurisdictions
- Separate paragraphs with blank lines for readability

---

## CONVERSATION HISTORY
{history_text}

## EPISODIC SUMMARY
{summary}

---

## DOCUMENTATION SOURCES PROVIDED
{context}

---

## CURRENT QUESTION
{current_query}

---

**Your Response:**
"""
