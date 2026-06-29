from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class Message(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message sender (e.g., 'user', 'assistant')",
    )
    content: str = Field(..., description="The content of the message")


class ChatRequest(BaseModel):
    conversation_id: UUID | None = Field(
        None,
        description="Optional conversation ID to continue existing conversation",
    )
    messages: list[Message] = Field(
        ...,
        description="A list of messages in the conversation",
    )
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    conversation_id: UUID = Field(..., description="The ID of the conversation")
    message_id: UUID = Field(..., description="The ID of the assistant's message")
    response: str = Field(..., description="The response from the assistant")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata including citations, retrieval diagnostics, and degraded mode flags",
    )


class ProfileMemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=240)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class PersonalizationUpdate(BaseModel):
    enabled: bool = Field(...)


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    metadata: dict[str, Any] | None = Field(None, validation_alias="message_metadata")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class ConversationListItem(BaseModel):
    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}


class ConversationDetail(BaseModel):
    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    items: list[ConversationListItem]
    total: int


class DocumentIngestRequest(BaseModel):
    legislation_id: UUID = Field(
        ...,
        description="UUID of the legislation record in KMS",
    )
    document_id: UUID = Field(..., description="UUID of the document metadata record")
    file_url: str = Field(
        ...,
        description="Presigned or public URL to download the PDF",
    )
    title: str = Field(..., description="Human-readable title of the document")
    publication_date: str | None = Field(
        None,
        description="ISO date string, e.g. '2024-01-15'",
    )


class IngestResponse(BaseModel):
    status: str = Field(..., description="'accepted' when the job was queued")
    message: str
    legislation_id: UUID
    document_id: UUID


class DocumentChatRequest(BaseModel):
    conversation_id: UUID | None = Field(
        None,
        description="Optional conversation ID to continue existing conversation",
    )
    messages: list[Message] = Field(
        ...,
        description="A list of messages in the conversation",
    )
    stream: bool = Field(False, description="Whether to stream the response")
    legislation_id: str | None = Field(
        None,
        description="Retrieve chunks only from the document with this exact legislation_id",
    )
    document_id: str | None = Field(
        None,
        description="Alias for legislation_id (legacy). Ignored when legislation_id is also provided.",
    )
    title: str | None = Field(
        None,
        description="Retrieve chunks only from the document with this exact title",
    )
    legislation_ids: list[str] | None = Field(
        None,
        description="Retrieve chunks from several documents (by legislation_id) to compare them. Takes precedence over legislation_id.",
    )
    titles: list[str] | None = Field(
        None,
        description="Retrieve chunks from several documents (by title) to compare them. Used when ids are unavailable.",
    )
    domain: str | None = Field(
        "legal",
        description="Exact-match domain filter for pinned-document retrieval. Defaults to 'legal'.",
    )

    @model_validator(mode="after")
    def at_least_one_filter(self) -> "DocumentChatRequest":
        if (
            not self.legislation_id
            and not self.document_id
            and not self.title
            and not self.legislation_ids
            and not self.titles
        ):
            raise ValueError(
                "At least one of legislation_id, document_id, title, legislation_ids, or titles must be provided",
            )
        return self

    @property
    def resolved_legislation_id(self) -> str | None:
        """Return legislation_id if set, otherwise fall back to document_id."""
        return self.legislation_id or self.document_id

    @property
    def is_comparison(self) -> bool:
        """True when more than one document is pinned (comparison mode)."""
        return bool(
            (self.legislation_ids and len(self.legislation_ids) > 1)
            or (self.titles and len(self.titles) > 1),
        )


class ConversationRenameRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="New conversation title",
    )
