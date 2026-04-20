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
    metadata: dict[str, Any] | None = Field(None, alias="message_metadata")
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
    document_id: str | None = Field(
        None,
        description="Retrieve chunks only from the document with this exact document_id",
    )
    title: str | None = Field(
        None,
        description="Retrieve chunks only from the document with this exact title",
    )

    @model_validator(mode="after")
    def at_least_one_filter(self) -> "DocumentChatRequest":
        if not self.document_id and not self.title:
            raise ValueError("At least one of document_id or title must be provided")
        return self


class ConversationRenameRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="New conversation title",
    )
