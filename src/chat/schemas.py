from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


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
