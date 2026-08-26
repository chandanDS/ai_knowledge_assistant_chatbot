from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    retriever_available: bool


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str


class ConversationCreate(BaseModel):
    user_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )


class MessageCreate(BaseModel):
    content: str = Field(
        min_length=1,
        max_length=4000,
    )

    model: Literal[
        "Automatic",
        "gpt-4o-mini",
        "gpt-4o",
    ] = Field(
        default="Automatic",
        description=(
            "Use Automatic for policy-based model routing, or select "
            "one of the supported explicit models."
        ),
    )

    temperature: float = Field(
        default=0,
        ge=0,
        le=2,
    )


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ConversationResponse(BaseModel):
    id: UUID
    user_id: str | None
    created_at: datetime
    messages: list[MessageResponse]


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatTurnResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    route: str
    selected_model: str
    follow_up_questions: list[str]
    token_usage: TokenUsage
    documents_retrieved: int
    estimated_cost_usd: float | None = None
    latency_seconds: float
    request_id: UUID
