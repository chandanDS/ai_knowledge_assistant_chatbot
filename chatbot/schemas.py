from typing import Literal

from pydantic import BaseModel, Field


class KnowledgeRoute(BaseModel):

    route: Literal[
        "RAG_KNOWLEDGE",
        "WEB_SEARCH",
        "GENERAL_LLM"
    ]


class ChatResponse(BaseModel):

    answer: str = Field(
        description="The answer to the user's question."
    )

    follow_up_questions: list[str] = Field(
        description="Exactly 3 relevant follow-up questions."
    )