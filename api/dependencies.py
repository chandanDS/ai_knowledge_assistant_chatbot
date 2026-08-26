from fastapi import Request

from api.repository import InMemoryConversationRepository


_repository = InMemoryConversationRepository()


def get_conversation_repository() -> (
    InMemoryConversationRepository
):
    return _repository


def get_api_retriever(request: Request):
    return getattr(
        request.app.state,
        "retriever",
        None,
    )