
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Request,
    Response,
    status,
)

from api.dependencies import (
    get_api_retriever,
    get_conversation_repository,
)
from api.repository import (
    InMemoryConversationRepository,
)
from api.schemas import (
    ChatTurnResponse,
    ConversationCreate,
    ConversationResponse,
    MessageCreate,
    MessageResponse,
)
from api.service import ConversationService


router = APIRouter()


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a conversation",
)
def create_conversation(
    payload: ConversationCreate,
    response: Response,
    repository: InMemoryConversationRepository = Depends(
        get_conversation_repository
    ),
) -> ConversationResponse:
    conversation = repository.create(
        user_id=payload.user_id
    )

    response.headers["Location"] = (
        f"/api/v1/conversations/"
        f"{conversation['id']}"
    )

    return ConversationResponse(
        **conversation
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Retrieve a conversation",
)
def get_conversation(
    conversation_id: UUID,
    repository: InMemoryConversationRepository = Depends(
        get_conversation_repository
    ),
) -> ConversationResponse:
    conversation = repository.get(
        conversation_id
    )

    return ConversationResponse(
        **conversation
    )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
def delete_conversation(
    conversation_id: UUID,
    repository: InMemoryConversationRepository = Depends(
        get_conversation_repository
    ),
) -> Response:
    repository.delete(conversation_id)

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
    summary="List conversation messages",
)
def list_messages(
    conversation_id: UUID,
    repository: InMemoryConversationRepository = Depends(
        get_conversation_repository
    ),
) -> list[MessageResponse]:
    messages = repository.list_messages(
        conversation_id
    )

    return [
        MessageResponse(**message)
        for message in messages
    ]


@router.post(
    "/{conversation_id}/messages",
    response_model=ChatTurnResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a message and chatbot response",
)
def create_message(
    conversation_id: UUID,
    payload: MessageCreate,
    request: Request,
    repository: InMemoryConversationRepository = Depends(
        get_conversation_repository
    ),
    retriever=Depends(get_api_retriever),
) -> ChatTurnResponse:
    request_id = UUID(
        request.state.request_id
    )

    service = ConversationService(
        repository=repository,
        retriever=retriever,
    )

    return service.create_message(
        conversation_id=conversation_id,
        request=payload,
        request_id=request_id,
    )