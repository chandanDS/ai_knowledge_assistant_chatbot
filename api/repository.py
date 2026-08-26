from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from uuid import UUID, uuid4

from api.exceptions import ConversationNotFoundError


class InMemoryConversationRepository:
    """
    Temporary repository for local development.

    Conversation data is lost whenever FastAPI restarts.
    Replace this with a shared database before running
    multiple Kubernetes replicas.
    """

    def __init__(self):
        self._conversations: dict[UUID, dict] = {}
        self._lock = Lock()

    def create(self, user_id: str | None) -> dict:
        conversation_id = uuid4()

        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "messages": [],
        }

        with self._lock:
            self._conversations[conversation_id] = conversation

        return deepcopy(conversation)

    def get(self, conversation_id: UUID) -> dict:
        with self._lock:
            conversation = self._conversations.get(conversation_id)

            if conversation is None:
                raise ConversationNotFoundError(
                    str(conversation_id)
                )

            return deepcopy(conversation)

    def delete(self, conversation_id: UUID) -> None:
        with self._lock:
            if conversation_id not in self._conversations:
                raise ConversationNotFoundError(
                    str(conversation_id)
                )

            del self._conversations[conversation_id]

    def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
    ) -> dict:
        message = {
            "id": uuid4(),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc),
        }

        with self._lock:
            conversation = self._conversations.get(conversation_id)

            if conversation is None:
                raise ConversationNotFoundError(
                    str(conversation_id)
                )

            conversation["messages"].append(message)

        return deepcopy(message)

    def list_messages(
        self,
        conversation_id: UUID,
    ) -> list[dict]:
        conversation = self.get(conversation_id)
        return conversation["messages"]
