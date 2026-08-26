class ConversationNotFoundError(Exception):
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        super().__init__(
            f"Conversation '{conversation_id}' was not found."
        )


class ChatbotUnavailableError(Exception):
    pass
