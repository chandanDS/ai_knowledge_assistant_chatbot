import os
from uuid import UUID

from openai import APIConnectionError, APIError

from api.exceptions import ChatbotUnavailableError
from api.repository import InMemoryConversationRepository
from api.schemas import (
    ChatTurnResponse,
    MessageCreate,
    MessageResponse,
    TokenUsage,
)
from chatbot.response_generator import generate_response
from logging_service.json_logger import log_interaction_json


class ConversationService:
    def __init__(
        self,
        repository: InMemoryConversationRepository,
        retriever,
    ):
        self.repository = repository
        self.retriever = retriever

    def create_message(
        self,
        conversation_id: UUID,
        request: MessageCreate,
        request_id: UUID,
    ) -> ChatTurnResponse:
        conversation = self.repository.get(
            conversation_id
        )

        history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in conversation["messages"]
        ]

        model = (
            request.model
            or os.getenv(
                "CHATBOT_DEFAULT_MODEL",
                "Automatic",
            )
        )

        max_history_messages = int(
            os.getenv(
                "CHATBOT_MAX_HISTORY_MESSAGES",
                "6",
            )
        )

        security_principal = (
            conversation["user_id"]
            or str(conversation_id)
        )

        try:
            (
                answer,
                follow_up_questions,
                route,
                documents,
                router_tokens,
                final_tokens,
                total_usage,
                latency,
            ) = generate_response(
                question=request.content,
                llm=model,
                temperature=request.temperature,
                chat_history=history,
                retriever=self.retriever,
                max_messages=max_history_messages,
                security_principal=security_principal,
                request_id=str(request_id),
            )

        except (APIConnectionError, APIError, ValueError) as exc:
            raise ChatbotUnavailableError(
                str(exc)
            ) from exc

        user_message = self.repository.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.content,
        )

        assistant_message = (
            self.repository.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
            )
        )

        selected_model = total_usage.get(
            "selected_model",
            model,
        )

        # Preserve the existing JSON interaction log.
        log_interaction_json(
            session_id=str(conversation_id),
            user_id=conversation["user_id"] or "anonymous",
            query=request.content,
            route=route,
            model=selected_model,
            response=answer,
            documents_retrieved=len(documents),
            router_input_tokens=router_tokens.get(
                "input_tokens",
                0,
            ),
            router_output_tokens=router_tokens.get(
                "output_tokens",
                0,
            ),
            final_input_tokens=final_tokens.get(
                "input_tokens",
                0,
            ),
            final_output_tokens=final_tokens.get(
                "output_tokens",
                0,
            ),
            total_tokens=total_usage.get(
                "total_tokens",
                0,
            ),
            latency_seconds=latency,
            request_id=str(request_id),
        )

        return ChatTurnResponse(
            user_message=MessageResponse(
                **user_message
            ),
            assistant_message=MessageResponse(
                **assistant_message
            ),
            route=route,
            selected_model=selected_model,
            follow_up_questions=follow_up_questions,
            token_usage=TokenUsage(
                input_tokens=total_usage.get(
                    "input_tokens",
                    0,
                ),
                output_tokens=total_usage.get(
                    "output_tokens",
                    0,
                ),
                total_tokens=total_usage.get(
                    "total_tokens",
                    0,
                ),
            ),
            documents_retrieved=len(documents),
            estimated_cost_usd=total_usage.get(
                "estimated_cost_usd"
            ),
            latency_seconds=latency,
            request_id=request_id,
        )
