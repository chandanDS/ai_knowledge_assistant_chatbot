import streamlit as st

from ui.api_client import (
    ChatbotApiClient,
    ChatbotApiError,
)


EXAMPLE_QUESTIONS = [
    "What is RAG and how does it work?",
    "How do I deploy a Streamlit application on AWS?",
    "Explain precision and recall with an example.",
    "What is the difference between Docker and Kubernetes?",
    "Who won the 2023 Cricket World Cup final?",
]


@st.cache_resource
def get_api_client() -> ChatbotApiClient:
    return ChatbotApiClient()


def display_chat_history() -> None:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content", "")

        avatar = "👤" if role == "user" else "🤖"

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)


def render_header() -> None:
    st.title("🤖 Intelligent Knowledge Assistant")
    st.caption(
        "Ask any question and the assistant will "
        "automatically select the appropriate knowledge source."
    )


def select_question(question: str) -> None:
    st.session_state.selected_question = question


def render_welcome() -> None:
    if st.session_state.messages:
        return

    st.markdown("### 👋 Welcome!")
    st.write(
        "Ask a question to get started. The assistant will "
        "choose between the knowledge base, web search, "
        "and general model knowledge."
    )

    st.markdown("### 💡 Try asking")

    for index, question in enumerate(EXAMPLE_QUESTIONS):
        st.button(
            f"👉 {question}",
            key=f"example_{index}",
            use_container_width=True,
            on_click=select_question,
            args=(question,),
        )


def ensure_conversation(
    client: ChatbotApiClient,
) -> str:
    conversation_id = st.session_state.get(
        "conversation_id"
    )

    if conversation_id:
        return conversation_id

    conversation = client.create_conversation(
        user_id=st.session_state.get("user_id")
    )

    conversation_id = conversation["id"]
    st.session_state.conversation_id = conversation_id

    return conversation_id


def update_metrics(result: dict) -> None:
    usage = result.get("token_usage") or {}
    total_tokens = usage.get("total_tokens", 0)

    # The API currently returns aggregate usage rather than
    # separate router and final-model usage.
    st.session_state.token_usage = {
        "router": 0,
        "final_llm": total_tokens,
        "total": total_tokens,
        "router_input": 0,
        "router_output": 0,
        "final_input": usage.get("input_tokens", 0),
        "final_output": usage.get("output_tokens", 0),
    }

    st.session_state.route = result.get(
        "route",
        "GENERAL_LLM",
    )

    st.session_state.selected_model = result.get(
        "selected_model",
        "Unknown",
    )

    request_cost = result.get("estimated_cost_usd")

    if request_cost is not None:
        request_cost = float(request_cost)

        st.session_state.last_estimated_cost_usd = (
            request_cost
        )

        st.session_state.total_estimated_cost_usd = (
            st.session_state.get(
                "total_estimated_cost_usd",
                0.0,
            )
            + request_cost
        )


def save_messages(
    question: str,
    answer: str,
    max_history_messages: int,
) -> None:
    st.session_state.messages.extend(
        [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
    )

    st.session_state.messages = (
        st.session_state.messages[
            -max_history_messages:
        ]
    )


def process_question(
    question: str,
    model: str,
    temperature: float,
    max_history_messages: int,
) -> None:
    client = get_api_client()

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Thinking..."):
            try:
                conversation_id = ensure_conversation(
                    client
                )

                result = client.send_message(
                    conversation_id=conversation_id,
                    content=question,
                    model=model,
                    temperature=temperature,
                )

            except ChatbotApiError as exc:
                st.error(
                    "The chatbot service could not process "
                    "your request."
                )
                st.caption(str(exc))
                return

        assistant_message = (
            result.get("assistant_message") or {}
        )

        answer = assistant_message.get(
            "content",
            "The API returned no answer.",
        )

        st.markdown(answer)

    save_messages(
        question=question,
        answer=answer,
        max_history_messages=max_history_messages,
    )

    update_metrics(result)

    follow_up_questions = result.get(
        "follow_up_questions"
    ) or []

    if follow_up_questions:
        st.markdown(
            "### 💡 Suggested follow-up questions"
        )

        for index, follow_up in enumerate(
            follow_up_questions
        ):
            st.button(
                f"👉 {follow_up}",
                key=(
                    f"followup_"
                    f"{len(st.session_state.messages)}_"
                    f"{index}"
                ),
                use_container_width=True,
                on_click=select_question,
                args=(follow_up,),
            )


def render_chat(
    llm: str,
    temperature: float,
    max_history_messages: int,
) -> None:
    render_header()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "📚 **RAG Knowledge Base**\n\n"
            "Uses your uploaded documents."
        )

    with col2:
        st.info(
            "🌐 **Web Search**\n\n"
            "Uses current external information."
        )

    with col3:
        st.info(
            "🧠 **General LLM**\n\n"
            "Uses general model knowledge."
        )

    render_welcome()
    display_chat_history()

    question = st.chat_input("💬 Ask your question...")

    selected_question = st.session_state.get(
        "selected_question"
    )

    if selected_question:
        question = selected_question
        st.session_state.selected_question = None

    if question:
        process_question(
            question=question,
            model=llm,
            temperature=temperature,
            max_history_messages=max_history_messages,
        )