# ============================================================
# CHAT UI
# ============================================================
#
# Responsible for:
#
#   1. Displaying conversation history
#   2. Accepting user questions
#   3. Calling response_generator.py
#   4. Displaying responses
#   5. Updating session history
#   6. Updating token statistics
#   7. Writing interaction logs
#
# ============================================================

import streamlit as st


from chatbot.response_generator import (
    generate_response
)


from logging_service.json_logger import (
    log_interaction_json
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

EXAMPLE_QUESTIONS = [

    "What is RAG and how does it work?",

    "How do I deploy a Streamlit application on AWS?",

    "Explain precision and recall with an example.",

    "What is the difference between Docker and Kubernetes?",

    "Who won the 2023 Cricket World Cup final?"
]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

def display_chat_history():
    """
    Display all messages currently stored in session state.
    """

    for message in (
        st.session_state.messages
    ):

        role = message.get(
            "role"
        )

        content = message.get(
            "content",
            ""
        )


        if role == "user":

            with st.chat_message(
                "user",
                avatar="👤"
            ):

                st.markdown(
                    content
                )


        elif role == "assistant":

            with st.chat_message(
                "assistant",
                avatar="🤖"
            ):

                st.markdown(
                    content
                )


# ============================================================
# WELCOME SCREEN
# ============================================================

def render_welcome():

    if len(
        st.session_state.messages
    ) != 0:

        return


    st.markdown(
        "### 👋 Welcome!"
    )


    st.write(
        "Ask a question to get started. "
        "The assistant will automatically decide "
        "whether your question should use the "
        "knowledge base, web search, or general "
        "LLM knowledge."
    )


    st.markdown(
        "### 💡 Try asking"
    )


    for i, question in enumerate(
        EXAMPLE_QUESTIONS
    ):

        if st.button(

            f"👉 {question}",

            use_container_width=True,

            key=f"example_{i}"

        ):

            st.session_state.selected_question = (
                question
            )

            st.rerun()


# ============================================================
# HEADER
# ============================================================

def render_header():
    """
    Render the chatbot header.

    User identity, session ID and logout controls are
    intentionally hidden from the UI.

    user_id and session_id remain in session state
    for backend logging.
    """

    st.title(
        "🤖 Intelligent Knowledge Assistant"
    )

    st.caption(
        "Ask any question and the assistant will "
        "automatically select the appropriate "
        "knowledge source."
    )
# ============================================================
# TOKEN USAGE UPDATE
# ============================================================

def update_token_usage(
        router_tokens,
        final_tokens,
        total_usage
        ):
        
        router_tokens = router_tokens or {}
        final_tokens = final_tokens or {}
        total_usage = total_usage or {}
        st.session_state.token_usage["router"] = (
            router_tokens.get("total_tokens", 0)
    )

        st.session_state.token_usage["final_llm"] = (
            final_tokens.get("total_tokens", 0)
    )

        st.session_state.token_usage["total"] = (
            total_usage.get("total_tokens", 0)
    )

        st.session_state.token_usage["router_input"] = (
            router_tokens.get("input_tokens", 0)
    )

        st.session_state.token_usage["router_output"] = (
            router_tokens.get("output_tokens", 0)
    )

        st.session_state.token_usage["final_input"] = (
            final_tokens.get("input_tokens", 0)
    )

        st.session_state.token_usage["final_output"] = (
            final_tokens.get("output_tokens", 0)
    )

# ============================================================
# SAVE CHAT HISTORY
# ============================================================

def save_chat_history(

    question,

    response,

    max_history_messages

):
    """
    Add the latest user/assistant interaction and keep
    only the configured number of messages.

    Example:

        max_history_messages = 6

    Stored:

        User 1
        AI 1
        User 2
        AI 2
        User 3
        AI 3

    Therefore 6 messages = 3 complete turns.
    """

    st.session_state.messages.append(

        {
            "role": "user",

            "content": question
        }
    )


    st.session_state.messages.append(

        {
            "role": "assistant",

            "content": response
        }
    )


    # --------------------------------------------------------
    # Keep only latest N messages
    # --------------------------------------------------------

    st.session_state.messages = (

        st.session_state.messages[
            -max_history_messages:
        ]

    )


# ============================================================
# LOG INTERACTION
# ============================================================

def save_interaction_log(

    question,

    response,

    route,

    docs,

    llm,

    router_tokens,

    final_tokens,

    total_usage,

    latency

):
    """
    Write one chatbot interaction to the JSON log.

    Logging failure must NEVER break the chatbot.
    """

    try:

        log_interaction_json(

            session_id=(
                st.session_state.session_id
            ),

            user_id=(
                st.session_state.user_id
            ),

            query=question,

            route=route,

            model=llm,

            response=response,

            documents_retrieved=len(
                docs
            ),

            router_input_tokens=(
                router_tokens.get(
                    "input_tokens",
                    0
                )
            ),

            router_output_tokens=(
                router_tokens.get(
                    "output_tokens",
                    0
                )
            ),

            final_input_tokens=(
                final_tokens.get(
                    "input_tokens",
                    0
                )
            ),

            final_output_tokens=(
                final_tokens.get(
                    "output_tokens",
                    0
                )
            ),

            total_tokens=(
                total_usage.get(
                    "total_tokens",
                    0
                )
            ),

            latency_seconds=latency
        )


    except Exception as exc:

        print(
            "WARNING: Unable to write "
            f"interaction log: {exc}"
        )

def select_followup(question):
    st.session_state.selected_question = question
# ============================================================
# PROCESS QUESTION
# ============================================================

def select_followup(question):
    """
    Store the selected follow-up question.

    The question will be picked up by render_chat()
    and processed exactly like a typed question.
    """
    st.session_state.selected_question = question


def process_question(
    question,
    llm,
    temperature,
    retriever,
    max_history_messages
):
    """
    Process one user question.
    """

    # ========================================================
    # USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user",
        avatar="👤"
    ):
        st.markdown(question)

    # ========================================================
    # AI RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "🤔 Thinking..."
        ):

            try:

                (
                    response,
                    follow_up_questions,
                    route,
                    docs,
                    router_tokens,
                    final_tokens,
                    total_usage,
                    latency
                ) = generate_response(

                    question=question,

                    llm=llm,

                    temperature=temperature,

                    chat_history=(
                        st.session_state.messages
                    ),

                    retriever=retriever,

                    max_messages=(
                        max_history_messages
                    )
                )

            except Exception as exc:

                st.error(
                    "Sorry, I was unable to "
                    "process your question."
                )

                st.exception(exc)

                return

        # ----------------------------------------------------
        # DISPLAY RESPONSE
        # ----------------------------------------------------

        st.markdown(response)

    # ========================================================
    # NORMALIZE TOKEN DATA
    # ========================================================

    router_tokens = (
        router_tokens or {}
    )

    final_tokens = (
        final_tokens or {}
    )

    total_usage = (
        total_usage or {}
    )

    # ========================================================
    # UPDATE ROUTE
    # ========================================================

    st.session_state.route = route

    # ========================================================
    # UPDATE TOKEN USAGE
    # ========================================================

    update_token_usage(

        router_tokens,

        final_tokens,

        total_usage

    )

    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    save_chat_history(

        question,

        response,

        max_history_messages

    )

    # ========================================================
    # SAVE JSON LOG
    # ========================================================

    save_interaction_log(

        question=question,

        response=response,

        route=route,

        docs=docs,

        llm=llm,

        router_tokens=router_tokens,

        final_tokens=final_tokens,

        total_usage=total_usage,

        latency=latency

    )

    # ========================================================
    # FOLLOW-UP QUESTIONS
    # ========================================================

    if follow_up_questions:

        st.markdown(
            "### 💡 Suggested follow-up questions"
        )

        for i, follow_up in enumerate(
            follow_up_questions
        ):

            st.button(

                f"👉 {follow_up}",

                key=(
                    f"followup_"
                    f"{len(st.session_state.messages)}_"
                    f"{i}"
                ),

                use_container_width=True,

                on_click=select_followup,

                args=(follow_up,)

            )

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

def render_chat(

    llm,

    temperature,

    retriever,

    max_history_messages

):
    """
    Render the complete chatbot interface.
    """

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    render_header()


    # --------------------------------------------------------
    # Capability cards
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Welcome
    # --------------------------------------------------------

    render_welcome()


    # --------------------------------------------------------
    # Existing history
    # --------------------------------------------------------

    display_chat_history()


    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    question = st.chat_input(
        "💬 Ask your question..."
    )


    # --------------------------------------------------------
    # Example / follow-up question
    # --------------------------------------------------------

    if st.session_state.get(
        "selected_question"
    ):

        question = (
            st.session_state.selected_question
        )

        st.session_state.selected_question = None


    # --------------------------------------------------------
    # Process question
    # --------------------------------------------------------

    if question:

        process_question(

            question=question,

            llm=llm,

            temperature=temperature,

            retriever=retriever,

            max_history_messages=(
                max_history_messages
            )
        )