# ============================================================
# STREAMLIT APPLICATION UI
# ============================================================
#
# Main presentation-layer orchestrator.
#
# Responsibilities:
#
#   1. Authentication check
#   2. Login rendering
#   3. Sidebar rendering
#   4. Chat rendering
#
# It deliberately does NOT contain:
#
#   - LLM logic
#   - RAG logic
#   - Web search logic
#   - Prompt logic
#   - Routing logic
#   - Logging implementation
#
# ============================================================

import streamlit as st


# from ui.login import (
#     render_login
# )


from ui.sidebar import (
    render_sidebar
)


from ui.chat import (
    render_chat
)


# ============================================================
# APPLICATION CSS
# ============================================================

def render_global_css():

    st.markdown(
        """
        <style>

        /* ====================================================
           MAIN CONTENT
        ==================================================== */

        .main {

            padding-top: 1rem;

        }


        /* ====================================================
           USER HEADER
        ==================================================== */

        .user-header {

            display: flex;

            justify-content: flex-end;

            align-items: center;

            gap: 10px;

        }


        /* ====================================================
           FOOTER
        ==================================================== */

        .footer {

            text-align: center;

            color: #9ca3af;

            font-size: 12px;

            padding: 20px 0;

            margin-top: 30px;

        }

        </style>
        """,

        unsafe_allow_html=True
    )


# ============================================================
# APPLICATION FOOTER
# ============================================================

def render_footer():

    st.markdown(
    """
    <div style="
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding: 20px 0 10px 0;
    ">
        Powered by LangChain & OpenAI
        &nbsp; | &nbsp;
        RAG + Web Search + General LLM
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN UI
# ============================================================

def render_application(
    max_history_messages=6,
):
    """
    Main Streamlit UI entry point.

    Parameters
    ----------
    retriever:
        Initialized RAG retriever.

    max_history_messages:
        Maximum conversation messages supplied to the
        chatbot.
    """

    # ========================================================
    # GLOBAL CSS
    # ========================================================

    render_global_css()


    # ========================================================
    # AUTHENTICATION CHECK
    # ========================================================

    # if not st.session_state.get(
    #     "logged_in",
    #     False
    # ):

    #     render_login()

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Do not render the chatbot when the user is not
        # authenticated.
        # ----------------------------------------------------

        #st.stop()


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    # if not st.session_state.get(
    #     "user_id"
    # ):

    #     st.error(
    #         "User session is invalid. "
    #         "Please log in again."
    #     )

    #     st.stop()


    # if not st.session_state.get(
    #     "session_id"
    # ):

    #     st.error(
    #         "Conversation session is invalid. "
    #         "Please log in again."
    #     )

    #     st.stop()


    # ========================================================
    # SIDEBAR
    # ========================================================

    llm, temperature = render_sidebar(

        max_history_messages=(
            max_history_messages
        )

    )


    # ========================================================
    # CHAT
    # ========================================================

    render_chat(
        llm=llm,
        temperature=temperature,
        max_history_messages=(
            max_history_messages
        )

    )


    # ========================================================
    # FOOTER
    # ========================================================

    render_footer()