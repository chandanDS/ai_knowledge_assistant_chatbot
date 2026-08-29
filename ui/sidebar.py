# ============================================================
# SIDEBAR UI
# ============================================================
#
# Responsible for:
#
#   - Model selection
#   - Temperature
#   - Conversation controls
#   - Current route
#   - Token usage
#   - Session information
#
# ============================================================

import streamlit as st

from auth.authentication import streamlit_login_enabled
from ui.login import logout


# ============================================================
# CSS
# ============================================================

def render_sidebar_css():

    st.markdown(
        """
        <style>

        /* ====================================================
           SIDEBAR
        ==================================================== */

        section[data-testid="stSidebar"] {

            background-color: #111827;

        }


        section[data-testid="stSidebar"] > div {

            padding-top: 0.8rem;

            padding-bottom: 0.8rem;

        }


        /* ====================================================
           SIDEBAR TEXT
        ==================================================== */

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {

            color: #e5e7eb !important;

        }


        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {

            color: #f9fafb !important;

        }


        /* ====================================================
           SELECT BOX
        ==================================================== */

        section[data-testid="stSidebar"]
        div[data-baseweb="select"] > div {

            background-color: #1f2937 !important;

            border: 1px solid #374151 !important;

            color: #f9fafb !important;

            border-radius: 8px;

        }


        section[data-testid="stSidebar"]
        div[data-baseweb="select"] span {

            color: #f9fafb !important;

        }


        /* ====================================================
           SIDEBAR BUTTON
        ==================================================== */

        section[data-testid="stSidebar"]
        .stButton button {

            background-color: #f3f4f6 !important;

            color: #374151 !important;

            border: none;

            border-radius: 8px;

            min-height: 34px;

            font-size: 12px;

        }


        section[data-testid="stSidebar"]
        .stButton button:hover {

            background-color: #ffffff !important;

            color: #111827 !important;

        }


        section[data-testid="stSidebar"]
        .stButton button p {

            color: #374151 !important;

        }


        /* ====================================================
           METRICS
        ==================================================== */

        section[data-testid="stSidebar"]
        [data-testid="stMetric"] {

            background-color: #1f2937 !important;

            border: 1px solid #374151;

            padding: 8px 10px;

            border-radius: 9px;

        }


        section[data-testid="stSidebar"]
        [data-testid="stMetricLabel"] {

            color: #9ca3af !important;

            font-size: 11px;

        }


        section[data-testid="stSidebar"]
        [data-testid="stMetricValue"] {

            color: #f9fafb !important;

            font-size: 18px;

        }


        /* ====================================================
           INFO
        ==================================================== */

        section[data-testid="stSidebar"]
        .stInfo {

            background-color: #1e3a8a !important;

            border: 1px solid #2563eb !important;

            border-radius: 8px;

        }


        /* ====================================================
           SUCCESS
        ==================================================== */

        section[data-testid="stSidebar"]
        .stSuccess {

            background-color: #064e3b !important;

            border: 1px solid #047857 !important;

            border-radius: 8px;

        }


        /* ====================================================
           DIVIDER
        ==================================================== */

        section[data-testid="stSidebar"] hr {

            border-color: #263244 !important;

            margin-top: 8px;

            margin-bottom: 8px;

        }


        /* ====================================================
           EXPANDER
        ==================================================== */

        section[data-testid="stSidebar"]
        details {

            background-color: #1f2937 !important;

            border: 1px solid #374151 !important;

            border-radius: 8px;

        }


        section[data-testid="stSidebar"]
        details summary {

            color: #e5e7eb !important;

        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(
        max_history_messages=6
):
    """
    Render the complete application sidebar.

    Returns
    -------
    tuple
        Selected LLM model and temperature.
    """

    render_sidebar_css()


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        st.markdown(
            "## ⚙️ Assistant Settings"
        )

        st.caption(
            "Configure your AI knowledge assistant"
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        llm = st.selectbox(

            "🤖 OpenAI Model",

            [
                "Automatic",
                "gpt-4o",
                "gpt-4o-mini"
            ],

            index=0
        )

        if llm == "Automatic":
            st.caption(
                "Automatically balances response quality, latency, and cost."
            )


        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        temperature = st.slider(

            "🌡️ Temperature",

            min_value=0.0,

            max_value=1.0,

            value=0.7,

            step=0.1
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        st.success(
            "🟢 AI Assistant Online"
        )


        # ====================================================
        # CONVERSATION
        # ====================================================

        st.divider()

        st.markdown(
            "### 💬 Conversation"
        )


        st.caption(
            f"Stored messages: "
            f"{len(st.session_state.messages)}"
        )


        st.caption(
            f"LLM history limit: "
            f"{max_history_messages}"
        )


        # ----------------------------------------------------
        # CLEAR CONVERSATION
        # ----------------------------------------------------

        if st.button(

            "🗑️ Clear Conversation",

            use_container_width=True,

            key="clear_conversation"

        ):

            st.session_state.messages = []
            st.session_state.conversation_id = None

            st.session_state.route = (
                "GENERAL_LLM"
            )

            st.session_state.selected_model = (
                "Not selected yet"
            )

            st.session_state.selected_question = (
                None
            )

            st.session_state.token_usage = {

                "router": 0,

                "final_llm": 0,

                "total": 0,

                "router_input": 0,

                "router_output": 0,

                "final_input": 0,

                "final_output": 0
            }

            st.session_state.last_estimated_cost_usd = 0.0
            st.session_state.total_estimated_cost_usd = 0.0

            st.rerun()


        # ====================================================
        # CURRENT ROUTE
        # ====================================================

        st.divider()

        st.markdown(
            "### 🧭 Current Route"
        )


        route = (
            st.session_state.route
        )

        selected_model = st.session_state.get(
            "selected_model",
            "Not selected yet"
        )

        st.caption(f"Final model: {selected_model}")


        if route == "RAG_KNOWLEDGE":

            st.success(
                "📚 RAG Knowledge Base"
            )


        elif route == "WEB_SEARCH":

            st.info(
                "🌐 Web Search"
            )


        else:

            st.info(
                "🧠 General LLM Knowledge"
            )


        # ====================================================
        # TOKEN USAGE
        # ====================================================

        st.divider()

        st.markdown(
            "### 📊 Token Usage"
        )


        usage = (
            st.session_state.token_usage
        )


        # ----------------------------------------------------
        # TOTAL / ROUTER
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Total",
                usage.get(
                    "total",
                    0
                )
            )


        with col2:

            st.metric(
                "Router",
                usage.get(
                    "router",
                    0
                )
            )


        # ----------------------------------------------------
        # FINAL LLM
        # ----------------------------------------------------

        st.metric(

            "Final LLM",

            usage.get(
                "final_llm",
                0
            )
        )

        st.metric(
            "Estimated session cost",
            f"${st.session_state.get('total_estimated_cost_usd', 0.0):.6f}"
        )

        st.caption(
            "Last request estimate: "
            f"${st.session_state.get('last_estimated_cost_usd', 0.0):.6f}"
        )


        # ----------------------------------------------------
        # TOKEN DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🔍 Token Details"
        ):

            st.caption(
                "Router Input: "
                f"{usage.get('router_input', 0)}"
            )

            st.caption(
                "Router Output: "
                f"{usage.get('router_output', 0)}"
            )

            st.caption(
                "Final LLM Input: "
                f"{usage.get('final_input', 0)}"
            )

            st.caption(
                "Final LLM Output: "
                f"{usage.get('final_output', 0)}"
            )


        # ====================================================
        # SESSION
        # ====================================================

        st.divider()

        st.markdown(
            "### 🔐 Session"
        )


        user_id = (
            st.session_state.get(
                "user_id",
                "Unknown"
            )
        )


        session_id = (
            st.session_state.get(
                "session_id",
                ""
            )
        )


        login_enabled = streamlit_login_enabled()

        if login_enabled:
            st.caption(f"Signed in as: {user_id}")
        else:
            st.caption("Public access enabled")


        if session_id:

            st.caption(
                "Session: "
                f"{session_id[:12]}..."
            )

        if login_enabled:
            if st.button(
                "↪️ Sign out",
                use_container_width=True,
                key="logout",
            ):
                logout()
    return llm, temperature
