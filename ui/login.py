"""Streamlit login page for the demo chatbot."""

import logging
import uuid

import streamlit as st

from auth.authentication import authenticate_credentials
from logging_service.operational_logger import fingerprint_text, log_event


def _reset_conversation_state() -> None:
    """Start a clean conversation while retaining authentication state."""

    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.route = "GENERAL_LLM"
    st.session_state.selected_model = "Not selected yet"
    st.session_state.selected_question = None
    st.session_state.token_usage = {
        "router": 0,
        "final_llm": 0,
        "total": 0,
        "router_input": 0,
        "router_output": 0,
        "final_input": 0,
        "final_output": 0,
    }
    st.session_state.last_estimated_cost_usd = 0.0
    st.session_state.total_estimated_cost_usd = 0.0


def render_login() -> None:
    """Render the login form and establish an authenticated session."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none; }
        .block-container { max-width: 520px; padding-top: 4.5rem; }
        .login-hero {
            padding: 1.8rem 1.5rem 1.2rem;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            background: linear-gradient(145deg, #f8fafc, #eef2ff);
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
            text-align: center;
            margin-bottom: 1.2rem;
        }
        .login-icon { font-size: 2.6rem; }
        .login-title { color: #0f172a; font-size: 1.75rem; font-weight: 750; }
        .login-subtitle { color: #64748b; margin-top: .35rem; }
        </style>
        <div class="login-hero">
            <div class="login-icon">🤖</div>
            <div class="login-title">Intelligent Knowledge Assistant</div>
            <div class="login-subtitle">Sign in to start a secure conversation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Username", placeholder="Enter your username", autocomplete="username"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            autocomplete="current-password",
        )
        submitted = st.form_submit_button(
            "Sign in", type="primary", use_container_width=True
        )

    st.caption("Demo access: `admin` / `admin123`")

    if not submitted:
        return

    normalized_username = username.strip()
    user_fingerprint = fingerprint_text(normalized_username or "empty-username")
    if authenticate_credentials(normalized_username, password):
        st.session_state.logged_in = True
        st.session_state.user_id = normalized_username
        _reset_conversation_state()
        log_event(
            "login_succeeded",
            user_fingerprint=user_fingerprint,
            session_fingerprint=fingerprint_text(st.session_state.session_id),
        )
        st.rerun()

    log_event(
        "login_failed",
        level=logging.WARNING,
        user_fingerprint=user_fingerprint,
        reason="invalid_credentials",
    )
    st.error("The username or password is incorrect. Please try again.")


def logout() -> None:
    """Clear authenticated and conversation state, then show login again."""
    user_id = str(st.session_state.get("user_id") or "unknown")
    session_id = str(st.session_state.get("session_id") or "unknown")
    log_event(
        "logout_completed",
        user_fingerprint=fingerprint_text(user_id),
        session_fingerprint=fingerprint_text(session_id),
    )
    st.session_state.clear()
    st.rerun()
