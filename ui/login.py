# ============================================================
# LOGIN UI
# ============================================================
#
# Responsible only for rendering the Streamlit login page.
#
# Authentication credentials:
#     auth/authentication.py
#
# ============================================================

import uuid

import streamlit as st

from auth.authentication import USERS


# ============================================================
# LOGIN PAGE
# ============================================================

def render_login():
    """
    Render the login page.

    Returns
    -------
    bool
        True if login is successful.
        False otherwise.
    """

    # --------------------------------------------------------
    # Page styling
    # --------------------------------------------------------

    st.html(
        """
        <style>

        /* Main login card */

        .login-container {

            max-width: 450px;

            margin: 80px auto 30px auto;

            padding: 35px;

            border-radius: 16px;

            background: #f8fafc;

            border: 1px solid #e5e7eb;

            box-shadow:
                0px 4px 18px
                rgba(0, 0, 0, 0.08);

        }


        /* Application title */

        .login-title {

            text-align: center;

            font-size: 28px;

            font-weight: 700;

            color: #111827;

            margin-bottom: 8px;

        }


        /* Application subtitle */

        .login-subtitle {

            text-align: center;

            color: #6b7280;

            font-size: 14px;

            margin-bottom: 10px;

        }

        </style>


        <div class="login-container">

            <div class="login-title">

                🤖 AI Knowledge Assistant

            </div>

            <div class="login-subtitle">

                Please sign in to continue

            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # Login form
    # --------------------------------------------------------

    with st.form("login_form"):

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        submitted = st.form_submit_button(
            "🚀 Login",
            use_container_width=True
        )


    # --------------------------------------------------------
    # Validate credentials
    # --------------------------------------------------------

    if submitted:

        username = username.strip()

        # ----------------------------------------------------
        # Check username/password
        # ----------------------------------------------------

        if (
            username in USERS
            and USERS[username] == password
        ):

            # ------------------------------------------------
            # Authentication successful
            # ------------------------------------------------

            st.session_state.logged_in = True

            st.session_state.user_id = username


            # ------------------------------------------------
            # Create NEW conversation session
            # ------------------------------------------------

            st.session_state.session_id = str(
                uuid.uuid4()
            )


            # ------------------------------------------------
            # Start fresh conversation
            # ------------------------------------------------

            st.session_state.messages = []


            # ------------------------------------------------
            # Reset route
            # ------------------------------------------------

            st.session_state.route = (
                "GENERAL_LLM"
            )


            # ------------------------------------------------
            # Reset token usage
            # ------------------------------------------------

            st.session_state.token_usage = {

                "router": 0,

                "final_llm": 0,

                "total": 0,

                "router_input": 0,

                "router_output": 0,

                "final_input": 0,

                "final_output": 0
            }


            # ------------------------------------------------
            # Login success
            # ------------------------------------------------

            st.success(
                "Login successful!"
            )

            st.rerun()


        else:

            st.error(
                "❌ Invalid username or password"
            )


# ============================================================
# LOGOUT
# ============================================================

def logout():
    """
    Clear the authenticated user's session.
    """

    st.session_state.logged_in = False

    st.session_state.user_id = None

    st.session_state.session_id = None

    st.session_state.messages = []

    st.session_state.route = (
        "GENERAL_LLM"
    )

    st.session_state.selected_question = None

    st.session_state.token_usage = {

        "router": 0,

        "final_llm": 0,

        "total": 0,

        "router_input": 0,

        "router_output": 0,

        "final_input": 0,

        "final_output": 0
    }

    st.rerun()