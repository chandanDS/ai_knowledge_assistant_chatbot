"""
Authentication and session management.
"""

import uuid
#import streamlit as st


# ============================================================
# DEVELOPMENT USERS
# ============================================================

USERS = {
    "chandan": "1234",
    "admin": "admin123",
    "testuser": "test123"
}


# ============================================================
# LOGIN
# ============================================================

# def login() -> None:

#     st.title("🔐 AI Assistant")

#     st.write(
#         "Please sign in to continue."
#     )

#     username = st.text_input(
#         "Username"
#     )

#     password = st.text_input(
#         "Password",
#         type="password"
#     )

#     if st.button(
#         "🚀 Login",
#         use_container_width=True
#     ):

#         if (
#             username in USERS
#             and USERS[username] == password
#         ):

#             st.session_state.logged_in = True

#             st.session_state.user_id = username

#             st.session_state.session_id = str(
#                 uuid.uuid4()
#             )

#             st.session_state.messages = []

#             st.rerun()

#         else:

#             st.error(
#                 "Invalid username or password."
#             )


# # ============================================================
# # LOGOUT
# # ============================================================

# def logout() -> None:
#     """
#     Clear the current authenticated session.
#     """

#     st.session_state.logged_in = False

#     st.session_state.pop(
#         "user_id",
#         None
#     )

#     st.session_state.pop(
#         "session_id",
#         None
#     )

#     st.session_state.pop(
#         "messages",
#         None
#     )

    # st.rerun()