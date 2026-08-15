from chatbot.response_generator import generate_response
# ============================================================
# AI KNOWLEDGE ASSISTANT
# APPLICATION ENTRY POINT
# ============================================================
#
# This file intentionally contains NO chatbot business logic
# and NO detailed Streamlit UI code.
#
# Responsibilities:
#
#   1. Load environment configuration
#   2. Configure Streamlit
#   3. Initialize session state
#   4. Load shared application resources
#   5. Start the Streamlit UI
#
# ------------------------------------------------------------
# Architecture
# ------------------------------------------------------------
#
# app.py
#    |
#    +----> ui/streamlit_app.py
#    |
#    +----> chatbot/
#    |          |
#    |          +---- router.py
#    |          +---- response_generator.py
#    |          +---- prompts.py
#    |          +---- schemas.py
#    |
#    +----> rag/
#    |
#    +----> web/
#    |
#    +----> logging_service/
#
# ============================================================


# ============================================================
# 1. STANDARD LIBRARY
# ============================================================

import uuid


# ============================================================
# 2. THIRD-PARTY LIBRARIES
# ============================================================

import streamlit as st

from dotenv import load_dotenv


# ============================================================
# 3. APPLICATION IMPORTS
# ============================================================

from ui.streamlit_app import render_application

from rag.retriever import get_retriever


# ============================================================
# 4. LOAD ENVIRONMENT VARIABLES
# ============================================================
#
# Loads:
#
#   OPENAI_API_KEY
#   TAVILY_API_KEY
#   LANGCHAIN_API_KEY
#   LANGCHAIN_PROJECT
#   etc.
#
# ============================================================

load_dotenv()


# ============================================================
# 5. STREAMLIT PAGE CONFIGURATION
# ============================================================
#
# This must be executed before other Streamlit UI commands.
#
# ============================================================

st.set_page_config(

    page_title=(
        "Intelligent Knowledge Assistant"
    ),

    page_icon="🤖",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ============================================================
# 6. APPLICATION CONFIGURATION
# ============================================================
#
# Central place for application-level settings.
#
# These values can later be moved to:
#
#   config.py
#
# or environment variables for cloud deployment.
#
# ============================================================

MAX_HISTORY_MESSAGES = 6


# ============================================================
# 7. SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state():
    """
    Initialize application-level Streamlit session state.

    Streamlit reruns the Python script on user interaction,
    therefore values that need to survive reruns must be stored
    in st.session_state.
    """

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    if "logged_in" not in st.session_state:

        st.session_state.logged_in = False


    if "user_id" not in st.session_state:

        st.session_state.user_id = None


    # --------------------------------------------------------
    # Unique conversation session
    # --------------------------------------------------------
    #
    # A new UUID represents one chatbot conversation session.
    #
    # Example:
    #
    # user:
    #     chandan
    #
    # session:
    #     7d9c7e2a-....
    #
    # This same session_id is used by the JSON logger.
    #
    # --------------------------------------------------------

    if "session_id" not in st.session_state:

        st.session_state.session_id = None


    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []


    # --------------------------------------------------------
    # Current route
    # --------------------------------------------------------

    if "route" not in st.session_state:

        st.session_state.route = (
            "GENERAL_LLM"
        )


    # --------------------------------------------------------
    # Token usage
    # --------------------------------------------------------

    if "token_usage" not in st.session_state:

        st.session_state.token_usage = {

            "router": 0,

            "final_llm": 0,

            "total": 0,

            "router_input": 0,

            "router_output": 0,

            "final_input": 0,

            "final_output": 0
        }


# ============================================================
# 8. INITIALIZE SESSION STATE
# ============================================================

initialize_session_state()


# ============================================================
# 9. LOAD SHARED RESOURCES
# ============================================================
#
# The FAISS/vector-store retriever is a shared resource.
#
# st.cache_resource ensures that it is not rebuilt every time
# Streamlit reruns the application.
#
# ============================================================

@st.cache_resource
def load_retriever():

    return get_retriever()


# ============================================================
# 10. LOAD RETRIEVER
# ============================================================

retriever = load_retriever()


# ============================================================
# 11. START STREAMLIT APPLICATION
# ============================================================
#
# All actual UI rendering happens inside:
#
#     ui/streamlit_app.py
#
# app.py simply passes the shared application resources.
#
# ============================================================

render_application(

    retriever=retriever,

    max_history_messages=(
        MAX_HISTORY_MESSAGES
    )
)