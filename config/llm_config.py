# ============================================================
# LLM CONFIGURATION
# ============================================================
#
# This module is responsible only for creating the LLM.
#
# The model name and temperature are selected by the user
# from the Streamlit UI and passed to get_llm().
#
# Example:
#
#     llm = get_llm(
#         model_name="gpt-4o",
#         temperature=0.7
#     )
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# OPENAI API KEY
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)


# ============================================================
# VALIDATE API KEY
# ============================================================

if not OPENAI_API_KEY:

    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Please add OPENAI_API_KEY to your .env file."
    )


# ============================================================
# CREATE LLM
# ============================================================

def get_llm(
    model_name: str,
    temperature: float
):
    """
    Create and return an OpenAI LLM.

    Parameters
    ----------
    model_name : str
        Model selected by the user from Streamlit UI.

        Example:
            gpt-4o
            gpt-4-turbo
            gpt-4

    temperature : float
        Temperature selected by the user from Streamlit UI.

    Returns
    -------
    ChatOpenAI
        Configured LangChain ChatOpenAI instance.
    """

    llm = ChatOpenAI(

        model=model_name,

        temperature=temperature,

        api_key=OPENAI_API_KEY

    )

    return llm