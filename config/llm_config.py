import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm(
    model_name: str,
    temperature: float,
):
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Please add OPENAI_API_KEY to your .env file."
        )

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=openai_api_key,
    )