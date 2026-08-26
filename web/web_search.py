"""
External web-search service.

Currently powered by Tavily.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import (
    TavilySearchResults,
)


load_dotenv()


@lru_cache(maxsize=1)
def get_web_search_tool() -> TavilySearchResults:
    """
    Create the Tavily tool only when web search is used.

    This keeps unrelated modules and automated tests
    importable when no Tavily credential is configured.
    """

    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY not found. "
            "Please add TAVILY_API_KEY to your .env file."
        )

    return TavilySearchResults(
        max_results=5
    )


def search_web(question: str) -> str:
    try:
        web_search_tool = get_web_search_tool()

        results = web_search_tool.invoke(
            {
                "query": question
            }
        )

        if not results:
            return "No web search results found."

        formatted_results = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            formatted_results.append(
                f"""
SOURCE {index}

Title:
{result.get("title", "Unknown")}

Content:
{result.get("content", "")}

URL:
{result.get("url", "")}
"""
            )

        return "\n".join(formatted_results)

    except Exception as exc:
        return f"Web search failed: {exc}"