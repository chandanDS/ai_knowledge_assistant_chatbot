"""
External web search service.

Currently powered by Tavily.
"""
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()


# =========================================================
# CHECK API KEY
# =========================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

from langchain_community.tools.tavily_search import (
    TavilySearchResults
)


web_search_tool = TavilySearchResults(
    max_results=5
)


def search_web(question: str) -> str:

    try:

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
            start=1
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

        return "\n".join(
            formatted_results
        )

    except Exception as exc:

        return (
            f"Web search failed: {exc}"
        )