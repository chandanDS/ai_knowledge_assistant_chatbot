"""
============================================================
CHATBOT RESPONSE GENERATOR
============================================================

Central orchestration layer for the chatbot.

Flow
----
User Question
      |
      +--> Router
      |
      +--> RAG_KNOWLEDGE
      |       |
      |       +--> Query Expansion
      |       +--> Similarity Search
      |       +--> Context Builder
      |
      +--> WEB_SEARCH
      |       |
      |       +--> Web Search
      |
      +--> GENERAL_LLM
      |
      +--> Final LLM
      |
      +--> Structured Response

Responsibilities
----------------
1. Identify user intent
2. Route the request
3. Expand RAG queries
4. Retrieve relevant documents
5. Build RAG context
6. Perform web search when required
7. Maintain conversation history
8. Call final LLM
9. Track token usage
10. Calculate latency

This module contains chatbot orchestration only.
It does not contain Streamlit UI code.
============================================================
"""

# ============================================================
# STANDARD LIBRARY
# ============================================================

import os
import time
from typing import Any


# ============================================================
# ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# LANGCHAIN
# ============================================================

from langchain_openai import ChatOpenAI

from langchain_core.callbacks import (
    UsageMetadataCallbackHandler
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)


# ============================================================
# PROJECT MODULES
# ============================================================

from chatbot.router import identify_route
from chatbot.schemas import ChatResponse
from chatbot.prompts import FINAL_PROMPT

from rag.query_expansion import expand_query
from rag.retriever import similarity_search
from rag.context_builder import build_context

from web.web_search import search_web


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MAX_HISTORY_MESSAGES = 6

DEFAULT_TEMPERATURE = 0


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def get_recent_history(
    messages: list[dict[str, Any]],
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
) -> list[dict[str, Any]]:
    """
    Return only the most recent conversation messages.

    Limiting history reduces final LLM input tokens.
    """

    if not messages or max_messages <= 0:
        return []

    return messages[-max_messages:]


def convert_to_langchain_messages(
    recent_history: list[dict[str, Any]]
) -> list:
    """
    Convert application conversation history into
    LangChain HumanMessage / AIMessage objects.
    """

    messages = []

    for message in recent_history:

        role = message.get("role")
        content = message.get("content", "")

        if not content:
            continue

        if role == "user":

            messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":

            messages.append(
                AIMessage(content=content)
            )

    return messages


# ============================================================
# TOKEN USAGE
# ============================================================

def extract_usage(
    usage: Any
) -> dict[str, int]:
    """
    Normalize LangChain usage metadata.

    Returns
    -------
    {
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int
    }
    """

    empty_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }

    if not usage:
        return empty_usage

    try:

        if isinstance(usage, dict):

            # Direct usage format
            if (
                "input_tokens" in usage
                or "output_tokens" in usage
                or "total_tokens" in usage
            ):

                return {
                    "input_tokens": int(
                        usage.get(
                            "input_tokens",
                            0
                        )
                    ),
                    "output_tokens": int(
                        usage.get(
                            "output_tokens",
                            0
                        )
                    ),
                    "total_tokens": int(
                        usage.get(
                            "total_tokens",
                            0
                        )
                    )
                }

            # Nested model usage
            if usage:

                model_usage = next(
                    iter(usage.values())
                )

                if isinstance(
                    model_usage,
                    dict
                ):

                    return {
                        "input_tokens": int(
                            model_usage.get(
                                "input_tokens",
                                0
                            )
                        ),
                        "output_tokens": int(
                            model_usage.get(
                                "output_tokens",
                                0
                            )
                        ),
                        "total_tokens": int(
                            model_usage.get(
                                "total_tokens",
                                0
                            )
                        )
                    }

    except Exception as exc:

        print(
            f"WARNING: Unable to extract "
            f"token usage: {exc}"
        )

    return empty_usage


# ============================================================
# RAG PROCESSING
# ============================================================

def retrieve_rag_context(
    question: str
):
    """
    Execute the complete RAG retrieval pipeline.

    Flow
    ----
    Question
        ↓
    Query Expansion
        ↓
    Similarity Search
        ↓
    Context Builder

    Returns
    -------
    expanded_query:
        Optimized search query.

    context:
        LLM-ready context.

    docs:
        Retrieved documents with scores.
    """

    print("-" * 60)
    print("RAG QUERY EXPANSION")

    # --------------------------------------------------------
    # 1. Query Expansion
    # --------------------------------------------------------

    expanded_query = expand_query(
        question
    )

    print(
        "EXPANDED QUERY:",
        expanded_query
    )

    # --------------------------------------------------------
    # 2. Similarity Search
    # --------------------------------------------------------

    print(
        "PERFORMING SIMILARITY SEARCH..."
    )

    results = similarity_search(
        expanded_query
    )

    print(
        "RETRIEVED DOCUMENTS:",
        len(results)
    )

    # --------------------------------------------------------
    # 3. Build Context
    # --------------------------------------------------------

    context = build_context(
        results
    )

    print(
        "CONTEXT LENGTH:",
        len(context)
    )

    return (
        expanded_query,
        context,
        results
    )


# ============================================================
# FINAL LLM
# ============================================================

def generate_final_response(
    llm,
    question: str,
    history: list,
    context: str,
    web_context: str
):
    """
    Generate final structured chatbot response.

    The final LLM receives:
        - conversation history
        - original user question
        - RAG context
        - web context
    """

    structured_model = (
        llm.with_structured_output(
            ChatResponse
        )
    )

    chain = (
        FINAL_PROMPT
        | structured_model
    )

    final_usage_callback = (
        UsageMetadataCallbackHandler()
    )

    result = chain.invoke(
        {
            "history": history,
            "question": question,
            "context": context,
            "web_context": web_context
        },
        config={
            "callbacks": [
                final_usage_callback
            ]
        }
    )

    final_usage = (
        final_usage_callback.usage_metadata
    )

    return (
        result,
        final_usage
    )


# ============================================================
# MAIN RESPONSE GENERATOR
# ============================================================

def generate_response(
    question: str,
    llm: str,
    temperature: float,
    chat_history: list[dict[str, Any]],
    retriever=None,
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
):
    """
    Main chatbot orchestration function.

    Parameters
    ----------
    question:
        User's current question.

    llm:
        OpenAI model name.

    temperature:
        LLM temperature.

    chat_history:
        Previous conversation messages.

    retriever:
        Retained for backward compatibility.
        The new RAG pipeline uses similarity_search()
        directly.

    max_messages:
        Number of previous messages sent to final LLM.

    Returns
    -------
    (
        answer,
        follow_up_questions,
        route,
        docs,
        router_tokens,
        final_tokens,
        total_usage,
        latency
    )
    """

    # ========================================================
    # START TIMER
    # ========================================================

    start_time = time.time()


    # ========================================================
    # INITIALIZE
    # ========================================================

    context = ""

    web_context = ""

    docs = []

    router_usage = {}

    final_usage = {}


    # ========================================================
    # API KEY
    # ========================================================

    openai_api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not openai_api_key:

        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )


    # ========================================================
    # CREATE LLM
    # ========================================================

    model = ChatOpenAI(
        model=llm,
        temperature=temperature,
        api_key=openai_api_key
    )


    # ========================================================
    # STEP 1: ROUTER
    # ========================================================

    print("=" * 60)

    print(
        "QUESTION:",
        question
    )

    print(
        "IDENTIFYING ROUTE..."
    )

    route, router_usage = identify_route(
        question,
        model
    )

    print(
        "DETECTED ROUTE:",
        route
    )


    # ========================================================
    # STEP 2: ROUTE PROCESSING
    # ========================================================

    if route == "RAG_KNOWLEDGE":

        print(
            "RAG ROUTE SELECTED"
        )

        try:

            (
                expanded_query,
                context,
                docs
            ) = retrieve_rag_context(
                question
            )

        except Exception as exc:

            print(
                f"ERROR: RAG pipeline failed: {exc}"
            )

            expanded_query = question
            context = ""
            docs = []

        print(
            "RAG PIPELINE COMPLETED"
        )

        print(
            "DOCUMENTS RETRIEVED:",
            len(docs)
        )


    elif route == "WEB_SEARCH":

        print(
            "WEB SEARCH ROUTE SELECTED"
        )

        try:

            web_context = search_web(
                question
            )

        except Exception as exc:

            print(
                f"WEB SEARCH ERROR: {exc}"
            )

            web_context = (
                "Web search was unavailable."
            )

        print(
            "WEB SEARCH COMPLETED"
        )

        print(
            "WEB CONTEXT LENGTH:",
            len(web_context)
        )


    elif route == "GENERAL_LLM":

        print(
            "GENERAL LLM ROUTE SELECTED"
        )

        context = ""
        web_context = ""
        docs = []


    else:

        print(
            "UNKNOWN ROUTE:",
            route
        )

        route = "GENERAL_LLM"

        context = ""
        web_context = ""
        docs = []


    # ========================================================
    # STEP 3: CONVERSATION HISTORY
    # ========================================================

    recent_history = get_recent_history(
        chat_history,
        max_messages=max_messages
    )

    print(
        "TOTAL STORED MESSAGES:",
        len(chat_history)
    )

    print(
        "MESSAGES SENT TO FINAL LLM:",
        len(recent_history)
    )


    # ========================================================
    # STEP 4: LANGCHAIN HISTORY
    # ========================================================

    history_messages = (
        convert_to_langchain_messages(
            recent_history
        )
    )


    # ========================================================
    # STEP 5: FINAL LLM
    # ========================================================

    print(
        "CALLING FINAL LLM..."
    )

    result, final_usage = (
        generate_final_response(
            llm=model,
            question=question,
            history=history_messages,
            context=context,
            web_context=web_context
        )
    )

    print(
        "FINAL LLM RESPONSE GENERATED"
    )


    # ========================================================
    # STEP 6: TOKEN USAGE
    # ========================================================

    router_tokens = extract_usage(
        router_usage
    )

    final_tokens = extract_usage(
        final_usage
    )


    # ========================================================
    # STEP 7: TOTAL TOKEN USAGE
    # ========================================================

    total_usage = {

        "input_tokens":
            (
                router_tokens["input_tokens"]
                +
                final_tokens["input_tokens"]
            ),

        "output_tokens":
            (
                router_tokens["output_tokens"]
                +
                final_tokens["output_tokens"]
            ),

        "total_tokens":
            (
                router_tokens["total_tokens"]
                +
                final_tokens["total_tokens"]
            )
    }


    # ========================================================
    # STEP 8: LATENCY
    # ========================================================

    latency = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # MONITORING
    # ========================================================

    print("-" * 60)

    print(
        "ROUTER TOKENS:",
        router_tokens
    )

    print(
        "FINAL TOKENS:",
        final_tokens
    )

    print(
        "TOTAL TOKENS:",
        total_usage
    )

    print(
        "LATENCY:",
        round(
            latency,
            3
        ),
        "seconds"
    )

    print("=" * 60)


    # ========================================================
    # RETURN
    # ========================================================

    return (
        result.answer,
        result.follow_up_questions,
        route,
        docs,
        router_tokens,
        final_tokens,
        total_usage,
        latency
    )