from chatbot.prompts import FINAL_PROMPT
from web.web_search import search_web


"""
============================================================
CHATBOT RESPONSE GENERATOR
============================================================

Responsibilities
----------------
1. Identify the user's intent/route
2. Retrieve RAG documents when required
3. Perform web search when required
4. Maintain only the latest N conversation messages
5. Call the final LLM
6. Generate structured response + follow-up questions
7. Track router and final LLM token usage
8. Calculate total latency

Routes
------
RAG_KNOWLEDGE
    -> FAISS / Retriever
    -> Final LLM

WEB_SEARCH
    -> Tavily
    -> Final LLM

GENERAL_LLM
    -> Final LLM directly

Important
---------
This module does NOT contain Streamlit UI code.

The UI layer (app.py) is responsible for:
- displaying the answer
- displaying follow-up questions
- updating Streamlit session state
- writing logs

This module only handles chatbot processing.
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

from web.web_search import search_web


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MAX_HISTORY_MESSAGES = 6


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def get_recent_history(
    messages: list[dict[str, Any]],
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
) -> list[dict[str, Any]]:
    """
    Return only the most recent conversation messages.

    Example
    -------
    If max_messages = 6 and the session contains:

        user
        assistant
        user
        assistant
        user
        assistant
        user
        assistant

    only the latest 6 messages are returned.

    This helps control LLM input token consumption.
    """

    if not messages:
        return []

    if max_messages <= 0:
        return []

    return messages[-max_messages:]


# ============================================================
# CONVERT STREAMLIT HISTORY TO LANGCHAIN MESSAGES
# ============================================================

def convert_to_langchain_messages(
    recent_history: list[dict[str, Any]]
) -> list:
    """
    Convert Streamlit session messages into
    LangChain HumanMessage / AIMessage objects.
    """

    messages = []

    for message in recent_history:

        role = message.get("role")

        content = message.get(
            "content",
            ""
        )

        if not content:
            continue

        if role == "user":

            messages.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "assistant":

            messages.append(
                AIMessage(
                    content=content
                )
            )

    return messages


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_context(
    question: str,
    retriever
) -> tuple[str, list]:
    """
    Retrieve relevant documents from the configured
    vector store.

    Returns
    -------
    context:
        Combined text from retrieved documents.

    docs:
        Original retrieved document objects.
    """

    if retriever is None:

        print(
            "WARNING: Retriever is not available."
        )

        return "", []

    try:

        docs = retriever.invoke(
            question
        )

    except Exception as exc:

        print(
            f"ERROR: RAG retrieval failed: {exc}"
        )

        return "", []

    if not docs:

        return "", []

    context = "\n\n".join(
        doc.page_content
        for doc in docs
        if getattr(
            doc,
            "page_content",
            None
        )
    )

    return context, docs


# ============================================================
# TOKEN USAGE EXTRACTION
# ============================================================

def extract_usage(
    usage: Any
) -> dict[str, int]:
    """
    Normalize LangChain UsageMetadataCallbackHandler
    output into a simple dictionary.

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

        # LangChain callback normally returns:

        # {
        #     "gpt-4o": {
        #         "input_tokens": ...,
        #         "output_tokens": ...,
        #         "total_tokens": ...
        #     }
        # }

        if isinstance(
            usage,
            dict
        ):

            # Case 1:
            # Usage is directly token metadata

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

            # Case 2:
            # Usage is nested under model name

            if usage:

                model_usage = next(
                    iter(
                        usage.values()
                    )
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
            f"WARNING: Unable to extract token usage: {exc}"
        )

    return empty_usage


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
    Invoke the final structured-output LLM.

    Returns
    -------
    result:
        ChatResponse object.

    usage:
        Raw LangChain usage metadata.
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
    web_results = search_web(question)
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
    llm,
    temperature: float,
    chat_history: list[dict[str, Any]],
    retriever,
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
):
    """
    Main chatbot orchestration function.

    Parameters
    ----------
    question:
        Current user question.

    llm:
        LLM model name, for example:
        "gpt-4o"

    temperature:
        LLM temperature.

    chat_history:
        Full conversation history stored in
        st.session_state.messages.

    retriever:
        FAISS/vector-store retriever.

    max_messages:
        Maximum number of previous messages sent
        to the final LLM.

    Returns
    -------
    (
        response,
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
    # INITIALIZE VARIABLES
    # ========================================================

    context = ""

    web_context = ""

    docs = []

    router_usage = {}

    final_usage = {}


    # ========================================================
    # OPENAI API KEY
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
    # STEP 1
    # IDENTIFY USER INTENT
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
    # STEP 2
    # ROUTE-SPECIFIC PROCESSING
    # ========================================================

    if route == "RAG_KNOWLEDGE":

        # ----------------------------------------------------
        # RAG ROUTE
        # ----------------------------------------------------

        print(
            "RAG ROUTE SELECTED"
        )

        context, docs = retrieve_context(
            question,
            retriever
        )

        print(
            "RAG RETRIEVER INVOKED"
        )

        print(
            "DOCUMENTS RETRIEVED:",
            len(docs)
        )


    elif route == "WEB_SEARCH":

        # ----------------------------------------------------
        # WEB SEARCH ROUTE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # GENERAL LLM ROUTE
        # ----------------------------------------------------

        print(
            "GENERAL LLM ROUTE SELECTED"
        )

        # No RAG retrieval.
        # No web search.

        context = ""
        web_results = search_web(question)
        web_context = "\n\n".join(
            [
                result.get("content", "")
                for result in web_results
                if isinstance(result, dict)
                ]
        )

        docs = []


    else:

        # ----------------------------------------------------
        # SAFETY FALLBACK
        # ----------------------------------------------------

        print(
            "UNKNOWN ROUTE:",
            route
        )

        route = "GENERAL_LLM"

        context = ""

        web_context = ""

        docs = []


    # ========================================================
    # STEP 3
    # LIMIT CONVERSATION HISTORY
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
    # STEP 4
    # CONVERT HISTORY TO LANGCHAIN FORMAT
    # ========================================================

    history_messages = (
        convert_to_langchain_messages(
            recent_history
        )
    )


    # ========================================================
    # STEP 5
    # FINAL LLM CALL
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
    # STEP 6
    # EXTRACT TOKEN USAGE
    # ========================================================

    router_tokens = extract_usage(
        router_usage
    )

    final_tokens = extract_usage(
        final_usage
    )


    # ========================================================
    # STEP 7
    # TOTAL TOKEN CALCULATION
    # ========================================================

    total_input_tokens = (
        router_tokens["input_tokens"]
        +
        final_tokens["input_tokens"]
    )

    total_output_tokens = (
        router_tokens["output_tokens"]
        +
        final_tokens["output_tokens"]
    )

    total_tokens = (
        router_tokens["total_tokens"]
        +
        final_tokens["total_tokens"]
    )


    total_usage = {

        "input_tokens":
            total_input_tokens,

        "output_tokens":
            total_output_tokens,

        "total_tokens":
            total_tokens
    }


    # ========================================================
    # STEP 8
    # LATENCY
    # ========================================================

    latency = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # DEBUG / MONITORING
    # ========================================================

    print("=" * 60)

    print(
        "ROUTER INPUT TOKENS:",
        router_tokens["input_tokens"]
    )

    print(
        "ROUTER OUTPUT TOKENS:",
        router_tokens["output_tokens"]
    )

    print(
        "ROUTER TOTAL TOKENS:",
        router_tokens["total_tokens"]
    )

    print(
        "FINAL INPUT TOKENS:",
        final_tokens["input_tokens"]
    )

    print(
        "FINAL OUTPUT TOKENS:",
        final_tokens["output_tokens"]
    )

    print(
        "FINAL TOTAL TOKENS:",
        final_tokens["total_tokens"]
    )

    print(
        "TOTAL INPUT TOKENS:",
        total_usage["input_tokens"]
    )

    print(
        "TOTAL OUTPUT TOKENS:",
        total_usage["output_tokens"]
    )

    print(
        "TOTAL TOKENS:",
        total_usage["total_tokens"]
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
    # STEP 9
    # RETURN RESPONSE
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
