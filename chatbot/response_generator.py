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
import logging
import uuid
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

from chatbot.cache_service import (
    cached_rag_invoke,
    cached_web_search,
)

from chatbot.cost_optimizer import (
    deduplicate_documents,
    estimate_request_cost,
    fit_history_to_budget,
    fit_text_to_budget,
)

from web.web_search import search_web

from security.audit import log_security_event
from security.guardrails import guardrails

from config.model_routing import (
    ModelRoutingConfig,
    select_final_model,
)

from config.cost_optimization import CostOptimizationConfig

from logging_service.operational_logger import (
    fingerprint_text,
    log_event,
)


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
    retriever,
    event_logger=None,
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

        docs, cache_hit = cached_rag_invoke(question, retriever)
        print("RAG CACHE:", "HIT" if cache_hit else "MISS")
        if event_logger:
            event_logger("rag_cache_result", cache_hit=cache_hit)

    except Exception as exc:

        print(
            f"ERROR: RAG retrieval failed: {exc}"
        )
        if event_logger:
            event_logger(
                "rag_retrieval_failed",
                level=logging.ERROR,
                exc_info=True,
                error_type=type(exc).__name__,
            )

        return "", []

    if not docs:

        return "", []

    docs, duplicates_removed = deduplicate_documents(docs)
    if duplicates_removed:
        print("RAG DEDUPLICATION: removed", duplicates_removed, "duplicate chunk(s)")
    if event_logger:
        event_logger(
            "rag_retrieval_completed",
            documents=len(docs),
            duplicate_chunks_removed=duplicates_removed,
        )

    safe_chunks = []

    for index, doc in enumerate(docs, start=1):
        content = getattr(doc, "page_content", "")
        sanitization = guardrails.sanitize_context(content)

        if sanitization.action != "ALLOW":
            print(
                "RAG SECURITY FILTER:",
                f"{sanitization.reason} "
                f"from document {index}; categories="
                f"{','.join(sanitization.categories)}"
            )
            log_security_event(
                stage="RAG_CONTEXT",
                action=sanitization.action,
                categories=sanitization.categories,
                risk_score=sanitization.risk_score,
                content=content,
            )
            if event_logger:
                event_logger(
                    "rag_context_sanitized",
                    document_index=index,
                    action=sanitization.action,
                    categories=sanitization.categories,
                )

        if sanitization.safe_text:
            safe_chunks.append(sanitization.safe_text)

    context = "\n\n".join(safe_chunks)

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
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    security_principal: str = "anonymous",
    request_id: str | None = None,
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

    request_id = request_id or str(uuid.uuid4())
    session_fingerprint = fingerprint_text(security_principal)

    def operational_event(
        event: str,
        *,
        level: int = logging.INFO,
        exc_info: bool = False,
        **fields,
    ):
        log_event(
            event,
            level=level,
            exc_info=exc_info,
            request_id=request_id,
            session_fingerprint=session_fingerprint,
            **fields,
        )

    operational_event(
        "request_received",
        question_length=len(question or ""),
        question_fingerprint=fingerprint_text(question),
        requested_model=llm,
        history_messages=len(chat_history or []),
    )


    # ========================================================
    # INITIALIZE VARIABLES
    # ========================================================

    context = ""

    web_context = ""

    docs = []

    router_usage = {}

    final_usage = {}

    cost_config = CostOptimizationConfig.from_env()

    # Reject direct prompt injection before creating a model client.  A
    # blocked request therefore consumes no API tokens and needs no API key.
    print("=" * 60)

    print("QUESTION:", question)
    print("CHECKING PROMPT INJECTION...")

    security_result = guardrails.inspect_input(question)

    operational_event(
        "input_guardrail_evaluated",
        allowed=security_result.allowed,
        action=security_result.action,
        categories=security_result.categories,
        risk_score=security_result.risk_score,
    )

    print("PROMPT INJECTION:", not security_result.allowed)
    print("PROMPT RISK SCORE:", security_result.risk_score)
    print("PROMPT SECURITY CATEGORY:", "+".join(security_result.categories))

    if not security_result.allowed:
        print("INPUT GUARDRAIL BLOCKED REQUEST")
        log_security_event(
            stage="INPUT",
            action=security_result.action,
            categories=security_result.categories,
            risk_score=security_result.risk_score,
            content=question,
        )
        operational_event(
            "request_blocked_by_input_guardrail",
            level=logging.WARNING,
            categories=security_result.categories,
            latency_seconds=round(time.time() - start_time, 4),
        )
        zero_tokens = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        return (
            security_result.safe_text,
            [],
            "BLOCKED_PROMPT_INJECTION",
            [],
            zero_tokens.copy(),
            zero_tokens.copy(),
            zero_tokens.copy(),
            time.time() - start_time,
        )

    question = security_result.safe_text

    rate_decision = guardrails.inspect_rate_limit(security_principal)
    operational_event(
        "rate_limit_evaluated",
        allowed=rate_decision.allowed,
        categories=rate_decision.categories,
    )
    if not rate_decision.allowed:
        print("RATE LIMIT GUARDRAIL BLOCKED REQUEST")
        log_security_event(
            stage="RATE_LIMIT",
            action=rate_decision.action,
            categories=rate_decision.categories,
            risk_score=rate_decision.risk_score,
            content=security_principal,
        )
        operational_event(
            "request_blocked_by_rate_limit",
            level=logging.WARNING,
            latency_seconds=round(time.time() - start_time, 4),
        )
        zero_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return (
            rate_decision.safe_text,
            [],
            "BLOCKED_RATE_LIMIT",
            [],
            zero_tokens.copy(),
            zero_tokens.copy(),
            zero_tokens.copy(),
            time.time() - start_time,
        )


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
    # CREATE ECONOMICAL INTENT-ROUTER MODEL
    # ========================================================

    model_routing_config = ModelRoutingConfig.from_env()

    router_model = ChatOpenAI(
        model=model_routing_config.router_model,
        temperature=0,
        api_key=openai_api_key,
        max_tokens=cost_config.router_max_output_tokens,
    )


    # ========================================================
    # STEP 1
    # IDENTIFY USER INTENT
    # ========================================================

    print(
        "IDENTIFYING ROUTE..."
    )
    operational_event(
        "intent_routing_started",
        router_model=model_routing_config.router_model,
    )
    try:
        route, router_usage = identify_route(
            question,
            router_model
        )
    except Exception as exc:
        operational_event(
            "intent_routing_failed",
            level=logging.ERROR,
            exc_info=True,
            error_type=type(exc).__name__,
        )
        raise

    print(
        "DETECTED ROUTE:",
        route
    )
    operational_event("intent_route_selected", route=route)


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
            retriever,
            event_logger=operational_event,
        )

        for i, doc in enumerate(docs):
            print(
                f"\n--- RETRIEVED DOCUMENT {i + 1} ---"
                )
            print(
                doc.page_content
                )

        print("RAG RETRIEVER INVOKED")
        print("DOCUMENTS RETRIEVED:", len(docs))


    elif route == "WEB_SEARCH":

        # ----------------------------------------------------
        # WEB SEARCH ROUTE
        # ----------------------------------------------------

        print(
            "WEB SEARCH ROUTE SELECTED"
        )

        try:

            operational_event("web_search_started")
            raw_web_context, cache_hit = cached_web_search(question, search_web)
            print("WEB CACHE:", "HIT" if cache_hit else "MISS")
            web_decision = guardrails.sanitize_context(raw_web_context)
            web_context = web_decision.safe_text
            if web_decision.action != "ALLOW":
                log_security_event(
                    stage="WEB_CONTEXT",
                    action=web_decision.action,
                    categories=web_decision.categories,
                    risk_score=web_decision.risk_score,
                    content=raw_web_context,
                )
            operational_event(
                "web_search_completed",
                cache_hit=cache_hit,
                result_chars=len(raw_web_context),
                sanitized=web_decision.action != "ALLOW",
            )

        except Exception as exc:

            print(
                f"WEB SEARCH ERROR: {exc}"
            )
            operational_event(
                "web_search_failed",
                level=logging.ERROR,
                exc_info=True,
                error_type=type(exc).__name__,
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
        web_context = ""

        print("COST OPTIMIZATION: skipped web search for GENERAL_LLM route")
        operational_event("web_search_skipped", reason="general_llm_route")

        docs = []


    else:

        # ----------------------------------------------------
        # SAFETY FALLBACK
        # ----------------------------------------------------

        print(
            "UNKNOWN ROUTE:",
            route
        )
        operational_event(
            "unknown_route_fallback",
            level=logging.WARNING,
            returned_route=route,
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
    recent_history = guardrails.sanitize_history(recent_history)

    context_budget = fit_text_to_budget(context, cost_config.rag_token_budget)
    web_budget = fit_text_to_budget(web_context, cost_config.web_token_budget)
    recent_history, history_budget = fit_history_to_budget(
        recent_history,
        cost_config.history_token_budget,
    )
    context = context_budget.text
    web_context = web_budget.text

    print(
        "COST TOKEN BUDGETS:",
        f"history={history_budget['tokens_after']}/{cost_config.history_token_budget},",
        f"rag={context_budget.estimated_tokens_after}/{cost_config.rag_token_budget},",
        f"web={web_budget.estimated_tokens_after}/{cost_config.web_token_budget}",
    )
    operational_event(
        "token_budgets_applied",
        history_tokens_before=history_budget["tokens_before"],
        history_tokens_after=history_budget["tokens_after"],
        rag_tokens_before=context_budget.estimated_tokens_before,
        rag_tokens_after=context_budget.estimated_tokens_after,
        web_tokens_before=web_budget.estimated_tokens_before,
        web_tokens_after=web_budget.estimated_tokens_after,
    )

    model_decision = select_final_model(
        requested_model=llm,
        route=route,
        question=question,
        history_message_count=len(recent_history),
        context_chars=len(context) + len(web_context),
        config=model_routing_config,
    )

    print("MODEL ROUTING MODE:", "AUTOMATIC" if model_decision.automatic else "MANUAL")
    print("MODEL ROUTING TIER:", model_decision.tier)
    print("SELECTED FINAL MODEL:", model_decision.selected_model)
    print("MODEL ROUTING REASON:", model_decision.reason)
    operational_event(
        "final_model_selected",
        selected_model=model_decision.selected_model,
        tier=model_decision.tier,
        automatic=model_decision.automatic,
        complexity_score=model_decision.complexity_score,
        reason=model_decision.reason,
    )

    model = ChatOpenAI(
        model=model_decision.selected_model,
        temperature=temperature,
        api_key=openai_api_key,
        max_tokens=cost_config.final_max_output_tokens,
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
    operational_event(
        "final_llm_started",
        model=model_decision.selected_model,
        history_messages=len(history_messages),
        context_chars=len(context),
        web_context_chars=len(web_context),
    )
    llm_start_time = time.time()
    try:
        result, final_usage = generate_final_response(
            llm=model,
            question=question,
            history=history_messages,
            context=context,
            web_context=web_context
        )
    except Exception as exc:
        operational_event(
            "final_llm_failed",
            level=logging.ERROR,
            exc_info=True,
            model=model_decision.selected_model,
            error_type=type(exc).__name__,
            llm_latency_seconds=round(time.time() - llm_start_time, 4),
        )
        raise

    print(
        "FINAL LLM RESPONSE GENERATED"
    )
    operational_event(
        "final_llm_completed",
        model=model_decision.selected_model,
        llm_latency_seconds=round(time.time() - llm_start_time, 4),
    )

    output_decision = guardrails.inspect_output(result.answer)
    safe_answer = output_decision.safe_text
    safe_follow_up_questions = [
        guardrails.inspect_output(question).safe_text
        for question in result.follow_up_questions
    ]

    if output_decision.action != "ALLOW":
        print(
            "OUTPUT GUARDRAIL:",
            output_decision.action,
            "+".join(output_decision.categories),
        )
        log_security_event(
            stage="OUTPUT",
            action=output_decision.action,
            categories=output_decision.categories,
            risk_score=output_decision.risk_score,
            content=result.answer,
        )
        operational_event(
            "output_guardrail_activated",
            level=logging.WARNING,
            action=output_decision.action,
            categories=output_decision.categories,
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
            total_tokens,

        "selected_model":
            model_decision.selected_model,

        "model_routing_tier":
            model_decision.tier,

        "model_routing_reason":
            model_decision.reason,

        "model_routing_automatic":
            model_decision.automatic,

        "history_tokens_before_budget":
            history_budget["tokens_before"],

        "history_tokens_after_budget":
            history_budget["tokens_after"],

        "rag_tokens_before_budget":
            context_budget.estimated_tokens_before,

        "rag_tokens_after_budget":
            context_budget.estimated_tokens_after,

        "web_tokens_before_budget":
            web_budget.estimated_tokens_before,

        "web_tokens_after_budget":
            web_budget.estimated_tokens_after
    }

    cost_estimate = estimate_request_cost(
        router_model=model_routing_config.router_model,
        router_tokens=router_tokens,
        final_model=model_decision.selected_model,
        final_tokens=final_tokens,
    )
    total_usage.update(cost_estimate)


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
        "ESTIMATED REQUEST COST (USD):",
        total_usage.get("estimated_cost_usd", "unavailable")
    )

    print(
        "LATENCY:",
        round(
            latency,
            3
        ),
        "seconds"
    )

    operational_event(
        "request_completed",
        route=route,
        selected_model=model_decision.selected_model,
        documents_retrieved=len(docs),
        router_tokens=router_tokens["total_tokens"],
        final_tokens=final_tokens["total_tokens"],
        total_tokens=total_usage["total_tokens"],
        estimated_cost_usd=total_usage.get("estimated_cost_usd"),
        latency_seconds=round(latency, 4),
    )

    print("=" * 60)


    # ========================================================
    # STEP 9
    # RETURN RESPONSE
    # ========================================================

    return (
        safe_answer,
        safe_follow_up_questions,
        route,
        docs,
        router_tokens,
        final_tokens,
        total_usage,
        latency
    )
