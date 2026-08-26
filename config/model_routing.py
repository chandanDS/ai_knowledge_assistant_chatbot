"""Deterministic, environment-configurable final-model routing policy."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


AUTOMATIC_MODEL = "Automatic"


@dataclass(frozen=True)
class ModelRoutingConfig:
    router_model: str = "gpt-4o-mini"
    fast_model: str = "gpt-4o-mini"
    standard_model: str = "gpt-4o"
    advanced_model: str = "gpt-4o"
    complex_question_chars: int = 300
    large_context_chars: int = 12_000

    @classmethod
    def from_env(cls) -> "ModelRoutingConfig":
        return cls(
            router_model=os.getenv("MODEL_ROUTER_MODEL", "gpt-4o-mini"),
            fast_model=os.getenv("MODEL_FAST", "gpt-4o-mini"),
            standard_model=os.getenv("MODEL_STANDARD", "gpt-4o"),
            advanced_model=os.getenv("MODEL_ADVANCED", "gpt-4o"),
            complex_question_chars=_positive_int("MODEL_COMPLEX_QUESTION_CHARS", 300),
            large_context_chars=_positive_int("MODEL_LARGE_CONTEXT_CHARS", 12_000),
        )


@dataclass(frozen=True)
class ModelRoutingDecision:
    selected_model: str
    tier: str
    reason: str
    automatic: bool
    complexity_score: int


COMPLEXITY_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:compare|contrast|evaluate|analyse|analyze)\b",
    r"\b(?:step[- ]by[- ]step|trade[- ]offs?|pros and cons)\b",
    r"\b(?:architecture|strategy|root cause|recommendation)\b",
    r"\b(?:first|second|third|finally)\b",
))


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def calculate_complexity_score(
    question: str,
    history_message_count: int,
    context_chars: int,
    config: ModelRoutingConfig,
) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    if len(question) >= config.complex_question_chars:
        score += 1
        reasons.append("long question")
    if history_message_count >= 4:
        score += 1
        reasons.append("multi-turn history")
    if context_chars >= config.large_context_chars:
        score += 1
        reasons.append("large retrieved context")
    if sum(bool(pattern.search(question)) for pattern in COMPLEXITY_PATTERNS) >= 2:
        score += 1
        reasons.append("multi-step reasoning language")

    return score, reasons


def select_final_model(
    *,
    requested_model: str,
    route: str,
    question: str,
    history_message_count: int,
    context_chars: int,
    config: ModelRoutingConfig | None = None,
) -> ModelRoutingDecision:
    policy = config or ModelRoutingConfig.from_env()

    if requested_model and requested_model != AUTOMATIC_MODEL:
        return ModelRoutingDecision(
            requested_model,
            "MANUAL",
            "User selected an explicit model override.",
            False,
            0,
        )

    complexity_score, complexity_reasons = calculate_complexity_score(
        question,
        history_message_count,
        context_chars,
        policy,
    )

    if route == "WEB_SEARCH":
        tier = "ADVANCED"
        model = policy.advanced_model
        reason = "Web synthesis uses the advanced tier."
    elif complexity_score >= 2:
        tier = "ADVANCED"
        model = policy.advanced_model
        reason = "Complex request uses the advanced tier."
    elif route == "RAG_KNOWLEDGE":
        tier = "STANDARD"
        model = policy.standard_model
        reason = "Knowledge-base synthesis uses the standard tier."
    else:
        tier = "FAST"
        model = policy.fast_model
        reason = "Straightforward general request uses the fast tier."

    if complexity_reasons:
        reason += " Signals: " + ", ".join(complexity_reasons) + "."

    return ModelRoutingDecision(model, tier, reason, True, complexity_score)
