"""Configuration for prompt budgets, output limits, and cost estimation."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _non_negative_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value >= 0 else default


@dataclass(frozen=True)
class CostOptimizationConfig:
    history_token_budget: int = 1_500
    rag_token_budget: int = 5_000
    web_token_budget: int = 4_000
    final_max_output_tokens: int = 600
    router_max_output_tokens: int = 32

    @classmethod
    def from_env(cls) -> "CostOptimizationConfig":
        return cls(
            history_token_budget=_positive_int("COST_HISTORY_TOKEN_BUDGET", 1_500),
            rag_token_budget=_positive_int("COST_RAG_TOKEN_BUDGET", 5_000),
            web_token_budget=_positive_int("COST_WEB_TOKEN_BUDGET", 4_000),
            final_max_output_tokens=_positive_int("COST_FINAL_MAX_OUTPUT_TOKENS", 600),
            router_max_output_tokens=_positive_int("COST_ROUTER_MAX_OUTPUT_TOKENS", 32),
        )


def configured_model_prices() -> dict[str, tuple[float, float]]:
    """Return USD-per-million-token prices used only for local estimates."""
    return {
        "gpt-4o-mini": (
            _non_negative_float("PRICE_GPT4O_MINI_INPUT_PER_M", 0.15),
            _non_negative_float("PRICE_GPT4O_MINI_OUTPUT_PER_M", 0.60),
        ),
        "gpt-4o": (
            _non_negative_float("PRICE_GPT4O_INPUT_PER_M", 2.50),
            _non_negative_float("PRICE_GPT4O_OUTPUT_PER_M", 10.00),
        ),
    }
