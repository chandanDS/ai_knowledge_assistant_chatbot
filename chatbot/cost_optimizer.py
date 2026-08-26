"""Pure helpers for controlling tokens and estimating request cost."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from config.cost_optimization import configured_model_prices
from security.prompt_injection import normalize_text


@dataclass(frozen=True)
class BudgetedText:
    text: str
    estimated_tokens_before: int
    estimated_tokens_after: int
    truncated: bool


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate: approximately four chars/token."""
    return math.ceil(len(str(text or "")) / 4)


def fit_text_to_budget(text: str, max_tokens: int) -> BudgetedText:
    original = str(text or "")
    before = estimate_tokens(original)
    if before <= max_tokens:
        return BudgetedText(original, before, before, False)

    character_limit = max_tokens * 4
    shortened = original[:character_limit]
    boundary = max(shortened.rfind("\n"), shortened.rfind(" "))
    if boundary >= int(character_limit * 0.8):
        shortened = shortened[:boundary]
    shortened = shortened.rstrip()
    return BudgetedText(
        shortened,
        before,
        estimate_tokens(shortened),
        True,
    )


def fit_history_to_budget(
    messages: list[dict[str, Any]], max_tokens: int
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Keep the newest messages within budget, trimming only the newest if needed."""
    selected_reversed = []
    remaining = max_tokens
    before = sum(estimate_tokens(message.get("content", "")) for message in messages)

    for message in reversed(messages):
        content = str(message.get("content", ""))
        tokens = estimate_tokens(content)
        if tokens <= remaining:
            selected_reversed.append({**message, "content": content})
            remaining -= tokens
        elif not selected_reversed and remaining > 0:
            budgeted = fit_text_to_budget(content, remaining)
            selected_reversed.append({**message, "content": budgeted.text})
            remaining = 0
        else:
            break

    selected = list(reversed(selected_reversed))
    after = sum(estimate_tokens(message.get("content", "")) for message in selected)
    return selected, {
        "tokens_before": before,
        "tokens_after": after,
        "messages_before": len(messages),
        "messages_after": len(selected),
        "truncated": after < before,
    }


def deduplicate_documents(documents: list) -> tuple[list, int]:
    unique = []
    seen = set()
    for document in documents:
        content = getattr(document, "page_content", "")
        key = normalize_text(content)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique, len(documents) - len(unique)


def estimate_request_cost(
    *,
    router_model: str,
    router_tokens: dict[str, int],
    final_model: str,
    final_tokens: dict[str, int],
    prices: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float | bool | None]:
    price_table = prices or configured_model_prices()

    def model_cost(model: str, usage: dict[str, int]) -> float | None:
        price = price_table.get(model)
        if price is None:
            return None
        input_price, output_price = price
        return (
            usage.get("input_tokens", 0) * input_price
            + usage.get("output_tokens", 0) * output_price
        ) / 1_000_000

    router_cost = model_cost(router_model, router_tokens)
    final_cost = model_cost(final_model, final_tokens)
    estimated = router_cost is not None and final_cost is not None
    return {
        "router_cost_usd": round(router_cost, 8) if router_cost is not None else None,
        "final_cost_usd": round(final_cost, 8) if final_cost is not None else None,
        "estimated_cost_usd": round(router_cost + final_cost, 8) if estimated else None,
        "cost_estimate_available": estimated,
    }
