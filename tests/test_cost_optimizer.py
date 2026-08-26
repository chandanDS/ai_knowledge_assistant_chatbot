from types import SimpleNamespace

from chatbot.cost_optimizer import (
    deduplicate_documents,
    estimate_request_cost,
    estimate_tokens,
    fit_history_to_budget,
    fit_text_to_budget,
)


def test_text_is_truncated_to_configured_token_budget():
    result = fit_text_to_budget("word " * 1_000, max_tokens=100)

    assert result.truncated is True
    assert result.estimated_tokens_before > 100
    assert result.estimated_tokens_after <= 100


def test_short_text_is_unchanged():
    text = "A short prompt"
    result = fit_text_to_budget(text, max_tokens=100)

    assert result.text == text
    assert result.truncated is False
    assert result.estimated_tokens_after == estimate_tokens(text)


def test_history_budget_keeps_newest_messages():
    messages = [
        {"role": "user", "content": "old " * 20},
        {"role": "assistant", "content": "middle " * 20},
        {"role": "user", "content": "newest " * 20},
    ]

    selected, metadata = fit_history_to_budget(messages, max_tokens=40)

    assert selected[-1]["content"].startswith("newest")
    assert metadata["tokens_after"] <= 40
    assert metadata["truncated"] is True


def test_duplicate_rag_chunks_are_removed_in_original_order():
    docs = [
        SimpleNamespace(page_content="Same fact"),
        SimpleNamespace(page_content=" same   fact "),
        SimpleNamespace(page_content="Different fact"),
    ]

    unique, removed = deduplicate_documents(docs)

    assert [doc.page_content for doc in unique] == ["Same fact", "Different fact"]
    assert removed == 1


def test_request_cost_uses_router_and_final_model_usage():
    result = estimate_request_cost(
        router_model="gpt-4o-mini",
        router_tokens={"input_tokens": 1_000, "output_tokens": 100},
        final_model="gpt-4o",
        final_tokens={"input_tokens": 2_000, "output_tokens": 200},
        prices={
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
        },
    )

    assert result["router_cost_usd"] == 0.00021
    assert result["final_cost_usd"] == 0.007
    assert result["estimated_cost_usd"] == 0.00721
    assert result["cost_estimate_available"] is True


def test_unknown_model_cost_is_reported_as_unavailable():
    result = estimate_request_cost(
        router_model="unknown",
        router_tokens={},
        final_model="unknown",
        final_tokens={},
        prices={"known": (1.0, 1.0)},
    )

    assert result["estimated_cost_usd"] is None
    assert result["cost_estimate_available"] is False
