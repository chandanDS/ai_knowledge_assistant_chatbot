from chatbot.schemas import ChatResponse
from config.model_routing import (
    AUTOMATIC_MODEL,
    ModelRoutingConfig,
    select_final_model,
)


CONFIG = ModelRoutingConfig(
    router_model="router-model",
    fast_model="fast-model",
    standard_model="standard-model",
    advanced_model="advanced-model",
    complex_question_chars=100,
    large_context_chars=500,
)


def select(**overrides):
    values = {
        "requested_model": AUTOMATIC_MODEL,
        "route": "GENERAL_LLM",
        "question": "What is RAG?",
        "history_message_count": 0,
        "context_chars": 0,
        "config": CONFIG,
    }
    values.update(overrides)
    return select_final_model(**values)


def test_general_simple_question_uses_fast_tier():
    decision = select()

    assert decision.selected_model == "fast-model"
    assert decision.tier == "FAST"
    assert decision.automatic is True


def test_rag_question_uses_standard_tier():
    decision = select(route="RAG_KNOWLEDGE")

    assert decision.selected_model == "standard-model"
    assert decision.tier == "STANDARD"


def test_web_question_uses_advanced_tier():
    decision = select(route="WEB_SEARCH")

    assert decision.selected_model == "advanced-model"
    assert decision.tier == "ADVANCED"


def test_multiple_complexity_signals_upgrade_to_advanced():
    decision = select(
        question="Compare the architecture and evaluate the trade-offs. " + ("x" * 120),
        history_message_count=4,
    )

    assert decision.selected_model == "advanced-model"
    assert decision.complexity_score >= 2
    assert "Signals:" in decision.reason


def test_explicit_model_is_never_overridden():
    decision = select(
        requested_model="my-approved-model",
        route="WEB_SEARCH",
        question="Compare many complex strategies " + ("x" * 500),
        history_message_count=10,
        context_chars=10_000,
    )

    assert decision.selected_model == "my-approved-model"
    assert decision.tier == "MANUAL"
    assert decision.automatic is False


def test_generate_response_uses_router_and_selected_final_model(monkeypatch):
    import chatbot.response_generator as response_generator

    created_models = []

    class FakeChatOpenAI:
        def __init__(self, *, model, temperature, api_key, max_tokens):
            created_models.append((model, temperature))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_ROUTER_MODEL", "router-model")
    monkeypatch.setenv("MODEL_FAST", "fast-model")
    monkeypatch.setattr(response_generator, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        response_generator,
        "identify_route",
        lambda question, model: ("GENERAL_LLM", {}),
    )
    web_calls = []
    monkeypatch.setattr(
        response_generator,
        "cached_web_search",
        lambda question, function: web_calls.append(question) or ("", False),
    )
    monkeypatch.setattr(
        response_generator,
        "generate_final_response",
        lambda **kwargs: (
            ChatResponse(answer="Safe answer", follow_up_questions=[]),
            {},
        ),
    )

    result = response_generator.generate_response(
        question="What is RAG?",
        llm=AUTOMATIC_MODEL,
        temperature=0.7,
        chat_history=[],
        retriever=None,
        security_principal="model-routing-test",
    )

    assert created_models == [("router-model", 0), ("fast-model", 0.7)]
    assert web_calls == []
    assert result[6]["selected_model"] == "fast-model"
    assert result[6]["model_routing_tier"] == "FAST"
