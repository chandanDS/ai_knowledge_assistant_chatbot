from types import SimpleNamespace

from chatbot.cache_service import (
    BoundedTTLCache,
    cached_rag_invoke,
    cached_web_search,
    clear_application_caches,
    get_cache_statistics,
)


def test_cache_hit_avoids_second_loader_call():
    cache = BoundedTTLCache(max_entries=2, ttl_seconds=10)
    calls = []

    first, first_hit = cache.get_or_load("key", lambda: calls.append(1) or "value", now=0)
    second, second_hit = cache.get_or_load("key", lambda: calls.append(2) or "other", now=1)

    assert first == second == "value"
    assert first_hit is False
    assert second_hit is True
    assert calls == [1]
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_expired_entry_is_refreshed():
    cache = BoundedTTLCache(max_entries=2, ttl_seconds=10)
    cache.get_or_load("key", lambda: "old", now=0)

    value, hit = cache.get_or_load("key", lambda: "new", now=11)

    assert value == "new"
    assert hit is False
    assert cache.stats()["expirations"] == 1


def test_lru_eviction_respects_maximum_size():
    cache = BoundedTTLCache(max_entries=2, ttl_seconds=10)
    cache.get_or_load("a", lambda: "A", now=0)
    cache.get_or_load("b", lambda: "B", now=0)
    cache.get_or_load("c", lambda: "C", now=0)

    assert cache.stats()["size"] == 2
    assert cache.stats()["evictions"] == 1
    _, hit = cache.get_or_load("a", lambda: "A2", now=1)
    assert hit is False


def test_disabled_cache_always_calls_loader():
    cache = BoundedTTLCache(max_entries=2, ttl_seconds=10, enabled=False)
    calls = []

    cache.get_or_load("key", lambda: calls.append(1), now=0)
    cache.get_or_load("key", lambda: calls.append(2), now=1)

    assert calls == [1, 2]
    assert cache.stats()["size"] == 0


def test_rag_cache_normalizes_equivalent_questions():
    clear_application_caches()
    calls = []
    retriever = SimpleNamespace(
        invoke=lambda question: calls.append(question) or ["document"]
    )

    first, first_hit = cached_rag_invoke(" What  is RAG? ", retriever)
    second, second_hit = cached_rag_invoke("what is rag?", retriever)

    assert first == second == ["document"]
    assert first_hit is False
    assert second_hit is True
    assert len(calls) == 1


def test_rag_cache_survives_reconstructed_retriever_wrapper():
    clear_application_caches()
    first_calls = []
    second_calls = []
    first_retriever = SimpleNamespace(
        invoke=lambda question: first_calls.append(question) or ["document"]
    )
    second_retriever = SimpleNamespace(
        invoke=lambda question: second_calls.append(question) or ["different"]
    )

    first, first_hit = cached_rag_invoke("What is RAG?", first_retriever)
    second, second_hit = cached_rag_invoke("What is RAG?", second_retriever)

    assert first == second == ["document"]
    assert first_hit is False
    assert second_hit is True
    assert len(first_calls) == 1
    assert second_calls == []


def test_web_failures_are_not_cached():
    clear_application_caches()
    calls = []

    def failing_search(question):
        calls.append(question)
        return "Web search failed: temporary error"

    _, first_hit = cached_web_search("latest news", failing_search)
    _, second_hit = cached_web_search("latest news", failing_search)

    assert first_hit is False
    assert second_hit is False
    assert len(calls) == 2


def test_cache_statistics_include_rag_and_web():
    statistics = get_cache_statistics()

    assert set(statistics) == {"rag", "web"}
    assert "hits" in statistics["rag"]
    assert "misses" in statistics["web"]
