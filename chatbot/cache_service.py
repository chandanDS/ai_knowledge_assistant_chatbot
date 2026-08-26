"""Thread-safe, bounded in-memory caches for retrieval and web search."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Hashable
from pathlib import Path

from config.cache_config import CacheConfig
from security.prompt_injection import normalize_text


@dataclass
class CacheStatistics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0


class BoundedTTLCache:
    """Small dependency-free TTL/LRU cache with stampede protection."""

    def __init__(self, max_entries: int, ttl_seconds: int, enabled: bool = True):
        if max_entries <= 0 or ttl_seconds <= 0:
            raise ValueError("max_entries and ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._values: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._statistics = CacheStatistics()

    def get_or_load(
        self,
        key: Hashable,
        loader: Callable[[], Any],
        *,
        should_cache: Callable[[Any], bool] | None = None,
        now: float | None = None,
    ) -> tuple[Any, bool]:
        if not self.enabled:
            return loader(), False

        current_time = time.monotonic() if now is None else now
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                expires_at, value = cached
                if expires_at > current_time:
                    self._values.move_to_end(key)
                    self._statistics.hits += 1
                    return value, True
                del self._values[key]
                self._statistics.expirations += 1

            self._statistics.misses += 1
            value = loader()
            if should_cache is not None and not should_cache(value):
                return value, False

            self._values[key] = (current_time + self.ttl_seconds, value)
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
                self._statistics.evictions += 1
            return value, False

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._statistics = CacheStatistics()

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                **asdict(self._statistics),
                "size": len(self._values),
                "enabled": self.enabled,
            }


_config = CacheConfig.from_env()
_project_root = Path(__file__).resolve().parent.parent
_faiss_index_dir = _project_root / "vector_store" / "faiss_index"
rag_cache = BoundedTTLCache(
    _config.max_entries, _config.rag_ttl_seconds, _config.enabled
)
web_cache = BoundedTTLCache(
    _config.max_entries, _config.web_ttl_seconds, _config.enabled
)


def _faiss_index_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Return a stable fingerprint that changes when the FAISS index changes."""
    if not _faiss_index_dir.exists():
        return ()
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(_faiss_index_dir.glob("index.*"))
    )


def cached_rag_invoke(question: str, retriever) -> tuple[list, bool]:
    key = (
        _config.rag_namespace,
        _faiss_index_fingerprint(),
        normalize_text(question),
    )
    docs, hit = rag_cache.get_or_load(key, lambda: list(retriever.invoke(question)))
    return list(docs), hit


def cached_web_search(question: str, search_function) -> tuple[str, bool]:
    key = (_config.web_namespace, normalize_text(question))
    return web_cache.get_or_load(
        key,
        lambda: search_function(question),
        should_cache=lambda result: bool(result)
        and not str(result).casefold().startswith("web search failed"),
    )


def clear_application_caches() -> None:
    rag_cache.clear()
    web_cache.clear()


def get_cache_statistics() -> dict[str, dict[str, int | bool]]:
    return {"rag": rag_cache.stats(), "web": web_cache.stats()}
