"""Environment-based cache configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    max_entries: int = 256
    rag_ttl_seconds: int = 900
    web_ttl_seconds: int = 300
    rag_namespace: str = "faiss-default-v1"
    web_namespace: str = "tavily-default-v1"

    @classmethod
    def from_env(cls) -> "CacheConfig":
        return cls(
            enabled=_boolean("CHATBOT_CACHE_ENABLED", True),
            max_entries=_positive_int("CHATBOT_CACHE_MAX_ENTRIES", 256),
            rag_ttl_seconds=_positive_int("RAG_CACHE_TTL_SECONDS", 900),
            web_ttl_seconds=_positive_int("WEB_CACHE_TTL_SECONDS", 300),
            rag_namespace=os.getenv("RAG_CACHE_NAMESPACE", "faiss-default-v1"),
            web_namespace=os.getenv("WEB_CACHE_NAMESPACE", "tavily-default-v1"),
        )
