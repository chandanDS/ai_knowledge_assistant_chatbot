"""Operational logging configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class OperationalLoggingConfig:
    log_file: Path = PROJECT_ROOT / "logs" / "chatbot_debug.jsonl"
    level: str = "INFO"
    max_bytes: int = 5_000_000
    backup_count: int = 5

    @classmethod
    def from_env(cls) -> "OperationalLoggingConfig":
        return cls(
            log_file=Path(os.getenv(
                "OPERATIONAL_LOG_FILE",
                str(PROJECT_ROOT / "logs" / "chatbot_debug.jsonl"),
            )),
            level=os.getenv("OPERATIONAL_LOG_LEVEL", "INFO").upper(),
            max_bytes=_positive_int("OPERATIONAL_LOG_MAX_BYTES", 5_000_000),
            backup_count=_positive_int("OPERATIONAL_LOG_BACKUP_COUNT", 5),
        )
