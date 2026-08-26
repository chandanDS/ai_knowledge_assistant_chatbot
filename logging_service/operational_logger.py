"""Rotating, structured, privacy-safe operational logging."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from config.logging_config import OperationalLoggingConfig


LOGGER_NAME = "chatbot.operational"
RESERVED_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)
SENSITIVE_FIELD_NAMES = {
    "api_key", "authorization", "credentials", "password", "query",
    "raw_content", "raw_prompt", "response", "secret",
}
SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)\b(?:password|secret|api[_ -]?key)\s*[=:]\s*\S+"),
)


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def _redact_value(value: Any, field_name: str | None = None) -> Any:
    if field_name and field_name.casefold() in SENSITIVE_FIELD_NAMES:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {key: _redact_value(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in SENSITIVE_VALUE_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in RESERVED_RECORD_FIELDS and key != "event":
                payload[key] = _redact_value(value, key)
        if record.exc_info:
            payload["exception"] = _redact_value(
                "".join(traceback.format_exception(*record.exc_info))
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_operational_logger(
    config: OperationalLoggingConfig | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    policy = config or OperationalLoggingConfig.from_env()
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers and not force:
        return logger

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    policy.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        policy.log_file,
        maxBytes=policy.max_bytes,
        backupCount=policy.backup_count,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(JsonLogFormatter())
    logger.setLevel(getattr(logging, policy.level, logging.INFO))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    target = logger or operational_logger
    target.log(level, event, extra={"event": event, **fields}, exc_info=exc_info)


def read_operational_events(
    log_file: Path | None = None,
    *,
    limit: int = 1_000,
) -> list[dict[str, Any]]:
    path = log_file or OperationalLoggingConfig.from_env().log_file
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events[-limit:]


operational_logger = configure_operational_logger()
