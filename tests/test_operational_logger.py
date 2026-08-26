import json
import logging

from config.logging_config import OperationalLoggingConfig
from logging_service.operational_logger import (
    configure_operational_logger,
    fingerprint_text,
    log_event,
    read_operational_events,
)


def make_logger(tmp_path, *, max_bytes=10_000, backup_count=2):
    config = OperationalLoggingConfig(
        log_file=tmp_path / "debug.jsonl",
        level="DEBUG",
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    return configure_operational_logger(config, force=True), config.log_file


def test_structured_event_contains_correlation_fields(tmp_path):
    logger, log_file = make_logger(tmp_path)

    log_event(
        "route_selected",
        logger=logger,
        request_id="request-123",
        session_fingerprint="session-hash",
        route="RAG_KNOWLEDGE",
    )

    event = json.loads(log_file.read_text(encoding="utf-8"))
    assert event["event"] == "route_selected"
    assert event["level"] == "INFO"
    assert event["request_id"] == "request-123"
    assert event["session_fingerprint"] == "session-hash"
    assert event["route"] == "RAG_KNOWLEDGE"


def test_sensitive_fields_and_values_are_redacted(tmp_path):
    logger, log_file = make_logger(tmp_path)
    secret_key = "sk-abcdefghijklmnopqrstuvwxyz123456"

    log_event(
        "security_test",
        logger=logger,
        query="raw user question",
        api_key=secret_key,
        safe_note=f"accidental value {secret_key}",
    )

    text = log_file.read_text(encoding="utf-8")
    event = json.loads(text)
    assert "raw user question" not in text
    assert secret_key not in text
    assert event["query"] == "[REDACTED]"
    assert "[REDACTED]" in event["safe_note"]


def test_exception_is_recorded_with_redaction(tmp_path):
    logger, log_file = make_logger(tmp_path)

    try:
        raise ValueError("api_key=super-secret-value")
    except ValueError:
        log_event(
            "request_failed",
            logger=logger,
            level=logging.ERROR,
            exc_info=True,
            error_type="ValueError",
        )

    event = json.loads(log_file.read_text(encoding="utf-8"))
    assert event["level"] == "ERROR"
    assert "ValueError" in event["exception"]
    assert "super-secret-value" not in event["exception"]


def test_log_rotation_creates_backup_file(tmp_path):
    logger, log_file = make_logger(tmp_path, max_bytes=250, backup_count=2)

    for index in range(20):
        log_event("large_event", logger=logger, index=index, detail="x" * 100)

    assert log_file.exists()
    assert (tmp_path / "debug.jsonl.1").exists()


def test_reader_skips_malformed_lines_and_respects_limit(tmp_path):
    log_file = tmp_path / "events.jsonl"
    log_file.write_text(
        '{"event":"one"}\nnot-json\n{"event":"two"}\n',
        encoding="utf-8",
    )

    assert read_operational_events(log_file, limit=1) == [{"event": "two"}]


def test_fingerprint_is_stable_and_does_not_expose_value():
    first = fingerprint_text("private session")
    second = fingerprint_text("private session")

    assert first == second
    assert first != "private session"
    assert len(first) == 16
