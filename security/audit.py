"""Privacy-safe JSONL security-event auditing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_AUDIT_FILE = Path(__file__).resolve().parent.parent / "data" / "security_events.jsonl"
_AUDIT_LOCK = threading.Lock()


def log_security_event(
    *,
    stage: str,
    action: str,
    categories: tuple[str, ...],
    risk_score: float,
    content: str,
    audit_file: Path | None = None,
) -> bool:
    """Append metadata only; raw user/document content is never persisted."""
    path = audit_file or DEFAULT_AUDIT_FILE
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event_id": str(uuid.uuid4()),
        "stage": stage,
        "action": action,
        "categories": list(categories),
        "risk_score": round(float(risk_score), 2),
        "content_length": len(content or ""),
        "content_sha256": hashlib.sha256((content or "").encode("utf-8")).hexdigest(),
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        fd, temporary = tempfile.mkstemp(prefix="security_event_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(line)
            with _AUDIT_LOCK:
                with path.open("a", encoding="utf-8") as destination:
                    destination.write(Path(temporary).read_text(encoding="utf-8"))
        finally:
            Path(temporary).unlink(missing_ok=True)
        return True
    except OSError as exc:
        print(f"[SECURITY AUDIT ERROR] {exc}")
        return False
