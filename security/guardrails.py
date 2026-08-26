"""Central, configurable guardrails for chatbot trust boundaries.

The service is deliberately deterministic and local.  It does not send
security-sensitive content to another model or third-party service.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from security.prompt_injection import (
    detect_prompt_injection,
    sanitize_untrusted_content,
)


OUTPUT_BLOCKED_MESSAGE = (
    "I’m unable to provide that response because it may contain protected "
    "internal information. Please rephrase your question."
)


def _positive_int_from_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class GuardrailConfig:
    max_input_chars: int = 8_000
    max_context_chars: int = 50_000
    max_output_chars: int = 20_000
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    @classmethod
    def from_env(cls) -> "GuardrailConfig":
        return cls(
            max_input_chars=_positive_int_from_env(
                "GUARDRAIL_MAX_INPUT_CHARS", 8_000
            ),
            max_context_chars=_positive_int_from_env(
                "GUARDRAIL_MAX_CONTEXT_CHARS", 50_000
            ),
            max_output_chars=_positive_int_from_env(
                "GUARDRAIL_MAX_OUTPUT_CHARS", 20_000
            ),
            rate_limit_requests=_positive_int_from_env(
                "GUARDRAIL_RATE_LIMIT_REQUESTS", 30
            ),
            rate_limit_window_seconds=_positive_int_from_env(
                "GUARDRAIL_RATE_LIMIT_WINDOW_SECONDS", 60
            ),
        )


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    action: str
    categories: tuple[str, ...]
    risk_score: float
    reason: str
    safe_text: str


CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

SENSITIVE_OUTPUT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OPENAI_API_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("BEARER_TOKEN", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.DOTALL,
    )),
    ("ENV_SECRET", re.compile(
        r"(?im)^\s*(?:OPENAI_API_KEY|TAVILY_API_KEY|LANGCHAIN_API_KEY|PASSWORD|SECRET)"
        r"\s*=\s*\S+\s*$"
    )),
)

SYSTEM_PROMPT_LEAKAGE_PATTERNS = (
    re.compile(r"(?i)\btrust and security rules\b"),
    re.compile(r"(?i)\bthe instructions in this system message are authoritative\b"),
    re.compile(r"(?i)\bknowledge_base_context\b"),
    re.compile(r"(?i)\bweb_search_results\b"),
)


class EnterpriseGuardrails:
    """Apply consistent policy to input, retrieved context, history and output."""

    def __init__(self, config: GuardrailConfig | None = None):
        self.config = config or GuardrailConfig.from_env()
        self._request_times: dict[str, deque[float]] = defaultdict(deque)
        self._rate_limit_lock = threading.Lock()

    def inspect_rate_limit(
        self, principal: str, now: float | None = None
    ) -> GuardrailDecision:
        current_time = time.monotonic() if now is None else now
        cutoff = current_time - self.config.rate_limit_window_seconds
        principal_key = str(principal or "anonymous")

        with self._rate_limit_lock:
            timestamps = self._request_times[principal_key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.config.rate_limit_requests:
                return GuardrailDecision(
                    False, "BLOCK", ("RATE_LIMIT_EXCEEDED",), 0.8,
                    "Too many requests in the configured time window.",
                    "Too many requests. Please wait a moment and try again.",
                )

            timestamps.append(current_time)

        return GuardrailDecision(
            True, "ALLOW", ("SAFE",), 0.0,
            "Request is within the rate limit.", "",
        )

    def inspect_input(self, text: str) -> GuardrailDecision:
        if not isinstance(text, str) or not text.strip():
            return GuardrailDecision(
                False, "BLOCK", ("INVALID_INPUT",), 1.0,
                "Input must be non-empty text.",
                "Please enter a question or message before submitting.",
            )

        if len(text) > self.config.max_input_chars:
            return GuardrailDecision(
                False, "BLOCK", ("INPUT_TOO_LARGE",), 1.0,
                f"Input exceeds {self.config.max_input_chars} characters.",
                "Your message is too long. Please shorten it to "
                f"{self.config.max_input_chars:,} characters or fewer.",
            )

        if CONTROL_CHARACTER_PATTERN.search(text):
            return GuardrailDecision(
                False, "BLOCK", ("CONTROL_CHARACTERS",), 0.9,
                "Input contains unsupported control characters.",
                "Your message contains unsupported hidden or control characters. "
                "Please remove them and try again.",
            )

        injection = detect_prompt_injection(text)
        if injection.is_injection:
            categories = tuple(injection.category.split("+"))

            if "SECRET_EXTRACTION" in categories:
                user_message = (
                    "I can’t help reveal API keys, passwords, credentials, "
                    "tokens, or other protected secrets."
                )
            elif "PROMPT_EXTRACTION" in categories:
                user_message = (
                    "I can’t reveal system prompts, developer instructions, "
                    "or other protected internal instructions."
                )
            elif "ROLE_MANIPULATION" in categories:
                user_message = (
                    "I can’t switch into an unrestricted, administrator, "
                    "developer, or jailbreak mode. Please ask a regular question."
                )
            elif "OUTPUT_MANIPULATION" in categories:
                user_message = (
                    "I can’t follow instructions intended to force a security-"
                    "bypass response. Please ask a regular question."
                )
            else:
                user_message = (
                    "I can’t follow instructions that attempt to override or "
                    "bypass the chatbot’s rules. Please rephrase your request."
                )

            return GuardrailDecision(
                False,
                "BLOCK",
                categories,
                injection.risk_score,
                injection.reason,
                user_message,
            )

        return GuardrailDecision(
            True, "ALLOW", (injection.category,), injection.risk_score,
            injection.reason, text.strip(),
        )

    def sanitize_context(self, text: str) -> GuardrailDecision:
        result = sanitize_untrusted_content(text)
        categories = list(result.detected_categories)
        safe_text = result.text

        if len(safe_text) > self.config.max_context_chars:
            safe_text = safe_text[: self.config.max_context_chars]
            categories.append("CONTEXT_TRUNCATED")

        action = "SANITIZE" if categories else "ALLOW"
        return GuardrailDecision(
            True,
            action,
            tuple(dict.fromkeys(categories)) or ("SAFE",),
            0.7 if categories else 0.0,
            (
                f"Removed {result.removed_line_count} suspicious line(s)."
                if result.removed_line_count
                else "No unsafe context detected."
            ),
            safe_text,
        )

    def sanitize_history(self, messages: list[dict]) -> list[dict]:
        safe_messages = []
        for message in messages:
            content = str(message.get("content", ""))
            role = message.get("role")

            if role == "user":
                decision = self.inspect_input(content)
                if not decision.allowed:
                    continue
                safe_content = decision.safe_text
            else:
                decision = self.inspect_output(content)
                safe_content = decision.safe_text

            if role in {"user", "assistant"} and safe_content:
                safe_messages.append({"role": role, "content": safe_content})
        return safe_messages

    def inspect_output(self, text: str) -> GuardrailDecision:
        output = str(text or "")

        if any(pattern.search(output) for pattern in SYSTEM_PROMPT_LEAKAGE_PATTERNS):
            return GuardrailDecision(
                False, "BLOCK", ("SYSTEM_PROMPT_LEAKAGE",), 1.0,
                "Output resembles protected internal instructions.",
                OUTPUT_BLOCKED_MESSAGE,
            )

        categories = []
        for category, pattern in SENSITIVE_OUTPUT_PATTERNS:
            if pattern.search(output):
                categories.append(category)
                output = pattern.sub("[REDACTED]", output)

        if len(output) > self.config.max_output_chars:
            output = output[: self.config.max_output_chars]
            categories.append("OUTPUT_TRUNCATED")

        return GuardrailDecision(
            True,
            "REDACT" if categories else "ALLOW",
            tuple(dict.fromkeys(categories)) or ("SAFE",),
            0.8 if categories else 0.0,
            "Sensitive output was redacted." if categories else "Output is safe.",
            output,
        )


guardrails = EnterpriseGuardrails()
