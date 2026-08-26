"""Fast local guardrails for direct and indirect prompt injection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptInjectionResult:
    is_injection: bool
    risk_score: float
    category: str
    reason: str


@dataclass(frozen=True)
class SanitizationResult:
    text: str
    removed_line_count: int
    detected_categories: tuple[str, ...]


SUSPICIOUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("INSTRUCTION_OVERRIDE", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,80}"
        r"\b(?:previous|prior|above|system|developer|instructions?|rules?|prompts?)\b"
    )),
    ("PROMPT_EXTRACTION", re.compile(
        r"\b(?:show|reveal|display|print|repeat|expose|provide|tell)\b.{0,100}"
        r"\b(?:system|developer|hidden|internal)\s+(?:prompt|message|instructions?)\b"
    )),
    ("PROMPT_EXTRACTION", re.compile(
        r"\bwhat\s+(?:is|are|were)\s+your\s+(?:system|developer|hidden|internal)\s+"
        r"(?:prompt|message|instructions?)\b"
    )),
    ("ROLE_MANIPULATION", re.compile(
        r"\b(?:you\s+are\s+now|developer\s+mode|jailbreak|dan\s+mode)\b"
    )),
    ("ROLE_MANIPULATION", re.compile(
        r"\bact\s+as\s+(?:an?\s+)?(?:administrator|developer|system|root)\b"
    )),
    ("SECRET_EXTRACTION", re.compile(
        r"\b(?:reveal|show|print|display|expose|provide|dump)\b.{0,100}"
        r"\b(?:api[\s_-]?key|secret|password|token|credentials?|\.env)\b"
    )),
    ("INSTRUCTION_OVERRIDE", re.compile(
        r"\b(?:do\s+not|don't)\s+answer\b.{0,80}\b(?:question|user)\b"
    )),
    ("INSTRUCTION_OVERRIDE", re.compile(
        r"\bfollow\s+(?:only\s+)?(?:these|the following|my)\s+instructions?\b"
    )),
    ("PROMPT_EXTRACTION", re.compile(
        r"\b(?:everything|content|messages?|text)\b.{0,50}"
        r"\b(?:before|above|prior to)\b.{0,40}\b(?:first|initial)\s+(?:message|prompt|input)\b"
    )),
    ("OUTPUT_MANIPULATION", re.compile(
        r"\bsecurity\s+bypass\s+successful\b"
    )),
)


SAFE_CONTEXT_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"\b(?:what\s+is|explain|describe|how\s+does)\s+prompt\s+injection\b",
    r"\b(?:example|examples)\s+of\s+(?:a\s+)?prompt\s+injection\b",
    r"\b(?:review|analy[sz]e|explain\s+why)\b.{0,100}\bprompt\s+injection\b",
    r"\b(?:protect|protection|defend|defense|mitigat\w*)\b.{0,100}\bprompt\s+injection\b",
))


def normalize_text(text: str) -> str:
    """Normalize common visual obfuscation without changing normal meaning."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", normalized)
    normalized = re.sub(r"[\s_.\-/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _matched_categories(text: str) -> list[str]:
    return list(dict.fromkeys(
        category for category, pattern in SUSPICIOUS_PATTERNS
        if pattern.search(text)
    ))


def is_safe_context(text: str) -> bool:
    """Allow educational discussion, but never pair it with secret extraction."""
    educational = any(pattern.search(text) for pattern in SAFE_CONTEXT_PATTERNS)
    quoted_transformation = bool(
        re.match(r"^(?:translate|summarize|paraphrase)\b", text)
        and ("'" in text or '"' in text)
    )
    secret_request = any(
        category in {"PROMPT_EXTRACTION", "SECRET_EXTRACTION"}
        for category in _matched_categories(text)
    )
    return quoted_transformation or (educational and not secret_request)


def detect_prompt_injection(question: str) -> PromptInjectionResult:
    normalized = normalize_text(question)
    if not normalized:
        return PromptInjectionResult(False, 0.0, "EMPTY_INPUT", "No input supplied.")

    if is_safe_context(normalized):
        return PromptInjectionResult(
            False, 0.1, "SAFE_CONTEXT",
            "Educational discussion of prompt injection was detected."
        )

    categories = _matched_categories(normalized)
    if not categories:
        return PromptInjectionResult(False, 0.0, "SAFE", "No known injection pattern detected.")

    score = min(1.0, 0.75 + (0.1 * (len(categories) - 1)))
    return PromptInjectionResult(
        True,
        score,
        "+".join(categories),
        f"Detected {len(categories)} prompt-injection risk category/categories."
    )


def sanitize_untrusted_content(content: str) -> SanitizationResult:
    """Remove malicious instruction lines while retaining benign factual lines."""
    if not content:
        return SanitizationResult("", 0, ())

    kept_lines: list[str] = []
    categories: list[str] = []
    removed = 0

    for line in str(content).splitlines():
        line_categories = _matched_categories(normalize_text(line))
        if line_categories:
            removed += 1
            categories.extend(line_categories)
            continue
        kept_lines.append(line)

    return SanitizationResult(
        "\n".join(kept_lines).strip(),
        removed,
        tuple(dict.fromkeys(categories)),
    )
