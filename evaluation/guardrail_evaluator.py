"""Run repeatable, offline guardrail evaluations and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from security.guardrails import EnterpriseGuardrails, GuardrailConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = Path(__file__).resolve().parent / "cases" / "guardrail_cases.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    control: str
    passed: bool
    expected: str
    actual: str
    categories: tuple[str, ...]


def _safe_divide(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _classification_metrics(results: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(not item["expected_allowed"] and not item["actual_allowed"] for item in results)
    fp = sum(item["expected_allowed"] and not item["actual_allowed"] for item in results)
    tn = sum(item["expected_allowed"] and item["actual_allowed"] for item in results)
    fn = sum(not item["expected_allowed"] and item["actual_allowed"] for item in results)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        "accuracy": _safe_divide(tp + tn, len(results)),
        "false_positive_rate": _safe_divide(fp, fp + tn),
    }


def evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    guardrails = EnterpriseGuardrails()
    case_results: list[CaseResult] = []
    input_predictions = []

    for case in dataset["input_cases"]:
        decision = guardrails.inspect_input(case["text"])
        expected_allowed = bool(case["expected_allowed"])
        passed = decision.allowed == expected_allowed
        input_predictions.append({
            "expected_allowed": expected_allowed,
            "actual_allowed": decision.allowed,
        })
        case_results.append(CaseResult(
            case["id"], "INPUT", passed,
            "ALLOW" if expected_allowed else "BLOCK",
            decision.action, decision.categories,
        ))

    for case in dataset["context_cases"]:
        decision = guardrails.sanitize_context(case["text"])
        preserved = all(value in decision.safe_text for value in case["must_preserve"])
        removed = all(value not in decision.safe_text for value in case["must_remove"])
        case_results.append(CaseResult(
            case["id"], "CONTEXT", preserved and removed,
            "preserve facts; remove malicious instructions",
            decision.action, decision.categories,
        ))

    for case in dataset["output_cases"]:
        decision = guardrails.inspect_output(case["text"])
        removed = all(value not in decision.safe_text for value in case["must_remove"])
        passed = decision.action == case["expected_action"] and removed
        case_results.append(CaseResult(
            case["id"], "OUTPUT", passed,
            case["expected_action"], decision.action, decision.categories,
        ))

    rate_guardrails = EnterpriseGuardrails(GuardrailConfig(
        rate_limit_requests=2,
        rate_limit_window_seconds=60,
    ))
    first = rate_guardrails.inspect_rate_limit("evaluation-session", now=1)
    second = rate_guardrails.inspect_rate_limit("evaluation-session", now=2)
    third = rate_guardrails.inspect_rate_limit("evaluation-session", now=3)
    rate_passed = first.allowed and second.allowed and not third.allowed
    case_results.append(CaseResult(
        "session_rate_limit", "RATE_LIMIT", rate_passed,
        "ALLOW, ALLOW, BLOCK",
        f"{first.action}, {second.action}, {third.action}",
        third.categories,
    ))

    passed_count = sum(result.passed for result in case_results)
    control_totals: dict[str, dict[str, int | float]] = {}
    for control in sorted({result.control for result in case_results}):
        selected = [result for result in case_results if result.control == control]
        passed = sum(result.passed for result in selected)
        control_totals[control] = {
            "passed": passed,
            "total": len(selected),
            "pass_rate": _safe_divide(passed, len(selected)),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(DEFAULT_DATASET.relative_to(PROJECT_ROOT)),
        "summary": {
            "passed": passed_count,
            "failed": len(case_results) - passed_count,
            "total": len(case_results),
            "pass_rate": _safe_divide(passed_count, len(case_results)),
        },
        "input_classification": _classification_metrics(input_predictions),
        "controls": control_totals,
        "cases": [asdict(result) for result in case_results],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    metrics = report["input_classification"]
    lines = [
        "# Guardrail Evaluation Report",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Summary",
        "",
        f"- Passed: {summary['passed']}/{summary['total']}",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        f"- Input precision: {metrics['precision']:.2%}",
        f"- Input recall: {metrics['recall']:.2%}",
        f"- Input F1: {metrics['f1_score']:.2%}",
        f"- False-positive rate: {metrics['false_positive_rate']:.2%}",
        "",
        "## Control results",
        "",
        "| Control | Passed | Total | Pass rate |",
        "|---|---:|---:|---:|",
    ]
    for control, values in report["controls"].items():
        lines.append(
            f"| {control} | {values['passed']} | {values['total']} | "
            f"{values['pass_rate']:.2%} |"
        )

    lines.extend(["", "## Failed cases", ""])
    failures = [case for case in report["cases"] if not case["passed"]]
    if not failures:
        lines.append("No cases failed.")
    else:
        for case in failures:
            lines.append(
                f"- `{case['case_id']}` ({case['control']}): expected "
                f"{case['expected']}; received {case['actual']}."
            )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest.json"
    markdown_path = output_dir / "latest.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = evaluate(dataset)
    json_path, markdown_path = write_reports(report, args.output_dir)
    summary = report["summary"]
    print(f"Guardrail evaluation: {summary['passed']}/{summary['total']} passed")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
