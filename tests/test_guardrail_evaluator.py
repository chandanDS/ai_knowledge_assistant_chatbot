import json

from evaluation.guardrail_evaluator import (
    DEFAULT_DATASET,
    evaluate,
    render_markdown,
    write_reports,
)


def load_dataset():
    return json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))


def test_evaluation_dataset_has_attack_and_safe_input_cases():
    cases = load_dataset()["input_cases"]

    assert any(case["expected_allowed"] is False for case in cases)
    assert any(case["expected_allowed"] is True for case in cases)
    assert len({case["id"] for case in cases}) == len(cases)


def test_evaluator_reports_expected_metrics_and_controls():
    report = evaluate(load_dataset())

    assert report["summary"]["total"] > 0
    assert 0 <= report["summary"]["pass_rate"] <= 1
    assert 0 <= report["input_classification"]["precision"] <= 1
    assert 0 <= report["input_classification"]["recall"] <= 1
    assert {"INPUT", "CONTEXT", "OUTPUT", "RATE_LIMIT"}.issubset(
        report["controls"]
    )


def test_report_writers_create_json_and_markdown(tmp_path):
    report = evaluate(load_dataset())

    json_path, markdown_path = write_reports(report, tmp_path)

    stored = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert stored["summary"] == report["summary"]
    assert "# Guardrail Evaluation Report" in markdown
    assert "Input precision" in markdown


def test_markdown_lists_failed_case_without_raw_prompt():
    report = evaluate(load_dataset())
    report["cases"][0]["passed"] = False

    markdown = render_markdown(report)

    assert report["cases"][0]["case_id"] in markdown
    assert load_dataset()["input_cases"][0]["text"] not in markdown
