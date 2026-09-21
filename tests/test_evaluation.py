"""The evaluation harness reports computed rather than hard-coded metrics."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.evaluate import DEFAULT_DATASET_PATH, run_evaluation


def test_evaluation_runs_against_synthetic_data(tmp_path: Path):
    results = run_evaluation(
        DEFAULT_DATASET_PATH,
        index_path=tmp_path / "policy-index.json",
    )

    for metric_name in (
        "risk_rule_correctness",
        "retrieval_quality",
        "recommendation_correctness",
        "grounding_checks",
        "escalation_behavior",
    ):
        metric = results[metric_name]
        assert metric["total"] > 0
        assert metric["correct"] <= metric["total"]
        assert metric["accuracy"] == round(
            metric["correct"] / metric["total"], 4
        )

    assert len(results["risk_rule_correctness"]["cases"]) == 4
    assert len(results["retrieval_quality"]["cases"]) == 4
    assert len(results["investigation_cases"]) == 4

