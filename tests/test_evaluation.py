"""The evaluation harness reports computed rather than hard-coded metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.evaluate import (
    DEFAULT_DATASET_PATH,
    policy_ids_match_expected,
    run_evaluation,
)


@pytest.mark.parametrize(
    ("cited_policy_ids", "expected_policy_ids", "expected_match"),
    [
        ([], [], True),
        (["POL-001", "POL-002"], ["POL-002", "POL-001"], True),
        (["POL-001"], ["POL-001", "POL-002"], False),
        (["POL-001", "POL-002"], ["POL-001"], False),
    ],
)
def test_policy_ids_must_exactly_match_expected_ids(
    cited_policy_ids: list[str],
    expected_policy_ids: list[str],
    expected_match: bool,
) -> None:
    assert (
        policy_ids_match_expected(cited_policy_ids, expected_policy_ids)
        is expected_match
    )


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

    grounding = results["grounding_checks"]
    assert grounding["correct"] == grounding["total"] == 4
    assert grounding["accuracy"] == 1.0
    assert all(case["grounding_passed"] for case in results["investigation_cases"])

    clean_case = results["investigation_cases"][0]
    assert clean_case["expected_policy_ids"] == []
    assert clean_case["referenced_policy_ids"] == []

