import pytest

from app.evaluation.retrieval_eval import calculate_retrieval_metrics


def test_retrieval_metrics_use_first_relevant_rank() -> None:
    rankings = [
        ["POL-A", "POL-B", "POL-C"],
        ["POL-C", "POL-B", "POL-A"],
        ["POL-D"],
    ]
    expected_policy_ids = [
        {"POL-A"},
        {"POL-B", "POL-X"},
        {"POL-A"},
    ]

    metrics = calculate_retrieval_metrics(rankings, expected_policy_ids)

    assert metrics["hit@1"] == pytest.approx(1 / 3)
    assert metrics["hit@3"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx(0.5)
