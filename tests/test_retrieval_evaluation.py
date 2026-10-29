import json
from collections import Counter

import pytest

from app.evaluation.retrieval_eval import (
    DEFAULT_CASES_PATH,
    DEFAULT_POLICY_PATH,
    calculate_retrieval_metrics,
)


def test_bundled_dataset_covers_each_synthetic_policy_twice() -> None:
    policies = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))["policies"]
    cases = json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8"))["cases"]
    policy_ids = {policy["policy_id"] for policy in policies}
    case_counts = Counter(
        case["expected_policy_ids"][0] for case in cases if len(case["expected_policy_ids"]) == 1
    )

    assert len(policies) == 30
    assert len(policy_ids) == 30
    assert len(cases) == 60
    assert all(policy["source"] == "synthetic - written for this demo" for policy in policies)
    assert all(80 <= len(policy["text"].split()) <= 150 for policy in policies)
    assert all(len(case["expected_policy_ids"]) == 1 for case in cases)
    assert case_counts == Counter({policy_id: 2 for policy_id in policy_ids})


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
