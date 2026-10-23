"""Pure metric tests for the live decision evaluation harness."""

from __future__ import annotations

import pytest

from app.evaluation.decision_eval import calculate_decision_metrics


def test_decision_metrics_cover_accuracy_escalations_latency_and_cost() -> None:
    observations = [
        {
            "name": "correct clean",
            "expected_recommendation": "CLEAR",
            "actual_recommendation": "CLEAR",
            "expected_requires_human_review": False,
            "actual_requires_human_review": False,
            "expected_policy_ids": [],
            "actual_policy_ids": [],
            "llm_fallback_used": False,
            "latency_ms": 10.0,
            "estimated_cost_usd": 0.0,
            "completed": True,
        },
        {
            "name": "missed escalation",
            "expected_recommendation": "ESCALATE_TO_HUMAN",
            "actual_recommendation": "CLEAR",
            "expected_requires_human_review": True,
            "actual_requires_human_review": False,
            "expected_policy_ids": ["POL-A"],
            "actual_policy_ids": [],
            "llm_fallback_used": True,
            "latency_ms": 20.0,
            "estimated_cost_usd": 0.02,
            "completed": True,
        },
        {
            "name": "correct review",
            "expected_recommendation": "REQUEST_MORE_INFO",
            "actual_recommendation": "REQUEST_MORE_INFO",
            "expected_requires_human_review": True,
            "actual_requires_human_review": True,
            "expected_policy_ids": ["POL-B"],
            "actual_policy_ids": ["POL-B"],
            "llm_fallback_used": False,
            "latency_ms": 30.0,
            "estimated_cost_usd": 0.03,
            "completed": True,
        },
        {
            "name": "unnecessary escalation",
            "expected_recommendation": "CLEAR",
            "actual_recommendation": "ESCALATE_TO_HUMAN",
            "expected_requires_human_review": False,
            "actual_requires_human_review": True,
            "expected_policy_ids": [],
            "actual_policy_ids": ["POL-C"],
            "llm_fallback_used": False,
            "latency_ms": 40.0,
            "estimated_cost_usd": 0.05,
            "completed": True,
        },
    ]

    metrics = calculate_decision_metrics(observations)

    assert metrics["recommendation_accuracy"] == {
        "correct": 2,
        "total": 4,
        "accuracy": 0.5,
    }
    assert metrics["human_review_routing_accuracy"] == {
        "correct": 2,
        "total": 4,
        "accuracy": 0.5,
    }
    assert metrics["missed_escalations"] == {
        "count": 1,
        "case_names": ["missed escalation"],
    }
    assert metrics["unnecessary_escalations"] == {
        "count": 1,
        "case_names": ["unnecessary escalation"],
    }
    assert metrics["exact_citation_match_rate"] == {
        "correct": 2,
        "total": 4,
        "accuracy": 0.5,
    }
    assert metrics["llm_fallback_count"] == 1
    assert metrics["workflow_failure_count"] == 0
    assert metrics["latency_ms"]["p50"] == pytest.approx(25.0)
    assert metrics["latency_ms"]["p95"] == pytest.approx(38.5)
    assert metrics["cost_usd"]["total"] == pytest.approx(0.10)
    assert metrics["cost_usd"]["average_per_case"] == pytest.approx(0.025)
