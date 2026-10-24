"""Pure metric tests for the live decision evaluation harness."""

from __future__ import annotations

import pytest

from app.evaluation.decision_eval import (
    DecisionObservation,
    calculate_decision_metrics,
)


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
            "confidence": 0.98,
            "evidence_sufficient": True,
            "routing_rule_fired": "automatic_completion_criteria_met",
            "policy_references": [],
            "policies_considered": ["POL-A"],
            "llm_reasons": ["No relevant concern was identified."],
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
            "confidence": 0.95,
            "evidence_sufficient": True,
            "routing_rule_fired": "automatic_completion_criteria_met",
            "policy_references": [],
            "policies_considered": ["POL-A"],
            "llm_reasons": ["The transaction appears routine."],
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
            "confidence": 0.75,
            "evidence_sufficient": True,
            "routing_rule_fired": "medium_risk_assessment_not_clear",
            "policy_references": ["POL-B"],
            "policies_considered": [],
            "llm_reasons": ["Additional evidence is needed."],
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
            "confidence": 0.8,
            "evidence_sufficient": True,
            "routing_rule_fired": "policy_assessment_requires_human_review",
            "policy_references": ["POL-C"],
            "policies_considered": [],
            "llm_reasons": ["Manual review was recommended."],
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


def test_decision_observation_records_diagnostics_and_failure_defaults() -> None:
    completed = DecisionObservation(
        name="completed case",
        expected_recommendation="REQUEST_MORE_INFO",
        actual_recommendation="REQUEST_MORE_INFO",
        expected_requires_human_review=True,
        actual_requires_human_review=True,
        expected_policy_ids=["POL-A"],
        actual_policy_ids=["POL-A"],
        confidence=0.86,
        evidence_sufficient=True,
        routing_rule_fired="medium_risk_assessment_not_clear",
        policy_references=["POL-A"],
        policies_considered=["POL-B"],
        llm_reasons=["One risk signal needs supporting information."],
        llm_fallback_used=False,
        latency_ms=12.0,
        estimated_cost_usd=0.001,
        completed=True,
    ).model_dump(mode="json")

    assert completed["confidence"] == pytest.approx(0.86)
    assert completed["evidence_sufficient"] is True
    assert completed["routing_rule_fired"] == (
        "medium_risk_assessment_not_clear"
    )
    assert completed["policy_references"] == ["POL-A"]
    assert completed["policies_considered"] == ["POL-B"]
    assert completed["llm_reasons"] == [
        "One risk signal needs supporting information."
    ]

    failed = DecisionObservation(
        name="failed case",
        expected_recommendation="ESCALATE_TO_HUMAN",
        actual_recommendation=None,
        expected_requires_human_review=True,
        actual_requires_human_review=None,
        expected_policy_ids=["POL-A"],
        actual_policy_ids=[],
        llm_fallback_used=False,
        latency_ms=5.0,
        estimated_cost_usd=0.0,
        completed=False,
        workflow_error="RuntimeError: workflow failed",
    ).model_dump(mode="json")

    assert failed["confidence"] is None
    assert failed["evidence_sufficient"] is None
    assert failed["routing_rule_fired"] is None
    assert failed["policy_references"] == []
    assert failed["policies_considered"] == []
    assert failed["llm_reasons"] == []
