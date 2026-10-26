"""Pure metric tests for the live decision evaluation harness."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evaluation.decision_eval import (
    DecisionObservation,
    _fallback_reason,
    _rejected_citations,
    _rejected_policies_considered,
    _retrieval_source_ids,
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
            "retrieved_policy_ids": ["POL-A"],
            "semantic_retrieved_policy_ids": ["POL-A"],
            "signal_included_policy_ids": [],
            "confidence": 0.98,
            "evidence_sufficient": True,
            "routing_rule_fired": "automatic_completion_criteria_met",
            "policy_references": [],
            "policies_considered": ["POL-A"],
            "rejected_citations": [],
            "llm_reasons": ["No relevant concern was identified."],
            "llm_fallback_used": False,
            "fallback_reason": None,
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
            "retrieved_policy_ids": [],
            "semantic_retrieved_policy_ids": [],
            "signal_included_policy_ids": [],
            "confidence": 0.95,
            "evidence_sufficient": True,
            "routing_rule_fired": "automatic_completion_criteria_met",
            "policy_references": [],
            "policies_considered": ["POL-A"],
            "rejected_citations": ["POL-NOT-PROVIDED"],
            "llm_reasons": ["The transaction appears routine."],
            "llm_fallback_used": True,
            "fallback_reason": "TimeoutError: assessment timed out",
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
            "retrieved_policy_ids": ["POL-B"],
            "semantic_retrieved_policy_ids": [],
            "signal_included_policy_ids": ["POL-B"],
            "confidence": 0.75,
            "evidence_sufficient": True,
            "routing_rule_fired": "medium_risk_assessment_not_clear",
            "policy_references": ["POL-B"],
            "policies_considered": [],
            "rejected_citations": [],
            "llm_reasons": ["Additional evidence is needed."],
            "llm_fallback_used": False,
            "fallback_reason": None,
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
            "retrieved_policy_ids": ["POL-C"],
            "semantic_retrieved_policy_ids": ["POL-C"],
            "signal_included_policy_ids": [],
            "confidence": 0.8,
            "evidence_sufficient": True,
            "routing_rule_fired": "policy_assessment_requires_human_review",
            "policy_references": ["POL-C"],
            "policies_considered": [],
            "rejected_citations": [],
            "llm_reasons": ["Manual review was recommended."],
            "llm_fallback_used": False,
            "fallback_reason": None,
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
    assert metrics["llm_recommendation_accuracy"] == {
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
    }
    assert metrics["recommendation_accuracy_excluding_fallbacks"] == {
        "correct": 2,
        "total": 3,
        "accuracy": pytest.approx(2 / 3),
    }
    assert metrics["human_review_routing_accuracy"] == {
        "correct": 2,
        "total": 4,
        "accuracy": 0.5,
    }
    assert metrics["human_review_routing_accuracy_excluding_fallbacks"] == {
        "correct": 2,
        "total": 3,
        "accuracy": pytest.approx(2 / 3),
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
    assert metrics["llm_fallback_count_by_reason"] == {
        "TimeoutError: assessment timed out": 1
    }
    assert metrics["rubric_override_count"] == 0
    assert metrics["rubric_override_case_names"] == []
    assert metrics["deterministic_citation_case_count"] == 0
    assert metrics["deterministic_citation_count"] == 0
    assert metrics["deterministic_citation_case_names"] == []
    assert metrics["retrieval_recall"] == {
        "retrieved_expected_policy_count": 1,
        "expected_policy_count": 2,
        "missing_expected_policy_count": 1,
        "recall": 0.5,
        "cases_with_complete_retrieval": 3,
        "case_count": 4,
        "missing_policy_ids": {"POL-A": 1},
    }
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
        original_llm_recommendation="CLEAR",
        rubric_override=True,
        deterministic_citations=["POL-A"],
        retrieved_policy_ids=["POL-A", "POL-B"],
        semantic_retrieved_policy_ids=["POL-B"],
        signal_included_policy_ids=["POL-A"],
        confidence=0.86,
        evidence_sufficient=True,
        routing_rule_fired="medium_risk_assessment_not_clear",
        policy_references=["POL-A"],
        policies_considered=["POL-B"],
        rejected_citations=["POL-X"],
        rejected_policies_considered=["POL-Y"],
        llm_reasons=["One risk signal needs supporting information."],
        llm_fallback_used=False,
        fallback_reason=None,
        latency_ms=12.0,
        estimated_cost_usd=0.001,
        completed=True,
    ).model_dump(mode="json")

    assert completed["confidence"] == pytest.approx(0.86)
    assert completed["original_llm_recommendation"] == "CLEAR"
    assert completed["rubric_override"] is True
    assert completed["deterministic_citations"] == ["POL-A"]
    assert completed["evidence_sufficient"] is True
    assert completed["routing_rule_fired"] == (
        "medium_risk_assessment_not_clear"
    )
    assert completed["policy_references"] == ["POL-A"]
    assert completed["policies_considered"] == ["POL-B"]
    assert completed["retrieved_policy_ids"] == ["POL-A", "POL-B"]
    assert completed["semantic_retrieved_policy_ids"] == ["POL-B"]
    assert completed["signal_included_policy_ids"] == ["POL-A"]
    assert completed["rejected_citations"] == ["POL-X"]
    assert completed["rejected_policies_considered"] == ["POL-Y"]
    assert completed["fallback_reason"] is None
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
    assert failed["original_llm_recommendation"] is None
    assert failed["rubric_override"] is False
    assert failed["deterministic_citations"] == []
    assert failed["evidence_sufficient"] is None
    assert failed["routing_rule_fired"] is None
    assert failed["policy_references"] == []
    assert failed["policies_considered"] == []
    assert failed["retrieved_policy_ids"] == []
    assert failed["semantic_retrieved_policy_ids"] == []
    assert failed["signal_included_policy_ids"] == []
    assert failed["rejected_citations"] == []
    assert failed["rejected_policies_considered"] == []
    assert failed["fallback_reason"] is None
    assert failed["llm_reasons"] == []


def test_decision_diagnostics_read_current_audit_shape() -> None:
    audit_record = {
        "policy_retrieval_sources": {
            "semantic_search": ["POL-SEMANTIC"],
            "signal_mapping": ["POL-SIGNAL", "POL-SEMANTIC"],
        },
        "policy_assessment": {
            "rejected_citations": ["POL-INVENTED"],
            "rejected_policies_considered": ["POL-NOT-REVIEWED"],
        },
        "fallbacks": [
            {
                "step": "policy_validation",
                "error_type": "ValidationError",
                "message": "invalid structured response",
            }
        ],
    }
    assessment = SimpleNamespace(
        fallback_reason=None,
        rejected_citations=[],
        rejected_policies_considered=[],
    )

    assert _retrieval_source_ids(audit_record) == (
        ["POL-SEMANTIC"],
        ["POL-SIGNAL", "POL-SEMANTIC"],
    )
    assert _fallback_reason(assessment, audit_record) == (
        "ValidationError: invalid structured response"
    )
    assert _rejected_citations(assessment, audit_record) == ["POL-INVENTED"]
    assert _rejected_policies_considered(assessment, audit_record) == [
        "POL-NOT-REVIEWED"
    ]


def test_metrics_keep_llm_accuracy_visible_before_rubric_enforcement() -> None:
    observations = [
        DecisionObservation(
            name="overridden escalation",
            expected_recommendation="ESCALATE_TO_HUMAN",
            actual_recommendation="ESCALATE_TO_HUMAN",
            original_llm_recommendation="CLEAR",
            rubric_override=True,
            deterministic_citations=["DEMO-AML-004", "DEMO-AML-005"],
            expected_requires_human_review=True,
            actual_requires_human_review=True,
            expected_policy_ids=["DEMO-AML-004", "DEMO-AML-005"],
            actual_policy_ids=["DEMO-AML-004", "DEMO-AML-005"],
            llm_fallback_used=False,
            latency_ms=10.0,
            estimated_cost_usd=0.01,
            completed=True,
        ),
        DecisionObservation(
            name="unaltered clear",
            expected_recommendation="CLEAR",
            actual_recommendation="CLEAR",
            original_llm_recommendation="CLEAR",
            expected_requires_human_review=False,
            actual_requires_human_review=False,
            expected_policy_ids=[],
            actual_policy_ids=[],
            llm_fallback_used=False,
            latency_ms=5.0,
            estimated_cost_usd=0.0,
            completed=True,
        ),
    ]

    metrics = calculate_decision_metrics(observations)

    assert metrics["recommendation_accuracy"] == {
        "correct": 2,
        "total": 2,
        "accuracy": 1.0,
    }
    assert metrics["llm_recommendation_accuracy"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert metrics["rubric_override_count"] == 1
    assert metrics["rubric_override_case_names"] == [
        "overridden escalation"
    ]
    assert metrics["deterministic_citation_case_count"] == 1
    assert metrics["deterministic_citation_count"] == 2
    assert metrics["deterministic_citation_case_names"] == [
        "overridden escalation"
    ]
