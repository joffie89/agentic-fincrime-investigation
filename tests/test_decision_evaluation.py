"""Pure metric tests for the live decision evaluation harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evaluation import decision_eval
from app.evaluation.decision_eval import (
    DecisionEvaluationCase,
    DecisionEvaluationDataset,
    DecisionObservation,
    _fallback_reason,
    _positive_int,
    _rejected_citations,
    _rejected_policies_considered,
    _results_filename,
    _retrieval_source_ids,
    _wilson_score_interval,
    calculate_decision_metrics,
    recompute_metrics_from_results,
    summarize_metric_runs,
)


@pytest.mark.parametrize(
    ("successes", "total", "expected_lower", "expected_upper"),
    [
        (0, 7, 0.0, 0.35433043506668743),
        (5, 10, 0.236593090512564, 0.7634069094874361),
        (20, 20, 0.8388748419471806, 1.0),
    ],
)
def test_wilson_score_interval(
    successes: int,
    total: int,
    expected_lower: float,
    expected_upper: float,
) -> None:
    interval = _wilson_score_interval(successes, total)

    assert interval == {
        "lower": pytest.approx(expected_lower),
        "upper": pytest.approx(expected_upper),
    }


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
        "confidence_interval_95": {
            "lower": pytest.approx(0.15003898915214947),
            "upper": pytest.approx(0.8499610108478506),
        },
    }
    assert metrics["llm_recommendation_accuracy"] == {
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
        "confidence_interval_95": {
            "lower": 0.0,
            "upper": 1.0,
        },
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
    assert metrics["llm_fallback_count_by_reason"] == {"TimeoutError: assessment timed out": 1}
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
    assert completed["routing_rule_fired"] == ("medium_risk_assessment_not_clear")
    assert completed["policy_references"] == ["POL-A"]
    assert completed["policies_considered"] == ["POL-B"]
    assert completed["retrieved_policy_ids"] == ["POL-A", "POL-B"]
    assert completed["semantic_retrieved_policy_ids"] == ["POL-B"]
    assert completed["signal_included_policy_ids"] == ["POL-A"]
    assert completed["rejected_citations"] == ["POL-X"]
    assert completed["rejected_policies_considered"] == ["POL-Y"]
    assert completed["fallback_reason"] is None
    assert completed["llm_reasons"] == ["One risk signal needs supporting information."]

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
    assert _rejected_policies_considered(assessment, audit_record) == ["POL-NOT-REVIEWED"]


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
        "confidence_interval_95": {
            "lower": pytest.approx(0.34238022750665303),
            "upper": 1.0,
        },
    }
    assert metrics["llm_recommendation_accuracy"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
        "confidence_interval_95": {
            "lower": pytest.approx(0.09453120573423074),
            "upper": pytest.approx(0.9054687942657693),
        },
    }
    assert metrics["rubric_override_count"] == 1
    assert metrics["rubric_override_case_names"] == ["overridden escalation"]
    assert metrics["deterministic_citation_case_count"] == 1
    assert metrics["deterministic_citation_count"] == 2
    assert metrics["deterministic_citation_case_names"] == ["overridden escalation"]


def test_llm_only_escalation_counts_use_original_recommendation() -> None:
    def observation(
        name: str,
        expected: str,
        original: str,
    ) -> DecisionObservation:
        return DecisionObservation(
            name=name,
            expected_recommendation=expected,
            actual_recommendation=expected,
            original_llm_recommendation=original,
            expected_requires_human_review=(expected == "ESCALATE_TO_HUMAN"),
            actual_requires_human_review=(expected == "ESCALATE_TO_HUMAN"),
            expected_policy_ids=[],
            actual_policy_ids=[],
            llm_fallback_used=False,
            latency_ms=1.0,
            estimated_cost_usd=0.0,
            completed=True,
        )

    metrics = calculate_decision_metrics(
        [
            observation("llm miss", "ESCALATE_TO_HUMAN", "CLEAR"),
            observation(
                "llm request miss",
                "ESCALATE_TO_HUMAN",
                "REQUEST_MORE_INFO",
            ),
            observation(
                "llm correct escalation",
                "ESCALATE_TO_HUMAN",
                "ESCALATE_TO_HUMAN",
            ),
            observation("llm over clear", "CLEAR", "ESCALATE_TO_HUMAN"),
            observation(
                "llm over request",
                "REQUEST_MORE_INFO",
                "ESCALATE_TO_HUMAN",
            ),
        ]
    )

    assert metrics["expected_escalation_count"] == 3
    assert metrics["llm_only_missed_escalations"] == {
        "count": 2,
        "case_names": ["llm miss", "llm request miss"],
    }
    assert metrics["llm_only_unnecessary_escalations"] == {
        "count": 2,
        "case_names": ["llm over clear", "llm over request"],
    }
    assert metrics["llm_only_missed_escalation_rate"] == {
        "count": 2,
        "total": 3,
        "rate": pytest.approx(2 / 3),
        "confidence_interval_95": {
            "lower": pytest.approx(0.2076596008020477),
            "upper": pytest.approx(0.9385080552796037),
        },
    }


def test_metric_run_summary_reports_mean_and_min_max() -> None:
    summary = summarize_metric_runs(
        [
            {
                "recommendation_accuracy": {"accuracy": 0.5},
                "llm_fallback_count": 0,
                "llm_fallback_count_by_reason": {},
                "latency_ms": {"p50": 10.0},
            },
            {
                "recommendation_accuracy": {"accuracy": 1.0},
                "llm_fallback_count": 2,
                "llm_fallback_count_by_reason": {"TimeoutError": 2},
                "latency_ms": {"p50": 30.0},
            },
        ]
    )

    assert summary["recommendation_accuracy.accuracy"] == {
        "mean": 0.75,
        "min": 0.5,
        "max": 1.0,
    }
    assert summary["llm_fallback_count"] == {
        "mean": 1.0,
        "min": 0.0,
        "max": 2.0,
    }
    assert summary["llm_fallback_count_by_reason.TimeoutError"] == {
        "mean": 1.0,
        "min": 0.0,
        "max": 2.0,
    }
    assert summary["latency_ms.p50"] == {
        "mean": 20.0,
        "min": 10.0,
        "max": 30.0,
    }


def test_case_set_specific_result_filename_preserves_default_name() -> None:
    assert _results_filename(Path("decision_cases.json"), "openai") == (
        "decision_results_openai.json"
    )
    assert (
        _results_filename(
            Path("decision_cases_holdout.json"),
            "openai",
        )
        == "decision_results_openai_holdout.json"
    )
    assert _results_filename(Path("custom_set.json"), "local") == (
        "decision_results_local_custom_set.json"
    )


def test_repeated_evaluation_isolates_runs_and_writes_aggregates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transaction_factory,
) -> None:
    case = DecisionEvaluationCase(
        name="holdout-clean-01",
        category="clean",
        transaction=transaction_factory(transaction_id="txn-holdout-clean-01"),
        expected_recommendation="CLEAR",
        expected_requires_human_review=False,
        expected_policy_ids=[],
    )
    dataset = DecisionEvaluationDataset(cases=[case])
    services: list[object] = []

    def fake_build_service(settings: object) -> object:
        del settings
        service = object()
        services.append(service)
        return service

    def fake_evaluate_case(
        service: object,
        evaluation_case: DecisionEvaluationCase,
    ) -> DecisionObservation:
        run_number = services.index(service) + 1
        return DecisionObservation(
            name=evaluation_case.name,
            expected_recommendation="CLEAR",
            actual_recommendation="CLEAR",
            original_llm_recommendation="CLEAR",
            expected_requires_human_review=False,
            actual_requires_human_review=False,
            expected_policy_ids=[],
            actual_policy_ids=[],
            llm_fallback_used=False,
            latency_ms=float(run_number * 10),
            estimated_cost_usd=float(run_number) / 100,
            completed=True,
        )

    monkeypatch.setattr(decision_eval, "_load_cases", lambda path: dataset)
    monkeypatch.setattr(
        decision_eval,
        "build_investigation_service",
        fake_build_service,
    )
    monkeypatch.setattr(decision_eval, "_evaluate_case", fake_evaluate_case)

    results = decision_eval.run_decision_evaluation(
        "evaluation/decision_cases_holdout.json",
        runs=2,
        output_dir=tmp_path,
        settings=decision_eval.Settings(
            api_key="test-api-key",
            service_mode="local",
            embedding_mode="local",
        ),
    )

    assert len(services) == 2
    assert results["run_count"] == 2
    assert results["case_count"] == 1
    assert results["total_case_evaluations"] == 2
    assert results["openai_assessment_temperature"] == 0.0
    assert len(results["run_results"]) == 2
    assert results["metric_summary"]["latency_ms.p50"] == {
        "mean": 15.0,
        "min": 10.0,
        "max": 20.0,
    }
    assert Path(results["results_path"]).name == ("decision_results_local_holdout.json")
    assert Path(results["results_path"]).is_file()


def test_cli_accepts_cases_and_runs_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    expected_results = {"sentinel": True}

    def fake_run(
        cases_path: Path,
        *,
        runs: int,
        output_dir: Path,
    ) -> dict[str, object]:
        captured.update(
            cases_path=cases_path,
            runs=runs,
            output_dir=output_dir,
        )
        return expected_results

    monkeypatch.setattr(decision_eval, "run_decision_evaluation", fake_run)
    monkeypatch.setattr(
        decision_eval,
        "_print_results",
        lambda results: captured.update(printed=results),
    )

    decision_eval.main(
        [
            "--cases",
            "evaluation/decision_cases_holdout.json",
            "--runs",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert captured == {
        "cases_path": Path("evaluation/decision_cases_holdout.json"),
        "runs": 3,
        "output_dir": tmp_path,
        "printed": expected_results,
    }


def test_from_results_recomputes_fixture_without_running_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observation = DecisionObservation(
        name="offline-clear",
        expected_recommendation="CLEAR",
        actual_recommendation="CLEAR",
        original_llm_recommendation="CLEAR",
        expected_requires_human_review=False,
        actual_requires_human_review=False,
        expected_policy_ids=[],
        actual_policy_ids=[],
        llm_fallback_used=False,
        latency_ms=4.0,
        estimated_cost_usd=0.0,
        completed=True,
    ).model_dump(mode="json")
    results_path = tmp_path / "saved-results.json"
    results_path.write_text(
        json.dumps(
            {
                "service_mode": "openai",
                "case_count": 1,
                "run_count": 1,
                "total_case_evaluations": 1,
                "metrics": {"stale": True},
                "cases": [observation],
                "metric_summary": {},
                "run_results": [
                    {
                        "run": 1,
                        "metrics": {"stale": True},
                        "cases": [observation],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    recomputed = recompute_metrics_from_results(results_path)

    assert recomputed["recomputed_from_results"] is True
    assert recomputed["metrics"]["recommendation_accuracy"]["accuracy"] == 1.0
    assert recomputed["metrics"]["llm_recommendation_accuracy"]["accuracy"] == 1.0

    def fail_if_workflow_runs(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("--from-results must not call run_decision_evaluation")

    printed: dict[str, object] = {}
    monkeypatch.setattr(decision_eval, "run_decision_evaluation", fail_if_workflow_runs)
    monkeypatch.setattr(
        decision_eval,
        "_print_results",
        lambda results: printed.update(results=results),
    )

    decision_eval.main(["--from-results", str(results_path)])

    printed_results = printed["results"]
    assert isinstance(printed_results, dict)
    assert printed_results["recomputed_from_results"] is True
    assert printed_results["metrics"]["recommendation_accuracy"]["accuracy"] == 1.0


def test_repeated_report_includes_no_fallback_metrics_and_reasons(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = {
        "service_mode": "openai",
        "case_count": 20,
        "run_count": 3,
        "total_case_evaluations": 60,
        "results_path": "decision_results_openai_holdout.json",
        "metric_summary": {
            "recommendation_accuracy_excluding_fallbacks.accuracy": {
                "mean": 0.9,
                "min": 0.85,
                "max": 0.95,
            },
            "human_review_routing_accuracy_excluding_fallbacks.accuracy": {
                "mean": 0.95,
                "min": 0.9,
                "max": 1.0,
            },
            "llm_fallback_count_by_reason.TimeoutError: timed out": {
                "mean": 1.0,
                "min": 0.0,
                "max": 2.0,
            },
        },
    }

    decision_eval._print_results(result)

    output = capsys.readouterr().out
    assert "Recommendation accuracy (no fallbacks)" in output
    assert "Human-review routing accuracy (no fallbacks)" in output
    assert "Fallback reason: TimeoutError: timed out" in output
    assert "90.0%" in output
    assert "85.0% - 95.0%" in output


@pytest.mark.parametrize("value", ["0", "-2", "not-a-number"])
def test_runs_option_requires_positive_integer(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int(value)
