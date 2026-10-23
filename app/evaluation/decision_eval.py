"""Run labelled decisions through the complete investigation workflow."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictBool

from app.config import PersistenceBackend, Settings
from app.models.base import DomainModel, Identifier
from app.models.domain import Recommendation
from app.models.transaction import Transaction
from app.services.investigation_service import (
    InvestigationService,
    build_investigation_service,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation" / "decision_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation"


class DecisionEvaluationCase(DomainModel):
    """One labelled synthetic transaction and its expected workflow outcome."""

    name: Identifier
    category: Literal["clean", "high", "medium"]
    transaction: Transaction
    expected_recommendation: Recommendation
    expected_requires_human_review: StrictBool
    expected_policy_ids: list[Identifier]


class DecisionEvaluationDataset(DomainModel):
    """Validated collection of live workflow decision cases."""

    cases: list[DecisionEvaluationCase] = Field(min_length=1)


class DecisionObservation(DomainModel):
    """Expected and observed values needed by the pure metric calculator."""

    name: Identifier
    expected_recommendation: Recommendation
    actual_recommendation: Recommendation | None
    expected_requires_human_review: StrictBool
    actual_requires_human_review: StrictBool | None
    expected_policy_ids: list[Identifier]
    actual_policy_ids: list[Identifier]
    llm_fallback_used: StrictBool
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    estimated_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    completed: StrictBool
    investigation_id: Identifier | None = None
    llm_usage: dict[str, Any] | None = None
    workflow_error: str | None = None


def _accuracy(correct: int, total: int) -> dict[str, int | float]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def calculate_decision_metrics(
    observations: Sequence[DecisionObservation | Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate decision, routing, citation, latency, fallback, and cost metrics."""

    if not observations:
        raise ValueError("At least one decision observation is required")
    validated = [
        item
        if isinstance(item, DecisionObservation)
        else DecisionObservation.model_validate(item)
        for item in observations
    ]
    total = len(validated)
    recommendation_correct = sum(
        item.completed
        and item.actual_recommendation is item.expected_recommendation
        for item in validated
    )
    routing_correct = sum(
        item.completed
        and item.actual_requires_human_review
        is item.expected_requires_human_review
        for item in validated
    )
    citation_correct = sum(
        item.completed
        and set(item.actual_policy_ids) == set(item.expected_policy_ids)
        for item in validated
    )
    missed = [
        item.name
        for item in validated
        if item.completed
        and item.expected_requires_human_review
        and item.actual_requires_human_review is False
    ]
    unnecessary = [
        item.name
        for item in validated
        if item.completed
        and not item.expected_requires_human_review
        and item.actual_requires_human_review is True
    ]
    latencies = [item.latency_ms for item in validated]
    total_cost = math.fsum(item.estimated_cost_usd for item in validated)

    return {
        "case_count": total,
        "recommendation_accuracy": _accuracy(recommendation_correct, total),
        "human_review_routing_accuracy": _accuracy(routing_correct, total),
        "missed_escalations": {
            "count": len(missed),
            "case_names": missed,
        },
        "unnecessary_escalations": {
            "count": len(unnecessary),
            "case_names": unnecessary,
        },
        "exact_citation_match_rate": _accuracy(citation_correct, total),
        "llm_fallback_count": sum(
            item.llm_fallback_used for item in validated
        ),
        "workflow_failure_count": sum(not item.completed for item in validated),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "cost_usd": {
            "total": total_cost,
            "average_per_case": total_cost / total,
        },
    }


def _load_cases(path: Path) -> DecisionEvaluationDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read decision evaluation cases: {path}") from exc
    return DecisionEvaluationDataset.model_validate(payload)


def _policy_fallback_used(audit_record: Mapping[str, Any] | None) -> bool:
    if audit_record is None:
        return False
    fallbacks = audit_record.get("fallbacks", [])
    return any(
        isinstance(item, Mapping) and item.get("step") == "policy_validation"
        for item in fallbacks
    )


def _evaluate_case(
    service: InvestigationService,
    case: DecisionEvaluationCase,
) -> DecisionObservation:
    started_at = time.perf_counter()
    try:
        result = service.investigate(case.transaction)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started_at) * 1_000
        logger.exception(
            "decision_evaluation_case_failed",
            extra={
                "event": "decision_evaluation_case_failed",
                "case_name": case.name,
                "transaction_id": case.transaction.transaction_id,
                "exception_type": type(exc).__name__,
            },
        )
        return DecisionObservation(
            name=case.name,
            expected_recommendation=case.expected_recommendation,
            actual_recommendation=None,
            expected_requires_human_review=(
                case.expected_requires_human_review
            ),
            actual_requires_human_review=None,
            expected_policy_ids=case.expected_policy_ids,
            actual_policy_ids=[],
            llm_fallback_used=False,
            latency_ms=float(latency_ms),
            estimated_cost_usd=0.0,
            completed=False,
            workflow_error=f"{type(exc).__name__}: {exc}",
        )

    latency_ms = (time.perf_counter() - started_at) * 1_000
    audit_record = service.get_investigation(result.investigation_id)
    usage = (
        result.llm_usage.model_dump(mode="json")
        if result.llm_usage is not None
        else None
    )
    cost = (
        result.llm_usage.estimated_cost_usd
        if result.llm_usage is not None
        else 0.0
    )
    return DecisionObservation(
        name=case.name,
        expected_recommendation=case.expected_recommendation,
        actual_recommendation=result.recommendation,
        expected_requires_human_review=case.expected_requires_human_review,
        actual_requires_human_review=result.requires_human_review,
        expected_policy_ids=case.expected_policy_ids,
        actual_policy_ids=list(result.policy_assessment.policy_references),
        llm_fallback_used=_policy_fallback_used(audit_record),
        latency_ms=float(latency_ms),
        estimated_cost_usd=float(cost),
        completed=True,
        investigation_id=result.investigation_id,
        llm_usage=usage,
    )


def run_decision_evaluation(
    cases_path: str | Path = DEFAULT_CASES_PATH,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run every case through the configured full workflow and write JSON results."""

    cases_source = Path(cases_path)
    dataset = _load_cases(cases_source)
    configured_settings = settings or Settings(
        persistence_backend=PersistenceBackend.MEMORY
    )
    evaluation_settings = configured_settings.model_copy(
        update={
            "persistence_backend": PersistenceBackend.MEMORY,
            "database_url": None,
        }
    )
    service = build_investigation_service(evaluation_settings)
    observations = [
        _evaluate_case(service, case)
        for case in dataset.cases
    ]
    metrics = calculate_decision_metrics(observations)
    output_path = Path(output_dir) / (
        f"decision_results_{evaluation_settings.service_mode.value}.json"
    )
    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(cases_source.resolve()),
        "service_mode": evaluation_settings.service_mode.value,
        "embedding_mode": evaluation_settings.embedding_mode.value,
        "retrieval_mode": evaluation_settings.retrieval_mode.value,
        "case_count": len(observations),
        "cost_scope": (
            "Policy-assessment LLM usage only; embedding API cost is not included."
        ),
        "metrics": metrics,
        "cases": [
            observation.model_dump(mode="json")
            for observation in observations
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results["results_path"] = str(output_path.resolve())
    return results


def _print_results(results: Mapping[str, Any]) -> None:
    metrics = results["metrics"]
    recommendation = metrics["recommendation_accuracy"]
    routing = metrics["human_review_routing_accuracy"]
    citations = metrics["exact_citation_match_rate"]
    latency = metrics["latency_ms"]
    cost = metrics["cost_usd"]
    rows = [
        (
            "Recommendation accuracy",
            f"{recommendation['correct']}/{recommendation['total']} "
            f"({recommendation['accuracy']:.1%})",
        ),
        (
            "Human-review routing accuracy",
            f"{routing['correct']}/{routing['total']} "
            f"({routing['accuracy']:.1%})",
        ),
        ("MISSED ESCALATIONS", str(metrics["missed_escalations"]["count"])),
        (
            "Unnecessary escalations",
            str(metrics["unnecessary_escalations"]["count"]),
        ),
        (
            "Exact citation match rate",
            f"{citations['correct']}/{citations['total']} "
            f"({citations['accuracy']:.1%})",
        ),
        ("LLM fallback count", str(metrics["llm_fallback_count"])),
        ("Workflow failure count", str(metrics["workflow_failure_count"])),
        ("End-to-end latency p50", f"{latency['p50']:.1f} ms"),
        ("End-to-end latency p95", f"{latency['p95']:.1f} ms"),
        ("Total estimated LLM cost", f"${cost['total']:.6f}"),
        (
            "Average LLM cost per case",
            f"${cost['average_per_case']:.6f}",
        ),
    ]
    label_width = max(len(label) for label, _ in rows)
    print(f"Decision evaluation ({results['service_mode']})")
    print(f"Cases: {results['case_count']}")
    print()
    print(f"{'Metric':<{label_width}}  Result")
    print(f"{'-' * label_width}  {'-' * 20}")
    for label, value in rows:
        print(f"{label:<{label_width}}  {value}")
    print()
    print(f"Results written to: {results['results_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to labelled decision evaluation cases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for decision_results_<mode>.json",
    )
    arguments = parser.parse_args()
    results = run_decision_evaluation(
        arguments.cases,
        output_dir=arguments.output_dir,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
