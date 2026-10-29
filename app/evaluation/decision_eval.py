"""Run labelled decisions through the complete investigation workflow."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictBool

from app.config import PersistenceBackend, Settings
from app.models.base import DomainModel, Identifier, ShortText, UnitInterval
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


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for repeatable CLI runs."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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
    original_llm_recommendation: Recommendation | None = None
    rubric_override: StrictBool = False
    deterministic_citations: list[Identifier] = Field(default_factory=list)
    retrieved_policy_ids: list[Identifier] = Field(default_factory=list)
    semantic_retrieved_policy_ids: list[Identifier] = Field(default_factory=list)
    signal_included_policy_ids: list[Identifier] = Field(default_factory=list)
    confidence: UnitInterval | None = None
    evidence_sufficient: StrictBool | None = None
    routing_rule_fired: Identifier | None = None
    policy_references: list[Identifier] = Field(default_factory=list)
    policies_considered: list[Identifier] = Field(default_factory=list)
    rejected_citations: list[str] = Field(default_factory=list)
    rejected_policies_considered: list[str] = Field(default_factory=list)
    llm_reasons: list[ShortText] = Field(default_factory=list)
    llm_fallback_used: StrictBool
    fallback_reason: str | None = None
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
        "accuracy": correct / total if total else 0.0,
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
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def calculate_decision_metrics(
    observations: Sequence[DecisionObservation | Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate decision, routing, citation, latency, fallback, and cost metrics."""

    if not observations:
        raise ValueError("At least one decision observation is required")
    validated = [
        item if isinstance(item, DecisionObservation) else DecisionObservation.model_validate(item)
        for item in observations
    ]
    total = len(validated)
    recommendation_correct = sum(
        item.completed and item.actual_recommendation is item.expected_recommendation
        for item in validated
    )
    llm_scored = [
        item
        for item in validated
        if item.completed and item.original_llm_recommendation is not None
    ]
    llm_recommendation_correct = sum(
        item.original_llm_recommendation is item.expected_recommendation for item in llm_scored
    )
    routing_correct = sum(
        item.completed and item.actual_requires_human_review is item.expected_requires_human_review
        for item in validated
    )
    without_fallbacks = [item for item in validated if not item.llm_fallback_used]
    recommendation_correct_without_fallbacks = sum(
        item.completed and item.actual_recommendation is item.expected_recommendation
        for item in without_fallbacks
    )
    routing_correct_without_fallbacks = sum(
        item.completed and item.actual_requires_human_review is item.expected_requires_human_review
        for item in without_fallbacks
    )
    citation_correct = sum(
        item.completed and set(item.actual_policy_ids) == set(item.expected_policy_ids)
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
    expected_policy_count = 0
    retrieved_expected_policy_count = 0
    missing_policy_counts: Counter[str] = Counter()
    cases_with_complete_retrieval = 0
    for item in validated:
        expected_ids = set(item.expected_policy_ids)
        retrieved_ids = set(item.retrieved_policy_ids)
        expected_policy_count += len(expected_ids)
        retrieved_expected_policy_count += len(expected_ids & retrieved_ids)
        missing_ids = expected_ids - retrieved_ids
        missing_policy_counts.update(missing_ids)
        if not missing_ids:
            cases_with_complete_retrieval += 1

    fallback_reasons: Counter[str] = Counter()
    for item in validated:
        if item.llm_fallback_used:
            fallback_reasons[item.fallback_reason or "unknown"] += 1

    rubric_override_cases = [item.name for item in validated if item.rubric_override]
    deterministic_citation_cases = [item.name for item in validated if item.deterministic_citations]

    return {
        "case_count": total,
        "recommendation_accuracy": _accuracy(recommendation_correct, total),
        "llm_recommendation_accuracy": _accuracy(
            llm_recommendation_correct,
            len(llm_scored),
        ),
        "recommendation_accuracy_excluding_fallbacks": _accuracy(
            recommendation_correct_without_fallbacks,
            len(without_fallbacks),
        ),
        "human_review_routing_accuracy": _accuracy(routing_correct, total),
        "human_review_routing_accuracy_excluding_fallbacks": _accuracy(
            routing_correct_without_fallbacks,
            len(without_fallbacks),
        ),
        "missed_escalations": {
            "count": len(missed),
            "case_names": missed,
        },
        "unnecessary_escalations": {
            "count": len(unnecessary),
            "case_names": unnecessary,
        },
        "exact_citation_match_rate": _accuracy(citation_correct, total),
        "llm_fallback_count": sum(item.llm_fallback_used for item in validated),
        "llm_fallback_count_by_reason": dict(sorted(fallback_reasons.items())),
        "rubric_override_count": len(rubric_override_cases),
        "rubric_override_case_names": rubric_override_cases,
        "deterministic_citation_case_count": len(deterministic_citation_cases),
        "deterministic_citation_count": sum(
            len(item.deterministic_citations) for item in validated
        ),
        "deterministic_citation_case_names": deterministic_citation_cases,
        "retrieval_recall": {
            "retrieved_expected_policy_count": retrieved_expected_policy_count,
            "expected_policy_count": expected_policy_count,
            "missing_expected_policy_count": (
                expected_policy_count - retrieved_expected_policy_count
            ),
            "recall": (
                retrieved_expected_policy_count / expected_policy_count
                if expected_policy_count
                else 1.0
            ),
            "cases_with_complete_retrieval": cases_with_complete_retrieval,
            "case_count": total,
            "missing_policy_ids": dict(sorted(missing_policy_counts.items())),
        },
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


def _flatten_numeric_metrics(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten numeric metric leaves into stable dotted paths."""

    flattened: dict[str, float] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_numeric_metrics(item, prefix=path))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            flattened[path] = float(item)
    return flattened


def summarize_metric_runs(
    run_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return mean and range for every numeric metric across repeated runs.

    Dynamic counter entries, such as individual fallback reasons, are treated
    as zero in runs where they do not occur. This makes their means and ranges
    describe all requested runs rather than only the runs containing the key.
    """

    if not run_metrics:
        raise ValueError("At least one run of metrics is required")
    flattened_runs = [_flatten_numeric_metrics(metrics) for metrics in run_metrics]
    metric_paths = sorted(set().union(*(set(metrics) for metrics in flattened_runs)))
    summary: dict[str, dict[str, float]] = {}
    for path in metric_paths:
        values = [metrics.get(path, 0.0) for metrics in flattened_runs]
        summary[path] = {
            "mean": math.fsum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _results_filename(cases_path: Path, service_mode: str) -> str:
    """Name results for their mode and case set without breaking the default."""

    case_stem = cases_path.stem
    default_stem = DEFAULT_CASES_PATH.stem
    if case_stem == default_stem:
        case_suffix = ""
    elif case_stem.startswith(f"{default_stem}_"):
        case_suffix = case_stem[len(default_stem) :]
    else:
        case_suffix = f"_{case_stem}"
    return f"decision_results_{service_mode}{case_suffix}.json"


def _policy_fallback_used(audit_record: Mapping[str, Any] | None) -> bool:
    if audit_record is None:
        return False
    fallbacks = audit_record.get("fallbacks", [])
    return any(
        isinstance(item, Mapping) and item.get("step") == "policy_validation" for item in fallbacks
    )


def _routing_rule_fired(audit_record: Mapping[str, Any] | None) -> str | None:
    if audit_record is None:
        return None
    routing = audit_record.get("routing")
    if not isinstance(routing, Mapping):
        return None
    rule = routing.get("triggered_rule")
    return rule if isinstance(rule, str) and rule else None


def _string_list(value: Any) -> list[str]:
    """Return de-duplicated string values from an audit/result field."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _retrieval_source_ids(
    audit_record: Mapping[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Read semantic and signal-derived policy IDs from current or older audits."""

    if audit_record is None:
        return [], []
    sources = audit_record.get("policy_retrieval_sources")
    if isinstance(sources, Mapping):
        semantic = _string_list(sources.get("semantic_search"))
        signal = _string_list(sources.get("signal_mapping"))
    else:
        semantic = []
        signal = []
    # Keep the evaluator able to read intermediate/older result files that
    # recorded these fields flat while the audit schema was being introduced.
    if not semantic:
        semantic = _string_list(audit_record.get("semantic_retrieved_policy_ids"))
    if not signal:
        signal = _string_list(audit_record.get("signal_included_policy_ids"))
    return semantic, signal


def _fallback_reason(
    policy_assessment: Any,
    audit_record: Mapping[str, Any] | None,
) -> str | None:
    """Read the safe exception summary attached to an assessment or audit."""

    reason = getattr(policy_assessment, "fallback_reason", None)
    if isinstance(reason, str) and reason:
        return reason
    if audit_record is None:
        return None
    recorded_assessment = audit_record.get("policy_assessment")
    if isinstance(recorded_assessment, Mapping):
        reason = recorded_assessment.get("fallback_reason")
        if isinstance(reason, str) and reason:
            return reason
    for fallback in audit_record.get("fallbacks", []):
        if not isinstance(fallback, Mapping):
            continue
        if fallback.get("step") != "policy_validation":
            continue
        reason = fallback.get("fallback_reason")
        if isinstance(reason, str) and reason:
            return reason
        exception_type = fallback.get("exception_type") or fallback.get("error_type")
        exception_message = fallback.get("exception_message") or fallback.get("message")
        if isinstance(exception_type, str) and exception_type:
            if isinstance(exception_message, str) and exception_message:
                return f"{exception_type}: {exception_message}"
            return exception_type
    return None


def _rejected_citations(
    policy_assessment: Any,
    audit_record: Mapping[str, Any] | None,
) -> list[str]:
    citations = _string_list(getattr(policy_assessment, "rejected_citations", None))
    if citations or audit_record is None:
        return citations
    recorded_assessment = audit_record.get("policy_assessment")
    if isinstance(recorded_assessment, Mapping):
        return _string_list(recorded_assessment.get("rejected_citations"))
    return []


def _rejected_policies_considered(
    policy_assessment: Any,
    audit_record: Mapping[str, Any] | None,
) -> list[str]:
    rejected = _string_list(getattr(policy_assessment, "rejected_policies_considered", None))
    if rejected or audit_record is None:
        return rejected
    recorded_assessment = audit_record.get("policy_assessment")
    if isinstance(recorded_assessment, Mapping):
        return _string_list(recorded_assessment.get("rejected_policies_considered"))
    return _string_list(audit_record.get("rejected_policies_considered"))


def _recorded_assessment_value(
    policy_assessment: Any,
    audit_record: Mapping[str, Any] | None,
    field_name: str,
) -> Any:
    """Read an enforcement diagnostic from the response or persisted audit."""

    value = getattr(policy_assessment, field_name, None)
    if value is not None:
        return value
    if audit_record is None:
        return None
    recorded_assessment = audit_record.get("policy_assessment")
    if isinstance(recorded_assessment, Mapping):
        value = recorded_assessment.get(field_name)
        if value is not None:
            return value
    return audit_record.get(field_name)


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
            expected_requires_human_review=(case.expected_requires_human_review),
            actual_requires_human_review=None,
            expected_policy_ids=case.expected_policy_ids,
            actual_policy_ids=[],
            original_llm_recommendation=None,
            rubric_override=False,
            deterministic_citations=[],
            retrieved_policy_ids=[],
            semantic_retrieved_policy_ids=[],
            signal_included_policy_ids=[],
            confidence=None,
            evidence_sufficient=None,
            routing_rule_fired=None,
            policy_references=[],
            policies_considered=[],
            rejected_citations=[],
            rejected_policies_considered=[],
            llm_reasons=[],
            llm_fallback_used=False,
            fallback_reason=None,
            latency_ms=float(latency_ms),
            estimated_cost_usd=0.0,
            completed=False,
            workflow_error=f"{type(exc).__name__}: {exc}",
        )

    latency_ms = (time.perf_counter() - started_at) * 1_000
    try:
        audit_record = service.get_investigation(result.investigation_id)
    except Exception as exc:
        logger.exception(
            "decision_evaluation_audit_lookup_failed",
            extra={
                "event": "decision_evaluation_audit_lookup_failed",
                "case_name": case.name,
                "investigation_id": result.investigation_id,
                "exception_type": type(exc).__name__,
            },
        )
        audit_record = None
    usage = result.llm_usage.model_dump(mode="json") if result.llm_usage is not None else None
    cost = result.llm_usage.estimated_cost_usd if result.llm_usage is not None else 0.0
    policy_references = list(result.policy_assessment.policy_references)
    semantic_policy_ids, signal_policy_ids = _retrieval_source_ids(audit_record)
    fallback_reason = _fallback_reason(result.policy_assessment, audit_record)
    original_llm_recommendation = _recorded_assessment_value(
        result.policy_assessment,
        audit_record,
        "original_llm_recommendation",
    )
    rubric_override = _recorded_assessment_value(
        result.policy_assessment,
        audit_record,
        "rubric_override",
    )
    deterministic_citations = _string_list(
        _recorded_assessment_value(
            result.policy_assessment,
            audit_record,
            "deterministic_citations",
        )
    )
    return DecisionObservation(
        name=case.name,
        expected_recommendation=case.expected_recommendation,
        actual_recommendation=result.recommendation,
        expected_requires_human_review=case.expected_requires_human_review,
        actual_requires_human_review=result.requires_human_review,
        expected_policy_ids=case.expected_policy_ids,
        actual_policy_ids=policy_references,
        original_llm_recommendation=original_llm_recommendation,
        rubric_override=(rubric_override if isinstance(rubric_override, bool) else False),
        deterministic_citations=deterministic_citations,
        retrieved_policy_ids=list(
            dict.fromkeys(item.policy_id for item in result.retrieved_policies)
        ),
        semantic_retrieved_policy_ids=semantic_policy_ids,
        signal_included_policy_ids=signal_policy_ids,
        confidence=result.policy_assessment.confidence,
        evidence_sufficient=result.policy_assessment.evidence_sufficient,
        routing_rule_fired=_routing_rule_fired(audit_record),
        policy_references=policy_references,
        policies_considered=list(result.policy_assessment.policies_considered),
        rejected_citations=_rejected_citations(
            result.policy_assessment,
            audit_record,
        ),
        rejected_policies_considered=_rejected_policies_considered(
            result.policy_assessment,
            audit_record,
        ),
        llm_reasons=list(result.policy_assessment.reasons),
        llm_fallback_used=(fallback_reason is not None or _policy_fallback_used(audit_record)),
        fallback_reason=fallback_reason,
        latency_ms=float(latency_ms),
        estimated_cost_usd=float(cost),
        completed=True,
        investigation_id=result.investigation_id,
        llm_usage=usage,
    )


def run_decision_evaluation(
    cases_path: str | Path = DEFAULT_CASES_PATH,
    *,
    runs: int = 1,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the case set one or more times and write JSON results."""

    if runs < 1:
        raise ValueError("runs must be at least 1")

    cases_source = Path(cases_path)
    dataset = _load_cases(cases_source)
    configured_settings = settings or Settings(persistence_backend=PersistenceBackend.MEMORY)
    evaluation_settings = configured_settings.model_copy(
        update={
            "persistence_backend": PersistenceBackend.MEMORY,
            "database_url": None,
        }
    )
    run_results: list[dict[str, Any]] = []
    for run_number in range(1, runs + 1):
        # Each run needs isolated graph/repository state because cases reuse
        # transaction IDs and a prior run would otherwise look like a replay.
        service = build_investigation_service(evaluation_settings)
        observations = [_evaluate_case(service, case) for case in dataset.cases]
        run_results.append(
            {
                "run": run_number,
                "metrics": calculate_decision_metrics(observations),
                "cases": [observation.model_dump(mode="json") for observation in observations],
            }
        )

    metric_summary = summarize_metric_runs([run["metrics"] for run in run_results])
    output_path = Path(output_dir) / _results_filename(
        cases_source,
        evaluation_settings.service_mode.value,
    )
    first_run = run_results[0]
    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(cases_source.resolve()),
        "service_mode": evaluation_settings.service_mode.value,
        "embedding_mode": evaluation_settings.embedding_mode.value,
        "retrieval_mode": evaluation_settings.retrieval_mode.value,
        "openai_assessment_temperature": (evaluation_settings.openai_assessment_temperature),
        "case_count": len(dataset.cases),
        "run_count": runs,
        "total_case_evaluations": len(dataset.cases) * runs,
        "cost_scope": ("Policy-assessment LLM usage only; embedding API cost is not included."),
        # Keep the original single-run fields for existing analysis scripts.
        # For repeated evaluations these are the first run; all runs and their
        # aggregate statistics are available in the fields below.
        "metrics": first_run["metrics"],
        "cases": first_run["cases"],
        "metric_summary": metric_summary,
        "run_results": run_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results["results_path"] = str(output_path.resolve())
    return results


def _print_results(results: Mapping[str, Any]) -> None:
    run_count = int(results.get("run_count", 1))
    if run_count > 1:
        _print_repeated_results(results)
        return

    metrics = results["metrics"]
    recommendation = metrics["recommendation_accuracy"]
    llm_recommendation = metrics["llm_recommendation_accuracy"]
    recommendation_without_fallbacks = metrics["recommendation_accuracy_excluding_fallbacks"]
    routing = metrics["human_review_routing_accuracy"]
    routing_without_fallbacks = metrics["human_review_routing_accuracy_excluding_fallbacks"]
    citations = metrics["exact_citation_match_rate"]
    retrieval = metrics["retrieval_recall"]
    latency = metrics["latency_ms"]
    cost = metrics["cost_usd"]
    rows = [
        (
            "Recommendation accuracy",
            f"{recommendation['correct']}/{recommendation['total']} "
            f"({recommendation['accuracy']:.1%})",
        ),
        (
            "LLM-own recommendation accuracy",
            f"{llm_recommendation['correct']}/"
            f"{llm_recommendation['total']} "
            f"({llm_recommendation['accuracy']:.1%})",
        ),
        (
            "Recommendation accuracy (no fallbacks)",
            f"{recommendation_without_fallbacks['correct']}/"
            f"{recommendation_without_fallbacks['total']} "
            f"({recommendation_without_fallbacks['accuracy']:.1%})",
        ),
        (
            "Human-review routing accuracy",
            f"{routing['correct']}/{routing['total']} ({routing['accuracy']:.1%})",
        ),
        (
            "Human-review routing accuracy (no fallbacks)",
            f"{routing_without_fallbacks['correct']}/"
            f"{routing_without_fallbacks['total']} "
            f"({routing_without_fallbacks['accuracy']:.1%})",
        ),
        ("MISSED ESCALATIONS", str(metrics["missed_escalations"]["count"])),
        (
            "Unnecessary escalations",
            str(metrics["unnecessary_escalations"]["count"]),
        ),
        (
            "Exact citation match rate",
            f"{citations['correct']}/{citations['total']} ({citations['accuracy']:.1%})",
        ),
        (
            "Expected-policy retrieval recall",
            f"{retrieval['retrieved_expected_policy_count']}/"
            f"{retrieval['expected_policy_count']} "
            f"({retrieval['recall']:.1%})",
        ),
        ("LLM fallback count", str(metrics["llm_fallback_count"])),
        ("Rubric override cases", str(metrics["rubric_override_count"])),
        (
            "Cases with deterministic citations",
            str(metrics["deterministic_citation_case_count"]),
        ),
        (
            "LLM fallback reasons",
            ", ".join(
                f"{reason} ({count})"
                for reason, count in metrics["llm_fallback_count_by_reason"].items()
            )
            or "none",
        ),
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


_REPEATED_RESULT_ROWS = (
    ("recommendation_accuracy.accuracy", "Recommendation accuracy", "percent"),
    (
        "llm_recommendation_accuracy.accuracy",
        "LLM-own recommendation accuracy",
        "percent",
    ),
    (
        "recommendation_accuracy_excluding_fallbacks.accuracy",
        "Recommendation accuracy (no fallbacks)",
        "percent",
    ),
    (
        "human_review_routing_accuracy.accuracy",
        "Human-review routing accuracy",
        "percent",
    ),
    (
        "human_review_routing_accuracy_excluding_fallbacks.accuracy",
        "Human-review routing accuracy (no fallbacks)",
        "percent",
    ),
    ("missed_escalations.count", "MISSED ESCALATIONS", "number"),
    ("unnecessary_escalations.count", "Unnecessary escalations", "number"),
    (
        "exact_citation_match_rate.accuracy",
        "Exact citation match rate",
        "percent",
    ),
    ("retrieval_recall.recall", "Expected-policy retrieval recall", "percent"),
    ("llm_fallback_count", "LLM fallback count", "number"),
    ("rubric_override_count", "Rubric override cases", "number"),
    (
        "deterministic_citation_case_count",
        "Cases with deterministic citations",
        "number",
    ),
    (
        "deterministic_citation_count",
        "Deterministic citations added",
        "number",
    ),
    ("workflow_failure_count", "Workflow failure count", "number"),
    ("latency_ms.p50", "End-to-end latency p50", "milliseconds"),
    ("latency_ms.p95", "End-to-end latency p95", "milliseconds"),
    ("cost_usd.total", "Total estimated LLM cost", "currency"),
    ("cost_usd.average_per_case", "Average LLM cost per case", "currency"),
)


def _format_summary_value(value: float, style: str) -> str:
    if style == "percent":
        return f"{value:.1%}"
    if style == "milliseconds":
        return f"{value:.1f} ms"
    if style == "currency":
        return f"${value:.6f}"
    return f"{value:.2f}"


def _print_repeated_results(results: Mapping[str, Any]) -> None:
    """Print aggregate mean and min-max values for repeated live runs."""

    summary = results["metric_summary"]
    rows: list[tuple[str, str, str]] = []
    for path, label, style in _REPEATED_RESULT_ROWS:
        values = summary.get(path)
        if not isinstance(values, Mapping):
            continue
        mean = _format_summary_value(float(values["mean"]), style)
        minimum = _format_summary_value(float(values["min"]), style)
        maximum = _format_summary_value(float(values["max"]), style)
        rows.append((label, mean, f"{minimum} - {maximum}"))

    fallback_prefix = "llm_fallback_count_by_reason."
    for path in sorted(summary):
        if not path.startswith(fallback_prefix):
            continue
        values = summary[path]
        reason = path.removeprefix(fallback_prefix)
        mean = _format_summary_value(float(values["mean"]), "number")
        minimum = _format_summary_value(float(values["min"]), "number")
        maximum = _format_summary_value(float(values["max"]), "number")
        rows.append(
            (
                f"Fallback reason: {reason}",
                mean,
                f"{minimum} - {maximum}",
            )
        )

    label_width = max(len(label) for label, _, _ in rows)
    print(f"Decision evaluation ({results['service_mode']})")
    print(
        f"Cases: {results['case_count']} x {results['run_count']} runs "
        f"= {results['total_case_evaluations']} evaluations"
    )
    print()
    print(f"{'Metric':<{label_width}}  {'Mean':>14}  Min - max")
    print(f"{'-' * label_width}  {'-' * 14}  {'-' * 24}")
    for label, mean, value_range in rows:
        print(f"{label:<{label_width}}  {mean:>14}  {value_range}")
    print()
    print(f"Results written to: {results['results_path']}")


def main(argv: Sequence[str] | None = None) -> None:
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
        help="Directory for the case-set-specific results JSON file",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=1,
        help="Number of independent repetitions (default: 1)",
    )
    arguments = parser.parse_args(argv)
    results = run_decision_evaluation(
        arguments.cases,
        runs=arguments.runs,
        output_dir=arguments.output_dir,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
