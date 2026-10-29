"""Run measurable checks over a small synthetic evaluation dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import Field

from app.config import PersistenceBackend, ServiceMode, Settings
from app.models.base import DomainModel, Identifier, LongText
from app.models.domain import Recommendation, RiskLevel
from app.models.transaction import Transaction
from app.rag.embeddings import LocalHashEmbeddingService
from app.rag.ingestion import PolicyIngestor
from app.rag.retriever import PolicyRetriever
from app.services.investigation_service import build_investigation_service
from app.services.risk_scorer import assess_transaction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "cases.json"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "data" / "policies.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "policy_index.json"


class RiskEvaluationCase(DomainModel):
    name: Identifier
    transaction: Transaction
    expected_score: int = Field(ge=0, le=100)
    expected_level: RiskLevel


class RetrievalEvaluationCase(DomainModel):
    name: Identifier
    query: LongText
    expected_policy_id: Identifier
    top_k: int = Field(default=3, ge=1, le=20)


class InvestigationEvaluationCase(DomainModel):
    name: Identifier
    transaction: Transaction
    expected_recommendation: Recommendation
    expect_human_review: bool
    expected_policy_ids: list[Identifier]


class EvaluationDataset(DomainModel):
    risk_cases: list[RiskEvaluationCase] = Field(min_length=1)
    retrieval_cases: list[RetrievalEvaluationCase] = Field(min_length=1)
    investigation_cases: list[InvestigationEvaluationCase] = Field(min_length=1)


def _metric(correct: int, total: int) -> dict[str, int | float]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
    }


def policy_ids_match_expected(
    cited_policy_ids: Collection[str],
    expected_policy_ids: Collection[str],
) -> bool:
    """Return whether the cited and expected policy ID sets match exactly."""

    return set(cited_policy_ids) == set(expected_policy_ids)


def _load_dataset(path: Path) -> EvaluationDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read evaluation dataset: {path}") from exc
    return EvaluationDataset.model_validate(payload)


def _ensure_index(policy_path: Path, index_path: Path) -> None:
    if not index_path.exists():
        PolicyIngestor(LocalHashEmbeddingService()).ingest_file(policy_path, index_path)


def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> dict[str, Any]:
    """Execute evaluation cases and return metrics actually observed."""

    dataset_source = Path(dataset_path)
    policy_source = Path(policy_path)
    index_source = Path(index_path)
    dataset = _load_dataset(dataset_source)
    _ensure_index(policy_source, index_source)

    risk_correct = 0
    risk_details: list[dict[str, Any]] = []
    for case in dataset.risk_cases:
        actual = assess_transaction(case.transaction)
        passed = (
            actual.risk_score == case.expected_score and actual.risk_level is case.expected_level
        )
        risk_correct += int(passed)
        risk_details.append(
            {
                "name": case.name,
                "passed": passed,
                "expected_score": case.expected_score,
                "actual_score": actual.risk_score,
                "expected_level": case.expected_level.value,
                "actual_level": actual.risk_level.value,
            }
        )

    retriever = PolicyRetriever.from_index(index_source, LocalHashEmbeddingService())
    retrieval_correct = 0
    retrieval_details: list[dict[str, Any]] = []
    for case in dataset.retrieval_cases:
        actual = retriever.retrieve(case.query, top_k=case.top_k)
        returned_ids = [item.policy_id for item in actual]
        passed = case.expected_policy_id in returned_ids
        retrieval_correct += int(passed)
        retrieval_details.append(
            {
                "name": case.name,
                "passed": passed,
                "expected_policy_id": case.expected_policy_id,
                "returned_policy_ids": returned_ids,
            }
        )

    settings = Settings(
        _env_file=None,
        service_mode=ServiceMode.LOCAL,
        persistence_backend=PersistenceBackend.MEMORY,
        policy_index_path=index_source,
    )
    service = build_investigation_service(settings)
    recommendation_correct = 0
    grounding_correct = 0
    escalation_correct = 0
    investigation_details: list[dict[str, Any]] = []
    for case in dataset.investigation_cases:
        result = service.investigate(case.transaction)
        recommendation_passed = result.recommendation is case.expected_recommendation
        escalation_passed = result.requires_human_review is case.expect_human_review
        retrieved_ids = {item.policy_id for item in result.retrieved_policies}
        referenced_ids = set(result.policy_assessment.policy_references)
        expected_policy_ids = set(case.expected_policy_ids)
        grounding_passed = policy_ids_match_expected(
            referenced_ids,
            expected_policy_ids,
        )
        recommendation_correct += int(recommendation_passed)
        grounding_correct += int(grounding_passed)
        escalation_correct += int(escalation_passed)
        investigation_details.append(
            {
                "name": case.name,
                "expected_recommendation": case.expected_recommendation.value,
                "actual_recommendation": result.recommendation.value,
                "recommendation_passed": recommendation_passed,
                "expected_human_review": case.expect_human_review,
                "actual_human_review": result.requires_human_review,
                "escalation_passed": escalation_passed,
                "grounding_passed": grounding_passed,
                "retrieved_policy_ids": sorted(retrieved_ids),
                "expected_policy_ids": sorted(expected_policy_ids),
                "referenced_policy_ids": sorted(referenced_ids),
            }
        )

    return {
        "dataset": str(dataset_source),
        "risk_rule_correctness": {
            **_metric(risk_correct, len(dataset.risk_cases)),
            "cases": risk_details,
        },
        "retrieval_quality": {
            **_metric(retrieval_correct, len(dataset.retrieval_cases)),
            "metric": "expected policy present in Top-K",
            "cases": retrieval_details,
        },
        "recommendation_correctness": {
            **_metric(
                recommendation_correct,
                len(dataset.investigation_cases),
            ),
        },
        "grounding_checks": {
            **_metric(grounding_correct, len(dataset.investigation_cases)),
            "metric": "cited policy IDs exactly match expected policy IDs",
        },
        "escalation_behavior": {
            **_metric(escalation_correct, len(dataset.investigation_cases)),
        },
        "investigation_cases": investigation_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the synthetic evaluation JSON file",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to the local policy vector index",
    )
    arguments = parser.parse_args()
    results = run_evaluation(arguments.dataset, index_path=arguments.index)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
