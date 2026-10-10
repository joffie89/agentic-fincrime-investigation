"""Evaluate policy retrieval with labelled analyst-style queries."""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from app.config import EmbeddingSettings, RetrievalMode
from app.models.base import DomainModel, Identifier, LongText
from app.models.domain import RetrievedPolicy
from app.rag.embedding_provider import (
    build_embedding_service,
    resolve_policy_index_path,
)
from app.rag.embeddings import EmbeddingService, LocalHashEmbeddingService
from app.rag.ingestion import PolicyIngestor
from app.rag.retriever import PolicyRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "data" / "policies.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "policy_index.json"
RETRIEVAL_DEPTH = 100
EVALUATION_MODES = (
    RetrievalMode.VECTOR,
    RetrievalMode.BM25,
    RetrievalMode.HYBRID,
)


class PolicyRetrievalCase(DomainModel):
    """One query and the policies considered relevant to it."""

    name: Identifier
    query: LongText
    expected_policy_ids: list[Identifier] = Field(min_length=1)


class PolicyRetrievalDataset(DomainModel):
    """Validated retrieval evaluation dataset."""

    cases: list[PolicyRetrievalCase] = Field(min_length=1)


def calculate_retrieval_metrics(
    rankings: Sequence[Sequence[str]],
    expected_policy_ids: Sequence[Collection[str]],
) -> dict[str, float]:
    """Calculate policy-level hit@1, hit@3, and mean reciprocal rank."""

    if len(rankings) != len(expected_policy_ids):
        raise ValueError("Rankings and expected policy IDs must have equal lengths")
    if not rankings:
        raise ValueError("At least one retrieval result is required")

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_total = 0.0

    for ranked_ids, expected_ids in zip(
        rankings,
        expected_policy_ids,
        strict=True,
    ):
        relevant_ids = set(expected_ids)
        if not relevant_ids:
            raise ValueError("Each query must have at least one expected policy ID")

        hit_at_1 += int(bool(ranked_ids) and ranked_ids[0] in relevant_ids)
        hit_at_3 += int(any(item in relevant_ids for item in ranked_ids[:3]))
        first_relevant_rank = next(
            (
                rank
                for rank, policy_id in enumerate(ranked_ids, start=1)
                if policy_id in relevant_ids
            ),
            None,
        )
        if first_relevant_rank is not None:
            reciprocal_rank_total += 1 / first_relevant_rank

    total = len(rankings)
    return {
        "hit@1": hit_at_1 / total,
        "hit@3": hit_at_3 / total,
        "mrr": reciprocal_rank_total / total,
    }


def _load_dataset(path: Path) -> PolicyRetrievalDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read retrieval evaluation cases: {path}") from exc
    return PolicyRetrievalDataset.model_validate(payload)


def _ensure_index(
    policy_path: Path,
    index_path: Path,
    embedding_service: EmbeddingService,
) -> None:
    if not index_path.exists():
        PolicyIngestor(embedding_service).ingest_file(
            policy_path,
            index_path,
        )


def _unique_policy_ids(results: Sequence[RetrievedPolicy]) -> list[str]:
    return list(dict.fromkeys(result.policy_id for result in results))


def run_retrieval_evaluation(
    cases_path: str | Path = DEFAULT_CASES_PATH,
    *,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, Any]:
    """Compare every retrieval mode over the same labelled queries."""

    cases_source = Path(cases_path)
    policy_source = Path(policy_path)
    index_source = Path(index_path)
    service = embedding_service or LocalHashEmbeddingService()
    dataset = _load_dataset(cases_source)
    _ensure_index(policy_source, index_source, service)
    results_by_mode: dict[str, dict[str, Any]] = {}
    for mode in EVALUATION_MODES:
        retriever = PolicyRetriever.from_index(
            index_source,
            service,
            policy_source_path=policy_source,
            retrieval_mode=mode,
        )

        rankings: list[list[str]] = []
        expected: list[list[str]] = []
        details: list[dict[str, Any]] = []
        for case in dataset.cases:
            results = retriever.retrieve(case.query, top_k=RETRIEVAL_DEPTH)
            ranked_policy_ids = _unique_policy_ids(results)
            rankings.append(ranked_policy_ids)
            expected.append(case.expected_policy_ids)
            details.append(
                {
                    "name": case.name,
                    "expected_policy_ids": case.expected_policy_ids,
                    "returned_policy_ids": ranked_policy_ids,
                }
            )

        results_by_mode[mode.value] = {
            "metrics": calculate_retrieval_metrics(rankings, expected),
            "cases": details,
        }

    hybrid_results = results_by_mode[RetrievalMode.HYBRID.value]
    return {
        "case_count": len(dataset.cases),
        "modes": results_by_mode,
        # Preserve the earlier top-level view for callers that consumed the
        # single-mode evaluator; it now represents the application default.
        "metrics": hybrid_results["metrics"],
        "cases": hybrid_results["cases"],
    }


def _print_results(results: dict[str, Any]) -> None:
    print("Policy RAG retrieval evaluation")
    print(f"Queries: {results['case_count']}")
    print()
    print("Mode    hit@1   hit@3   MRR")
    print("------  ------  ------  ------")
    for mode in EVALUATION_MODES:
        metrics = results["modes"][mode.value]["metrics"]
        print(
            f"{mode.value:<6}  {metrics['hit@1']:.4f}  "
            f"{metrics['hit@3']:.4f}  {metrics['mrr']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to labelled retrieval evaluation cases",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Path to source policy documents",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to the policy retrieval index (defaults by provider)",
    )
    arguments = parser.parse_args()
    settings = EmbeddingSettings()
    embedding_service = build_embedding_service(settings)
    index_path = resolve_policy_index_path(settings, arguments.index)
    if not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path
    results = run_retrieval_evaluation(
        arguments.cases,
        policy_path=arguments.policies,
        index_path=index_path,
        embedding_service=embedding_service,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
