"""Offline CI gate for decision-evaluation routing and escalation regressions."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.config import (
    EmbeddingMode,
    PersistenceBackend,
    RetrievalMode,
    ServiceMode,
    Settings,
)
from app.evaluation.decision_eval import run_decision_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_INDEX_PATH = PROJECT_ROOT / "data" / "policy_index.json"
CASE_PATHS = (
    PROJECT_ROOT / "evaluation" / "decision_cases.json",
    PROJECT_ROOT / "evaluation" / "decision_cases_adversarial.json",
)


def _regression_failures(dataset_name: str, metrics: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    routing = metrics["human_review_routing_accuracy"]
    routing_correct = int(routing["correct"])
    routing_total = int(routing["total"])
    routing_misses = int(metrics["missed_escalations"]["count"])

    if routing_misses > 0:
        failures.append(f"{dataset_name}: system missed escalations = {routing_misses}, expected 0")
    if routing_correct != routing_total:
        failures.append(
            f"{dataset_name}: routing accuracy = {routing_correct}/{routing_total}, expected 100%"
        )
    return failures


def run_regression_gate() -> None:
    """Run both labelled sets through the deterministic local workflow."""

    if not LOCAL_INDEX_PATH.is_file():
        raise SystemExit(f"Local policy index is missing: {LOCAL_INDEX_PATH}")

    settings = Settings(
        _env_file=None,
        api_key="ci-local-decision-gate-key",
        service_mode=ServiceMode.LOCAL,
        llm_assessment_enabled=True,
        embedding_mode=EmbeddingMode.LOCAL,
        retrieval_mode=RetrievalMode.HYBRID,
        persistence_backend=PersistenceBackend.MEMORY,
        policy_index_path=LOCAL_INDEX_PATH,
        qa_sample_rate=0.0,
    )
    failures: list[str] = []

    with TemporaryDirectory(prefix="fincrime-decision-gate-") as temporary_directory:
        output_dir = Path(temporary_directory)
        for cases_path in CASE_PATHS:
            results = run_decision_evaluation(
                cases_path,
                output_dir=output_dir,
                settings=settings,
            )
            metrics = results["metrics"]
            failures.extend(_regression_failures(cases_path.name, metrics))
            routing = metrics["human_review_routing_accuracy"]
            print(
                f"{cases_path.name}: routing "
                f"{routing['correct']}/{routing['total']}; "
                f"missed escalations {metrics['missed_escalations']['count']}"
            )

    if failures:
        raise SystemExit("Decision regression gate failed:\n- " + "\n- ".join(failures))

    print("Decision regression gate passed.")


if __name__ == "__main__":
    run_regression_gate()
