"""Integrity checks for the generated 100-case held-out decision set."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from app.evaluation.decision_eval import DecisionEvaluationDataset
from app.models.domain import Recommendation
from evaluation.generate_decision_cases_holdout_v2 import generate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_V2_PATH = PROJECT_ROOT / "evaluation" / "decision_cases_holdout_v2.json"
COMPARISON_PATHS = (
    PROJECT_ROOT / "evaluation" / "decision_cases.json",
    PROJECT_ROOT / "evaluation" / "decision_cases_holdout.json",
    PROJECT_ROOT / "evaluation" / "decision_cases_adversarial.json",
)
ACCOUNT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset() -> DecisionEvaluationDataset:
    return DecisionEvaluationDataset.model_validate(_load_payload(HOLDOUT_V2_PATH))


def _risk_signals(transaction) -> set[str]:
    signals: set[str] = set()
    if transaction.account_age_days < 30:
        signals.add("YOUNG_ACCOUNT")
    if transaction.new_payee:
        signals.add("NEW_PAYEE")
    if transaction.amount > 5 * transaction.average_transaction_amount:
        signals.add("UNUSUAL_AMOUNT")
    if transaction.transactions_last_24h > 10:
        signals.add("HIGH_VELOCITY")
    return signals


def _risk_score(signals: set[str]) -> int:
    points = {
        "YOUNG_ACCOUNT": 20,
        "NEW_PAYEE": 15,
        "UNUSUAL_AMOUNT": 30,
        "HIGH_VELOCITY": 20,
    }
    return sum(points[signal] for signal in signals)


def _rubric_recommendation(signals: set[str]) -> Recommendation:
    if len(signals) >= 2:
        return Recommendation.ESCALATE_TO_HUMAN
    if len(signals) == 1:
        return Recommendation.REQUEST_MORE_INFO
    return Recommendation.CLEAR


def _rubric_policy_ids(transaction, signals: set[str]) -> set[str]:
    policy_ids: set[str] = set()
    if signals & {"YOUNG_ACCOUNT", "NEW_PAYEE", "UNUSUAL_AMOUNT"}:
        policy_ids.add("DEMO-AML-001")
    if "HIGH_VELOCITY" in signals:
        policy_ids.add("DEMO-AML-002")
    if transaction.origin_country != transaction.destination_country:
        policy_ids.add("DEMO-AML-004")
    if _risk_score(signals) >= 60:
        policy_ids.add("DEMO-AML-005")
    if "NEW_PAYEE" in signals:
        policy_ids.add("DEMO-AML-006")
    return policy_ids


def _group(case) -> str:
    prefix = "holdout-v2-"
    transaction_id = case.transaction.transaction_id
    assert transaction_id.startswith(prefix)
    return transaction_id.removeprefix(prefix).split("-", maxsplit=1)[0]


def _all_raw_cases(path: Path) -> list[dict]:
    payload = _load_payload(path)
    return [
        *payload.get("cases", []),
        *payload.get("validation_rejection_cases", []),
    ]


def _canonical_transaction(case: dict) -> str:
    transaction = {
        key: value
        for key, value in case["transaction"].items()
        if key
        not in {
            "transaction_id",
            "origin_account_id",
            "destination_account_id",
        }
    }
    return json.dumps(transaction, sort_keys=True, separators=(",", ":"))


def test_holdout_v2_is_exactly_reproducible_from_fixed_seed() -> None:
    assert generate_dataset() == _load_payload(HOLDOUT_V2_PATH)


def test_holdout_v2_distribution_boundaries_and_identifiers() -> None:
    dataset = _load_dataset()

    assert len(dataset.cases) == 100
    assert dataset.validation_rejection_cases == []
    grouped = Counter(_group(case) for case in dataset.cases)
    assert grouped == {"clean": 25, "single": 25, "multi": 40, "boundary": 10}

    clean = [case for case in dataset.cases if _group(case) == "clean"]
    single = [case for case in dataset.cases if _group(case) == "single"]
    multi = [case for case in dataset.cases if _group(case) == "multi"]
    boundaries = [case for case in dataset.cases if _group(case) == "boundary"]

    assert all(not _risk_signals(case.transaction) for case in clean)
    assert all(len(_risk_signals(case.transaction)) == 1 for case in single)
    assert sum(_risk_signals(case.transaction) == {"HIGH_VELOCITY"} for case in single) >= 8
    assert all(len(_risk_signals(case.transaction)) >= 2 for case in multi)
    assert all(case.expected_recommendation is Recommendation.ESCALATE_TO_HUMAN for case in multi)

    assert all(
        case.transaction.account_age_days in {29, 30}
        or case.transaction.transactions_last_24h in {10, 11}
        or case.transaction.amount == 5 * case.transaction.average_transaction_amount
        for case in boundaries
    )
    assert {29, 30} <= {case.transaction.account_age_days for case in boundaries}
    assert {10, 11} <= {case.transaction.transactions_last_24h for case in boundaries}
    assert any(
        case.transaction.amount == 5 * case.transaction.average_transaction_amount
        for case in boundaries
    )

    for case in dataset.cases:
        transaction = case.transaction
        assert transaction.origin_account_id is not None
        assert transaction.destination_account_id is not None
        assert ACCOUNT_ID_PATTERN.fullmatch(transaction.origin_account_id)
        assert ACCOUNT_ID_PATTERN.fullmatch(transaction.destination_account_id)

    amounts = {case.transaction.amount for case in dataset.cases}
    averages = {case.transaction.average_transaction_amount for case in dataset.cases}
    ages = {case.transaction.account_age_days for case in dataset.cases}
    countries = {
        country
        for case in dataset.cases
        for country in (
            case.transaction.origin_country,
            case.transaction.destination_country,
        )
    }
    assert len(amounts) >= 75
    assert len(averages) >= 9
    assert len(ages) >= 60
    assert len(countries) >= 8
    assert all(amount > 0 for amount in amounts)
    assert all(0 <= age <= 3650 for age in ages)


def test_holdout_v2_labels_recompute_from_written_rubric() -> None:
    dataset = _load_dataset()

    for case in dataset.cases:
        transaction = case.transaction
        signals = _risk_signals(transaction)
        expected_recommendation = _rubric_recommendation(signals)
        score = _risk_score(signals)
        expected_category = "clean" if score == 0 else "high" if score >= 60 else "medium"

        assert case.expected_recommendation is expected_recommendation
        assert case.expected_requires_human_review is (
            expected_recommendation is Recommendation.ESCALATE_TO_HUMAN
        )
        assert set(case.expected_policy_ids) == _rubric_policy_ids(transaction, signals)
        assert case.category == expected_category


def test_holdout_v2_has_no_duplicates_with_existing_decision_sets() -> None:
    generated = _all_raw_cases(HOLDOUT_V2_PATH)
    existing = [case for path in COMPARISON_PATHS for case in _all_raw_cases(path)]

    generated_names = [case["name"] for case in generated]
    generated_ids = [case["transaction"]["transaction_id"] for case in generated]
    generated_transactions = [_canonical_transaction(case) for case in generated]
    assert len(generated_names) == len(set(generated_names))
    assert len(generated_ids) == len(set(generated_ids))
    assert len(generated_transactions) == len(set(generated_transactions))

    assert set(generated_names).isdisjoint(case["name"] for case in existing)
    assert set(generated_ids).isdisjoint(case["transaction"]["transaction_id"] for case in existing)
    assert set(generated_transactions).isdisjoint(_canonical_transaction(case) for case in existing)
