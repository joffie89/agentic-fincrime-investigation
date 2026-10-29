"""Integrity checks for the held-out decision-evaluation cases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.evaluation.decision_eval import DecisionEvaluationDataset
from app.models.domain import Recommendation, RiskLevel
from app.services.risk_scorer import assess_transaction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CASES_PATH = PROJECT_ROOT / "evaluation" / "decision_cases.json"
HOLDOUT_CASES_PATH = PROJECT_ROOT / "evaluation" / "decision_cases_holdout.json"


def _load_dataset(path: Path) -> DecisionEvaluationDataset:
    return DecisionEvaluationDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _risk_signals(case) -> set[str]:
    transaction = case.transaction
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


def _expected_policy_ids(case, signals: set[str]) -> set[str]:
    transaction = case.transaction
    risk = assess_transaction(transaction)
    policy_ids: set[str] = set()
    if signals & {"YOUNG_ACCOUNT", "NEW_PAYEE", "UNUSUAL_AMOUNT"}:
        policy_ids.add("DEMO-AML-001")
    if "NEW_PAYEE" in signals:
        policy_ids.add("DEMO-AML-006")
    if "HIGH_VELOCITY" in signals:
        policy_ids.add("DEMO-AML-002")
    if transaction.origin_country != transaction.destination_country:
        policy_ids.add("DEMO-AML-004")
    if risk.risk_level is RiskLevel.HIGH:
        policy_ids.add("DEMO-AML-005")
    return policy_ids


def test_holdout_dataset_is_new_and_has_requested_group_sizes() -> None:
    training = _load_dataset(TRAINING_CASES_PATH)
    holdout = _load_dataset(HOLDOUT_CASES_PATH)

    assert len(holdout.cases) == 20
    group_counts = Counter(case.transaction.transaction_id.split("-")[1] for case in holdout.cases)
    assert group_counts == {
        "clean": 5,
        "single": 6,
        "multi": 5,
        "boundary": 4,
    }

    training_ids = {case.transaction.transaction_id for case in training.cases}
    holdout_ids = [case.transaction.transaction_id for case in holdout.cases]
    assert len(holdout_ids) == len(set(holdout_ids))
    assert training_ids.isdisjoint(holdout_ids)

    training_amounts = {case.transaction.amount for case in training.cases}
    holdout_amounts = {case.transaction.amount for case in holdout.cases}
    assert training_amounts.isdisjoint(holdout_amounts)

    training_countries = {
        country
        for case in training.cases
        for country in (
            case.transaction.origin_country,
            case.transaction.destination_country,
        )
    }
    holdout_countries = {
        country
        for case in holdout.cases
        for country in (
            case.transaction.origin_country,
            case.transaction.destination_country,
        )
    }
    assert training_countries.isdisjoint(holdout_countries)


def test_holdout_labels_follow_rubric_boundaries_and_citation_rules() -> None:
    holdout = _load_dataset(HOLDOUT_CASES_PATH)
    app_patterns = 0

    for case in holdout.cases:
        transaction = case.transaction
        signals = _risk_signals(case)
        if "NETWORK_LINK" in signals or len(signals) >= 2:
            expected_recommendation = Recommendation.ESCALATE_TO_HUMAN
        elif len(signals) == 1:
            expected_recommendation = Recommendation.REQUEST_MORE_INFO
        else:
            expected_recommendation = Recommendation.CLEAR

        assert case.expected_recommendation is expected_recommendation
        assert set(case.expected_policy_ids) == _expected_policy_ids(
            case,
            signals,
        )

        transaction_group = transaction.transaction_id.split("-")[1]
        if transaction_group == "single":
            assert len(signals) == 1
        elif transaction_group == "multi":
            assert len(signals) >= 2
            if transaction.new_payee and "UNUSUAL_AMOUNT" in signals:
                app_patterns += 1
        elif transaction_group in {"clean", "boundary"}:
            assert signals == set()

    assert app_patterns >= 1

    boundaries = {
        case.transaction.transaction_id: case.transaction
        for case in holdout.cases
        if "-boundary-" in case.transaction.transaction_id
    }
    assert boundaries["holdout-boundary-001"].account_age_days == 30
    assert boundaries["holdout-boundary-002"].transactions_last_24h == 10
    amount_boundary = boundaries["holdout-boundary-003"]
    assert amount_boundary.amount == 5 * amount_boundary.average_transaction_amount
    cross_border = boundaries["holdout-boundary-004"]
    assert cross_border.origin_country != cross_border.destination_country
