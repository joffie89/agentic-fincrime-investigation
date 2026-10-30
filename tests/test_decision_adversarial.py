"""Integrity checks for the adversarial decision-evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.decision_eval import DecisionEvaluationDataset
from app.models.domain import Recommendation, RiskLevel
from app.services.risk_scorer import assess_transaction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADVERSARIAL_CASES_PATH = PROJECT_ROOT / "evaluation" / "decision_cases_adversarial.json"


def _load_dataset() -> DecisionEvaluationDataset:
    return DecisionEvaluationDataset.model_validate(
        json.loads(ADVERSARIAL_CASES_PATH.read_text(encoding="utf-8"))
    )


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
    if "HIGH_VELOCITY" in signals:
        policy_ids.add("DEMO-AML-002")
    if transaction.origin_country != transaction.destination_country:
        policy_ids.add("DEMO-AML-004")
    if risk.risk_level is RiskLevel.HIGH:
        policy_ids.add("DEMO-AML-005")
    if transaction.new_payee:
        policy_ids.add("DEMO-AML-006")
    return policy_ids


def test_adversarial_cases_follow_rubric_and_policy_labels() -> None:
    dataset = _load_dataset()

    assert len(dataset.cases) == 10
    for case in dataset.cases:
        signals = _risk_signals(case)
        if len(signals) >= 2:
            expected_recommendation = Recommendation.ESCALATE_TO_HUMAN
        elif len(signals) == 1:
            expected_recommendation = Recommendation.REQUEST_MORE_INFO
        else:
            expected_recommendation = Recommendation.CLEAR

        assert case.expected_recommendation is expected_recommendation
        assert set(case.expected_policy_ids) == _expected_policy_ids(case, signals)


def test_adversarial_cases_cover_injection_boundaries_and_contradictions() -> None:
    dataset = _load_dataset()
    transactions = [case.transaction for case in dataset.cases]
    string_values = [
        value
        for transaction in transactions
        for value in transaction.model_dump(mode="python").values()
        if isinstance(value, str)
    ]

    assert any("ignore previous instructions" in value.casefold() for value in string_values)
    assert {29, 30} <= {transaction.account_age_days for transaction in transactions}
    assert {10, 11} <= {transaction.transactions_last_24h for transaction in transactions}
    assert any(
        transaction.amount == 5 * transaction.average_transaction_amount
        for transaction in transactions
    )
    assert any(
        transaction.account_age_days >= 365 and transaction.transactions_last_24h > 10
        for transaction in transactions
    )
    assert any(
        assess_transaction(transaction).risk_level is RiskLevel.HIGH
        and transaction.origin_country != transaction.destination_country
        for transaction in transactions
    )
