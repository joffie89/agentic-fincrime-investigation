"""Deterministic and explainable transaction risk rules."""

from typing import TypedDict

from app.models.domain import RiskAssessment, RiskLevel
from app.models.transaction import Transaction


class RiskScoreResult(TypedDict):
    """Legacy scorer payload retained for API backwards compatibility."""

    score: int
    reasons: list[str]


def score_transaction(transaction: Transaction) -> RiskScoreResult:
    """Return the legacy risk score and ordered explanations.

    Thresholds and response keys intentionally remain unchanged because the
    existing ``/assess-risk`` endpoint exposes this function's output.
    """

    score = 0
    reasons: list[str] = []
    if transaction.account_age_days < 30:
        score += 20
        reasons.append("Account is less than 30 days old")
    if transaction.new_payee:
        score += 15
        reasons.append("Transaction is to a new payee")
    if transaction.amount > 5 * transaction.average_transaction_amount:
        score += 30
        reasons.append("Transaction amount is more than 5 times the average")
    if transaction.transactions_last_24h > 10:
        score += 20
        reasons.append("More than 10 transactions in the last 24 hours")

    return {
        "score": score,
        "reasons": reasons,
    }


def assess_transaction(transaction: Transaction) -> RiskAssessment:
    """Adapt the legacy score payload to the typed workflow contract."""

    result = score_transaction(transaction)
    score = result["score"]
    if score >= 60:
        risk_level = RiskLevel.HIGH
    elif score >= 30:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW

    return RiskAssessment(
        risk_score=score,
        risk_level=risk_level,
        risk_reasons=result["reasons"],
        investigate=risk_level is RiskLevel.HIGH,
    )
