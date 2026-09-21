"""Shared deterministic fixtures for the test suite."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from app.models.domain import GraphAnalysisResult, RetrievedPolicy
from app.models.transaction import Transaction


@pytest.fixture
def transaction_factory():
    def factory(**updates: object) -> Transaction:
        values: dict[str, object] = {
            "transaction_id": "txn-test-low",
            "amount": 100.0,
            "account_age_days": 365,
            "new_payee": False,
            "average_transaction_amount": 100.0,
            "transactions_last_24h": 2,
            "origin_country": "GB",
            "destination_country": "GB",
            "origin_account_id": "acct-test-origin",
            "destination_account_id": "acct-test-destination",
        }
        values.update(updates)
        return Transaction.model_validate(values)

    return factory


@pytest.fixture
def clean_graph_result() -> GraphAnalysisResult:
    return GraphAnalysisResult(
        origin_account_id="acct-test-origin",
        destination_account_id="acct-test-destination",
        counterparty_count=1,
        origin_degree=1,
        shared_counterparty_count=0,
        repeated_transaction_count=1,
        concentration_ratio=1.0,
        signals=[],
        suspicious=False,
    )


@pytest.fixture
def retrieved_policy() -> RetrievedPolicy:
    return RetrievedPolicy(
        policy_id="DEMO-AML-001",
        chunk_id="DEMO-AML-001#001",
        text="Synthetic policy evidence for unusual payments and new payees.",
        relevance_score=0.75,
    )

