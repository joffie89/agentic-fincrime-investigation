from types import SimpleNamespace

import pytest

from app.graph import AccountTransactionEdge, GraphAnalyzer
from app.models.transaction import Transaction


def transaction(
    transaction_id: str = "current-001",
    origin: str = "acct-a",
    destination: str = "acct-b",
    amount: float = 200.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        transaction_id=transaction_id,
        origin_account_id=origin,
        destination_account_id=destination,
        amount=amount,
    )


def test_analyzer_returns_explainable_network_metrics() -> None:
    history = [
        AccountTransactionEdge("h-1", "acct-a", "acct-b", 100.0),
        AccountTransactionEdge("h-2", "acct-a", "acct-b", 50.0),
        AccountTransactionEdge("h-3", "acct-a", "acct-c", 25.0),
        AccountTransactionEdge("h-4", "acct-d", "acct-a", 25.0),
        AccountTransactionEdge("h-5", "acct-b", "acct-c", 10.0),
        AccountTransactionEdge("h-6", "acct-b", "acct-d", 10.0),
    ]

    result = GraphAnalyzer(history).analyze(transaction())

    assert result.origin_account_id == "acct-a"
    assert result.destination_account_id == "acct-b"
    assert result.counterparty_count == 3
    assert result.origin_degree == 5
    assert result.shared_counterparty_count == 2
    assert result.repeated_transaction_count == 3
    assert result.concentration_ratio == pytest.approx(0.875)
    assert result.signals == [
        "SHARED_COUNTERPARTIES",
        "REPEATED_DIRECT_TRANSACTIONS",
        "COUNTERPARTY_CONCENTRATION",
    ]
    assert result.suspicious is True


def test_isolated_first_transfer_is_not_flagged_for_concentration_alone() -> None:
    result = GraphAnalyzer([]).analyze(transaction())

    assert result.counterparty_count == 1
    assert result.origin_degree == 1
    assert result.repeated_transaction_count == 1
    assert result.concentration_ratio == 1.0
    assert result.signals == []
    assert result.suspicious is False


def test_analysis_does_not_mutate_history_or_inflate_repeated_calls() -> None:
    analyzer = GraphAnalyzer(
        [AccountTransactionEdge("h-1", "acct-a", "acct-b", 100.0)]
    )

    first = analyzer.analyze(transaction())
    second = analyzer.analyze(transaction())

    assert first == second
    assert analyzer.graph.number_of_edges() == 1


def test_analyzer_supports_legacy_transaction_without_account_ids() -> None:
    legacy_transaction = Transaction(
        transaction_id="legacy-001",
        amount=125.0,
        account_age_days=100,
        new_payee=False,
        average_transaction_amount=100.0,
        transactions_last_24h=1,
        origin_country="GB",
        destination_country="GB",
    )

    result = GraphAnalyzer([]).analyze(legacy_transaction)

    assert result.origin_account_id == "origin:GB"
    assert result.destination_account_id == "destination:GB"
    assert result.origin_account_id != result.destination_account_id
    assert result.suspicious is False


@pytest.mark.parametrize(
    ("transaction_id", "origin", "destination", "amount", "message"),
    [
        ("current-001", "acct-a", "acct-a", 10.0, "must differ"),
        ("", "acct-a", "acct-b", 10.0, "must not be blank"),
        ("current-001", "acct-a", "acct-b", 0.0, "positive finite"),
        ("current-001", "acct-a", "acct-b", float("inf"), "positive finite"),
    ],
)
def test_invalid_transaction_edges_are_rejected(
    transaction_id: str,
    origin: str,
    destination: str,
    amount: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GraphAnalyzer([]).analyze(
            transaction(
                transaction_id=transaction_id,
                origin=origin,
                destination=destination,
                amount=amount,
            )
        )
