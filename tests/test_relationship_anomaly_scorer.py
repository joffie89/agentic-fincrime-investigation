import pytest

from app.models.domain import GraphAnalysisResult
from app.services.relationship_anomaly_scorer import RelationshipAnomalyScorer


@pytest.fixture
def typical_relationship() -> GraphAnalysisResult:
    return GraphAnalysisResult(
        origin_account_id="acct-typical-origin",
        destination_account_id="acct-typical-destination",
        counterparty_count=3,
        origin_degree=3,
        shared_counterparty_count=0,
        # The current transfer is included, so one means there are no repeats.
        repeated_transaction_count=1,
        concentration_ratio=0.3,
        signals=[],
        suspicious=False,
    )


def test_score_is_reproducible_across_independent_scorers(
    typical_relationship: GraphAnalysisResult,
) -> None:
    first = RelationshipAnomalyScorer().score(typical_relationship)
    second = RelationshipAnomalyScorer().score(typical_relationship)

    assert first == second


def test_typical_relationship_is_not_anomalous(
    typical_relationship: GraphAnalysisResult,
) -> None:
    anomaly_score, is_anomalous = RelationshipAnomalyScorer().score(typical_relationship)

    assert isinstance(anomaly_score, float)
    assert is_anomalous is False


def test_isolated_first_transfer_is_not_anomalous() -> None:
    graph_result = GraphAnalysisResult(
        origin_account_id="acct-new-origin",
        destination_account_id="acct-new-destination",
        counterparty_count=1,
        origin_degree=1,
        shared_counterparty_count=0,
        repeated_transaction_count=1,
        concentration_ratio=1.0,
        signals=[],
        suspicious=False,
    )

    _, is_anomalous = RelationshipAnomalyScorer().score(graph_result)

    assert is_anomalous is False


def test_concentrated_repeated_relationship_is_anomalous() -> None:
    graph_result = GraphAnalysisResult(
        origin_account_id="acct-a",
        destination_account_id="acct-b",
        counterparty_count=3,
        origin_degree=5,
        shared_counterparty_count=2,
        repeated_transaction_count=3,
        concentration_ratio=0.875,
        signals=[
            "SHARED_COUNTERPARTIES",
            "REPEATED_DIRECT_TRANSACTIONS",
            "COUNTERPARTY_CONCENTRATION",
        ],
        suspicious=True,
    )

    anomaly_score, is_anomalous = RelationshipAnomalyScorer().score(graph_result)

    assert anomaly_score > 0.0
    assert is_anomalous is True
