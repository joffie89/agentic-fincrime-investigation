"""FastAPI contract and error-path tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import app.repositories.postgres as postgres_module
from app.config import Settings
from app.graph.analyzer import AccountTransactionEdge, GraphAnalyzer
from app.main import create_app
from app.models.domain import GraphAnalysisResult, RetrievedPolicy
from app.repositories.memory import InMemoryInvestigationRepository
from app.repositories.postgres import PostgresInvestigationRepository
from app.services.investigation_service import InvestigationService
from app.services.policy_assessment import DemoPolicyAssessmentService
from app.services.risk_scorer import assess_transaction
from app.workflow.investigation_workflow import InvestigationWorkflow


class CleanGraphAnalyzer:
    def analyze(self, transaction):
        return GraphAnalysisResult(
            origin_account_id=transaction.origin_account_id,
            destination_account_id=transaction.destination_account_id,
            counterparty_count=1,
            origin_degree=1,
            shared_counterparty_count=0,
            repeated_transaction_count=1,
            concentration_ratio=1.0,
            signals=[],
            suspicious=False,
        )


class StaticRelationshipAnomalyScorer:
    def score(self, _graph_analysis_result):
        return -0.1, False


class StaticRetriever:
    def retrieve(self, _query: str, top_k: int = 3):
        policies = [
            RetrievedPolicy(
                policy_id="DEMO-AML-001",
                chunk_id="DEMO-AML-001#001",
                text="Synthetic policy evidence for unusual payments.",
                relevance_score=0.8,
            )
        ]
        return policies[:top_k]


class RaisingService:
    def investigate(self, _transaction):
        raise RuntimeError("internal details must not leak")


def transaction_payload(**updates):
    payload = {
        "transaction_id": "txn-api-low",
        "amount": 100.0,
        "account_age_days": 365,
        "new_payee": False,
        "average_transaction_amount": 100.0,
        "transactions_last_24h": 2,
        "origin_country": "GB",
        "destination_country": "GB",
        "origin_account_id": "acct-api-origin",
        "destination_account_id": "acct-api-destination",
    }
    payload.update(updates)
    return payload


def service(repository=None, graph_analyzer=None) -> InvestigationService:
    workflow = InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=graph_analyzer or CleanGraphAnalyzer(),
        policy_retriever=StaticRetriever(),
        policy_assessor=DemoPolicyAssessmentService(),
        repository=repository or InMemoryInvestigationRepository(),
        retrieval_top_k=3,
        human_review_confidence_threshold=0.6,
        relationship_anomaly_scorer=StaticRelationshipAnomalyScorer(),
    )
    return InvestigationService(workflow)


def client_for(investigation_service) -> TestClient:
    settings = Settings(_env_file=None)
    return TestClient(
        create_app(settings=settings, investigation_service=investigation_service)
    )


def test_health_endpoint():
    response = client_for(service()).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "mode": "local",
        "persistence_backend": "memory",
    }


def test_assess_risk_preserves_legacy_contract():
    response = client_for(service()).post(
        "/assess-risk",
        json=transaction_payload(
            amount=6000.0,
            average_transaction_amount=1000.0,
            account_age_days=10,
            new_payee=True,
            transactions_last_24h=11,
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 85
    assert body["risk_level"] == "HIGH"
    assert body["investigate"] is True
    assert len(body["reasons"]) == 4


def test_investigate_low_risk_executes_workflow():
    response = client_for(service()).post("/investigate", json=transaction_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["risk_level"] == "LOW"
    assert body["recommendation"] == "CLEAR"
    assert body["requires_human_review"] is False
    assert body["persistence"]["succeeded"] is True
    assert body["retrieved_policies"][0]["policy_id"] == "DEMO-AML-001"
    assert body["relationship_anomaly"] == {
        "anomaly_score": -0.1,
        "is_anomalous": False,
    }


def test_investigate_high_risk_routes_to_human_review():
    response = client_for(service()).post(
        "/investigate",
        json=transaction_payload(
            transaction_id="txn-api-high",
            amount=6000.0,
            average_transaction_amount=1000.0,
            account_age_days=10,
            new_payee=True,
            transactions_last_24h=11,
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["risk_score"] == 85
    assert body["status"] == "PENDING_HUMAN_REVIEW"
    assert body["recommendation"] == "ESCALATE_TO_HUMAN"
    assert body["requires_human_review"] is True


def test_invalid_transaction_returns_422():
    response = client_for(service()).post(
        "/investigate",
        json=transaction_payload(amount=0.0),
    )

    assert response.status_code == 422


def test_same_origin_and_destination_account_returns_422():
    response = client_for(service()).post(
        "/investigate",
        json=transaction_payload(
            destination_account_id="acct-api-origin",
        ),
    )

    assert response.status_code == 422
    assert (
        "origin_account_id and destination_account_id must differ"
        in response.json()["detail"][0]["msg"]
    )


def test_reused_transaction_id_with_different_details_returns_409():
    analyzer = GraphAnalyzer(
        [
            AccountTransactionEdge(
                "txn-api-low",
                "acct-historical-origin",
                "acct-historical-destination",
                50.0,
            )
        ]
    )

    response = client_for(service(graph_analyzer=analyzer)).post(
        "/investigate",
        json=transaction_payload(),
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "transaction_id already exists with different details"
    }


def test_unknown_transaction_field_returns_422():
    response = client_for(service()).post(
        "/investigate",
        json={**transaction_payload(), "secret_extra": "ignored-no-longer"},
    )

    assert response.status_code == 422


def test_postgres_failure_returns_sanitized_503(monkeypatch):
    database_error = "connection to db.internal.example as admin_user refused"
    fake_psycopg = SimpleNamespace(
        connect=MagicMock(side_effect=RuntimeError(database_error))
    )
    monkeypatch.setattr(postgres_module, "psycopg", fake_psycopg)
    repository = PostgresInvestigationRepository(
        "postgresql://admin_user:secret@db.internal.example/investigations"
    )

    response = client_for(service(repository)).post(
        "/investigate", json=transaction_payload()
    )

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["persistence"] == {
        "succeeded": False,
        "backend": "postgres",
        "error": "Investigation could not be saved",
    }
    assert database_error not in response.text


def test_unexpected_workflow_error_returns_sanitized_500():
    response = client_for(RaisingService()).post(
        "/investigate", json=transaction_payload()
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Investigation could not be completed"}
    assert "internal details" not in response.text
