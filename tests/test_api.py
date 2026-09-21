"""FastAPI contract and error-path tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.domain import GraphAnalysisResult, PersistenceResult, RetrievedPolicy
from app.repositories.memory import InMemoryInvestigationRepository
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


class FailedRepository:
    def save(self, _record):
        return PersistenceResult(
            succeeded=False,
            backend="postgres",
            error="connection refused",
        )


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


def service(repository=None) -> InvestigationService:
    workflow = InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=CleanGraphAnalyzer(),
        policy_retriever=StaticRetriever(),
        policy_assessor=DemoPolicyAssessmentService(),
        repository=repository or InMemoryInvestigationRepository(),
        retrieval_top_k=3,
        human_review_confidence_threshold=0.6,
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


def test_unknown_transaction_field_returns_422():
    response = client_for(service()).post(
        "/investigate",
        json={**transaction_payload(), "secret_extra": "ignored-no-longer"},
    )

    assert response.status_code == 422


def test_persistence_failure_returns_503_with_truthful_body():
    response = client_for(service(FailedRepository())).post(
        "/investigate", json=transaction_payload()
    )

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["persistence"] == {
        "succeeded": False,
        "backend": "postgres",
        "error": "connection refused",
    }


def test_unexpected_workflow_error_returns_sanitized_500():
    response = client_for(RaisingService()).post(
        "/investigate", json=transaction_payload()
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Investigation could not be completed"}
    assert "internal details" not in response.text

