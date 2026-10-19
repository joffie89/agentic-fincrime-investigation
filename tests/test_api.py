"""FastAPI contract and error-path tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.repositories.postgres as postgres_module
from app.config import Settings
from app.graph.analyzer import AccountTransactionEdge, GraphAnalyzer
from app.main import create_app
from app.models.domain import GraphAnalysisResult, Recommendation, RetrievedPolicy
from app.repositories.memory import InMemoryInvestigationRepository
from app.repositories.postgres import PostgresInvestigationRepository
from app.services.investigation_service import InvestigationService
from app.services.policy_assessment import (
    DemoPolicyAssessmentService,
    OpenAIPolicyAssessmentService,
)
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


def service(
    repository=None,
    graph_analyzer=None,
    policy_assessor=None,
) -> InvestigationService:
    workflow = InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=graph_analyzer or CleanGraphAnalyzer(),
        policy_retriever=StaticRetriever(),
        policy_assessor=policy_assessor or DemoPolicyAssessmentService(),
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


def test_api_key_setting_is_required(monkeypatch):
    monkeypatch.delenv("API_KEY")

    with pytest.raises(ValidationError, match="api_key"):
        Settings(
            _env_file=None,
            service_mode="local",
            embedding_mode="local",
            persistence_backend="memory",
        )


def test_health_endpoint_works_without_api_key():
    response = client_for(service()).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "mode": "local",
        "persistence_backend": "memory",
    }


def test_response_has_generated_request_id():
    client = client_for(service())

    with patch("app.middleware.logger.info") as log_info:
        response = client.get("/health")

    request_id = response.headers["X-Request-ID"]
    parsed_request_id = UUID(request_id)
    assert parsed_request_id.version == 4
    assert str(parsed_request_id) == request_id

    log_info.assert_called_once()
    assert log_info.call_args.args == ("http_request",)
    fields = log_info.call_args.kwargs["extra"]
    assert fields == {
        "event": "http_request",
        "request_id": request_id,
        "method": "GET",
        "path": "/health",
        "status_code": 200,
        "duration_ms": fields["duration_ms"],
        "llm_usage": None,
    }
    assert isinstance(fields["duration_ms"], float)
    assert fields["duration_ms"] >= 0
    assert fields["duration_ms"] == round(fields["duration_ms"], 1)


def test_incoming_request_id_is_echoed_unchanged():
    request_id = "gateway-Req-ABC_123"

    response = client_for(service()).get(
        "/health",
        headers={"X-Request-ID": request_id},
    )

    assert response.headers["X-Request-ID"] == request_id


def test_openapi_documents_api_key_authentication():
    schema = client_for(service()).get("/openapi.json").json()
    security_schemes = schema["components"]["securitySchemes"]
    api_key_schemes = {
        name: definition
        for name, definition in security_schemes.items()
        if definition == {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }

    assert len(api_key_schemes) == 1
    scheme_name = next(iter(api_key_schemes))
    for path in ("/assess-risk", "/investigate"):
        operation = schema["paths"][path]["post"]
        assert {scheme_name: []} in operation["security"]
        assert operation["responses"]["401"] == {
            "description": "Invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or missing API key"}
                }
            },
        }
    assert "security" not in schema["paths"]["/health"]["get"]


@pytest.mark.parametrize("path", ["/assess-risk", "/investigate"])
def test_missing_api_key_returns_401(path):
    response = client_for(service()).post(path, json=transaction_payload())

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


@pytest.mark.parametrize("path", ["/assess-risk", "/investigate"])
def test_wrong_api_key_returns_401(path):
    response = client_for(service()).post(
        path,
        json=transaction_payload(),
        headers={"X-API-Key": "wrong-api-key"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


@pytest.mark.parametrize("path", ["/assess-risk", "/investigate"])
def test_correct_api_key_works(path, api_key_headers):
    response = client_for(service()).post(
        path,
        json=transaction_payload(),
        headers=api_key_headers,
    )

    assert response.status_code == 200


def test_assess_risk_preserves_legacy_contract(api_key_headers):
    response = client_for(service()).post(
        "/assess-risk",
        json=transaction_payload(
            amount=6000.0,
            average_transaction_amount=1000.0,
            account_age_days=10,
            new_payee=True,
            transactions_last_24h=11,
        ),
        headers=api_key_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 85
    assert body["risk_level"] == "HIGH"
    assert body["investigate"] is True
    assert len(body["reasons"]) == 4


def test_investigate_low_risk_executes_workflow(api_key_headers):
    with patch("app.middleware.logger.info") as log_info:
        response = client_for(service()).post(
            "/investigate",
            json=transaction_payload(),
            headers=api_key_headers,
        )

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
    assert body["llm_usage"] is None
    assert log_info.call_args.kwargs["extra"]["llm_usage"] is None


def test_investigate_returns_and_logs_mocked_llm_usage(api_key_headers):
    mocked_usage = SimpleNamespace(input_tokens=1_200, output_tokens=300)
    mocked_response = SimpleNamespace(
        output_parsed=SimpleNamespace(
            recommendation=Recommendation.CLEAR,
            reasons=["The supplied evidence supports clearing the payment."],
            confidence=0.95,
            cited_policy_ids=[],
            missing_information=[],
            evidence_sufficient=True,
        ),
        usage=mocked_usage,
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=MagicMock(return_value=mocked_response),
        )
    )
    assessor = OpenAIPolicyAssessmentService(
        client,
        "gpt-4.1-mini",
        input_price_per_1m=0.4,
        output_price_per_1m=1.6,
    )
    repository = InMemoryInvestigationRepository()

    with (
        patch("app.middleware.logger.info") as log_info,
        patch("app.services.investigation_service.logger.info") as investigation_log,
    ):
        response = client_for(
            service(repository=repository, policy_assessor=assessor)
        ).post(
            "/investigate",
            json=transaction_payload(transaction_id="txn-api-llm-usage"),
            headers=api_key_headers,
        )

    assert response.status_code == 200
    usage = response.json()["llm_usage"]
    assert usage["input_tokens"] == 1_200
    assert usage["output_tokens"] == 300
    assert usage["latency_ms"] >= 0
    assert usage["estimated_cost_usd"] == pytest.approx(0.00096)
    log_info.assert_called_once()
    assert log_info.call_args.kwargs["extra"]["llm_usage"] == usage
    finished_log = next(
        call
        for call in investigation_log.call_args_list
        if call.args == ("investigation_finished",)
    )
    assert finished_log.kwargs["extra"]["llm_usage"] == usage
    assert repository.records[0]["llm_usage"] == usage


def test_investigate_high_risk_routes_to_human_review(api_key_headers):
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
        headers=api_key_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["risk_score"] == 85
    assert body["status"] == "PENDING_HUMAN_REVIEW"
    assert body["recommendation"] == "ESCALATE_TO_HUMAN"
    assert body["requires_human_review"] is True


def test_invalid_transaction_returns_422(api_key_headers):
    response = client_for(service()).post(
        "/investigate",
        json=transaction_payload(amount=0.0),
        headers=api_key_headers,
    )

    assert response.status_code == 422


def test_same_origin_and_destination_account_returns_422(api_key_headers):
    response = client_for(service()).post(
        "/investigate",
        json=transaction_payload(
            destination_account_id="acct-api-origin",
        ),
        headers=api_key_headers,
    )

    assert response.status_code == 422
    assert (
        "origin_account_id and destination_account_id must differ"
        in response.json()["detail"][0]["msg"]
    )


def test_reused_transaction_id_with_different_details_returns_409(api_key_headers):
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
        headers=api_key_headers,
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "transaction_id already exists with different details"
    }


def test_unknown_transaction_field_returns_422(api_key_headers):
    response = client_for(service()).post(
        "/investigate",
        json={**transaction_payload(), "secret_extra": "ignored-no-longer"},
        headers=api_key_headers,
    )

    assert response.status_code == 422


def test_postgres_failure_returns_sanitized_503(monkeypatch, api_key_headers):
    database_error = "connection to db.internal.example as admin_user refused"
    fake_psycopg = SimpleNamespace(
        connect=MagicMock(side_effect=RuntimeError(database_error))
    )
    monkeypatch.setattr(postgres_module, "psycopg", fake_psycopg)
    repository = PostgresInvestigationRepository(
        "postgresql://admin_user:secret@db.internal.example/investigations"
    )

    response = client_for(service(repository)).post(
        "/investigate",
        json=transaction_payload(),
        headers=api_key_headers,
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


def test_unexpected_workflow_error_returns_sanitized_500(api_key_headers):
    response = client_for(RaisingService()).post(
        "/investigate",
        json=transaction_payload(),
        headers=api_key_headers,
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Investigation could not be completed"}
    assert "internal details" not in response.text
