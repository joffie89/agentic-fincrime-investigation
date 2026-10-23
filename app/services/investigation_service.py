"""Investigation service construction and legacy risk helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import ServiceMode, Settings
from app.graph.analyzer import GraphAnalyzer
from app.models.domain import InvestigationResponse, RiskLevel
from app.models.transaction import Transaction
from app.rag.embedding_provider import (
    build_embedding_service,
    resolve_policy_index_path,
)
from app.rag.ingestion import PolicyIngestor
from app.rag.retriever import PolicyRetriever
from app.repositories import build_repository
from app.services.policy_assessment import (
    AzureOpenAIPolicyAssessmentService,
    DemoPolicyAssessmentService,
    OpenAIPolicyAssessmentService,
    PolicyAssessmentService,
)
from app.services.risk_scorer import assess_transaction
from app.workflow.investigation_workflow import InvestigationWorkflow

logger = logging.getLogger(__name__)

_POLICY_SOURCE_PATH = Path(__file__).resolve().parents[2] / "data" / "policies.json"


def determine_risk_level(score: int) -> str:
    """Return the legacy risk band used by ``/assess-risk``."""

    if score >= 60:
        return RiskLevel.HIGH.value
    if score >= 30:
        return RiskLevel.MEDIUM.value
    return RiskLevel.LOW.value


def should_investigate(risk_level: str) -> bool:
    """Retain the original rule: only HIGH risk automatically investigates."""

    return risk_level == RiskLevel.HIGH.value


class InvestigationService:
    """Application-facing wrapper around the compiled LangGraph."""

    def __init__(self, workflow: InvestigationWorkflow) -> None:
        self.workflow = workflow

    def investigate(self, transaction: Transaction) -> InvestigationResponse:
        logger.info(
            "investigation_started",
            extra={
                "event": "investigation_started",
                "transaction_id": transaction.transaction_id,
            },
        )
        result = self.workflow.invoke(transaction)
        logger.info(
            "investigation_finished",
            extra={
                "event": "investigation_finished",
                "transaction_id": transaction.transaction_id,
                "investigation_id": result.investigation_id,
                "status": result.status.value,
                "requires_human_review": result.requires_human_review,
                "persistence_succeeded": result.persistence.succeeded,
                "llm_usage": (
                    result.llm_usage.model_dump(mode="json")
                    if result.llm_usage is not None
                    else None
                ),
            },
        )
        return result

    def get_investigation(self, investigation_id: str) -> dict[str, Any] | None:
        """Return a persisted investigation audit trail by its identifier."""

        return self.workflow.get_investigation(investigation_id)


def _build_policy_assessor(settings: Settings) -> PolicyAssessmentService:
    if settings.service_mode is ServiceMode.LOCAL:
        return DemoPolicyAssessmentService()
    if settings.service_mode is ServiceMode.OPENAI:
        if settings.openai_api_key is None:  # Defensive; config validation covers this.
            raise ValueError("OPENAI_API_KEY is required when SERVICE_MODE=openai")
        return OpenAIPolicyAssessmentService(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_chat_model,
            input_price_per_1m=settings.openai_input_price_per_1m,
            output_price_per_1m=settings.openai_output_price_per_1m,
        )

    # Import only for the configured hosted path so local/demo mode remains
    # independent of credentials and provider availability.
    client = _build_azure_client(settings)
    return AzureOpenAIPolicyAssessmentService(
        client=client,
        deployment=settings.azure_openai_chat_deployment or "",
        input_price_per_1m=settings.openai_input_price_per_1m,
        output_price_per_1m=settings.openai_output_price_per_1m,
    )


def _build_policy_retriever(settings: Settings) -> PolicyRetriever:
    embedding_service = build_embedding_service(settings)
    index_path = resolve_policy_index_path(settings)
    if not index_path.is_absolute():
        index_path = Path.cwd() / index_path
    if not index_path.exists():
        logger.info(
            "policy_index_missing_ingesting_demo_policies",
            extra={"event": "policy_ingestion", "index_path": str(index_path)},
        )
        PolicyIngestor(embedding_service).ingest_file(_POLICY_SOURCE_PATH, index_path)
    return PolicyRetriever.from_index(
        index_path,
        embedding_service,
        minimum_relevance_score=settings.retrieval_min_relevance_score,
        policy_source_path=_POLICY_SOURCE_PATH,
        retrieval_mode=settings.retrieval_mode,
    )


def _build_azure_client(settings: Settings):
    """Create an SDK client only when an Azure-backed component is selected."""

    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def build_investigation_service(settings: Settings) -> InvestigationService:
    """Construct all workflow dependencies from validated settings."""

    workflow = InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=GraphAnalyzer(),
        policy_retriever=_build_policy_retriever(settings),
        policy_assessor=_build_policy_assessor(settings),
        repository=build_repository(settings),
        retrieval_top_k=settings.retrieval_top_k,
        human_review_confidence_threshold=(
            settings.human_review_confidence_threshold
        ),
    )
    return InvestigationService(workflow)
