"""FastAPI entry point for risk assessment and investigations."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict

from app.config import Settings, get_settings
from app.graph.analyzer import TransactionConflictError
from app.logging_config import configure_logging
from app.models.domain import InvestigationResponse, RiskLevel
from app.models.transaction import Transaction
from app.services.investigation_service import (
    InvestigationService,
    build_investigation_service,
    determine_risk_level,
    should_investigate,
)
from app.services.risk_scorer import score_transaction

logger = logging.getLogger(__name__)


class LegacyRiskResponse(BaseModel):
    """Validated response while retaining the original public field names."""

    model_config = ConfigDict(extra="forbid")

    score: int
    reasons: list[str]
    risk_level: RiskLevel
    investigate: bool


def create_app(
    *,
    settings: Settings | None = None,
    investigation_service: InvestigationService | None = None,
) -> FastAPI:
    """Application factory supporting explicit test dependencies."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    service = investigation_service or build_investigation_service(resolved_settings)

    application = FastAPI(
        title="Agentic Financial Crime Investigation Platform",
        version="1.0.0",
        description=(
            "Synthetic portfolio demo. Recommendations require appropriate human "
            "oversight and must not be treated as autonomous financial decisions."
        ),
    )
    application.state.settings = resolved_settings
    application.state.investigation_service = service

    @application.get("/health", tags=["operations"])
    def health_check() -> dict[str, str]:
        return {
            "status": "healthy",
            "mode": resolved_settings.service_mode.value,
            "persistence_backend": resolved_settings.persistence_backend.value,
        }

    @application.post(
        "/assess-risk",
        response_model=LegacyRiskResponse,
        tags=["investigations"],
    )
    def assess_risk(transaction: Transaction) -> LegacyRiskResponse:
        result = score_transaction(transaction)
        risk_level = determine_risk_level(result["score"])
        return LegacyRiskResponse(
            **result,
            risk_level=risk_level,
            investigate=should_investigate(risk_level),
        )

    @application.post(
        "/investigate",
        response_model=InvestigationResponse,
        tags=["investigations"],
    )
    def investigate(
        transaction: Transaction,
        response: Response,
    ) -> InvestigationResponse:
        try:
            result = service.investigate(transaction)
        except TransactionConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="transaction_id already exists with different details",
            ) from exc
        except Exception as exc:
            logger.exception(
                "investigation_request_failed",
                extra={
                    "event": "investigation_request_failed",
                    "transaction_id": transaction.transaction_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Investigation could not be completed",
            ) from exc
        if not result.persistence.succeeded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    return application


app = create_app()

