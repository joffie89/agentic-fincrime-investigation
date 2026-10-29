"""FastAPI entry point for risk assessment and investigations."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict

from app.auth import require_api_key
from app.config import Settings, get_settings
from app.graph.analyzer import TransactionConflictError
from app.logging_config import configure_logging
from app.middleware import HttpRequestLoggingMiddleware
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

_UNAUTHORIZED_RESPONSE = {
    "description": "Invalid or missing API key",
    "content": {
        "application/json": {
            "example": {"detail": "Invalid or missing API key"},
        }
    },
}

_NOT_FOUND_RESPONSE = {
    "description": "Investigation not found",
    "content": {
        "application/json": {
            "example": {"detail": "Investigation not found"},
        }
    },
}

_RETRIEVAL_UNAVAILABLE_RESPONSE = {
    "description": "Investigation could not be retrieved",
    "content": {
        "application/json": {
            "example": {"detail": "Investigation could not be retrieved"},
        }
    },
}


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
    if not resolved_settings.llm_assessment_enabled:
        logger.warning(
            "llm_assessment_disabled",
            extra={
                "event": "llm_assessment_disabled",
                "service_mode": resolved_settings.service_mode.value,
            },
        )
    service = investigation_service or build_investigation_service(resolved_settings)

    application = FastAPI(
        title="Agentic Financial Crime Investigation Platform",
        version="1.0.0",
        description=(
            "Synthetic portfolio demo. Recommendations require appropriate human "
            "oversight and must not be treated as autonomous financial decisions."
        ),
    )
    application.add_middleware(HttpRequestLoggingMiddleware)
    application.state.settings = resolved_settings
    application.state.investigation_service = service

    @application.get("/health", tags=["operations"])
    def health_check() -> dict[str, str | bool]:
        return {
            "status": "healthy",
            "mode": resolved_settings.service_mode.value,
            "persistence_backend": resolved_settings.persistence_backend.value,
            "llm_assessment_enabled": resolved_settings.llm_assessment_enabled,
        }

    @application.post(
        "/assess-risk",
        response_model=LegacyRiskResponse,
        tags=["investigations"],
        dependencies=[Depends(require_api_key)],
        responses={status.HTTP_401_UNAUTHORIZED: _UNAUTHORIZED_RESPONSE},
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
        dependencies=[Depends(require_api_key)],
        responses={status.HTTP_401_UNAUTHORIZED: _UNAUTHORIZED_RESPONSE},
    )
    def investigate(
        transaction: Transaction,
        request: Request,
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
        request.state.llm_usage = (
            result.llm_usage.model_dump(mode="json")
            if result.llm_usage is not None
            else None
        )
        if not result.persistence.succeeded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    @application.get(
        "/investigations/{investigation_id}",
        response_model=dict[str, Any],
        tags=["investigations"],
        dependencies=[Depends(require_api_key)],
        responses={
            status.HTTP_401_UNAUTHORIZED: _UNAUTHORIZED_RESPONSE,
            status.HTTP_404_NOT_FOUND: _NOT_FOUND_RESPONSE,
            status.HTTP_503_SERVICE_UNAVAILABLE: _RETRIEVAL_UNAVAILABLE_RESPONSE,
        },
    )
    def get_investigation(investigation_id: str) -> dict[str, Any]:
        try:
            record = service.get_investigation(investigation_id)
        except Exception as exc:
            logger.exception(
                "investigation_retrieval_failed",
                extra={
                    "event": "investigation_retrieval_failed",
                    "investigation_id": investigation_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Investigation could not be retrieved",
            ) from exc

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Investigation not found",
            )
        return record

    return application


app = create_app()

