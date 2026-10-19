"""Validated models exchanged throughout an investigation workflow."""

from enum import Enum
from typing import Annotated

from pydantic import AwareDatetime, Field, StrictBool, StrictFloat, model_validator

from app.models.base import (
    DomainModel,
    Identifier,
    LongText,
    NonNegativeInt,
    RiskScore,
    ShortText,
    UnitInterval,
)


class RiskLevel(str, Enum):
    """Deterministic severity bands for a transaction risk score."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Recommendation(str, Enum):
    """Non-consequential recommendations produced by the demo workflow."""

    CLEAR = "CLEAR"
    REQUEST_MORE_INFO = "REQUEST_MORE_INFO"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class InvestigationStatus(str, Enum):
    """Lifecycle state returned by an investigation request."""

    COMPLETED = "COMPLETED"
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    FAILED = "FAILED"


ReasonList = Annotated[list[ShortText], Field(min_length=1)]
NonNegativeFiniteFloat = Annotated[
    StrictFloat,
    Field(ge=0, allow_inf_nan=False),
]


class RiskAssessment(DomainModel):
    """Explainable output of the deterministic rule-based scorer."""

    risk_score: RiskScore
    risk_level: RiskLevel
    risk_reasons: list[ShortText]
    investigate: StrictBool

    @model_validator(mode="after")
    def validate_legacy_thresholds(self) -> "RiskAssessment":
        if self.risk_score >= 60:
            expected_level = RiskLevel.HIGH
        elif self.risk_score >= 30:
            expected_level = RiskLevel.MEDIUM
        else:
            expected_level = RiskLevel.LOW

        if self.risk_level is not expected_level:
            raise ValueError(
                f"risk_level must be {expected_level.value} for score "
                f"{self.risk_score}"
            )

        expected_investigate = expected_level is RiskLevel.HIGH
        if self.investigate is not expected_investigate:
            raise ValueError(
                "investigate must be true exactly when risk_level is HIGH"
            )
        return self


class GraphAnalysisResult(DomainModel):
    """Small, explainable set of graph-derived transaction signals."""

    origin_account_id: Identifier | None = None
    destination_account_id: Identifier | None = None
    counterparty_count: NonNegativeInt
    origin_degree: NonNegativeInt
    shared_counterparty_count: NonNegativeInt
    repeated_transaction_count: NonNegativeInt
    concentration_ratio: UnitInterval
    signals: list[ShortText]
    suspicious: StrictBool


class RelationshipAnomalyResult(DomainModel):
    """Unsupervised second opinion over graph-derived relationship features."""

    anomaly_score: Annotated[StrictFloat, Field(allow_inf_nan=False)]
    is_anomalous: StrictBool


class RetrievedPolicy(DomainModel):
    """A policy chunk returned by retrieval with its relevance score."""

    policy_id: Identifier
    chunk_id: Identifier
    text: LongText
    relevance_score: UnitInterval


class PolicyAssessment(DomainModel):
    """Structured, evidence-grounded output from a policy assessor."""

    recommendation: Recommendation
    reasons: ReasonList
    confidence: UnitInterval
    policy_references: list[Identifier]
    policies_considered: list[Identifier]
    missing_information: list[ShortText]
    requires_human_review: StrictBool
    evidence_sufficient: StrictBool


class LLMUsage(DomainModel):
    """Token, latency, and estimated-cost metadata for one LLM call."""

    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    latency_ms: NonNegativeFiniteFloat
    estimated_cost_usd: NonNegativeFiniteFloat


class InvestigationRecommendation(DomainModel):
    """Final recommendation after deterministic escalation routing."""

    recommendation: Recommendation
    reasons: ReasonList
    confidence: UnitInterval
    policy_references: list[Identifier]
    missing_information: list[ShortText]
    requires_human_review: StrictBool


class PersistenceResult(DomainModel):
    """Truthful result of the attempt to persist an investigation record."""

    succeeded: StrictBool
    backend: ShortText
    error: ShortText | None = None


class InvestigationResponse(DomainModel):
    """Complete API response for a finished or escalated investigation."""

    investigation_id: Identifier
    transaction_id: Identifier
    risk_assessment: RiskAssessment
    graph_analysis: GraphAnalysisResult
    relationship_anomaly: RelationshipAnomalyResult | None = None
    retrieved_policies: list[RetrievedPolicy]
    policy_assessment: PolicyAssessment
    llm_usage: LLMUsage | None = None
    recommendation: Recommendation
    reasons: ReasonList
    requires_human_review: StrictBool
    status: InvestigationStatus
    persistence: PersistenceResult
    created_at: AwareDatetime
