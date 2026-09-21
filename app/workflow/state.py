"""Typed state passed between LangGraph nodes."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from app.models.domain import (
    GraphAnalysisResult,
    InvestigationRecommendation,
    InvestigationStatus,
    PersistenceResult,
    PolicyAssessment,
    RelationshipAnomalyResult,
    RetrievedPolicy,
    RiskAssessment,
)
from app.models.transaction import Transaction


class InvestigationState(TypedDict, total=False):
    """State is intentionally explicit so every node's contract is visible."""

    investigation_id: str
    created_at: datetime
    transaction: Transaction
    risk_assessment: RiskAssessment
    graph_analysis: GraphAnalysisResult
    relationship_anomaly: RelationshipAnomalyResult
    retrieved_policies: list[RetrievedPolicy]
    policy_assessment: PolicyAssessment
    recommendation: InvestigationRecommendation
    requires_human_review: bool
    status: InvestigationStatus
    persistence: PersistenceResult
    audit_events: list[str]
    errors: list[str]
