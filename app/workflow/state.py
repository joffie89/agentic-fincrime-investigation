"""Typed state passed between LangGraph nodes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from app.models.domain import (
    GraphAnalysisResult,
    InvestigationRecommendation,
    InvestigationStatus,
    LLMUsage,
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
    policy_retrieval_sources: dict[str, list[str]]
    policy_assessment: PolicyAssessment
    llm_usage: LLMUsage | None
    llm_disabled: bool
    qa_sampled: bool
    recommendation: InvestigationRecommendation
    requires_human_review: bool
    status: InvestigationStatus
    persistence: PersistenceResult
    routing: dict[str, Any]
    workflow_steps: dict[str, dict[str, Any]]
    audit_events: list[str]
    errors: list[str]
    fallbacks: list[dict[str, Any]]
