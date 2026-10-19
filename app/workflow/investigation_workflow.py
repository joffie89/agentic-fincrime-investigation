"""Actual LangGraph orchestration for an investigation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Callable, Protocol
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.models.domain import (
    InvestigationRecommendation,
    InvestigationResponse,
    InvestigationStatus,
    PersistenceResult,
    PolicyAssessment,
    Recommendation,
    RelationshipAnomalyResult,
    RiskAssessment,
    RiskLevel,
)
from app.models.transaction import Transaction
from app.services.policy_assessment import PolicyAssessmentError, PolicyAssessmentService
from app.services.relationship_anomaly_scorer import RelationshipAnomalyScorer
from app.workflow.state import InvestigationState

logger = logging.getLogger(__name__)


class GraphAnalyzerProtocol(Protocol):
    def analyze(self, transaction: Transaction) -> Any: ...


class PolicyRetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> list[Any]: ...


class RelationshipAnomalyScorerProtocol(Protocol):
    def score(self, graph_analysis_result: Any) -> tuple[float, bool]: ...


class InvestigationRepositoryProtocol(Protocol):
    def save(self, record: dict[str, Any]) -> PersistenceResult: ...


class InvestigationWorkflow:
    """Small, explainable graph whose compiled graph is invoked per request."""

    def __init__(
        self,
        *,
        risk_assessor: Callable[[Transaction], RiskAssessment],
        graph_analyzer: GraphAnalyzerProtocol,
        policy_retriever: PolicyRetrieverProtocol,
        policy_assessor: PolicyAssessmentService,
        repository: InvestigationRepositoryProtocol,
        retrieval_top_k: int = 3,
        human_review_confidence_threshold: float = 0.6,
        relationship_anomaly_scorer: RelationshipAnomalyScorerProtocol | None = None,
    ) -> None:
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1")
        if not 0 <= human_review_confidence_threshold <= 1:
            raise ValueError("human_review_confidence_threshold must be between 0 and 1")
        self._risk_assessor = risk_assessor
        self._graph_analyzer = graph_analyzer
        self._policy_retriever = policy_retriever
        self._policy_assessor = policy_assessor
        self._repository = repository
        self._retrieval_top_k = retrieval_top_k
        self._confidence_threshold = human_review_confidence_threshold
        self._relationship_anomaly_scorer = (
            relationship_anomaly_scorer
            if relationship_anomaly_scorer is not None
            else RelationshipAnomalyScorer()
        )
        self.graph = self._compile_graph()

    def _compile_graph(self) -> Any:
        builder = StateGraph(InvestigationState)
        builder.add_node("risk_assessment", self._risk_assessment_node)
        builder.add_node("graph_analysis", self._graph_analysis_node)
        builder.add_node(
            "relationship_anomaly_scoring",
            self._relationship_anomaly_scoring_node,
        )
        builder.add_node("policy_retrieval", self._policy_retrieval_node)
        builder.add_node("policy_validation", self._policy_validation_node)
        builder.add_node("human_review_required", self._human_review_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_node("persist", self._persistence_node)

        builder.add_edge(START, "risk_assessment")
        builder.add_edge("risk_assessment", "graph_analysis")
        builder.add_edge("graph_analysis", "relationship_anomaly_scoring")
        builder.add_edge("relationship_anomaly_scoring", "policy_retrieval")
        builder.add_edge("policy_retrieval", "policy_validation")
        builder.add_conditional_edges(
            "policy_validation",
            self._route_after_policy_validation,
            {
                "human_review": "human_review_required",
                "complete": "finalize",
            },
        )
        builder.add_edge("human_review_required", "persist")
        builder.add_edge("finalize", "persist")
        builder.add_edge("persist", END)
        return builder.compile()

    def _risk_assessment_node(self, state: InvestigationState) -> dict[str, Any]:
        assessment = self._risk_assessor(state["transaction"])
        return {
            "risk_assessment": assessment,
            "audit_events": [*state.get("audit_events", []), "risk_assessment_completed"],
        }

    def _graph_analysis_node(self, state: InvestigationState) -> dict[str, Any]:
        result = self._graph_analyzer.analyze(state["transaction"])
        return {
            "graph_analysis": result,
            "audit_events": [*state.get("audit_events", []), "graph_analysis_completed"],
        }

    def _relationship_anomaly_scoring_node(
        self,
        state: InvestigationState,
    ) -> dict[str, Any]:
        anomaly_score, is_anomalous = self._relationship_anomaly_scorer.score(
            state["graph_analysis"]
        )
        result = RelationshipAnomalyResult(
            anomaly_score=float(anomaly_score),
            is_anomalous=bool(is_anomalous),
        )
        return {
            "relationship_anomaly": result,
            "audit_events": [
                *state.get("audit_events", []),
                "relationship_anomaly_scoring_completed",
            ],
        }

    def _policy_retrieval_node(self, state: InvestigationState) -> dict[str, Any]:
        query = self._build_policy_query(state)
        policies = self._policy_retriever.retrieve(query, top_k=self._retrieval_top_k)
        return {
            "retrieved_policies": policies,
            "audit_events": [*state.get("audit_events", []), "policy_retrieval_completed"],
        }

    def _policy_validation_node(self, state: InvestigationState) -> dict[str, Any]:
        try:
            assessment = self._policy_assessor.assess(
                state["transaction"],
                state["risk_assessment"],
                state["graph_analysis"],
                state.get("retrieved_policies", []),
                state.get("relationship_anomaly"),
            )
            events = [*state.get("audit_events", []), "policy_assessment_completed"]
            return {
                "policy_assessment": assessment,
                "llm_usage": getattr(assessment, "llm_usage", None),
                "audit_events": events,
            }
        except PolicyAssessmentError as exc:
            logger.exception(
                "policy_assessment_failed",
                extra={"investigation_id": state["investigation_id"]},
            )
            retrieved_ids = list(
                dict.fromkeys(item.policy_id for item in state.get("retrieved_policies", []))
            )
            fallback = PolicyAssessment(
                recommendation=Recommendation.REQUEST_MORE_INFO,
                reasons=["A validated policy assessment could not be produced."],
                confidence=0.0,
                policy_references=[],
                policies_considered=retrieved_ids,
                missing_information=["Validated policy assessment"],
                requires_human_review=True,
                evidence_sufficient=False,
            )
            return {
                "policy_assessment": fallback,
                "llm_usage": exc.llm_usage,
                "audit_events": [*state.get("audit_events", []), "policy_assessment_failed"],
                "errors": [*state.get("errors", []), str(exc)],
            }

    def _route_after_policy_validation(self, state: InvestigationState) -> str:
        risk = state["risk_assessment"]
        graph = state["graph_analysis"]
        assessment = state["policy_assessment"]
        conflicting_signals = graph.suspicious and risk.risk_level is RiskLevel.LOW
        relationship_anomaly_disagreement = self._relationship_anomaly_disagreement(
            state
        )
        medium_risk_cannot_complete_automatically = (
            risk.risk_level is RiskLevel.MEDIUM
            and (
                assessment.recommendation is not Recommendation.CLEAR
                or assessment.confidence < 0.9
                or not assessment.evidence_sufficient
            )
        )
        requires_review = (
            risk.risk_level is RiskLevel.HIGH
            or medium_risk_cannot_complete_automatically
            or assessment.confidence < self._confidence_threshold
            or not assessment.evidence_sufficient
            or assessment.requires_human_review
            or conflicting_signals
            or relationship_anomaly_disagreement
        )
        return "human_review" if requires_review else "complete"

    @staticmethod
    def _relationship_anomaly_disagreement(state: InvestigationState) -> bool:
        return (
            state["risk_assessment"].risk_level
            in {RiskLevel.LOW, RiskLevel.MEDIUM}
            and not state["graph_analysis"].suspicious
            and state["relationship_anomaly"].is_anomalous
        )

    def _human_review_node(self, state: InvestigationState) -> dict[str, Any]:
        assessment = state["policy_assessment"]
        reasons = list(assessment.reasons)
        if state["risk_assessment"].risk_level is RiskLevel.HIGH:
            reasons.append("HIGH deterministic risk requires human review.")
        if assessment.confidence < self._confidence_threshold:
            reasons.append("Assessment confidence is below the human-review threshold.")
        if not assessment.evidence_sufficient:
            reasons.append("Retrieved policy evidence is insufficient.")
        if (
            state["graph_analysis"].suspicious
            and state["risk_assessment"].risk_level is RiskLevel.LOW
        ):
            reasons.append("Graph and deterministic risk signals conflict.")
        if self._relationship_anomaly_disagreement(state):
            reasons.append(
                "Relationship anomaly model flags atypical structure despite "
                "low or medium deterministic risk."
            )
        recommendation_value = assessment.recommendation
        if state["risk_assessment"].risk_level is RiskLevel.HIGH:
            recommendation_value = Recommendation.ESCALATE_TO_HUMAN
        recommendation = InvestigationRecommendation(
            recommendation=recommendation_value,
            reasons=list(dict.fromkeys(reasons)),
            confidence=assessment.confidence,
            policy_references=assessment.policy_references,
            missing_information=assessment.missing_information,
            requires_human_review=True,
        )
        return {
            "recommendation": recommendation,
            "requires_human_review": True,
            "status": InvestigationStatus.PENDING_HUMAN_REVIEW,
            "audit_events": [*state.get("audit_events", []), "human_review_required"],
        }

    def _finalize_node(self, state: InvestigationState) -> dict[str, Any]:
        assessment = state["policy_assessment"]
        recommendation = InvestigationRecommendation(
            recommendation=assessment.recommendation,
            reasons=assessment.reasons,
            confidence=assessment.confidence,
            policy_references=assessment.policy_references,
            missing_information=assessment.missing_information,
            requires_human_review=False,
        )
        return {
            "recommendation": recommendation,
            "requires_human_review": False,
            "status": InvestigationStatus.COMPLETED,
            "audit_events": [*state.get("audit_events", []), "investigation_completed"],
        }

    def _persistence_node(self, state: InvestigationState) -> dict[str, Any]:
        record = self._record_from_state(state)
        try:
            result = self._repository.save(record)
        except Exception as exc:
            logger.exception(
                "investigation_persistence_failed",
                extra={"investigation_id": state["investigation_id"]},
            )
            result = PersistenceResult(
                succeeded=False,
                backend=type(self._repository).__name__,
                error=str(exc),
            )
        updates: dict[str, Any] = {
            "persistence": result,
            "audit_events": [
                *state.get("audit_events", []),
                "persistence_succeeded" if result.succeeded else "persistence_failed",
            ],
        }
        if not result.succeeded:
            updates["status"] = InvestigationStatus.FAILED
            if result.error:
                updates["errors"] = [*state.get("errors", []), result.error]
        return updates

    @staticmethod
    def _build_policy_query(state: InvestigationState) -> str:
        transaction = state["transaction"]
        risk = state["risk_assessment"]
        graph = state["graph_analysis"]
        terms = [
            "financial crime transaction review policy",
            f"risk level {risk.risk_level.value}",
            *risk.risk_reasons,
            *graph.signals,
            f"origin country {transaction.origin_country}",
            f"destination country {transaction.destination_country}",
        ]
        return ". ".join(term for term in terms if term)

    @staticmethod
    def _record_from_state(state: InvestigationState) -> dict[str, Any]:
        return {
            "investigation_id": state["investigation_id"],
            "transaction_id": state["transaction"].transaction_id,
            "risk_score": state["risk_assessment"].risk_score,
            "risk_level": state["risk_assessment"].risk_level.value,
            "risk_reasons": state["risk_assessment"].risk_reasons,
            "graph_signals": state["graph_analysis"].model_dump(mode="json"),
            "retrieved_policy_references": [
                item.policy_id for item in state.get("retrieved_policies", [])
            ],
            "recommendation": state["recommendation"].recommendation.value,
            "human_review_required": state["requires_human_review"],
            "status": state["status"].value,
            "created_at": state["created_at"].isoformat(),
            "llm_usage": (
                state["llm_usage"].model_dump(mode="json")
                if state.get("llm_usage") is not None
                else None
            ),
            "audit_events": state.get("audit_events", []),
            "errors": state.get("errors", []),
        }

    def invoke(self, transaction: Transaction) -> InvestigationResponse:
        """Invoke the compiled LangGraph and return the public response model."""

        initial_state: InvestigationState = {
            "investigation_id": str(uuid4()),
            "created_at": datetime.now(UTC),
            "transaction": transaction,
            "audit_events": ["investigation_started"],
            "errors": [],
        }
        final_state: InvestigationState = self.graph.invoke(initial_state)
        recommendation = final_state["recommendation"]
        return InvestigationResponse(
            investigation_id=final_state["investigation_id"],
            transaction_id=transaction.transaction_id,
            risk_assessment=final_state["risk_assessment"],
            graph_analysis=final_state["graph_analysis"],
            relationship_anomaly=final_state["relationship_anomaly"],
            retrieved_policies=final_state.get("retrieved_policies", []),
            policy_assessment=final_state["policy_assessment"],
            llm_usage=final_state.get("llm_usage"),
            recommendation=recommendation.recommendation,
            reasons=recommendation.reasons,
            requires_human_review=final_state["requires_human_review"],
            status=final_state["status"],
            persistence=final_state["persistence"],
            created_at=final_state["created_at"],
        )
