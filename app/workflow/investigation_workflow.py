"""Actual LangGraph orchestration for an investigation."""

from __future__ import annotations

import logging
import time
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
from app.services.policy_assessment import (
    DEMO_POLICY_TRIGGER_SIGNALS,
    PolicyAssessmentError,
    PolicyAssessmentService,
    _active_demo_policy_signals,
    safe_exception_details,
)
from app.services.relationship_anomaly_scorer import RelationshipAnomalyScorer
from app.workflow.state import InvestigationState

logger = logging.getLogger(__name__)


class GraphAnalyzerProtocol(Protocol):
    def analyze(self, transaction: Transaction) -> Any: ...


class PolicyRetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> list[Any]: ...

    def get_by_policy_ids(self, policy_ids: list[str]) -> list[Any]: ...


class RelationshipAnomalyScorerProtocol(Protocol):
    def score(self, graph_analysis_result: Any) -> tuple[float, bool]: ...


class InvestigationRepositoryProtocol(Protocol):
    def save(self, record: dict[str, Any]) -> PersistenceResult: ...

    def get(self, investigation_id: str) -> dict[str, Any] | None: ...


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
        builder.add_node(
            "risk_assessment",
            self._timed_node("risk_assessment", self._risk_assessment_node),
        )
        builder.add_node(
            "graph_analysis",
            self._timed_node("graph_analysis", self._graph_analysis_node),
        )
        builder.add_node(
            "relationship_anomaly_scoring",
            self._timed_node(
                "relationship_anomaly_scoring",
                self._relationship_anomaly_scoring_node,
            ),
        )
        builder.add_node(
            "policy_retrieval",
            self._timed_node("policy_retrieval", self._policy_retrieval_node),
        )
        builder.add_node(
            "policy_validation",
            self._timed_node("policy_validation", self._policy_validation_node),
        )
        builder.add_node(
            "routing_decision",
            self._timed_node("routing_decision", self._routing_decision_node),
        )
        builder.add_node(
            "human_review_required",
            self._timed_node("human_review_required", self._human_review_node),
        )
        builder.add_node(
            "finalize",
            self._timed_node("finalize", self._finalize_node),
        )
        builder.add_node("persist", self._persistence_node)

        builder.add_edge(START, "risk_assessment")
        builder.add_edge("risk_assessment", "graph_analysis")
        builder.add_edge("graph_analysis", "relationship_anomaly_scoring")
        builder.add_edge("relationship_anomaly_scoring", "policy_retrieval")
        builder.add_edge("policy_retrieval", "policy_validation")
        builder.add_edge("policy_validation", "routing_decision")
        builder.add_conditional_edges(
            "routing_decision",
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

    def _timed_node(
        self,
        step_name: str,
        node: Callable[[InvestigationState], dict[str, Any]],
    ) -> Callable[[InvestigationState], dict[str, Any]]:
        """Record step timing and persist a partial audit record on failure."""

        def invoke_node(state: InvestigationState) -> dict[str, Any]:
            started_at = datetime.now(UTC)
            started_counter = time.perf_counter()
            try:
                updates = node(state)
            except Exception as exc:
                completed_at = datetime.now(UTC)
                step = self._step_timing(
                    started_at,
                    completed_at,
                    started_counter,
                    status="failed",
                    error=exc,
                )
                error = (
                    f"{step_name} failed with {type(exc).__name__}: {exc}"
                )
                failure_rule = f"{step_name}_failed"
                failed_state: InvestigationState = {
                    **state,
                    "status": InvestigationStatus.FAILED,
                    "requires_human_review": True,
                    "routing": {
                        "decision": "failed",
                        "triggered_rule": failure_rule,
                        "triggered_rules": [failure_rule],
                    },
                    "workflow_steps": {
                        **state.get("workflow_steps", {}),
                        step_name: step,
                    },
                    "audit_events": [
                        *state.get("audit_events", []),
                        f"{step_name}_failed",
                    ],
                    "errors": [*state.get("errors", []), error],
                }
                self._persist_partial_failure(failed_state)
                raise

            completed_at = datetime.now(UTC)
            return {
                **updates,
                "workflow_steps": {
                    **state.get("workflow_steps", {}),
                    step_name: self._step_timing(
                        started_at,
                        completed_at,
                        started_counter,
                        status="completed",
                    ),
                },
            }

        return invoke_node

    @staticmethod
    def _step_timing(
        started_at: datetime,
        completed_at: datetime,
        started_counter: float,
        *,
        status: str,
        error: Exception | None = None,
    ) -> dict[str, Any]:
        timing: dict[str, Any] = {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_ms": round(
                max(0.0, (time.perf_counter() - started_counter) * 1_000),
                1,
            ),
            "status": status,
        }
        if error is not None:
            timing["error_type"] = type(error).__name__
            timing["error_message"] = str(error)
        return timing

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
        semantic_policies = self._policy_retriever.retrieve(
            query,
            top_k=self._retrieval_top_k,
        )
        semantic_policy_ids = list(
            dict.fromkeys(item.policy_id for item in semantic_policies)
        )
        active_signals = _active_demo_policy_signals(
            state["transaction"],
            state["risk_assessment"],
            state["graph_analysis"],
        )
        mapped_policy_ids = [
            policy_id
            for policy_id, trigger_signals in DEMO_POLICY_TRIGGER_SIGNALS.items()
            if trigger_signals & active_signals
        ]
        missing_mapped_ids = [
            policy_id
            for policy_id in mapped_policy_ids
            if policy_id not in semantic_policy_ids
        ]
        signal_policies = (
            self._policy_retriever.get_by_policy_ids(missing_mapped_ids)
            if missing_mapped_ids
            else []
        )

        policies = list(
            {
                item.policy_id: item
                for item in [*semantic_policies, *signal_policies]
            }.values()
        )
        combined_policy_ids = {item.policy_id for item in policies}
        signal_policy_ids = [
            policy_id
            for policy_id in mapped_policy_ids
            if policy_id in combined_policy_ids
        ]
        return {
            "retrieved_policies": policies,
            "policy_retrieval_sources": {
                "semantic_search": semantic_policy_ids,
                "signal_mapping": signal_policy_ids,
            },
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
            updates: dict[str, Any] = {
                "policy_assessment": assessment,
                "llm_usage": getattr(assessment, "llm_usage", None),
                "audit_events": [
                    *state.get("audit_events", []),
                    "policy_assessment_completed",
                ],
            }
            if assessment.fallback_reason is not None or getattr(
                assessment,
                "fallback_used",
                False,
            ):
                error_type = getattr(
                    assessment,
                    "fallback_error_type",
                    "PolicyAssessmentError",
                )
                message = getattr(
                    assessment,
                    "fallback_error_message",
                    "AI policy assessment failed",
                )
                fallback = self._fallback_entry(
                    error_type=error_type or "PolicyAssessmentError",
                    message=message or "AI policy assessment failed",
                    fallback_reason=assessment.fallback_reason,
                )
                updates["fallbacks"] = [
                    *state.get("fallbacks", []),
                    fallback,
                ]
                updates["errors"] = [
                    *state.get("errors", []),
                    (
                        "policy_validation used fallback after "
                        f"{fallback['error_type']}: {fallback['message']}"
                    ),
                ]
                updates["audit_events"] = [
                    *state.get("audit_events", []),
                    "policy_assessment_fallback",
                ]
            return updates
        except PolicyAssessmentError as exc:
            error_type, error_message = safe_exception_details(exc)
            fallback_reason = f"{error_type}: {error_message}"
            logger.error(
                "policy_assessment_failed",
                extra={
                    "investigation_id": state["investigation_id"],
                    "exception_type": error_type,
                    "exception_message": error_message,
                },
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
                fallback_reason=fallback_reason,
            )
            fallback_entry = self._fallback_entry(
                error_type=error_type,
                message=error_message,
                fallback_reason=fallback_reason,
            )
            return {
                "policy_assessment": fallback,
                "llm_usage": exc.llm_usage,
                "audit_events": [
                    *state.get("audit_events", []),
                    "policy_assessment_fallback",
                ],
                "errors": [
                    *state.get("errors", []),
                    "policy_validation used fallback after "
                    f"{fallback_reason}",
                ],
                "fallbacks": [
                    *state.get("fallbacks", []),
                    fallback_entry,
                ],
            }

    @staticmethod
    def _fallback_entry(
        *,
        error_type: str,
        message: str,
        fallback_reason: str | None = None,
    ) -> dict[str, str]:
        return {
            "step": "policy_validation",
            "error_type": error_type,
            "message": message,
            "fallback_reason": (
                fallback_reason or f"{error_type}: {message}"
            ),
            "occurred_at": datetime.now(UTC).isoformat(),
        }

    def _routing_decision_node(
        self,
        state: InvestigationState,
    ) -> dict[str, Any]:
        risk = state["risk_assessment"]
        graph = state["graph_analysis"]
        assessment = state["policy_assessment"]
        triggered_rules: list[str] = []

        def add_rule(rule: str) -> None:
            if rule not in triggered_rules:
                triggered_rules.append(rule)

        if risk.risk_level is RiskLevel.HIGH:
            add_rule("high_risk_requires_human_review")
        if risk.risk_level is RiskLevel.MEDIUM:
            if assessment.recommendation is not Recommendation.CLEAR:
                add_rule("medium_risk_assessment_not_clear")
            if assessment.confidence < 0.9:
                add_rule("medium_risk_confidence_below_0_9")
            if not assessment.evidence_sufficient:
                add_rule("medium_risk_evidence_insufficient")
        if assessment.confidence < self._confidence_threshold:
            add_rule("assessment_confidence_below_threshold")
        if not assessment.evidence_sufficient:
            add_rule("insufficient_policy_evidence")
        if assessment.requires_human_review:
            add_rule("policy_assessment_requires_human_review")
        if graph.suspicious and risk.risk_level is RiskLevel.LOW:
            add_rule("graph_risk_conflict")
        if self._relationship_anomaly_disagreement(state):
            add_rule("relationship_anomaly_disagreement")

        decision = "human_review" if triggered_rules else "complete"
        if not triggered_rules:
            triggered_rules.append("automatic_completion_criteria_met")
        routing = {
            "decision": decision,
            "triggered_rule": triggered_rules[0],
            "triggered_rules": triggered_rules,
        }
        return {
            "routing": routing,
            "audit_events": [
                *state.get("audit_events", []),
                f"routing_decision_{decision}",
            ],
        }

    def _route_after_policy_validation(self, state: InvestigationState) -> str:
        return (
            "human_review"
            if state["routing"]["decision"] == "human_review"
            else "complete"
        )

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
        try:
            persistence_state = self._state_with_persistence_audit(state)
            record = self._record_from_state(persistence_state)
            result = self._repository.save(record)
        except Exception:
            logger.exception(
                "investigation_persistence_failed",
                extra={"investigation_id": state["investigation_id"]},
            )
            result = PersistenceResult(
                succeeded=False,
                backend=self._repository_backend(),
                error="Investigation could not be saved",
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

    def _state_with_persistence_audit(
        self,
        state: InvestigationState,
    ) -> InvestigationState:
        """Prepare the canonical record written by a successful save."""

        started_at = datetime.now(UTC)
        started_counter = time.perf_counter()
        completed_at = datetime.now(UTC)
        return {
            **state,
            "persistence": PersistenceResult(
                succeeded=True,
                backend=self._repository_backend(),
            ),
            "workflow_steps": {
                **state.get("workflow_steps", {}),
                "persist": self._step_timing(
                    started_at,
                    completed_at,
                    started_counter,
                    status="completed",
                ),
            },
            "audit_events": [
                *state.get("audit_events", []),
                "persistence_succeeded",
            ],
        }

    def _persist_partial_failure(self, state: InvestigationState) -> None:
        """Best-effort persistence that never hides the original step error."""

        try:
            persistence_state = self._state_with_persistence_audit(state)
            record = self._record_from_state(persistence_state)
            result = self._repository.save(record)
        except Exception:
            logger.exception(
                "partial_investigation_persistence_failed",
                extra={
                    "event": "partial_investigation_persistence_failed",
                    "investigation_id": state["investigation_id"],
                },
            )
            return
        if not result.succeeded:
            logger.error(
                "partial_investigation_persistence_failed",
                extra={
                    "event": "partial_investigation_persistence_failed",
                    "investigation_id": state["investigation_id"],
                    "backend": result.backend,
                },
            )

    def _repository_backend(self) -> str:
        return str(
            getattr(self._repository, "backend", type(self._repository).__name__)
        )

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
        transaction = state["transaction"]
        risk = state.get("risk_assessment")
        graph = state.get("graph_analysis")
        relationship_anomaly = state.get("relationship_anomaly")
        policy_assessment = state.get("policy_assessment")
        recommendation = state.get("recommendation")
        persistence = state.get("persistence")
        status_value = (
            state["status"].value if state.get("status") is not None else None
        )
        retrieved_policies = [
            {
                "policy_id": item.policy_id,
                "chunk_id": item.chunk_id,
                "relevance_score": item.relevance_score,
            }
            for item in state.get("retrieved_policies", [])
        ]
        return {
            "schema_version": 1,
            "investigation_id": state["investigation_id"],
            "transaction_id": transaction.transaction_id,
            "transaction": transaction.model_dump(mode="json"),
            "risk_assessment": (
                risk.model_dump(mode="json") if risk is not None else None
            ),
            "graph_analysis": (
                graph.model_dump(mode="json") if graph is not None else None
            ),
            "relationship_anomaly": (
                relationship_anomaly.model_dump(mode="json")
                if relationship_anomaly is not None
                else None
            ),
            "retrieved_policies": retrieved_policies,
            "policy_retrieval_sources": state.get(
                "policy_retrieval_sources",
                {"semantic_search": [], "signal_mapping": []},
            ),
            "policy_assessment": (
                policy_assessment.model_dump(mode="json")
                if policy_assessment is not None
                else None
            ),
            "fallback_reason": (
                policy_assessment.fallback_reason
                if policy_assessment is not None
                else None
            ),
            "rejected_citations": (
                policy_assessment.rejected_citations
                if policy_assessment is not None
                else []
            ),
            "rejected_policies_considered": (
                policy_assessment.rejected_policies_considered
                if policy_assessment is not None
                else []
            ),
            "routing": state.get("routing"),
            "recommendation": (
                recommendation.model_dump(mode="json")
                if recommendation is not None
                else None
            ),
            "requires_human_review": state.get("requires_human_review"),
            "status": status_value,
            "llm_usage": (
                state["llm_usage"].model_dump(mode="json")
                if state.get("llm_usage") is not None
                else None
            ),
            "workflow_steps": state.get("workflow_steps", {}),
            "persistence": (
                persistence.model_dump(mode="json")
                if persistence is not None
                else None
            ),
            "created_at": state["created_at"].isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "audit_events": state.get("audit_events", []),
            "errors": state.get("errors", []),
            "fallbacks": state.get("fallbacks", []),
            # Keep the summary fields used by the PostgreSQL table alongside
            # the complete JSON record for backwards-compatible indexing.
            "risk_score": risk.risk_score if risk is not None else None,
            "risk_level": risk.risk_level.value if risk is not None else None,
            "risk_reasons": risk.risk_reasons if risk is not None else [],
            "graph_signals": (
                graph.model_dump(mode="json") if graph is not None else {}
            ),
            "retrieved_policy_references": [
                item.policy_id for item in state.get("retrieved_policies", [])
            ],
            "human_review_required": state.get("requires_human_review", False),
        }

    def invoke(self, transaction: Transaction) -> InvestigationResponse:
        """Invoke the compiled LangGraph and return the public response model."""

        initial_state: InvestigationState = {
            "investigation_id": str(uuid4()),
            "created_at": datetime.now(UTC),
            "transaction": transaction,
            "audit_events": ["investigation_started"],
            "errors": [],
            "fallbacks": [],
            "workflow_steps": {},
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

    def get_investigation(self, investigation_id: str) -> dict[str, Any] | None:
        """Return the persisted audit record for ``investigation_id``."""

        return self._repository.get(investigation_id)
