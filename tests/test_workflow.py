"""LangGraph orchestration and conditional-route tests."""

from __future__ import annotations

from app.models.domain import (
    GraphAnalysisResult,
    InvestigationStatus,
    PersistenceResult,
    PolicyAssessment,
    Recommendation,
)
from app.repositories.memory import InMemoryInvestigationRepository
from app.services.policy_assessment import (
    DemoPolicyAssessmentService,
    PolicyAssessmentError,
)
from app.services.risk_scorer import assess_transaction
from app.workflow.investigation_workflow import InvestigationWorkflow


class StaticGraphAnalyzer:
    def __init__(self, result: GraphAnalysisResult) -> None:
        self.result = result

    def analyze(self, _transaction):
        return self.result


class StaticRelationshipAnomalyScorer:
    def __init__(self, result: tuple[float, bool]) -> None:
        self.result = result

    def score(self, _graph_analysis_result):
        return self.result


class StaticRetriever:
    def __init__(self, policies) -> None:
        self.policies = policies
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int = 3):
        self.queries.append(query)
        return self.policies[:top_k]


class StaticAssessor:
    def __init__(self, assessment: PolicyAssessment) -> None:
        self.assessment = assessment

    def assess(self, *_args):
        return self.assessment


class FailingAssessor:
    def assess(self, *_args):
        raise PolicyAssessmentError("provider unavailable")


class FailingRepository:
    def save(self, _record):
        return PersistenceResult(
            succeeded=False,
            backend="postgres",
            error="database unavailable",
        )


def build_workflow(
    *,
    graph_result,
    policies,
    assessor=None,
    repository=None,
    threshold=0.6,
    anomaly_result=(-0.1, False),
):
    return InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=StaticGraphAnalyzer(graph_result),
        policy_retriever=StaticRetriever(policies),
        policy_assessor=assessor or DemoPolicyAssessmentService(),
        repository=repository or InMemoryInvestigationRepository(),
        retrieval_top_k=3,
        human_review_confidence_threshold=threshold,
        relationship_anomaly_scorer=StaticRelationshipAnomalyScorer(
            anomaly_result
        ),
    )


def test_compiled_graph_contains_real_nodes(
    clean_graph_result, retrieved_policy
):
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
    )

    node_names = set(workflow.graph.get_graph().nodes)

    assert {
        "risk_assessment",
        "graph_analysis",
        "relationship_anomaly_scoring",
        "policy_retrieval",
        "policy_validation",
        "human_review_required",
        "finalize",
        "persist",
    } <= node_names


def test_low_risk_routes_to_completed(
    transaction_factory, clean_graph_result, retrieved_policy
):
    repository = InMemoryInvestigationRepository()
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        repository=repository,
    )

    result = workflow.invoke(transaction_factory())

    assert result.status is InvestigationStatus.COMPLETED
    assert result.recommendation is Recommendation.CLEAR
    assert result.requires_human_review is False
    assert result.relationship_anomaly is not None
    assert result.relationship_anomaly.anomaly_score == -0.1
    assert result.relationship_anomaly.is_anomalous is False
    assert result.persistence.succeeded is True
    assert len(repository.records) == 1


def test_high_risk_routes_to_human_review(
    transaction_factory, clean_graph_result, retrieved_policy
):
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
    )
    transaction = transaction_factory(
        transaction_id="txn-high",
        amount=6000.0,
        average_transaction_amount=1000.0,
        account_age_days=10,
        new_payee=True,
        transactions_last_24h=11,
    )

    result = workflow.invoke(transaction)

    assert result.risk_assessment.risk_score == 85
    assert result.status is InvestigationStatus.PENDING_HUMAN_REVIEW
    assert result.recommendation is Recommendation.ESCALATE_TO_HUMAN
    assert result.requires_human_review is True
    assert any("HIGH" in reason for reason in result.reasons)


def test_insufficient_policy_evidence_routes_to_human_review(
    transaction_factory, clean_graph_result
):
    workflow = build_workflow(graph_result=clean_graph_result, policies=[])

    result = workflow.invoke(transaction_factory())

    assert result.recommendation is Recommendation.REQUEST_MORE_INFO
    assert result.requires_human_review is True
    assert result.policy_assessment.evidence_sufficient is False
    assert any("insufficient" in reason.lower() for reason in result.reasons)


def test_low_confidence_routes_to_human_review(
    transaction_factory, clean_graph_result, retrieved_policy
):
    assessment = PolicyAssessment(
        recommendation=Recommendation.REQUEST_MORE_INFO,
        reasons=["The evidence is ambiguous."],
        confidence=0.4,
        policy_references=[retrieved_policy.policy_id],
        missing_information=["Payment purpose"],
        requires_human_review=False,
        evidence_sufficient=True,
    )
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        assessor=StaticAssessor(assessment),
        threshold=0.6,
    )

    result = workflow.invoke(transaction_factory())

    assert result.requires_human_review is True
    assert any("confidence" in reason.lower() for reason in result.reasons)


def test_conflicting_graph_signal_routes_to_human_review(
    transaction_factory, clean_graph_result, retrieved_policy
):
    suspicious_graph = clean_graph_result.model_copy(
        update={"signals": ["SHARED_COUNTERPARTIES"], "suspicious": True}
    )
    assessment = PolicyAssessment(
        recommendation=Recommendation.CLEAR,
        reasons=["Policy evidence alone does not trigger escalation."],
        confidence=0.95,
        policy_references=[retrieved_policy.policy_id],
        missing_information=[],
        requires_human_review=False,
        evidence_sufficient=True,
    )
    workflow = build_workflow(
        graph_result=suspicious_graph,
        policies=[retrieved_policy],
        assessor=StaticAssessor(assessment),
    )

    result = workflow.invoke(transaction_factory())

    assert result.requires_human_review is True
    assert any("conflict" in reason.lower() for reason in result.reasons)


def test_relationship_anomaly_disagreement_routes_to_human_review(
    transaction_factory, clean_graph_result, retrieved_policy
):
    assessment = PolicyAssessment(
        recommendation=Recommendation.CLEAR,
        reasons=["Policy evidence alone does not trigger escalation."],
        confidence=0.95,
        policy_references=[retrieved_policy.policy_id],
        missing_information=[],
        requires_human_review=False,
        evidence_sufficient=True,
    )
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        assessor=StaticAssessor(assessment),
        anomaly_result=(0.2, True),
    )

    result = workflow.invoke(transaction_factory())

    assert result.status is InvestigationStatus.PENDING_HUMAN_REVIEW
    assert result.requires_human_review is True
    assert result.relationship_anomaly is not None
    assert result.relationship_anomaly.is_anomalous is True
    assert any("anomaly model" in reason.lower() for reason in result.reasons)


def test_provider_failure_uses_safe_validated_fallback(
    transaction_factory, clean_graph_result, retrieved_policy
):
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        assessor=FailingAssessor(),
    )

    result = workflow.invoke(transaction_factory())

    assert result.status is InvestigationStatus.PENDING_HUMAN_REVIEW
    assert result.requires_human_review is True
    assert result.policy_assessment.confidence == 0.0
    assert result.policy_assessment.evidence_sufficient is False


def test_persistence_failure_is_reported_truthfully(
    transaction_factory, clean_graph_result, retrieved_policy
):
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        repository=FailingRepository(),
    )

    result = workflow.invoke(transaction_factory())

    assert result.status is InvestigationStatus.FAILED
    assert result.persistence.succeeded is False
    assert result.persistence.backend == "postgres"
    assert result.persistence.error == "database unavailable"
