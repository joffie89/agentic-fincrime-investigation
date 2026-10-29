"""LangGraph orchestration and conditional-route tests."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.models.domain import (
    GraphAnalysisResult,
    InvestigationStatus,
    PersistenceResult,
    PolicyAssessment,
    Recommendation,
    RetrievedPolicy,
)
from app.repositories.memory import InMemoryInvestigationRepository
from app.services.policy_assessment import (
    DemoPolicyAssessmentService,
    PolicyAssessmentError,
)
from app.services.risk_scorer import assess_transaction
from app.workflow.investigation_workflow import InvestigationWorkflow
from app.workflow.investigation_workflow import _should_qa_sample


class StaticGraphAnalyzer:
    def __init__(self, result: GraphAnalysisResult) -> None:
        self.result = result

    def analyze(self, _transaction):
        return self.result


class FailingGraphAnalyzer:
    def analyze(self, _transaction):
        raise RuntimeError("graph analysis exploded")


class StaticRelationshipAnomalyScorer:
    def __init__(self, result: tuple[float, bool]) -> None:
        self.result = result

    def score(self, _graph_analysis_result):
        return self.result


class StaticRetriever:
    def __init__(self, policies, catalog=None) -> None:
        self.policies = policies
        self.catalog = {
            item.policy_id: item
            for item in (catalog if catalog is not None else policies)
        }
        self.queries: list[str] = []
        self.lookup_requests: list[list[str]] = []

    def retrieve(self, query: str, top_k: int = 3):
        self.queries.append(query)
        return self.policies[:top_k]

    def get_by_policy_ids(self, policy_ids: list[str]):
        self.lookup_requests.append(policy_ids)
        return [
            self.catalog[policy_id]
            for policy_id in policy_ids
            if policy_id in self.catalog
        ]


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
    llm_assessment_enabled=True,
    qa_sample_rate=0.0,
):
    return InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=StaticGraphAnalyzer(graph_result),
        policy_retriever=StaticRetriever(policies),
        policy_assessor=assessor or DemoPolicyAssessmentService(),
        repository=repository or InMemoryInvestigationRepository(),
        retrieval_top_k=3,
        human_review_confidence_threshold=threshold,
        llm_assessment_enabled=llm_assessment_enabled,
        qa_sample_rate=qa_sample_rate,
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
    assert result.llm_disabled is False
    assert result.qa_sampled is False
    assert len(repository.records) == 1


def test_successful_investigation_persists_complete_audit_trail(
    transaction_factory, clean_graph_result, retrieved_policy
):
    repository = InMemoryInvestigationRepository()
    assessment = PolicyAssessment(
        recommendation=Recommendation.CLEAR,
        reasons=["The available evidence supports clearing the payment."],
        confidence=0.95,
        policy_references=[retrieved_policy.policy_id],
        policies_considered=[],
        missing_information=[],
        requires_human_review=False,
        evidence_sufficient=True,
    )
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        assessor=StaticAssessor(assessment),
        repository=repository,
    )
    transaction = transaction_factory(transaction_id="txn-audit-success")

    result = workflow.invoke(transaction)

    record = repository.records[0]
    assert record["investigation_id"] == result.investigation_id
    assert record["transaction"] == transaction.model_dump(mode="json")
    assert record["risk_assessment"] == result.risk_assessment.model_dump(mode="json")
    assert record["graph_analysis"] == clean_graph_result.model_dump(mode="json")
    assert record["relationship_anomaly"] == (
        result.relationship_anomaly.model_dump(mode="json")
    )
    assert record["retrieved_policies"] == [
        {
            "policy_id": retrieved_policy.policy_id,
            "chunk_id": retrieved_policy.chunk_id,
            "relevance_score": retrieved_policy.relevance_score,
        }
    ]
    assert record["policy_retrieval_sources"] == {
        "semantic_search": [retrieved_policy.policy_id],
        "signal_mapping": [],
    }
    assert record["policy_assessment"] == assessment.model_dump(mode="json")
    assert record["original_llm_recommendation"] is None
    assert record["rubric_override"] is False
    assert record["deterministic_citations"] == []
    assert record["fallback_reason"] is None
    assert record["rejected_citations"] == []
    assert record["rejected_policies_considered"] == []
    assert record["routing"]["decision"] == "complete"
    assert record["routing"]["triggered_rule"]
    assert record["routing"]["triggered_rule"] in record["routing"][
        "triggered_rules"
    ]
    assert record["llm_usage"] is None
    assert record["llm_disabled"] is False
    assert record["qa_sampled"] is False
    assert record["errors"] == []
    assert record["fallbacks"] == []

    expected_steps = {
        "risk_assessment",
        "graph_analysis",
        "relationship_anomaly_scoring",
        "policy_retrieval",
        "policy_validation",
        "routing_decision",
        "finalize",
        "persist",
    }
    assert expected_steps <= record["workflow_steps"].keys()
    for step_name in expected_steps:
        step = record["workflow_steps"][step_name]
        assert step["status"] == "completed"
        assert datetime.fromisoformat(step["started_at"]).tzinfo is not None
        assert datetime.fromisoformat(step["completed_at"]).tzinfo is not None


def test_signal_mapped_policy_is_added_to_semantic_results(
    transaction_factory, clean_graph_result, retrieved_policy
):
    repository = InMemoryInvestigationRepository()
    cross_border_policy = RetrievedPolicy(
        policy_id="DEMO-AML-004",
        chunk_id="DEMO-AML-004#001",
        text="Synthetic policy evidence for cross-border payment context.",
        relevance_score=1.0,
    )
    retriever = StaticRetriever(
        [retrieved_policy],
        catalog=[retrieved_policy, cross_border_policy],
    )
    workflow = InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=StaticGraphAnalyzer(clean_graph_result),
        policy_retriever=retriever,
        policy_assessor=DemoPolicyAssessmentService(),
        repository=repository,
        retrieval_top_k=3,
        relationship_anomaly_scorer=StaticRelationshipAnomalyScorer(
            (-0.1, False)
        ),
    )

    result = workflow.invoke(
        transaction_factory(
            transaction_id="txn-cross-border-policy-inclusion",
            destination_country="FR",
        )
    )

    assert [item.policy_id for item in result.retrieved_policies] == [
        "DEMO-AML-001",
        "DEMO-AML-004",
    ]
    assert retriever.lookup_requests == [["DEMO-AML-004"]]
    assert repository.records[0]["policy_retrieval_sources"] == {
        "semantic_search": ["DEMO-AML-001"],
        "signal_mapping": ["DEMO-AML-004"],
    }


def test_suspicious_graph_always_adds_network_policy(
    transaction_factory, clean_graph_result, retrieved_policy
):
    repository = InMemoryInvestigationRepository()
    network_policy = RetrievedPolicy(
        policy_id="DEMO-AML-003",
        chunk_id="DEMO-AML-003#001",
        text="Synthetic policy evidence for suspicious account links.",
        relevance_score=1.0,
    )
    retriever = StaticRetriever(
        [retrieved_policy],
        catalog=[retrieved_policy, network_policy],
    )
    suspicious_graph = clean_graph_result.model_copy(
        update={"signals": [], "suspicious": True}
    )
    workflow = InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=StaticGraphAnalyzer(suspicious_graph),
        policy_retriever=retriever,
        policy_assessor=DemoPolicyAssessmentService(),
        repository=repository,
        retrieval_top_k=3,
        relationship_anomaly_scorer=StaticRelationshipAnomalyScorer(
            (-0.1, False)
        ),
    )

    result = workflow.invoke(
        transaction_factory(transaction_id="txn-network-policy-inclusion")
    )

    assert [item.policy_id for item in result.retrieved_policies] == [
        "DEMO-AML-001",
        "DEMO-AML-003",
    ]
    assert retriever.lookup_requests == [["DEMO-AML-003"]]
    assert repository.records[0]["policy_retrieval_sources"] == {
        "semantic_search": ["DEMO-AML-001"],
        "signal_mapping": ["DEMO-AML-003"],
    }


def test_graph_failure_persists_partial_audit_trail(
    transaction_factory, retrieved_policy
):
    repository = InMemoryInvestigationRepository()
    workflow = InvestigationWorkflow(
        risk_assessor=assess_transaction,
        graph_analyzer=FailingGraphAnalyzer(),
        policy_retriever=StaticRetriever([retrieved_policy]),
        policy_assessor=DemoPolicyAssessmentService(),
        repository=repository,
        retrieval_top_k=3,
        human_review_confidence_threshold=0.6,
        relationship_anomaly_scorer=StaticRelationshipAnomalyScorer(
            (-0.1, False)
        ),
    )
    transaction = transaction_factory(transaction_id="txn-audit-partial")

    with pytest.raises(RuntimeError, match="graph analysis exploded"):
        workflow.invoke(transaction)

    assert len(repository.records) == 1
    record = repository.records[0]
    assert record["transaction"] == transaction.model_dump(mode="json")
    assert record["risk_assessment"] is not None
    assert record["graph_analysis"] is None
    assert record["relationship_anomaly"] is None
    assert record["retrieved_policies"] == []
    assert record["policy_assessment"] is None
    assert record["recommendation"] is None
    assert record["status"] == InvestigationStatus.FAILED.value
    assert record["routing"]["decision"] == "failed"
    assert record["routing"]["triggered_rule"] == "graph_analysis_failed"
    assert record["workflow_steps"]["risk_assessment"]["status"] == "completed"
    assert record["workflow_steps"]["graph_analysis"]["status"] == "failed"
    assert any("RuntimeError" in error for error in record["errors"])
    assert any("graph analysis exploded" in error for error in record["errors"])
    assert record["fallbacks"] == []


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


def test_medium_clear_high_confidence_with_sufficient_evidence_completes(
    transaction_factory, clean_graph_result, retrieved_policy
):
    assessment = PolicyAssessment(
        recommendation=Recommendation.CLEAR,
        reasons=["The available evidence supports clearing the payment."],
        confidence=0.9,
        policy_references=[retrieved_policy.policy_id],
        policies_considered=[],
        missing_information=[],
        requires_human_review=False,
        evidence_sufficient=True,
    )
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        assessor=StaticAssessor(assessment),
    )

    result = workflow.invoke(
        transaction_factory(transaction_id="txn-medium-clear", amount=600.0)
    )

    assert result.risk_assessment.risk_score == 30
    assert result.status is InvestigationStatus.COMPLETED
    assert result.recommendation is Recommendation.CLEAR
    assert result.requires_human_review is False


def test_medium_clear_low_confidence_routes_to_human_review(
    transaction_factory, clean_graph_result, retrieved_policy
):
    assessment = PolicyAssessment(
        recommendation=Recommendation.CLEAR,
        reasons=["The available evidence tentatively supports clearing the payment."],
        confidence=0.89,
        policy_references=[retrieved_policy.policy_id],
        policies_considered=[],
        missing_information=[],
        requires_human_review=False,
        evidence_sufficient=True,
    )
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        assessor=StaticAssessor(assessment),
    )

    result = workflow.invoke(
        transaction_factory(transaction_id="txn-medium-low-confidence", amount=600.0)
    )

    assert result.risk_assessment.risk_score == 30
    assert result.status is InvestigationStatus.PENDING_HUMAN_REVIEW
    assert result.recommendation is Recommendation.CLEAR
    assert result.requires_human_review is True


def test_medium_request_more_info_routes_to_human_review(
    transaction_factory, clean_graph_result, retrieved_policy
):
    assessment = PolicyAssessment(
        recommendation=Recommendation.REQUEST_MORE_INFO,
        reasons=["The payment purpose needs confirmation."],
        confidence=0.95,
        policy_references=[retrieved_policy.policy_id],
        policies_considered=[],
        missing_information=["Payment purpose"],
        requires_human_review=False,
        evidence_sufficient=True,
    )
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        assessor=StaticAssessor(assessment),
    )

    result = workflow.invoke(
        transaction_factory(transaction_id="txn-medium-more-info", amount=600.0)
    )

    assert result.risk_assessment.risk_score == 30
    assert result.status is InvestigationStatus.PENDING_HUMAN_REVIEW
    assert result.recommendation is Recommendation.REQUEST_MORE_INFO
    assert result.requires_human_review is True


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
        policies_considered=[],
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
        policies_considered=[],
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
        policies_considered=[],
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
    repository = InMemoryInvestigationRepository()
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        assessor=FailingAssessor(),
        repository=repository,
    )

    result = workflow.invoke(transaction_factory())

    assert result.status is InvestigationStatus.PENDING_HUMAN_REVIEW
    assert result.requires_human_review is True
    assert result.policy_assessment.confidence == 0.0
    assert result.policy_assessment.evidence_sufficient is False
    assert result.policy_assessment.policy_references == []
    assert result.policy_assessment.policies_considered == [retrieved_policy.policy_id]
    assert (
        result.policy_assessment.fallback_reason
        == "PolicyAssessmentError: provider unavailable"
    )
    record = repository.records[0]
    assert (
        record["fallback_reason"]
        == "PolicyAssessmentError: provider unavailable"
    )
    assert (
        record["policy_assessment"]["fallback_reason"]
        == "PolicyAssessmentError: provider unavailable"
    )
    assert record["fallbacks"]
    assert record["fallbacks"][0]["step"] == "policy_validation"
    assert (
        record["fallbacks"][0]["fallback_reason"]
        == "PolicyAssessmentError: provider unavailable"
    )
    assert "provider unavailable" in record["fallbacks"][0]["message"]
    assert any("provider unavailable" in error for error in record["errors"])


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


@pytest.mark.parametrize(
    ("transaction_id", "updates", "expected_risk_level"),
    [
        (
            "txn-llm-disabled-medium",
            {"amount": 600.0, "average_transaction_amount": 100.0},
            "MEDIUM",
        ),
        (
            "txn-llm-disabled-high",
            {
                "amount": 600.0,
                "average_transaction_amount": 100.0,
                "account_age_days": 10,
                "new_payee": True,
            },
            "HIGH",
        ),
        (
            "txn-llm-disabled-low-signal",
            {"new_payee": True},
            "LOW",
        ),
    ],
)
def test_llm_disabled_routes_nontrivial_risk_to_human_review(
    transaction_factory,
    clean_graph_result,
    retrieved_policy,
    transaction_id,
    updates,
    expected_risk_level,
):
    repository = InMemoryInvestigationRepository()
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        repository=repository,
        llm_assessment_enabled=False,
    )

    result = workflow.invoke(
        transaction_factory(transaction_id=transaction_id, **updates)
    )

    assert result.risk_assessment.risk_level.value == expected_risk_level
    assert result.status is InvestigationStatus.PENDING_HUMAN_REVIEW
    assert result.requires_human_review is True
    assert result.llm_disabled is True
    assert result.qa_sampled is False
    record = repository.records[0]
    assert record["llm_disabled"] is True
    assert record["qa_sampled"] is False
    assert "llm_disabled_risk_requires_human_review" in record["routing"][
        "triggered_rules"
    ]


def test_llm_disabled_allows_clean_low_risk_case_to_complete(
    transaction_factory, clean_graph_result, retrieved_policy
):
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        llm_assessment_enabled=False,
    )

    result = workflow.invoke(
        transaction_factory(transaction_id="txn-llm-disabled-clean")
    )

    assert result.status is InvestigationStatus.COMPLETED
    assert result.requires_human_review is False
    assert result.llm_disabled is True


def test_qa_sample_rate_zero_never_samples_eligible_case(
    transaction_factory, clean_graph_result, retrieved_policy
):
    repository = InMemoryInvestigationRepository()
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        repository=repository,
        qa_sample_rate=0.0,
    )

    result = workflow.invoke(transaction_factory(transaction_id="txn-qa-none"))

    assert result.status is InvestigationStatus.COMPLETED
    assert result.qa_sampled is False
    assert repository.records[0]["qa_sampled"] is False
    assert "qa_sample" not in repository.records[0]["routing"]["triggered_rules"]


def test_qa_sample_rate_one_samples_every_eligible_case_without_rerouting(
    transaction_factory, clean_graph_result, retrieved_policy
):
    repository = InMemoryInvestigationRepository()
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        repository=repository,
        qa_sample_rate=1.0,
    )

    result = workflow.invoke(transaction_factory(transaction_id="txn-qa-all"))

    assert result.status is InvestigationStatus.COMPLETED
    assert result.recommendation is Recommendation.CLEAR
    assert result.requires_human_review is False
    assert result.qa_sampled is True
    record = repository.records[0]
    assert record["qa_sampled"] is True
    assert record["routing"]["decision"] == "complete"
    assert record["routing"]["triggered_rule"] == (
        "automatic_completion_criteria_met"
    )
    assert record["routing"]["triggered_rules"] == [
        "automatic_completion_criteria_met",
        "qa_sample",
    ]


def test_human_review_case_is_never_qa_sampled_even_at_rate_one(
    transaction_factory, clean_graph_result, retrieved_policy
):
    repository = InMemoryInvestigationRepository()
    workflow = build_workflow(
        graph_result=clean_graph_result,
        policies=[retrieved_policy],
        repository=repository,
        qa_sample_rate=1.0,
    )

    result = workflow.invoke(
        transaction_factory(
            transaction_id="txn-qa-human-review",
            amount=600.0,
            average_transaction_amount=100.0,
            account_age_days=10,
            new_payee=True,
        )
    )

    assert result.status is InvestigationStatus.PENDING_HUMAN_REVIEW
    assert result.requires_human_review is True
    assert result.qa_sampled is False
    assert repository.records[0]["qa_sampled"] is False
    assert "qa_sample" not in repository.records[0]["routing"]["triggered_rules"]


def test_qa_sampling_is_stable_for_same_investigation_id():
    investigation_id = "inv-stable-qa-decision"

    first = _should_qa_sample(investigation_id, 0.37)
    repeated = [_should_qa_sample(investigation_id, 0.37) for _ in range(10)]

    assert repeated == [first] * 10
