"""Structured local and Azure policy-assessment tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.models.domain import (
    PolicyAssessment,
    Recommendation,
    RetrievedPolicy,
    RiskAssessment,
    RiskLevel,
)
from app.services.policy_assessment import (
    AzureOpenAIPolicyAssessmentService,
    DemoPolicyAssessmentService,
    PolicyAssessmentError,
)
from app.services.risk_scorer import assess_transaction


def low_risk() -> RiskAssessment:
    return RiskAssessment(
        risk_score=0,
        risk_level=RiskLevel.LOW,
        risk_reasons=[],
        investigate=False,
    )


def high_risk() -> RiskAssessment:
    return RiskAssessment(
        risk_score=85,
        risk_level=RiskLevel.HIGH,
        risk_reasons=["All deterministic demo rules fired"],
        investigate=True,
    )


def all_demo_policies() -> list[RetrievedPolicy]:
    return [
        RetrievedPolicy(
            policy_id=f"DEMO-AML-{number:03d}",
            chunk_id=f"DEMO-AML-{number:03d}#001",
            text=f"Synthetic policy {number} evidence.",
            relevance_score=0.75,
        )
        for number in range(1, 6)
    ]


def test_demo_assessor_clears_clean_transaction_without_policy_citations(
    transaction_factory, clean_graph_result
):
    result = DemoPolicyAssessmentService().assess(
        transaction_factory(),
        low_risk(),
        clean_graph_result,
        all_demo_policies(),
    )

    assert result.recommendation is Recommendation.CLEAR
    assert result.evidence_sufficient is True
    assert result.requires_human_review is False
    assert result.policy_references == []


def test_demo_assessor_cites_only_policies_for_active_risk_signals(
    transaction_factory, clean_graph_result
):
    transaction = transaction_factory(
        new_payee=True,
        transactions_last_24h=11,
    )

    result = DemoPolicyAssessmentService().assess(
        transaction,
        assess_transaction(transaction),
        clean_graph_result,
        all_demo_policies(),
    )

    assert result.policy_references == ["DEMO-AML-001", "DEMO-AML-002"]


def test_demo_assessor_escalates_high_risk(
    transaction_factory, clean_graph_result, retrieved_policy
):
    result = DemoPolicyAssessmentService().assess(
        transaction_factory(), high_risk(), clean_graph_result, [retrieved_policy]
    )

    assert result.recommendation is Recommendation.ESCALATE_TO_HUMAN
    assert result.requires_human_review is True


def test_demo_assessor_marks_missing_evidence(
    transaction_factory, clean_graph_result
):
    result = DemoPolicyAssessmentService().assess(
        transaction_factory(), low_risk(), clean_graph_result, []
    )

    assert result.recommendation is Recommendation.REQUEST_MORE_INFO
    assert result.confidence == 0.3
    assert result.evidence_sufficient is False
    assert result.requires_human_review is True
    assert result.policy_references == []


class StubCompletions:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(self.payload))
                )
            ]
        )


def azure_client(payload: dict[str, object]) -> tuple[SimpleNamespace, StubCompletions]:
    completions = StubCompletions(payload)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def valid_azure_payload() -> dict[str, object]:
    return PolicyAssessment(
        recommendation=Recommendation.REQUEST_MORE_INFO,
        reasons=["The supplied evidence calls for supporting context."],
        confidence=0.8,
        policy_references=["DEMO-AML-001"],
        missing_information=["Payment purpose"],
        requires_human_review=False,
        evidence_sufficient=True,
    ).model_dump(mode="json")


def test_azure_assessor_validates_structured_output_and_prompt(
    transaction_factory, clean_graph_result, retrieved_policy
):
    client, completions = azure_client(valid_azure_payload())
    service = AzureOpenAIPolicyAssessmentService(client, "chat-deployment")

    result = service.assess(
        transaction_factory(), low_risk(), clean_graph_result, [retrieved_policy]
    )

    assert result.policy_references == ["DEMO-AML-001"]
    assert completions.kwargs is not None
    assert completions.kwargs["temperature"] == 0
    messages = completions.kwargs["messages"]
    assert "Never invent policy identifiers" in messages[0]["content"]
    assert "DEMO-AML-001" in messages[1]["content"]


def test_azure_assessor_rejects_fabricated_policy_reference(
    transaction_factory, clean_graph_result, retrieved_policy
):
    payload = valid_azure_payload()
    payload["policy_references"] = ["FABRICATED-POLICY"]
    client, _ = azure_client(payload)
    service = AzureOpenAIPolicyAssessmentService(client, "chat-deployment")

    with pytest.raises(PolicyAssessmentError, match="not retrieved"):
        service.assess(
            transaction_factory(), low_risk(), clean_graph_result, [retrieved_policy]
        )


def test_azure_assessor_wraps_invalid_json(
    transaction_factory, clean_graph_result, retrieved_policy
):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
                )
            )
        )
    )
    service = AzureOpenAIPolicyAssessmentService(client, "chat-deployment")

    with pytest.raises(PolicyAssessmentError, match="assessment failed"):
        service.assess(
            transaction_factory(), low_risk(), clean_graph_result, [retrieved_policy]
        )

