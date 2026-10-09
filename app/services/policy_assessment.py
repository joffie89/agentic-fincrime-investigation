"""Policy-grounded assessment services.

The deterministic implementation is deliberately conservative and keeps the
demo runnable without paid services.  The Azure implementation is an adapter:
it supplies the same validated :class:`PolicyAssessment` contract and never
allows an unvalidated model response into the workflow.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final, Protocol, Sequence

from app.models.domain import (
    GraphAnalysisResult,
    PolicyAssessment,
    Recommendation,
    RetrievedPolicy,
    RiskAssessment,
    RiskLevel,
)
from app.models.transaction import Transaction


class PolicyAssessmentError(RuntimeError):
    """Raised when a policy-assessment provider cannot return valid output."""


# Single applicability map for the synthetic policies in data/policies.json.
# Graph signal names deliberately match GraphAnalyzer's public output.
DEMO_POLICY_TRIGGER_SIGNALS: Final[Mapping[str, frozenset[str]]] = {
    "DEMO-AML-001": frozenset(
        {"YOUNG_ACCOUNT", "NEW_PAYEE", "UNUSUAL_AMOUNT"}
    ),
    "DEMO-AML-002": frozenset({"HIGH_VELOCITY"}),
    "DEMO-AML-003": frozenset(
        {
            "HIGH_CONNECTIVITY",
            "SHARED_COUNTERPARTIES",
            "REPEATED_DIRECT_TRANSACTIONS",
            "COUNTERPARTY_CONCENTRATION",
        }
    ),
    "DEMO-AML-004": frozenset({"CROSS_BORDER"}),
    "DEMO-AML-005": frozenset({"HIGH_RISK", "RISK_GRAPH_CONFLICT"}),
}


class PolicyAssessmentService(Protocol):
    """Boundary between orchestration and an LLM or deterministic substitute."""

    def assess(
        self,
        transaction: Transaction,
        risk_assessment: RiskAssessment,
        graph_analysis: GraphAnalysisResult,
        policies: Sequence[RetrievedPolicy],
    ) -> PolicyAssessment:
        """Return a structured, evidence-grounded assessment."""


class DemoPolicyAssessmentService:
    """Deterministic local substitute for Azure OpenAI.

    This is explicitly a rules-based demo implementation, not a simulated
    model call.  It is useful for tests, local development, and interviews
    where the complete workflow must run without credentials.
    """

    def assess(
        self,
        transaction: Transaction,
        risk_assessment: RiskAssessment,
        graph_analysis: GraphAnalysisResult,
        policies: Sequence[RetrievedPolicy],
    ) -> PolicyAssessment:
        active_signals = _active_demo_policy_signals(
            transaction,
            risk_assessment,
            graph_analysis,
        )
        policy_references = list(
            dict.fromkeys(
                item.policy_id
                for item in policies
                if DEMO_POLICY_TRIGGER_SIGNALS.get(item.policy_id, frozenset())
                & active_signals
            )
        )
        # Whether evidence was retrieved remains separate from whether a policy
        # applies. A clean payment can have sufficient context without citing an
        # inapplicable AML policy.
        evidence_sufficient = bool(policies)

        if risk_assessment.risk_level is RiskLevel.HIGH:
            recommendation = Recommendation.ESCALATE_TO_HUMAN
            reasons = [
                "Deterministic risk is HIGH; a person must review the evidence before any action."
            ]
        elif not evidence_sufficient:
            recommendation = Recommendation.REQUEST_MORE_INFO
            reasons = ["No relevant policy evidence was retrieved for this assessment."]
        elif graph_analysis.suspicious:
            recommendation = Recommendation.ESCALATE_TO_HUMAN
            reasons = [
                "Explainable graph signals indicate a relationship pattern requiring review."
            ]
        elif risk_assessment.risk_level is RiskLevel.MEDIUM:
            recommendation = Recommendation.REQUEST_MORE_INFO
            reasons = ["The deterministic signals warrant additional contextual information."]
        else:
            recommendation = Recommendation.CLEAR
            reasons = [
                "The supplied deterministic and graph signals do not meet the demo escalation rules."
            ]

        confidence = 0.9 if evidence_sufficient else 0.3
        requires_human_review = (
            risk_assessment.risk_level is RiskLevel.HIGH
            or graph_analysis.suspicious
            or not evidence_sufficient
        )
        missing_information = [] if evidence_sufficient else [
            "Relevant policy evidence",
            "Additional transaction context",
        ]

        return PolicyAssessment(
            recommendation=recommendation,
            reasons=reasons,
            confidence=confidence,
            policy_references=policy_references,
            missing_information=missing_information,
            requires_human_review=requires_human_review,
            evidence_sufficient=evidence_sufficient,
        )


def _active_demo_policy_signals(
    transaction: Transaction,
    risk_assessment: RiskAssessment,
    graph_analysis: GraphAnalysisResult,
) -> set[str]:
    """Derive policy applicability signals from validated workflow inputs."""

    signals = set(graph_analysis.signals)
    if transaction.account_age_days < 30:
        signals.add("YOUNG_ACCOUNT")
    if transaction.new_payee:
        signals.add("NEW_PAYEE")
    if transaction.amount > 5 * transaction.average_transaction_amount:
        signals.add("UNUSUAL_AMOUNT")
    if transaction.transactions_last_24h > 10:
        signals.add("HIGH_VELOCITY")
    if (
        transaction.origin_country.strip().casefold()
        != transaction.destination_country.strip().casefold()
    ):
        signals.add("CROSS_BORDER")
    if risk_assessment.risk_level is RiskLevel.HIGH:
        signals.add("HIGH_RISK")
    if graph_analysis.suspicious and risk_assessment.risk_level is RiskLevel.LOW:
        signals.add("RISK_GRAPH_CONFLICT")
    return signals


class AzureOpenAIPolicyAssessmentService:
    """Azure OpenAI adapter using a strict JSON-schema response contract."""

    SYSTEM_PROMPT = """You support a financial-crime investigator in a demo system.
Use only the supplied policy excerpts for policy claims. If evidence is missing or
insufficient, say so explicitly. Never invent policy identifiers. Never execute or
recommend an autonomous consequential action such as blocking an account or filing
a report. Return only the requested structured JSON assessment. HIGH deterministic
risk, low confidence, insufficient evidence, or conflicting signals must require
human review."""

    def __init__(self, client: Any, deployment: str) -> None:
        if not deployment.strip():
            raise ValueError("An Azure OpenAI chat deployment is required")
        self._client = client
        self._deployment = deployment

    def assess(
        self,
        transaction: Transaction,
        risk_assessment: RiskAssessment,
        graph_analysis: GraphAnalysisResult,
        policies: Sequence[RetrievedPolicy],
    ) -> PolicyAssessment:
        evidence = [item.model_dump(mode="json") for item in policies]
        payload = {
            "transaction": transaction.model_dump(mode="json"),
            "deterministic_risk_assessment": risk_assessment.model_dump(mode="json"),
            "graph_analysis": graph_analysis.model_dump(mode="json"),
            "retrieved_policy_evidence": evidence,
        }

        try:
            completion = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "policy_assessment",
                        "strict": True,
                        "schema": PolicyAssessment.model_json_schema(),
                    },
                },
            )
            content = completion.choices[0].message.content
            if not content:
                raise PolicyAssessmentError("Azure OpenAI returned an empty response")
            assessment = PolicyAssessment.model_validate_json(content)
        except PolicyAssessmentError:
            raise
        except Exception as exc:
            raise PolicyAssessmentError(f"Azure OpenAI assessment failed: {exc}") from exc

        allowed_references = {item.policy_id for item in policies}
        unknown = set(assessment.policy_references) - allowed_references
        if unknown:
            raise PolicyAssessmentError(
                "Azure OpenAI returned policy references that were not retrieved: "
                + ", ".join(sorted(unknown))
            )
        if not policies and assessment.evidence_sufficient:
            raise PolicyAssessmentError(
                "Azure OpenAI claimed sufficient evidence when no evidence was supplied"
            )
        return assessment

