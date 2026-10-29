"""Boundary tests for deterministic scoring and its domain contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models import (
    GraphAnalysisResult,
    InvestigationRecommendation,
    InvestigationResponse,
    InvestigationStatus,
    PersistenceResult,
    PolicyAssessment,
    Recommendation,
    RetrievedPolicy,
    RiskAssessment,
    RiskLevel,
    Transaction,
)
from app.services.risk_scorer import assess_transaction, score_transaction


def transaction_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "transaction_id": "txn-001",
        "amount": 100.0,
        "account_age_days": 365,
        "new_payee": False,
        "average_transaction_amount": 100.0,
        "transactions_last_24h": 1,
        "origin_country": "GB",
        "destination_country": "GB",
    }
    data.update(overrides)
    return data


def make_transaction(**overrides: object) -> Transaction:
    return Transaction(**transaction_data(**overrides))


def test_score_transaction_preserves_legacy_shape() -> None:
    result = score_transaction(make_transaction())

    assert result == {"score": 0, "reasons": []}
    assert set(result) == {"score", "reasons"}


@pytest.mark.parametrize(
    ("overrides", "expected_score", "expected_reason"),
    [
        (
            {"account_age_days": 29},
            20,
            "Account is less than 30 days old",
        ),
        (
            {"new_payee": True},
            15,
            "Transaction is to a new payee",
        ),
        (
            {"amount": 500.01},
            30,
            "Transaction amount is more than 5 times the average",
        ),
        (
            {"transactions_last_24h": 11},
            20,
            "More than 10 transactions in the last 24 hours",
        ),
    ],
)
def test_each_rule_fires_independently(
    overrides: dict[str, object],
    expected_score: int,
    expected_reason: str,
) -> None:
    assert score_transaction(make_transaction(**overrides)) == {
        "score": expected_score,
        "reasons": [expected_reason],
    }


def test_rule_boundaries_are_strict_and_unchanged() -> None:
    transaction = make_transaction(
        account_age_days=30,
        amount=500.0,
        average_transaction_amount=100.0,
        transactions_last_24h=10,
    )

    assert score_transaction(transaction) == {"score": 0, "reasons": []}


def test_all_rules_accumulate_in_stable_explanation_order() -> None:
    transaction = make_transaction(
        account_age_days=0,
        new_payee=True,
        amount=501.0,
        transactions_last_24h=11,
    )

    assert score_transaction(transaction) == {
        "score": 85,
        "reasons": [
            "Account is less than 30 days old",
            "Transaction is to a new payee",
            "Transaction amount is more than 5 times the average",
            "More than 10 transactions in the last 24 hours",
        ],
    }


@pytest.mark.parametrize(
    ("overrides", "expected_score", "expected_level", "investigate"),
    [
        ({}, 0, RiskLevel.LOW, False),
        ({"amount": 500.01}, 30, RiskLevel.MEDIUM, False),
        (
            {
                "account_age_days": 29,
                "new_payee": True,
                "amount": 500.01,
            },
            65,
            RiskLevel.HIGH,
            True,
        ),
    ],
)
def test_assess_transaction_adapts_legacy_result(
    overrides: dict[str, object],
    expected_score: int,
    expected_level: RiskLevel,
    investigate: bool,
) -> None:
    transaction = make_transaction(**overrides)
    legacy = score_transaction(transaction)

    assessment = assess_transaction(transaction)

    assert assessment.risk_score == expected_score
    assert assessment.risk_level is expected_level
    assert assessment.risk_reasons == legacy["reasons"]
    assert assessment.investigate is investigate


def test_country_fields_do_not_change_the_legacy_score() -> None:
    domestic = make_transaction(origin_country="GB", destination_country="GB")
    cross_border = make_transaction(
        origin_country="GB",
        destination_country="Synthetic Jurisdiction",
    )

    assert score_transaction(domestic) == score_transaction(cross_border)


def test_transaction_preserves_original_payload_and_optional_ids() -> None:
    original = make_transaction()
    enriched = make_transaction(
        transaction_id="  txn-002  ",
        origin_country="  GB ",
        origin_account_id="  acct-origin  ",
        destination_account_id="acct-destination",
    )

    assert original.origin_account_id is None
    assert original.destination_account_id is None
    assert enriched.transaction_id == "txn-002"
    assert enriched.origin_country == "GB"
    assert enriched.origin_account_id == "acct-origin"
    assert enriched.destination_account_id == "acct-destination"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("amount", 0.0),
        ("amount", -0.01),
        ("amount", float("inf")),
        ("amount", float("nan")),
        ("amount", "100.0"),
        ("average_transaction_amount", 0.0),
        ("average_transaction_amount", float("inf")),
        ("account_age_days", -1),
        ("account_age_days", True),
        ("transactions_last_24h", -1),
        ("transactions_last_24h", 1.5),
        ("new_payee", 1),
        ("new_payee", "false"),
        ("transaction_id", "   "),
        ("origin_country", ""),
        ("destination_country", "   "),
        ("origin_account_id", "  "),
    ],
)
def test_transaction_rejects_invalid_or_coerced_values(
    field: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValidationError):
        Transaction(**transaction_data(**{field: invalid_value}))


def test_transaction_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Transaction(**transaction_data(undeclared_signal=True))


@pytest.mark.parametrize(
    ("score", "level", "investigate"),
    [
        (0, RiskLevel.LOW, False),
        (29, RiskLevel.LOW, False),
        (30, RiskLevel.MEDIUM, False),
        (59, RiskLevel.MEDIUM, False),
        (60, RiskLevel.HIGH, True),
        (100, RiskLevel.HIGH, True),
    ],
)
def test_risk_assessment_enforces_legacy_level_boundaries(
    score: int,
    level: RiskLevel,
    investigate: bool,
) -> None:
    assessment = RiskAssessment(
        risk_score=score,
        risk_level=level.value,
        risk_reasons=[],
        investigate=investigate,
    )

    assert assessment.risk_level is level
    assert assessment.investigate is investigate


@pytest.mark.parametrize(
    "payload",
    [
        {
            "risk_score": -1,
            "risk_level": "LOW",
            "risk_reasons": [],
            "investigate": False,
        },
        {
            "risk_score": 101,
            "risk_level": "HIGH",
            "risk_reasons": [],
            "investigate": True,
        },
        {
            "risk_score": 60,
            "risk_level": "MEDIUM",
            "risk_reasons": [],
            "investigate": True,
        },
        {
            "risk_score": 60,
            "risk_level": "HIGH",
            "risk_reasons": [],
            "investigate": False,
        },
    ],
)
def test_risk_assessment_rejects_inconsistent_payloads(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        RiskAssessment(**payload)


def test_graph_analysis_model_validates_counts_and_ratio() -> None:
    result = GraphAnalysisResult(
        origin_account_id="acct-1",
        destination_account_id="acct-2",
        counterparty_count=3,
        origin_degree=4,
        shared_counterparty_count=1,
        repeated_transaction_count=2,
        concentration_ratio=0.75,
        signals=["Repeated transfers"],
        suspicious=True,
    )

    assert result.concentration_ratio == 0.75
    assert result.suspicious is True

    for invalid_ratio in (-0.01, 1.01, float("inf"), float("nan"), "0.5"):
        with pytest.raises(ValidationError):
            GraphAnalysisResult(
                counterparty_count=0,
                origin_degree=0,
                shared_counterparty_count=0,
                repeated_transaction_count=0,
                concentration_ratio=invalid_ratio,
                signals=[],
                suspicious=False,
            )

    with pytest.raises(ValidationError):
        GraphAnalysisResult(
            counterparty_count=-1,
            origin_degree=0,
            shared_counterparty_count=0,
            repeated_transaction_count=0,
            concentration_ratio=0.0,
            signals=[],
            suspicious=False,
        )


@pytest.mark.parametrize("relevance_score", [0.0, 1.0])
def test_retrieved_policy_accepts_similarity_boundaries(
    relevance_score: float,
) -> None:
    policy = RetrievedPolicy(
        policy_id="policy-1",
        chunk_id="policy-1-chunk-1",
        text="Synthetic policy evidence.",
        relevance_score=relevance_score,
    )

    assert policy.relevance_score == relevance_score


@pytest.mark.parametrize("relevance_score", [-0.01, 1.01, float("inf"), "0.5"])
def test_retrieved_policy_rejects_invalid_similarity(
    relevance_score: object,
) -> None:
    with pytest.raises(ValidationError):
        RetrievedPolicy(
            policy_id="policy-1",
            chunk_id="chunk-1",
            text="Synthetic policy evidence.",
            relevance_score=relevance_score,
        )


def make_policy_assessment() -> PolicyAssessment:
    return PolicyAssessment(
        recommendation="CLEAR",
        reasons=["Signals are below the deterministic escalation threshold."],
        confidence=0.9,
        policy_references=["policy-1"],
        policies_considered=[],
        missing_information=[],
        requires_human_review=False,
        evidence_sufficient=True,
    )


def test_policy_assessment_and_final_recommendation_use_enums() -> None:
    policy_assessment = make_policy_assessment()
    final = InvestigationRecommendation(
        recommendation="ESCALATE_TO_HUMAN",
        reasons=["A human must review the high-risk transaction."],
        confidence=0.9,
        policy_references=["policy-1"],
        missing_information=[],
        requires_human_review=True,
    )

    assert policy_assessment.recommendation is Recommendation.CLEAR
    assert final.recommendation is Recommendation.ESCALATE_TO_HUMAN


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("recommendation", "APPROVE_PAYMENT"),
        ("reasons", []),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("confidence", "0.9"),
        ("requires_human_review", "false"),
        ("evidence_sufficient", 1),
    ],
)
def test_policy_assessment_rejects_invalid_output(
    field: str,
    invalid_value: object,
) -> None:
    data = make_policy_assessment().model_dump()
    data[field] = invalid_value

    with pytest.raises(ValidationError):
        PolicyAssessment(**data)


def test_persistence_result_is_explicit_and_strict() -> None:
    failed = PersistenceResult(
        succeeded=False,
        backend="postgresql",
        error="Database unavailable",
    )

    assert failed.succeeded is False
    assert failed.error == "Database unavailable"

    with pytest.raises(ValidationError):
        PersistenceResult(succeeded="false", backend="postgresql")
    with pytest.raises(ValidationError):
        PersistenceResult(succeeded=False, backend="   ")


def test_investigation_response_serializes_as_an_api_payload() -> None:
    risk = RiskAssessment(
        risk_score=0,
        risk_level="LOW",
        risk_reasons=[],
        investigate=False,
    )
    graph = GraphAnalysisResult(
        counterparty_count=1,
        origin_degree=1,
        shared_counterparty_count=0,
        repeated_transaction_count=0,
        concentration_ratio=1.0,
        signals=[],
        suspicious=False,
    )
    policy = RetrievedPolicy(
        policy_id="policy-1",
        chunk_id="chunk-1",
        text="Synthetic low-risk handling policy.",
        relevance_score=0.8,
    )
    response = InvestigationResponse(
        investigation_id="inv-001",
        transaction_id="txn-001",
        risk_assessment=risk,
        graph_analysis=graph,
        retrieved_policies=[policy],
        policy_assessment=make_policy_assessment(),
        recommendation="CLEAR",
        reasons=["No escalation condition was met."],
        requires_human_review=False,
        status="COMPLETED",
        persistence=PersistenceResult(succeeded=True, backend="memory"),
        created_at=datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    payload = response.model_dump(mode="json")
    assert payload["recommendation"] == Recommendation.CLEAR.value
    assert payload["status"] == InvestigationStatus.COMPLETED.value
    assert payload["risk_assessment"]["risk_level"] == RiskLevel.LOW.value
    assert payload["created_at"] == "2025-01-02T03:04:05Z"


def test_investigation_response_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        InvestigationResponse(
            investigation_id="inv-001",
            transaction_id="txn-001",
            risk_assessment=RiskAssessment(
                risk_score=0,
                risk_level="LOW",
                risk_reasons=[],
                investigate=False,
            ),
            graph_analysis=GraphAnalysisResult(
                counterparty_count=0,
                origin_degree=0,
                shared_counterparty_count=0,
                repeated_transaction_count=0,
                concentration_ratio=0.0,
                signals=[],
                suspicious=False,
            ),
            retrieved_policies=[],
            policy_assessment=make_policy_assessment(),
            recommendation="CLEAR",
            reasons=["No escalation condition was met."],
            requires_human_review=False,
            status="COMPLETED",
            persistence=PersistenceResult(succeeded=True, backend="memory"),
            created_at=datetime(2025, 1, 2, 3, 4, 5),
        )
