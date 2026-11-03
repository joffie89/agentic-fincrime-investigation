"""Generate the deterministic 100-case second held-out decision set."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

SEED = 20261103
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "decision_cases_holdout_v2.json"
COMPARISON_PATHS = (
    PROJECT_ROOT / "evaluation" / "decision_cases.json",
    PROJECT_ROOT / "evaluation" / "decision_cases_holdout.json",
    PROJECT_ROOT / "evaluation" / "decision_cases_adversarial.json",
)

ACCOUNT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
COUNTRY_PAIRS = (
    ("GB", "GB"),
    ("US", "US"),
    ("CA", "CA"),
    ("AU", "AU"),
    ("DE", "DE"),
    ("FR", "FR"),
    ("GB", "IE"),
    ("US", "CA"),
    ("DE", "NL"),
    ("AU", "NZ"),
    ("FR", "BE"),
    ("SG", "JP"),
)
AVERAGE_AMOUNTS = (35.0, 60.0, 95.0, 140.0, 225.0, 375.0, 650.0, 900.0, 1500.0)
SIGNAL_POINTS = {
    "young": 20,
    "new_payee": 15,
    "unusual_amount": 30,
    "velocity": 20,
}
IDENTIFIER_FIELDS = {
    "transaction_id",
    "origin_account_id",
    "destination_account_id",
}


def _substantive_fingerprint(transaction: dict[str, Any]) -> str:
    """Identify duplicate transaction facts independently of synthetic IDs."""

    substantive = {key: value for key, value in transaction.items() if key not in IDENTIFIER_FIELDS}
    return json.dumps(substantive, sort_keys=True, separators=(",", ":"))


def _risk_signals(transaction: dict[str, Any]) -> set[str]:
    signals: set[str] = set()
    if transaction["account_age_days"] < 30:
        signals.add("young")
    if transaction["new_payee"]:
        signals.add("new_payee")
    if transaction["amount"] > 5 * transaction["average_transaction_amount"]:
        signals.add("unusual_amount")
    if transaction["transactions_last_24h"] > 10:
        signals.add("velocity")
    return signals


def _label_case(name: str, transaction: dict[str, Any]) -> dict[str, Any]:
    """Apply the README rubric without consulting a model."""

    signals = _risk_signals(transaction)
    score = sum(SIGNAL_POINTS[signal] for signal in signals)
    if not signals:
        recommendation = "CLEAR"
        category = "clean"
    elif len(signals) == 1:
        recommendation = "REQUEST_MORE_INFO"
        category = "medium"
    else:
        recommendation = "ESCALATE_TO_HUMAN"
        category = "high" if score >= 60 else "medium"

    policy_ids: list[str] = []
    if signals & {"young", "new_payee", "unusual_amount"}:
        policy_ids.append("DEMO-AML-001")
    if "velocity" in signals:
        policy_ids.append("DEMO-AML-002")
    if transaction["origin_country"] != transaction["destination_country"]:
        policy_ids.append("DEMO-AML-004")
    if score >= 60:
        policy_ids.append("DEMO-AML-005")
    if "new_payee" in signals:
        policy_ids.append("DEMO-AML-006")

    return {
        "name": name,
        "category": category,
        "transaction": transaction,
        "expected_recommendation": recommendation,
        "expected_requires_human_review": recommendation == "ESCALATE_TO_HUMAN",
        "expected_policy_ids": policy_ids,
    }


def _base_transaction(
    rng: random.Random,
    group: str,
    index: int,
) -> dict[str, Any]:
    average = rng.choice(AVERAGE_AMOUNTS)
    origin_country, destination_country = rng.choice(COUNTRY_PAIRS)
    return {
        "transaction_id": f"holdout-v2-{group}-{index:03d}",
        "amount": round(average * rng.uniform(0.35, 2.4), 2),
        "account_age_days": rng.randint(90, 3650),
        "new_payee": False,
        "average_transaction_amount": average,
        "transactions_last_24h": rng.randint(0, 10),
        "origin_country": origin_country,
        "destination_country": destination_country,
        "origin_account_id": f"hv2_{group}_origin_{index:03d}",
        "destination_account_id": f"hv2_{group}_destination_{index:03d}",
    }


def _clean_cases(rng: random.Random) -> list[dict[str, Any]]:
    return [
        _label_case(
            f"holdout v2 clean routine varied payment {index:03d}",
            _base_transaction(rng, "clean", index),
        )
        for index in range(1, 26)
    ]


def _single_signal_cases(rng: random.Random) -> list[dict[str, Any]]:
    signal_plan = ["velocity"] * 10 + ["young"] * 5 + ["new_payee"] * 5 + ["unusual_amount"] * 5
    cases: list[dict[str, Any]] = []
    for index, signal in enumerate(signal_plan, start=1):
        transaction = _base_transaction(rng, "single", index)
        if signal == "velocity":
            transaction["transactions_last_24h"] = rng.randint(11, 45)
        elif signal == "young":
            transaction["account_age_days"] = rng.randint(1, 29)
        elif signal == "new_payee":
            transaction["new_payee"] = True
        elif signal == "unusual_amount":
            transaction["amount"] = round(
                transaction["average_transaction_amount"] * rng.uniform(5.25, 9.0),
                2,
            )
        cases.append(
            _label_case(
                f"holdout v2 single {signal.replace('_', ' ')} {index:03d}",
                transaction,
            )
        )
    return cases


def _multi_signal_cases(rng: random.Random) -> list[dict[str, Any]]:
    patterns = (
        ("young", "new_payee"),
        ("young", "unusual_amount"),
        ("young", "velocity"),
        ("new_payee", "unusual_amount"),
        ("new_payee", "velocity"),
        ("unusual_amount", "velocity"),
        ("young", "new_payee", "unusual_amount"),
        ("young", "new_payee", "velocity"),
        ("young", "unusual_amount", "velocity"),
        ("new_payee", "unusual_amount", "velocity"),
        ("young", "new_payee", "unusual_amount", "velocity"),
    )
    cases: list[dict[str, Any]] = []
    for index in range(1, 41):
        pattern = patterns[(index - 1) % len(patterns)]
        transaction = _base_transaction(rng, "multi", index)
        if "young" in pattern:
            transaction["account_age_days"] = rng.randint(1, 29)
        if "new_payee" in pattern:
            transaction["new_payee"] = True
        if "unusual_amount" in pattern:
            transaction["amount"] = round(
                transaction["average_transaction_amount"] * rng.uniform(5.25, 10.0),
                2,
            )
        if "velocity" in pattern:
            transaction["transactions_last_24h"] = rng.randint(11, 50)
        pattern_name = " and ".join(signal.replace("_", " ") for signal in pattern)
        cases.append(
            _label_case(
                f"holdout v2 multi {pattern_name} {index:03d}",
                transaction,
            )
        )
    return cases


def _boundary_cases() -> list[dict[str, Any]]:
    specifications = (
        ("account age 29", 29, 3, 90.0, 100.0, "GB", "GB"),
        ("account age 30", 30, 3, 125.0, 100.0, "US", "US"),
        ("transactions 10", 640, 10, 180.0, 150.0, "CA", "CA"),
        ("transactions 11", 730, 11, 210.0, 175.0, "AU", "AU"),
        ("amount exactly 5x one", 900, 4, 500.0, 100.0, "DE", "DE"),
        ("amount exactly 5x two", 1200, 5, 1250.0, 250.0, "FR", "BE"),
        ("age 29 and transactions 10", 29, 10, 75.0, 125.0, "GB", "IE"),
        ("age 30 and transactions 11", 30, 11, 320.0, 200.0, "US", "CA"),
        ("age 29 and amount exactly 5x", 29, 2, 750.0, 150.0, "SG", "JP"),
        ("age 30 transactions 10 amount exactly 5x", 30, 10, 2000.0, 400.0, "NZ", "NZ"),
    )
    cases: list[dict[str, Any]] = []
    for index, specification in enumerate(specifications, start=1):
        label, age, velocity, amount, average, origin, destination = specification
        transaction = {
            "transaction_id": f"holdout-v2-boundary-{index:03d}",
            "amount": amount,
            "account_age_days": age,
            "new_payee": False,
            "average_transaction_amount": average,
            "transactions_last_24h": velocity,
            "origin_country": origin,
            "destination_country": destination,
            "origin_account_id": f"hv2_boundary_origin_{index:03d}",
            "destination_account_id": f"hv2_boundary_destination_{index:03d}",
        }
        cases.append(_label_case(f"holdout v2 boundary {label} {index:03d}", transaction))
    return cases


def _all_comparison_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in COMPARISON_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.extend(payload.get("cases", []))
        cases.extend(payload.get("validation_rejection_cases", []))
    return cases


def _validate(cases: list[dict[str, Any]]) -> None:
    group_counts = Counter(case["transaction"]["transaction_id"].split("-")[2] for case in cases)
    assert group_counts == {"clean": 25, "single": 25, "multi": 40, "boundary": 10}
    assert len(cases) == 100

    names = [case["name"] for case in cases]
    transaction_ids = [case["transaction"]["transaction_id"] for case in cases]
    transactions = [
        json.dumps(case["transaction"], sort_keys=True, separators=(",", ":")) for case in cases
    ]
    substantive_transactions = [_substantive_fingerprint(case["transaction"]) for case in cases]
    assert len(names) == len(set(names))
    assert len(transaction_ids) == len(set(transaction_ids))
    assert len(transactions) == len(set(transactions))
    assert len(substantive_transactions) == len(set(substantive_transactions))

    comparison_cases = _all_comparison_cases()
    existing_names = {case["name"] for case in comparison_cases}
    existing_ids = {case["transaction"]["transaction_id"] for case in comparison_cases}
    existing_transactions = {
        json.dumps(case["transaction"], sort_keys=True, separators=(",", ":"))
        for case in comparison_cases
    }
    existing_substantive_transactions = {
        _substantive_fingerprint(case["transaction"]) for case in comparison_cases
    }
    assert not set(names) & existing_names
    assert not set(transaction_ids) & existing_ids
    assert not set(transactions) & existing_transactions
    assert not set(substantive_transactions) & existing_substantive_transactions

    for case in cases:
        transaction = case["transaction"]
        assert ACCOUNT_ID_PATTERN.fullmatch(transaction["origin_account_id"])
        assert ACCOUNT_ID_PATTERN.fullmatch(transaction["destination_account_id"])
        assert case == _label_case(case["name"], transaction)

    single_cases = [
        case for case in cases if "holdout-v2-single" in case["transaction"]["transaction_id"]
    ]
    assert all(len(_risk_signals(case["transaction"])) == 1 for case in single_cases)
    assert sum(_risk_signals(case["transaction"]) == {"velocity"} for case in single_cases) >= 8
    multi_cases = [
        case for case in cases if "holdout-v2-multi" in case["transaction"]["transaction_id"]
    ]
    assert all(len(_risk_signals(case["transaction"])) >= 2 for case in multi_cases)


def generate_dataset(seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    """Return the complete dataset without writing to disk."""

    rng = random.Random(seed)
    cases = [
        *_clean_cases(rng),
        *_single_signal_cases(rng),
        *_multi_signal_cases(rng),
        *_boundary_cases(),
    ]
    _validate(cases)
    return {"cases": cases}


def main() -> None:
    payload = generate_dataset()
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payload['cases'])} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
