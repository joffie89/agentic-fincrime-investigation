from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.repositories.memory import InMemoryInvestigationRepository
from app.repositories.postgres import PostgresInvestigationRepository
import app.repositories.postgres as postgres_module


def _record() -> dict:
    return {
        "investigation_id": "inv-001",
        "transaction": {"transaction_id": "txn-001", "amount": 8_000},
        "risk_assessment": {
            "risk_score": 85,
            "risk_level": "HIGH",
            "risk_reasons": ["young_account", "unusual_amount"],
        },
        "graph_analysis": {"counterparty_count": 4, "signals": ["shared_payee"]},
        "retrieved_policies": [
            {"policy_id": "POL-HIGH-RISK", "text": "Escalate high-risk cases."}
        ],
        "recommendation": {
            "recommendation": "ESCALATE_TO_HUMAN",
            "requires_human_review": True,
        },
        "requires_human_review": True,
        "status": "PENDING_HUMAN_REVIEW",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }


def test_in_memory_repository_stores_defensive_copy() -> None:
    repository = InMemoryInvestigationRepository()
    record = _record()

    result = repository.save(record)
    record["status"] = "MUTATED"
    returned = repository.get("inv-001")

    assert result.succeeded is True
    assert result.backend == "memory"
    assert result.error is None
    assert returned is not None
    assert returned["status"] == "PENDING_HUMAN_REVIEW"

    returned["status"] = "ALSO_MUTATED"
    assert repository.records[0]["status"] == "PENDING_HUMAN_REVIEW"


def test_in_memory_repository_rejects_missing_investigation_id() -> None:
    repository = InMemoryInvestigationRepository()

    result = repository.save({"transaction_id": "txn-001"})

    assert result.succeeded is False
    assert result.backend == "memory"
    assert "investigation_id" in (result.error or "")
    assert repository.records == []


def test_postgres_repository_commits_successful_write(monkeypatch) -> None:
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    connection = MagicMock()
    connection.cursor.return_value = cursor
    fake_psycopg = SimpleNamespace(connect=MagicMock(return_value=connection))
    monkeypatch.setattr(postgres_module, "psycopg", fake_psycopg)
    repository = PostgresInvestigationRepository(
        "postgresql://demo:demo@postgres/demo"
    )

    result = repository.save(_record())

    assert result.succeeded is True
    assert result.backend == "postgres"
    assert result.error is None
    fake_psycopg.connect.assert_called_once_with(
        "postgresql://demo:demo@postgres/demo", connect_timeout=5
    )
    assert cursor.execute.call_count == 2
    create_call, insert_call = cursor.execute.call_args_list
    assert "CREATE TABLE IF NOT EXISTS investigations" in create_call.args[0]
    assert "INSERT INTO investigations" in insert_call.args[0]
    assert insert_call.args[1][0] == "inv-001"
    assert insert_call.args[1][1] == "txn-001"
    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()
    connection.close.assert_called_once_with()


def test_postgres_repository_reports_connection_failure(monkeypatch) -> None:
    fake_psycopg = SimpleNamespace(
        connect=MagicMock(side_effect=RuntimeError("database unavailable"))
    )
    monkeypatch.setattr(postgres_module, "psycopg", fake_psycopg)
    repository = PostgresInvestigationRepository(
        "postgresql://demo:demo@postgres/demo"
    )

    result = repository.save(_record())

    assert result.succeeded is False
    assert result.backend == "postgres"
    assert "database unavailable" in (result.error or "")


def test_postgres_repository_rolls_back_failed_write(monkeypatch) -> None:
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    cursor.execute.side_effect = [None, RuntimeError("write rejected")]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    fake_psycopg = SimpleNamespace(connect=MagicMock(return_value=connection))
    monkeypatch.setattr(postgres_module, "psycopg", fake_psycopg)
    repository = PostgresInvestigationRepository(
        "postgresql://demo:demo@postgres/demo"
    )

    result = repository.save(_record())

    assert result.succeeded is False
    assert "write rejected" in (result.error or "")
    connection.commit.assert_not_called()
    connection.rollback.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_postgres_repository_does_not_connect_for_invalid_record(monkeypatch) -> None:
    fake_psycopg = SimpleNamespace(connect=MagicMock())
    monkeypatch.setattr(postgres_module, "psycopg", fake_psycopg)
    repository = PostgresInvestigationRepository(
        "postgresql://demo:demo@postgres/demo"
    )

    result = repository.save({"transaction_id": "txn-001"})

    assert result.succeeded is False
    assert "investigation_id" in (result.error or "")
    fake_psycopg.connect.assert_not_called()
