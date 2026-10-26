"""PostgreSQL-backed investigation audit repository."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from app.models.domain import PersistenceResult

try:  # Keep local/demo imports usable before optional infrastructure is installed.
    import psycopg
except ImportError:  # pragma: no cover - exercised only without installed extras
    psycopg = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    transaction_id TEXT,
    risk_score DOUBLE PRECISION,
    risk_level TEXT,
    risk_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    graph_signals JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieved_policy_references JSONB NOT NULL DEFAULT '[]'::jsonb,
    recommendation TEXT,
    human_review_required BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    full_record JSONB NOT NULL
)
"""


_UPSERT_SQL = """
INSERT INTO investigations (
    investigation_id,
    transaction_id,
    risk_score,
    risk_level,
    risk_reasons,
    graph_signals,
    retrieved_policy_references,
    recommendation,
    human_review_required,
    status,
    created_at,
    updated_at,
    full_record
) VALUES (
    %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s,
    %s::jsonb
)
ON CONFLICT (investigation_id) DO UPDATE SET
    transaction_id = EXCLUDED.transaction_id,
    risk_score = EXCLUDED.risk_score,
    risk_level = EXCLUDED.risk_level,
    risk_reasons = EXCLUDED.risk_reasons,
    graph_signals = EXCLUDED.graph_signals,
    retrieved_policy_references = EXCLUDED.retrieved_policy_references,
    recommendation = EXCLUDED.recommendation,
    human_review_required = EXCLUDED.human_review_required,
    status = EXCLUDED.status,
    updated_at = EXCLUDED.updated_at,
    full_record = EXCLUDED.full_record
"""


_SELECT_SQL = """
SELECT full_record
FROM investigations
WHERE investigation_id = %s
"""


def _plain_value(value: Any) -> Any:
    """Convert common domain values to JSON-compatible Python structures."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, UUID)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(_plain_value(value), default=str, ensure_ascii=False)


def _as_dict(value: Any) -> dict[str, Any]:
    plain = _plain_value(value)
    return plain if isinstance(plain, dict) else {}


def _policy_references(record: dict[str, Any]) -> list[str]:
    explicit = record.get("retrieved_policy_references") or record.get(
        "policy_references"
    )
    if explicit:
        normalized = _plain_value(explicit)
        if isinstance(normalized, (list, tuple, set)):
            return [str(item) for item in normalized]
        return [str(normalized)]

    references: list[str] = []
    for policy in record.get("retrieved_policies", []) or []:
        policy_data = _as_dict(policy)
        reference = (
            policy_data.get("policy_id")
            or policy_data.get("source")
            or policy_data.get("source_id")
        )
        if reference and str(reference) not in references:
            references.append(str(reference))
    return references


class PostgresInvestigationRepository:
    """Persist investigation summaries and their complete JSON audit record."""

    backend = "postgres"

    def __init__(self, database_url: str) -> None:
        if not database_url or not database_url.strip():
            raise ValueError("database_url must not be empty")
        self._database_url = database_url.strip()

    def save(self, record: dict[str, Any]) -> PersistenceResult:
        normalized = _as_dict(record)
        investigation_id = normalized.get("investigation_id")
        if investigation_id is None or not str(investigation_id).strip():
            return PersistenceResult(
                succeeded=False,
                backend=self.backend,
                error="investigation_id is required for persistence",
            )

        if psycopg is None:
            error = "PostgreSQL persistence unavailable: psycopg is not installed"
            logger.error(
                error,
                extra={"event": "persistence_failed", "backend": self.backend},
            )
            return PersistenceResult(
                succeeded=False,
                backend=self.backend,
                error=error,
            )

        connection: Any | None = None
        try:
            connection = psycopg.connect(self._database_url, connect_timeout=5)
            with connection.cursor() as cursor:
                cursor.execute(_CREATE_TABLE_SQL)
                cursor.execute(_UPSERT_SQL, self._parameters(normalized))
            connection.commit()
        except Exception:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    logger.exception(
                        "PostgreSQL rollback failed",
                        extra={
                            "event": "persistence_rollback_failed",
                            "backend": self.backend,
                            "investigation_id": str(investigation_id),
                        },
                    )
            logger.exception(
                "PostgreSQL investigation persistence failed",
                extra={
                    "event": "persistence_failed",
                    "backend": self.backend,
                    "investigation_id": str(investigation_id),
                },
            )
            return PersistenceResult(
                succeeded=False,
                backend=self.backend,
                error="Investigation could not be saved",
            )
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.exception(
                        "PostgreSQL connection close failed",
                        extra={
                            "event": "persistence_close_failed",
                            "backend": self.backend,
                            "investigation_id": str(investigation_id),
                        },
                    )

        logger.info(
            "Investigation persisted",
            extra={
                "event": "investigation_persisted",
                "backend": self.backend,
                "investigation_id": str(investigation_id),
            },
        )
        return PersistenceResult(succeeded=True, backend=self.backend)

    def get(self, investigation_id: str) -> dict[str, Any] | None:
        """Retrieve the complete JSON audit record by investigation ID."""

        if not investigation_id or not investigation_id.strip():
            return None

        if psycopg is None:
            logger.error(
                "PostgreSQL investigation retrieval unavailable",
                extra={
                    "event": "persistence_retrieval_failed",
                    "backend": self.backend,
                    "investigation_id": investigation_id,
                },
            )
            raise RuntimeError("Investigation could not be retrieved")

        connection: Any | None = None
        try:
            connection = psycopg.connect(self._database_url, connect_timeout=5)
            with connection.cursor() as cursor:
                # A fresh database should return a normal not-found result rather
                # than exposing a missing-table error to the API caller.
                cursor.execute(_CREATE_TABLE_SQL)
                cursor.execute(_SELECT_SQL, (investigation_id,))
                row = cursor.fetchone()
            connection.commit()

            if row is None:
                return None

            payload = row[0]
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise ValueError("Stored investigation record is not a JSON object")
            return deepcopy(_as_dict(payload))
        except Exception as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    logger.exception(
                        "PostgreSQL rollback failed after retrieval",
                        extra={
                            "event": "persistence_retrieval_rollback_failed",
                            "backend": self.backend,
                            "investigation_id": investigation_id,
                        },
                    )
            logger.exception(
                "PostgreSQL investigation retrieval failed",
                extra={
                    "event": "persistence_retrieval_failed",
                    "backend": self.backend,
                    "investigation_id": investigation_id,
                },
            )
            raise RuntimeError("Investigation could not be retrieved") from exc
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.exception(
                        "PostgreSQL connection close failed after retrieval",
                        extra={
                            "event": "persistence_retrieval_close_failed",
                            "backend": self.backend,
                            "investigation_id": investigation_id,
                        },
                    )

    @staticmethod
    def _parameters(record: dict[str, Any]) -> tuple[Any, ...]:
        transaction = _as_dict(record.get("transaction"))
        risk = _as_dict(record.get("risk_assessment") or record.get("risk"))
        graph = _as_dict(record.get("graph_analysis") or record.get("graph_signals"))
        recommendation_data = _as_dict(record.get("recommendation"))

        recommendation = record.get("recommendation")
        if recommendation_data:
            recommendation = recommendation_data.get("recommendation")
        if isinstance(recommendation, Enum):
            recommendation = recommendation.value

        human_review = bool(
            record.get(
                "requires_human_review",
                record.get(
                    "human_review_required",
                    recommendation_data.get("requires_human_review", False),
                ),
            )
        )
        now = datetime.now(timezone.utc)
        created_at = record.get("created_at") or record.get("timestamp") or now
        updated_at = record.get("updated_at") or now
        status = record.get("status") or (
            "PENDING_HUMAN_REVIEW" if human_review else "COMPLETED"
        )

        return (
            str(record["investigation_id"]),
            str(record.get("transaction_id") or transaction.get("transaction_id") or "")
            or None,
            risk.get("risk_score", record.get("risk_score")),
            risk.get("risk_level", record.get("risk_level")),
            _json(risk.get("risk_reasons", record.get("risk_reasons", []))),
            _json(graph),
            _json(_policy_references(record)),
            str(recommendation) if recommendation is not None else None,
            human_review,
            str(status),
            created_at,
            updated_at,
            _json(record),
        )
