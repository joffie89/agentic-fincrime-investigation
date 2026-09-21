"""Process-local investigation repository for local/demo mode and tests."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any

from app.models.domain import PersistenceResult


class InMemoryInvestigationRepository:
    """Retain defensive copies of audit records for this process only."""

    backend = "memory"

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._lock = RLock()

    def save(self, record: dict[str, Any]) -> PersistenceResult:
        investigation_id = record.get("investigation_id")
        if investigation_id is None or not str(investigation_id).strip():
            return PersistenceResult(
                succeeded=False,
                backend=self.backend,
                error="investigation_id is required for persistence",
            )

        try:
            stored_record = deepcopy(record)
            with self._lock:
                self._records.append(stored_record)
        except Exception as exc:  # pragma: no cover - defensive for exotic values
            error = f"In-memory persistence failed: {exc}"[:256]
            return PersistenceResult(
                succeeded=False,
                backend=self.backend,
                error=error,
            )

        return PersistenceResult(succeeded=True, backend=self.backend)

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return a snapshot so callers cannot mutate the repository indirectly."""

        with self._lock:
            return deepcopy(self._records)

    def get(self, investigation_id: str) -> dict[str, Any] | None:
        """Return the newest matching record, if one exists."""

        with self._lock:
            for record in reversed(self._records):
                if str(record.get("investigation_id")) == investigation_id:
                    return deepcopy(record)
        return None
