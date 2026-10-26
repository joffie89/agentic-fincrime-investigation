"""Persistence contracts for investigation audit records."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.models.domain import PersistenceResult


@runtime_checkable
class InvestigationRepository(Protocol):
    """Store and retrieve investigation audit records."""

    def save(self, record: dict[str, Any]) -> PersistenceResult:
        """Persist ``record`` and truthfully report whether it succeeded."""

        ...

    def get(self, investigation_id: str) -> dict[str, Any] | None:
        """Return the stored audit record, or ``None`` when it does not exist."""

        ...
