"""Persistence contracts for investigation audit records."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.models.domain import PersistenceResult


@runtime_checkable
class InvestigationRepository(Protocol):
    """Store a completed or escalated investigation record."""

    def save(self, record: dict[str, Any]) -> PersistenceResult:
        """Persist ``record`` and truthfully report whether it succeeded."""

        ...
