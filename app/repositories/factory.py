"""Repository construction from validated application settings."""

from __future__ import annotations

from app.config import PersistenceBackend, Settings
from app.repositories.base import InvestigationRepository
from app.repositories.memory import InMemoryInvestigationRepository
from app.repositories.postgres import PostgresInvestigationRepository


def build_repository(settings: Settings) -> InvestigationRepository:
    """Build the selected backend without silently changing durability."""

    if settings.persistence_backend is PersistenceBackend.MEMORY:
        return InMemoryInvestigationRepository()
    if settings.persistence_backend is PersistenceBackend.POSTGRES:
        if not settings.database_url:  # Also guarded by Settings validation.
            raise ValueError(
                "DATABASE_URL is required when PERSISTENCE_BACKEND=postgres"
            )
        return PostgresInvestigationRepository(settings.database_url)
    raise ValueError(f"Unsupported persistence backend: {settings.persistence_backend}")
