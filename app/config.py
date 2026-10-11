"""Centralized, environment-driven application configuration."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceMode(str, Enum):
    """Policy-assessment provider used by the application."""

    LOCAL = "local"
    OPENAI = "openai"
    AZURE = "azure"


class EmbeddingMode(str, Enum):
    """Embedding provider used for policy ingestion and retrieval."""

    LOCAL = "local"
    OPENAI = "openai"
    AZURE = "azure"


class RetrievalMode(str, Enum):
    """Ranking strategy used for policy retrieval."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class PersistenceBackend(str, Enum):
    """Supported investigation persistence implementations."""

    MEMORY = "memory"
    POSTGRES = "postgres"


class EmbeddingSettings(BaseSettings):
    """Embedding configuration shared by the app and offline RAG commands."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    embedding_mode: EmbeddingMode = EmbeddingMode.LOCAL
    policy_index_path: Path = Path("data/policy_index.json")

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = Field(default=None, repr=False, exclude=True)
    azure_openai_embedding_deployment: str | None = None
    azure_openai_embedding_dimensions: int = Field(default=1536, ge=1, le=4096)
    azure_openai_api_version: str = "2024-10-21"
    openai_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    openai_embedding_model: str = "text-embedding-3-small"

    @field_validator(
        "azure_openai_endpoint",
        "azure_openai_api_key",
        "azure_openai_embedding_deployment",
        "openai_api_key",
        mode="before",
    )
    @classmethod
    def empty_embedding_strings_are_unset(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("openai_embedding_model", mode="before")
    @classmethod
    def openai_embedding_model_must_not_be_blank(cls, value: object) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("OPENAI_EMBEDDING_MODEL must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_selected_embedding_provider(self) -> "EmbeddingSettings":
        if self.embedding_mode is EmbeddingMode.AZURE:
            common_required = {
                "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            }
            missing = [
                name for name, value in common_required.items() if not value
            ]
            if missing:
                raise ValueError(
                    "An Azure provider requires these environment variables: "
                    + ", ".join(missing)
                )
            if not self.azure_openai_embedding_deployment:
                raise ValueError(
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required when "
                    "EMBEDDING_MODE=azure"
                )
        if self.embedding_mode is EmbeddingMode.OPENAI and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_MODE=openai"
            )
        return self

    @property
    def azure_endpoint(self) -> str | None:
        return self.azure_openai_endpoint

    @property
    def azure_api_key(self) -> str | None:
        return self.azure_openai_api_key

    @property
    def azure_embedding_deployment(self) -> str | None:
        return self.azure_openai_embedding_deployment

    @property
    def azure_api_version(self) -> str:
        return self.azure_openai_api_version


def _default_retrieval_mode(
    validated_data: dict[str, object],
) -> RetrievalMode:
    if validated_data.get("embedding_mode") is EmbeddingMode.OPENAI:
        return RetrievalMode.VECTOR
    return RetrievalMode.HYBRID


class Settings(EmbeddingSettings):
    """Application settings loaded from environment variables or ``.env``.

    Local/demo mode is the safe default: it neither requires hosted-provider
    credentials nor claims durable persistence. Selecting an external backend
    validates its configuration at startup instead of silently falling back.
    """

    app_env: str = "development"
    log_level: str = "INFO"
    service_mode: ServiceMode = ServiceMode.LOCAL
    retrieval_mode: RetrievalMode = Field(default_factory=_default_retrieval_mode)
    persistence_backend: PersistenceBackend = PersistenceBackend.MEMORY
    api_key: SecretStr = Field(repr=False, exclude=True)
    database_url: str | None = Field(default=None, repr=False, exclude=True)

    retrieval_top_k: int = Field(default=3, ge=1, le=20)
    retrieval_min_relevance_score: float = Field(default=0.05, ge=0.0, le=1.0)
    human_review_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    azure_openai_chat_deployment: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_app_env(cls, value: object) -> str:
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("APP_ENV must not be empty")
        return normalized

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> str:
        normalized = str(value).strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return normalized

    @field_validator("openai_chat_model", mode="before")
    @classmethod
    def openai_chat_model_must_not_be_blank(cls, value: object) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("OPENAI_CHAT_MODEL must not be empty")
        return normalized

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("API_KEY must not be empty")
        return value

    @field_validator("database_url", "azure_openai_chat_deployment", mode="before")
    @classmethod
    def empty_application_strings_are_unset(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def validate_selected_backends(self) -> "Settings":
        if self.service_mode is ServiceMode.OPENAI and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when SERVICE_MODE=openai"
            )

        if self.persistence_backend is PersistenceBackend.POSTGRES and not self.database_url:
            raise ValueError(
                "DATABASE_URL is required when PERSISTENCE_BACKEND=postgres"
            )

        if self.service_mode is ServiceMode.AZURE:
            common_required = {
                "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            }
            missing = [
                name for name, value in common_required.items() if not value
            ]
            if missing:
                raise ValueError(
                    "An Azure provider requires these environment variables: "
                    + ", ".join(missing)
                )
        if (
            self.service_mode is ServiceMode.AZURE
            and not self.azure_openai_chat_deployment
        ):
            raise ValueError(
                "AZURE_OPENAI_CHAT_DEPLOYMENT is required when SERVICE_MODE=azure"
            )
        return self

    @property
    def azure_chat_deployment(self) -> str | None:
        return self.azure_openai_chat_deployment


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one validated settings object per application process."""

    return Settings()
