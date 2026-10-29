"""Compose embedding providers and keep their index files isolated."""

from __future__ import annotations

from pathlib import Path

from app.config import EmbeddingMode, EmbeddingSettings
from app.rag.embeddings import (
    AzureOpenAIEmbeddingService,
    EmbeddingService,
    LocalHashEmbeddingService,
    OpenAIEmbeddingService,
)

LOCAL_INDEX_PATH = Path("data/policy_index.json")
OPENAI_INDEX_PATH = Path("data/policy_index_openai.json")
AZURE_INDEX_PATH = Path("data/policy_index_azure.json")

_DEFAULT_INDEX_PATHS = {
    EmbeddingMode.LOCAL: LOCAL_INDEX_PATH,
    EmbeddingMode.OPENAI: OPENAI_INDEX_PATH,
    EmbeddingMode.AZURE: AZURE_INDEX_PATH,
}


def _reject_another_provider_default(
    path: Path,
    embedding_mode: EmbeddingMode,
) -> None:
    selected_absolute = path.resolve()
    for mode, reserved_path in _DEFAULT_INDEX_PATHS.items():
        if mode is not embedding_mode and selected_absolute == reserved_path.resolve():
            raise ValueError(
                f"{path} is reserved for EMBEDDING_MODE={mode.value}; "
                f"choose a separate {embedding_mode.value} index path"
            )


def build_embedding_service(settings: EmbeddingSettings) -> EmbeddingService:
    if settings.embedding_mode is EmbeddingMode.LOCAL:
        return LocalHashEmbeddingService()
    if settings.embedding_mode is EmbeddingMode.OPENAI:
        if settings.openai_api_key is None:  # Defensive; config validation covers this.
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_MODE=openai")
        return OpenAIEmbeddingService(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_embedding_model,
        )

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    return AzureOpenAIEmbeddingService(
        client=client,
        deployment=settings.azure_openai_embedding_deployment or "",
        dimension=settings.azure_openai_embedding_dimensions,
    )


def resolve_policy_index_path(
    settings: EmbeddingSettings,
    explicit_path: str | Path | None = None,
) -> Path:
    """Resolve a provider-specific index path unless one is explicitly supplied.

    ``POLICY_INDEX_PATH`` historically defaults to the local filename.  Treating
    that exact default as provider-neutral lets OpenAI and Azure select their own
    files while preserving custom paths supplied by deployments.
    """

    if explicit_path is not None:
        selected = Path(explicit_path)
        _reject_another_provider_default(selected, settings.embedding_mode)
        return selected

    configured = Path(settings.policy_index_path)
    if configured != LOCAL_INDEX_PATH:
        _reject_another_provider_default(configured, settings.embedding_mode)
        return configured
    return _DEFAULT_INDEX_PATHS[settings.embedding_mode]
