"""Local policy ingestion and retrieval components.

Exports are loaded lazily so ``python -m app.rag.ingestion`` remains a clean,
explicit ingestion command rather than importing the target module twice.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EmbeddingService": ("app.rag.embeddings", "EmbeddingService"),
    "AzureOpenAIEmbeddingService": (
        "app.rag.embeddings",
        "AzureOpenAIEmbeddingService",
    ),
    "LocalHashEmbeddingService": (
        "app.rag.embeddings",
        "LocalHashEmbeddingService",
    ),
    "OpenAIEmbeddingService": (
        "app.rag.embeddings",
        "OpenAIEmbeddingService",
    ),
    "PolicyIngestor": ("app.rag.ingestion", "PolicyIngestor"),
    "PolicyRetriever": ("app.rag.retriever", "PolicyRetriever"),
    "build_policy_query": ("app.rag.retriever", "build_policy_query"),
    "reciprocal_rank_fusion": (
        "app.rag.retriever",
        "reciprocal_rank_fusion",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
