"""Embedding abstractions and a deterministic local implementation."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any, Protocol, Sequence


class EmbeddingService(Protocol):
    """Minimal interface supported by local and hosted embedding providers."""

    @property
    def dimension(self) -> int:
        """Length of each returned vector."""

    def embed(self, text: str) -> list[float]:
        """Embed one text value."""

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple text values."""


class LocalHashEmbeddingService:
    """Network-free hashing-vectorizer embeddings for local/demo mode.

    Tokens and adjacent token pairs are mapped into a fixed-size vector with a
    stable BLAKE2 hash, then L2-normalized.  This is intentionally a lexical
    baseline rather than a semantic ML model, but it is deterministic across
    processes and needs no downloaded model or credentials.
    """

    VERSION = "1"
    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 32:
            raise ValueError("Embedding dimension must be at least 32")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("Text to embed must be a string")

        tokens = self._TOKEN_PATTERN.findall(text.lower())
        vector = [0.0] * self.dimension
        features = tokens + [
            f"{left}_{right}" for left, right in zip(tokens, tokens[1:])
        ]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, byteorder="big") % self.dimension
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class AzureOpenAIEmbeddingService:
    """Adapter for an injected Azure OpenAI-compatible client.

    The adapter deliberately does not create a client or read credentials. The
    application composition layer owns endpoint, API-version, and credential
    configuration, then injects a client exposing ``embeddings.create``. This
    keeps ingestion testable and prevents secrets from entering this module.
    """

    def __init__(self, client: object, deployment: str, dimension: int) -> None:
        if not deployment.strip():
            raise ValueError("Azure OpenAI embedding deployment must not be blank")
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        embeddings_endpoint = getattr(client, "embeddings", None)
        if embeddings_endpoint is None or not callable(
            getattr(embeddings_endpoint, "create", None)
        ):
            raise TypeError("Client must expose embeddings.create")
        self._client = client
        self.deployment = deployment
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("Azure OpenAI embedding inputs must be non-empty strings")

        response = self._client.embeddings.create(  # type: ignore[attr-defined]
            model=self.deployment,
            input=values,
        )
        raw_data = self._read(response, "data")
        if not isinstance(raw_data, (list, tuple)) or len(raw_data) != len(values):
            raise ValueError("Azure OpenAI returned an unexpected embedding count")

        ordered = sorted(
            raw_data,
            key=lambda item: int(self._read(item, "index", 0)),
        )
        vectors: list[list[float]] = []
        for item in ordered:
            raw_vector = self._read(item, "embedding")
            if not isinstance(raw_vector, (list, tuple)):
                raise ValueError("Azure OpenAI returned a malformed embedding")
            vector = [float(value) for value in raw_vector]
            if len(vector) != self.dimension or not all(
                math.isfinite(value) for value in vector
            ):
                raise ValueError("Azure OpenAI returned an unexpected embedding dimension")
            vectors.append(vector)
        return vectors

    @staticmethod
    def _read(value: object, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)
