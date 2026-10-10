"""Embedding abstractions and local or hosted implementations."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any, Protocol, Sequence

from openai import OpenAI


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


class OpenAIEmbeddingService:
    """Batched adapter for the OpenAI embeddings API.

    A client can be injected for deterministic tests.  In normal use the
    adapter constructs the official SDK client with a 30-second timeout and at
    most three retries.
    """

    _MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: object | None = None,
        model: str = "text-embedding-3-small",
        dimension: int | None = None,
        batch_size: int = 100,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("OpenAI embedding model must not be blank")
        if dimension is None:
            dimension = self._MODEL_DIMENSIONS.get(normalized_model, 1536)
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("OpenAI embedding batch size must be a positive integer")
        if client is None:
            if not api_key or not api_key.strip():
                raise ValueError("OpenAI API key must not be blank")
            client = OpenAI(
                api_key=api_key,
                timeout=30.0,
                max_retries=3,
            )
        embeddings_endpoint = getattr(client, "embeddings", None)
        if embeddings_endpoint is None or not callable(
            getattr(embeddings_endpoint, "create", None)
        ):
            raise TypeError("Client must expose embeddings.create")

        self._client = client
        self.model = normalized_model
        self._dimension = dimension
        self.batch_size = batch_size

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
            raise ValueError("OpenAI embedding inputs must be non-empty strings")

        vectors: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            response = self._client.embeddings.create(  # type: ignore[attr-defined]
                model=self.model,
                input=batch,
            )
            raw_data = self._read(response, "data")
            if not isinstance(raw_data, (list, tuple)) or len(raw_data) != len(batch):
                raise ValueError("OpenAI returned an unexpected embedding count")

            indexed_items: list[tuple[int, object]] = []
            for item in raw_data:
                raw_index = self._read(item, "index")
                if isinstance(raw_index, bool):
                    raise ValueError("OpenAI returned a malformed embedding index")
                try:
                    index = int(raw_index)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "OpenAI returned a malformed embedding index"
                    ) from exc
                indexed_items.append((index, item))
            indexed_items.sort(key=lambda pair: pair[0])
            if [index for index, _ in indexed_items] != list(range(len(batch))):
                raise ValueError("OpenAI returned malformed embedding indexes")

            for _, item in indexed_items:
                raw_vector = self._read(item, "embedding")
                if not isinstance(raw_vector, (list, tuple)):
                    raise ValueError("OpenAI returned a malformed embedding")
                try:
                    vector = [float(value) for value in raw_vector]
                except (TypeError, ValueError) as exc:
                    raise ValueError("OpenAI returned a malformed embedding") from exc
                if len(vector) != self.dimension or not all(
                    math.isfinite(value) for value in vector
                ):
                    raise ValueError("OpenAI returned an unexpected embedding dimension")
                vectors.append(vector)
        return vectors

    @staticmethod
    def _read(value: object, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)


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
