"""Query-time loading and ranking of the local JSON policy index."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.models.domain import RetrievedPolicy
from app.rag.embeddings import EmbeddingService, LocalHashEmbeddingService


class PolicyRetriever:
    """Cosine-similarity retrieval over an index produced by PolicyIngestor."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedding_service: EmbeddingService,
        *,
        minimum_relevance_score: float = 0.05,
    ) -> None:
        if not 0 <= minimum_relevance_score <= 1:
            raise ValueError("minimum_relevance_score must be between 0 and 1")
        self._chunks = chunks
        self.embedding_service = embedding_service
        self.minimum_relevance_score = minimum_relevance_score

    @classmethod
    def from_index(
        cls,
        index_path: str | Path,
        embedding_service: EmbeddingService | None = None,
        *,
        minimum_relevance_score: float = 0.05,
        policy_source_path: str | Path | None = None,
    ) -> "PolicyRetriever":
        source = Path(index_path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read policy index: {source}") from exc

        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported or malformed policy index")
        metadata = payload.get("embedding")
        chunks = payload.get("chunks")
        if not isinstance(metadata, dict) or not isinstance(chunks, list):
            raise ValueError("Malformed policy index metadata")
        try:
            dimension = int(metadata["dimension"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Policy index has no valid embedding dimension") from exc

        implementation = metadata.get("implementation")
        if embedding_service is None:
            if implementation != "LocalHashEmbeddingService":
                raise ValueError(
                    "A matching embedding service is required for this policy index"
                )
            service: EmbeddingService = LocalHashEmbeddingService(dimension=dimension)
        else:
            service = embedding_service
            if implementation != type(service).__name__:
                raise ValueError(
                    "Embedding implementation mismatch: "
                    f"index={implementation}, service={type(service).__name__}"
                )
        if service.dimension != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: index={dimension}, service={service.dimension}"
            )

        if implementation == "LocalHashEmbeddingService":
            indexed_version = metadata.get("implementation_version")
            if indexed_version != LocalHashEmbeddingService.VERSION:
                raise ValueError(
                    "Local embedding implementation version does not match the index"
                )

        if policy_source_path is not None:
            policy_source = Path(policy_source_path)
            source_metadata = payload.get("source")
            expected_digest = (
                source_metadata.get("sha256")
                if isinstance(source_metadata, dict)
                else None
            )
            try:
                source_text = policy_source.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(
                    f"Unable to verify policy source: {policy_source}"
                ) from exc
            actual_digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            if not expected_digest or expected_digest != actual_digest:
                raise ValueError(
                    "Policy index is stale or does not match its source document"
                )

        validated_chunks: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("Malformed policy chunk")
            policy_id = chunk.get("policy_id")
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            vector = chunk.get("embedding")
            if not all(isinstance(value, str) and value for value in (policy_id, chunk_id, text)):
                raise ValueError("Policy chunk is missing identifying text fields")
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id in policy index: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            if (
                not isinstance(vector, list)
                or len(vector) != dimension
                or not all(isinstance(value, (int, float)) for value in vector)
            ):
                raise ValueError(f"Invalid embedding for policy chunk: {chunk_id}")
            validated_chunks.append(chunk)

        return cls(
            validated_chunks,
            service,
            minimum_relevance_score=minimum_relevance_score,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Policy query must not be blank")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self._chunks:
            return []

        query_vector = self.embedding_service.embed(query)
        if len(query_vector) != self.embedding_service.dimension:
            raise ValueError("Embedding service returned an unexpected dimension")

        ranked = sorted(
            (
                (
                    self._cosine_similarity(query_vector, chunk["embedding"]),
                    chunk,
                )
                for chunk in self._chunks
            ),
            key=lambda item: (-item[0], item[1]["policy_id"], item[1]["chunk_id"]),
        )
        return [
            RetrievedPolicy(
                policy_id=chunk["policy_id"],
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                relevance_score=round(score, 6),
            )
            for score, chunk in ranked
            if score >= self.minimum_relevance_score
        ][:top_k]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
        if not left_norm or not right_norm:
            return 0.0
        score = sum(a * float(b) for a, b in zip(left, right, strict=True)) / (
            left_norm * right_norm
        )
        # The local vectors are non-negative, but clamp tiny floating-point
        # overshoots to the score range promised by the API model.
        return max(0.0, min(1.0, score))


def build_policy_query(
    transaction: object,
    risk_assessment: object,
    graph_analysis: object,
) -> str:
    """Create an evidence-search query from structured investigation signals.

    Objects may be Pydantic models or mappings, which keeps this helper usable
    with typed workflow state as well as small unit-test fixtures.
    """

    parts = ["financial crime transaction investigation policy"]
    for field in ("origin_country", "destination_country"):
        value = _field(transaction, field)
        if value:
            parts.append(str(value))
    if _field(transaction, "new_payee"):
        parts.append("new payee")

    reasons = _field(risk_assessment, "risk_reasons") or _field(
        risk_assessment, "reasons"
    )
    signals = _field(graph_analysis, "signals")
    for values in (reasons, signals):
        if isinstance(values, (list, tuple, set)):
            parts.extend(str(value).replace("_", " ") for value in values)
    return " ".join(parts)


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
