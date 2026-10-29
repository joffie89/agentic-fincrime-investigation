"""Query-time loading and ranking of a JSON policy index."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import RetrievalMode
from app.models.domain import RetrievedPolicy
from app.rag.embeddings import EmbeddingService, LocalHashEmbeddingService

RRF_RANK_CONSTANT = 60
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
) -> list[tuple[str, float]]:
    """Fuse result identifiers using ``1 / (60 + rank)`` per ranked list."""

    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        seen: set[str] = set()
        for rank, item_id in enumerate(ranked_ids, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (RRF_RANK_CONSTANT + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class PolicyRetriever:
    """Vector, BM25, or fused retrieval over a policy index."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedding_service: EmbeddingService,
        *,
        minimum_relevance_score: float = 0.05,
        retrieval_mode: RetrievalMode | str = RetrievalMode.HYBRID,
    ) -> None:
        if not 0 <= minimum_relevance_score <= 1:
            raise ValueError("minimum_relevance_score must be between 0 and 1")
        try:
            resolved_mode = RetrievalMode(retrieval_mode)
        except ValueError as exc:
            raise ValueError("retrieval_mode must be one of: vector, bm25, hybrid") from exc
        self._chunks = chunks
        self.embedding_service = embedding_service
        self.minimum_relevance_score = minimum_relevance_score
        self.retrieval_mode = resolved_mode
        self._tokenized_chunks = [self._tokenize(str(chunk["text"])) for chunk in chunks]
        self._bm25 = (
            BM25Okapi(self._tokenized_chunks)
            if self._tokenized_chunks and any(self._tokenized_chunks)
            else None
        )

    @classmethod
    def from_index(
        cls,
        index_path: str | Path,
        embedding_service: EmbeddingService | None = None,
        *,
        minimum_relevance_score: float = 0.05,
        policy_source_path: str | Path | None = None,
        retrieval_mode: RetrievalMode | str = RetrievalMode.HYBRID,
    ) -> PolicyRetriever:
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
                raise ValueError("A matching embedding service is required for this policy index")
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
                raise ValueError("Local embedding implementation version does not match the index")
        if implementation == "OpenAIEmbeddingService":
            indexed_model = metadata.get("model")
            service_model = getattr(service, "model", None)
            if not isinstance(indexed_model, str) or not indexed_model:
                raise ValueError("OpenAI policy index has no valid embedding model")
            if indexed_model != service_model:
                raise ValueError(
                    f"Embedding model mismatch: index={indexed_model}, service={service_model}"
                )

        if policy_source_path is not None:
            policy_source = Path(policy_source_path)
            source_metadata = payload.get("source")
            expected_digest = (
                source_metadata.get("sha256") if isinstance(source_metadata, dict) else None
            )
            try:
                source_text = policy_source.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Unable to verify policy source: {policy_source}") from exc
            actual_digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            if not expected_digest or expected_digest != actual_digest:
                raise ValueError("Policy index is stale or does not match its source document")

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
            retrieval_mode=retrieval_mode,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Policy query must not be blank")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self._chunks:
            return []

        if self.retrieval_mode is RetrievalMode.VECTOR:
            return self._retrieve_vector(query, top_k)
        if self.retrieval_mode is RetrievalMode.BM25:
            return self._retrieve_bm25(query, top_k)
        return self._retrieve_hybrid(query, top_k)

    def get_by_policy_ids(self, policy_ids: Sequence[str]) -> list[RetrievedPolicy]:
        """Return indexed policy passages without running an embedding search.

        Signal-based inclusion is an exact policy lookup rather than another
        relevance search. A score of ``1.0`` denotes that deterministic match;
        semantic results retain their original search scores when the workflow
        de-duplicates the combined evidence.
        """

        chunks_by_policy_id: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks:
            chunks_by_policy_id.setdefault(str(chunk["policy_id"]), chunk)

        policies: list[RetrievedPolicy] = []
        for policy_id in dict.fromkeys(policy_ids):
            chunk = chunks_by_policy_id.get(policy_id)
            if chunk is not None:
                policies.append(self._retrieved_policy(chunk, 1.0))
        return policies

    def _retrieve_vector(self, query: str, top_k: int) -> list[RetrievedPolicy]:
        ranked = [
            item for item in self._vector_ranking(query) if item[0] >= self.minimum_relevance_score
        ]
        return [self._retrieved_policy(chunk, score) for score, chunk in ranked[:top_k]]

    def _retrieve_bm25(self, query: str, top_k: int) -> list[RetrievedPolicy]:
        ranked = self._bm25_ranking(query)
        if not ranked:
            return []
        maximum_score = ranked[0][0]
        score_floor = min(0.0, ranked[-1][0])
        score_range = maximum_score - score_floor
        return [
            self._retrieved_policy(
                chunk,
                (score - score_floor) / score_range if score_range else 1.0,
            )
            for score, chunk in ranked[:top_k]
        ]

    def _retrieve_hybrid(self, query: str, top_k: int) -> list[RetrievedPolicy]:
        vector_ranked = [
            item for item in self._vector_ranking(query) if item[0] >= self.minimum_relevance_score
        ]
        bm25_ranked = self._bm25_ranking(query)
        fused = reciprocal_rank_fusion(
            [
                [str(chunk["chunk_id"]) for _, chunk in vector_ranked],
                [str(chunk["chunk_id"]) for _, chunk in bm25_ranked],
            ]
        )
        chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in self._chunks}
        return [
            self._retrieved_policy(chunks_by_id[chunk_id], score)
            for chunk_id, score in fused[:top_k]
        ]

    def _vector_ranking(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        query_vector = self.embedding_service.embed(query)
        if len(query_vector) != self.embedding_service.dimension:
            raise ValueError("Embedding service returned an unexpected dimension")

        return sorted(
            (
                (
                    self._cosine_similarity(query_vector, chunk["embedding"]),
                    chunk,
                )
                for chunk in self._chunks
            ),
            key=lambda item: (-item[0], item[1]["policy_id"], item[1]["chunk_id"]),
        )

    def _bm25_ranking(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        if self._bm25 is None:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        return sorted(
            (
                (float(score), chunk)
                for score, chunk, chunk_tokens in zip(
                    scores,
                    self._chunks,
                    self._tokenized_chunks,
                    strict=True,
                )
                if query_token_set.intersection(chunk_tokens)
            ),
            key=lambda item: (-item[0], item[1]["policy_id"], item[1]["chunk_id"]),
        )

    @staticmethod
    def _retrieved_policy(
        chunk: Mapping[str, Any],
        score: float,
    ) -> RetrievedPolicy:
        return RetrievedPolicy(
            policy_id=chunk["policy_id"],
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            relevance_score=round(float(score), 6),
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text.lower())

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

    reasons = _field(risk_assessment, "risk_reasons") or _field(risk_assessment, "reasons")
    signals = _field(graph_analysis, "signals")
    for values in (reasons, signals):
        if isinstance(values, (list, tuple, set)):
            parts.extend(str(value).replace("_", " ") for value in values)
    return " ".join(parts)


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
