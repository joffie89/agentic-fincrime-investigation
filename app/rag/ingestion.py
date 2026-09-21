"""Offline ingestion of demo policies into a portable JSON vector index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.rag.embeddings import EmbeddingService, LocalHashEmbeddingService


class PolicyIngestor:
    """Chunk and embed policy documents separately from query-time retrieval."""

    INDEX_VERSION = 1

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        *,
        chunk_size_words: int = 120,
        overlap_words: int = 20,
    ) -> None:
        if chunk_size_words < 10:
            raise ValueError("chunk_size_words must be at least 10")
        if overlap_words < 0 or overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be non-negative and smaller than chunk size")
        self.embedding_service = embedding_service or LocalHashEmbeddingService()
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def ingest_file(self, policy_path: str | Path, index_path: str | Path) -> int:
        """Read policy JSON, then write embeddings to ``index_path``.

        Returns the number of indexed chunks.  Source documents accept either a
        top-level list or ``{"policies": [...]}``; every policy needs a unique
        ``policy_id`` and non-empty ``text``.  ``title`` is optional.
        """

        source = Path(policy_path)
        destination = Path(index_path)
        try:
            source_text = source.read_text(encoding="utf-8")
            payload = json.loads(source_text)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read policy file: {source}") from exc

        documents = payload.get("policies") if isinstance(payload, dict) else payload
        if not isinstance(documents, list) or not documents:
            raise ValueError("Policy file must contain a non-empty policies list")

        chunks: list[dict[str, Any]] = []
        policy_ids: set[str] = set()
        for raw_document in documents:
            if not isinstance(raw_document, dict):
                raise ValueError("Each policy must be a JSON object")
            policy_id = str(raw_document.get("policy_id", "")).strip()
            title = str(raw_document.get("title", "")).strip()
            text = str(raw_document.get("text", "")).strip()
            if not policy_id or not text:
                raise ValueError("Each policy requires non-empty policy_id and text")
            if policy_id in policy_ids:
                raise ValueError(f"Duplicate policy_id: {policy_id}")
            policy_ids.add(policy_id)

            document_chunks = self._chunk_text(text)
            evidence_chunks = [
                f"{title}\n{chunk}" if title else chunk for chunk in document_chunks
            ]
            vectors = self.embedding_service.embed_many(evidence_chunks)
            for position, (evidence, vector) in enumerate(
                zip(evidence_chunks, vectors, strict=True),
                start=1,
            ):
                if len(vector) != self.embedding_service.dimension:
                    raise ValueError("Embedding service returned an unexpected dimension")
                chunks.append(
                    {
                        "policy_id": policy_id,
                        "chunk_id": f"{policy_id}#{position:03d}",
                        "text": evidence,
                        "embedding": vector,
                    }
                )

        index = {
            "version": self.INDEX_VERSION,
            "embedding": {
                "implementation": type(self.embedding_service).__name__,
                "implementation_version": getattr(
                    self.embedding_service, "VERSION", "provider-managed"
                ),
                "dimension": self.embedding_service.dimension,
            },
            "source": {
                "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            },
            "chunking": {
                "chunk_size_words": self.chunk_size_words,
                "overlap_words": self.overlap_words,
            },
            "chunks": chunks,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return len(chunks)

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            raise ValueError("Policy text must not be blank")
        step = self.chunk_size_words - self.overlap_words
        chunks: list[str] = []
        for start in range(0, len(words), step):
            chunk = words[start : start + self.chunk_size_words]
            chunks.append(" ".join(chunk))
            # Do not emit a final chunk containing only overlap already present
            # in the preceding complete chunk.
            if start + self.chunk_size_words >= len(words):
                break
        return chunks


def main() -> None:
    """Build the deterministic local index as an explicit offline step."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/policies.json"),
        help="Path to the synthetic policy documents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/policy_index.json"),
        help="Destination for the JSON vector index",
    )
    arguments = parser.parse_args()
    count = PolicyIngestor(LocalHashEmbeddingService()).ingest_file(
        arguments.source,
        arguments.output,
    )
    print(json.dumps({"indexed_chunks": count, "index_path": str(arguments.output)}))


if __name__ == "__main__":
    main()
