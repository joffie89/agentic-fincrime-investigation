import json

import pytest

from app.rag import (
    AzureOpenAIEmbeddingService,
    LocalHashEmbeddingService,
    PolicyIngestor,
    PolicyRetriever,
)


@pytest.fixture
def policy_file(tmp_path):
    source = tmp_path / "policies.json"
    source.write_text(
        json.dumps(
            {
                "policies": [
                    {
                        "policy_id": "POL-VELOCITY",
                        "title": "Rapid transaction velocity",
                        "text": (
                            "A rapid burst of payments in twenty four hours is a "
                            "transaction velocity indicator requiring investigation."
                        ),
                    },
                    {
                        "policy_id": "POL-NETWORK",
                        "title": "Network relationships",
                        "text": (
                            "Repeated direct transfers and shared counterparties are "
                            "account network indicators that require graph review."
                        ),
                    },
                    {
                        "policy_id": "POL-GEOGRAPHY",
                        "title": "Cross border context",
                        "text": (
                            "Cross border payments require country and beneficiary "
                            "context; geography alone does not establish wrongdoing."
                        ),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return source


def test_local_hash_embeddings_are_deterministic_and_normalized() -> None:
    service = LocalHashEmbeddingService(dimension=64)

    first = service.embed("Repeated payments to a shared counterparty")
    second = service.embed("Repeated payments to a shared counterparty")

    assert first == second
    assert len(first) == 64
    assert sum(value * value for value in first) == pytest.approx(1.0)
    assert service.embed("") == [0.0] * 64


def test_azure_embedding_adapter_uses_injected_client_without_live_call() -> None:
    class FakeEmbeddings:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            }

    class FakeClient:
        def __init__(self) -> None:
            self.embeddings = FakeEmbeddings()

    client = FakeClient()
    service = AzureOpenAIEmbeddingService(
        client=client,
        deployment="demo-embedding-deployment",
        dimension=3,
    )

    vectors = service.embed_many(["first policy", "second policy"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert client.embeddings.calls == [
        {
            "model": "demo-embedding-deployment",
            "input": ["first policy", "second policy"],
        }
    ]


def test_ingestion_chunks_and_persists_a_portable_json_index(
    policy_file, tmp_path
) -> None:
    index_path = tmp_path / "nested" / "index.json"
    service = LocalHashEmbeddingService(dimension=64)
    ingestor = PolicyIngestor(
        service,
        chunk_size_words=10,
        overlap_words=2,
    )

    count = ingestor.ingest_file(policy_file, index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert count > 3
    assert index["version"] == 1
    assert index["embedding"]["dimension"] == 64
    assert len(index["chunks"]) == count
    assert len({chunk["chunk_id"] for chunk in index["chunks"]}) == count
    assert all(len(chunk["embedding"]) == 64 for chunk in index["chunks"])


def test_retrieval_returns_ranked_policy_sources_and_scores(
    policy_file, tmp_path
) -> None:
    index_path = tmp_path / "index.json"
    service = LocalHashEmbeddingService(dimension=256)
    PolicyIngestor(service).ingest_file(policy_file, index_path)
    retriever = PolicyRetriever.from_index(index_path, service)

    results = retriever.retrieve(
        "rapid burst payment transaction velocity twenty four hours",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].policy_id == "POL-VELOCITY"
    assert results[0].chunk_id.startswith("POL-VELOCITY#")
    assert "velocity" in results[0].text.lower()
    assert 0.0 <= results[0].relevance_score <= 1.0
    assert results[0].relevance_score >= results[1].relevance_score


def test_retriever_can_recreate_matching_local_embedder_from_metadata(
    policy_file, tmp_path
) -> None:
    index_path = tmp_path / "index.json"
    PolicyIngestor(LocalHashEmbeddingService(dimension=96)).ingest_file(
        policy_file, index_path
    )

    retriever = PolicyRetriever.from_index(index_path)

    assert retriever.embedding_service.dimension == 96
    assert retriever.retrieve("shared counterparties", top_k=1)[0].policy_id == "POL-NETWORK"


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_retrieval_rejects_invalid_top_k(policy_file, tmp_path, top_k) -> None:
    index_path = tmp_path / "index.json"
    PolicyIngestor().ingest_file(policy_file, index_path)
    retriever = PolicyRetriever.from_index(index_path)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("velocity", top_k=top_k)


def test_loading_index_rejects_embedding_dimension_mismatch(
    policy_file, tmp_path
) -> None:
    index_path = tmp_path / "index.json"
    PolicyIngestor(LocalHashEmbeddingService(dimension=64)).ingest_file(
        policy_file, index_path
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        PolicyRetriever.from_index(
            index_path,
            LocalHashEmbeddingService(dimension=128),
        )
