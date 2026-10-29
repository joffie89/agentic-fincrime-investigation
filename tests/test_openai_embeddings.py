"""OpenAI embedding configuration and adapter tests with no network calls."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from app.config import EmbeddingSettings, RetrievalMode, Settings
from app.rag.embedding_provider import resolve_policy_index_path
from app.rag.embeddings import OpenAIEmbeddingService
from app.rag.ingestion import PolicyIngestor
from app.rag.retriever import PolicyRetriever


class FakeEmbeddingsEndpoint:
    """Return deterministic vectors while retaining each SDK-style request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        inputs = list(kwargs["input"])
        rows = [
            SimpleNamespace(
                index=index,
                embedding=[float(len(value)), float(index + 1)],
            )
            for index, value in enumerate(inputs)
        ]
        # The API's response order is not part of the adapter's contract. Return
        # it backwards to verify that the explicit index is respected.
        return SimpleNamespace(data=list(reversed(rows)))


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsEndpoint()


@pytest.fixture
def policy_file(tmp_path):
    source = tmp_path / "policies.json"
    source.write_text(
        json.dumps(
            {
                "policies": [
                    {
                        "policy_id": "POL-OPENAI-TEST",
                        "title": "Mock policy",
                        "text": "Synthetic policy text used to test index metadata.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return source


def test_openai_embedding_service_batches_inputs_and_restores_response_order() -> None:
    client = FakeOpenAIClient()
    service = OpenAIEmbeddingService(
        client=client,
        model="mock-embedding-model",
        dimension=2,
        batch_size=2,
    )

    vectors = service.embed_many(["one", "second", "three", "fourth", "five"])

    assert vectors == [
        [3.0, 1.0],
        [6.0, 2.0],
        [5.0, 1.0],
        [6.0, 2.0],
        [4.0, 1.0],
    ]
    assert client.embeddings.calls == [
        {
            "model": "mock-embedding-model",
            "input": ["one", "second"],
        },
        {
            "model": "mock-embedding-model",
            "input": ["three", "fourth"],
        },
        {
            "model": "mock-embedding-model",
            "input": ["five"],
        },
    ]


def test_openai_embedding_service_configures_sdk_timeout_and_retries() -> None:
    client = FakeOpenAIClient()
    with patch("app.rag.embeddings.OpenAI", return_value=client) as openai_client:
        service = OpenAIEmbeddingService(
            api_key="not-a-real-key",
            model="mock-embedding-model",
            dimension=2,
        )

    assert service.model == "mock-embedding-model"
    openai_client.assert_called_once_with(
        api_key="not-a-real-key",
        timeout=30.0,
        max_retries=3,
    )


def test_local_embedding_mode_does_not_require_openai_key() -> None:
    settings = Settings(
        _env_file=None,
        api_key="test-api-key",
        embedding_mode="local",
        openai_api_key=None,
    )

    assert settings.openai_api_key is None
    assert settings.openai_embedding_model == "text-embedding-3-small"


def test_offline_embedding_settings_do_not_require_http_api_key() -> None:
    settings = EmbeddingSettings(
        _env_file=None,
        embedding_mode="local",
    )

    assert settings.embedding_mode.value == "local"


def test_openai_embedding_mode_requires_openai_key() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(
            _env_file=None,
            api_key="test-api-key",
            embedding_mode="openai",
            openai_api_key=None,
        )


def test_openai_embedding_settings_keep_key_secret() -> None:
    settings = Settings(
        _env_file=None,
        api_key="test-api-key",
        embedding_mode="openai",
        openai_api_key="not-a-real-key",
    )

    assert isinstance(settings.openai_api_key, SecretStr)
    assert settings.openai_api_key.get_secret_value() == "not-a-real-key"
    assert settings.openai_embedding_model == "text-embedding-3-small"


def test_openai_mode_defaults_to_vector_but_explicit_retrieval_mode_wins(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    values = {
        "_env_file": None,
        "api_key": "test-api-key",
        "embedding_mode": "openai",
        "openai_api_key": "not-a-real-key",
    }

    automatic = Settings(**values)
    explicit_argument = Settings(**values, retrieval_mode="hybrid")
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    explicit_environment = Settings(**values)

    assert automatic.retrieval_mode is RetrievalMode.VECTOR
    assert explicit_argument.retrieval_mode is RetrievalMode.HYBRID
    assert explicit_environment.retrieval_mode is RetrievalMode.BM25


def test_openai_and_local_modes_use_separate_default_index_paths() -> None:
    local_settings = Settings(
        _env_file=None,
        api_key="test-api-key",
        embedding_mode="local",
        openai_api_key=None,
    )
    openai_settings = Settings(
        _env_file=None,
        api_key="test-api-key",
        embedding_mode="openai",
        openai_api_key="not-a-real-key",
    )

    local_path = resolve_policy_index_path(local_settings)
    openai_path = resolve_policy_index_path(openai_settings)

    assert local_path.name == "policy_index.json"
    assert openai_path.name == "policy_index_openai.json"
    assert openai_path != local_path

    with pytest.raises(ValueError, match="reserved for EMBEDDING_MODE=local"):
        resolve_policy_index_path(openai_settings, "data/policy_index.json")


def test_openai_index_records_model_and_rejects_different_model(policy_file, tmp_path) -> None:
    index_path = tmp_path / "policy_index_openai.json"
    service = OpenAIEmbeddingService(
        client=FakeOpenAIClient(),
        model="model-used-for-index",
        dimension=2,
    )

    PolicyIngestor(service).ingest_file(policy_file, index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["embedding"]["implementation"] == "OpenAIEmbeddingService"
    assert index["embedding"]["model"] == "model-used-for-index"

    other_model = OpenAIEmbeddingService(
        client=FakeOpenAIClient(),
        model="different-model",
        dimension=2,
    )
    with pytest.raises(ValueError, match="[Mm]odel mismatch"):
        PolicyRetriever.from_index(index_path, other_model)
