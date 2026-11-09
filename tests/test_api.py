import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.api import Settings, create_app
from app.generation import COMPLETIONS_URL
from app.retrieval import SearchEngine
from scripts.build_corpus import build_corpus

ROOT = Path(__file__).resolve().parents[1]
QUERY = {"question": "Что означает ImagePullBackOff?", "document_id": "images"}


def test_default_search_mode_prefers_hybrid_then_qdrant():
    assert Settings().default_mode == "hybrid"
    assert Settings(modes=("bm25", "qdrant")).default_mode == "qdrant"


@pytest.fixture(scope="module")
def chunks():
    return build_corpus(ROOT)


def client_for(chunks, handler=None):
    settings = Settings(modes=("bm25",), llm_api_key="test-key", llm_folder_id="test-folder") if handler else Settings(
        modes=("bm25",)
    )
    transport = httpx.MockTransport(handler) if handler else None
    return TestClient(create_app(
        settings, engine_factory=lambda config: SearchEngine(chunks, config.modes), llm_transport=transport
    ))


def completion(status, answer="", citation_ids=None):
    content = json.dumps({"status": status, "answer": answer, "citation_ids": citation_ids or []})
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})


def test_search_health_and_swagger_work_without_credentials(chunks):
    with client_for(chunks) as client:
        assert client.get("/health").json() == {
            "status": "ok", "index_ready": True, "llm_configured": False,
            "default_mode": "bm25", "search_modes": ["bm25"],
        }
        assert client.get("/docs").status_code == 200
        result = client.post("/search", json=QUERY).json()
        assert result["mode"] == "bm25"
        assert result["results"][0]["section_id"] == "images:89"
        assert result["results"][0]["source_url"].startswith("https://github.com/kubernetes/website/")
        assert all(hit["document_id"] == "images" for hit in result["results"])
        missing_key = client.post("/ask", json=QUERY)
        assert missing_key.status_code == 503
        assert missing_key.json()["detail"]["code"] == "llm_not_configured"


@pytest.mark.parametrize("payload", [
    {"question": "  "},
    {**QUERY, "mode": "qdrant"},
    {**QUERY, "limit": 0},
    {**QUERY, "limit": 11},
])
def test_search_rejects_invalid_requests(chunks, payload):
    with client_for(chunks) as client:
        assert client.post("/search", json=payload).status_code == 422


def test_ask_builds_yandex_request_and_resolves_citations(chunks):
    def upstream(request):
        assert str(request.url) == COMPLETIONS_URL
        assert request.headers["Authorization"] == "Api-Key test-key"
        assert request.headers["OpenAI-Project"] == "test-folder"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt://test-folder/yandexgpt/rc"
        assert payload["response_format"]["type"] == "json_schema"
        user = json.loads(payload["messages"][1]["content"])
        assert user["question"] == QUERY["question"]
        assert [source["id"] for source in user["sources"]][:2] == ["S1", "S2"]
        assert "source_url" not in user["sources"][1]
        return completion("answered", "Контейнер не смог скачать образ.", ["S2", "S2"])

    with client_for(chunks, upstream) as client:
        search = client.post("/search", json=QUERY).json()["results"]
        response = client.post("/ask", json=QUERY)
        assert response.status_code == 200
        answer = response.json()
        assert answer["status"] == "answered"
        assert answer["answer"] == "Контейнер не смог скачать образ."
        assert len(answer["sources"]) == 1
        assert answer["sources"][0]["chunk_id"] == search[1]["chunk_id"]
        assert answer["sources"][0]["source_url"] == search[1]["source_url"]


def test_insufficient_evidence_with_and_without_hits(chunks):
    requests = []

    def upstream(request):
        requests.append(request)
        return completion("insufficient_evidence")

    with client_for(chunks, upstream) as client:
        no_hits = client.post("/ask", json={**QUERY, "document_id": "missing"})
        assert no_hits.status_code == 200
        assert not requests
        model_refusal = client.post("/ask", json=QUERY)
        assert len(requests) == 1
        assert model_refusal.json() == no_hits.json()
        assert no_hits.json()["status"] == "insufficient_evidence"
        assert no_hits.json()["sources"] == []


@pytest.mark.parametrize("upstream_response", [
    lambda: completion("answered", "Текст", ["S99"]),
    lambda: completion("answered", "Текст"),
    lambda: completion("answered", " ", ["S1"]),
    lambda: completion("insufficient_evidence", citation_ids=["S1"]),
    lambda: httpx.Response(200, json={"choices": []}),
])
def test_invalid_llm_output_is_a_gateway_error(chunks, upstream_response):
    with client_for(chunks, lambda request: upstream_response()) as client:
        response = client.post("/ask", json=QUERY)
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "llm_invalid_response"


def test_timeout_is_distinct_from_insufficient_evidence(chunks):
    def upstream(request):
        raise httpx.ReadTimeout("internal detail", request=request)

    with client_for(chunks, upstream) as client:
        response = client.post("/ask", json=QUERY)
        assert response.status_code == 504
        assert response.json()["detail"]["code"] == "llm_timeout"
        assert "internal detail" not in response.text


def test_upstream_errors_are_sanitized(chunks):
    with client_for(chunks, lambda request: httpx.Response(401, text="sensitive upstream body")) as client:
        response = client.post("/ask", json=QUERY)
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "llm_unavailable"
        assert "sensitive upstream body" not in response.text


def test_unavailable_index_returns_503():
    class UnavailableEngine:
        def is_ready(self):
            raise ResponseHandlingException(OSError("connection lost"))

        def search(self, *args, **kwargs):
            raise ResponseHandlingException(OSError("connection lost"))

        def close(self):
            pass

    with TestClient(create_app(Settings(), engine_factory=lambda settings: UnavailableEngine())) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["index_ready"] is False
        search = client.post("/search", json=QUERY)
        assert search.status_code == 503
        assert search.json()["detail"]["code"] == "index_unavailable"
