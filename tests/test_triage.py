from pathlib import Path

from fastapi.testclient import TestClient

from app.api import Settings, create_app
from app.retrieval import EvidenceCoverage, Hit, SearchEngine, read_jsonl, section_fusion
from scripts.build_corpus import build_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_section_fusion_keeps_only_one_slot_per_section():
    a1 = Hit({"chunk_id": "a1", "section_id": "a"}, 10)
    a2 = Hit({"chunk_id": "a2", "section_id": "a"}, 9)
    b1 = Hit({"chunk_id": "b1", "section_id": "b"}, 8)
    hits = section_fusion(((a1, a2, b1), (b1, a2)), 5)
    assert len(hits) == 2
    assert {hit.chunk["section_id"] for hit in hits} == {"a", "b"}
    assert all(hit.score > 0 for hit in hits)


def test_unseen_terms_lower_evidence_coverage():
    chunk = {"chunk_id": "a", "heading_path": ["Образы"], "text": "контейнер и ImagePullBackOff"}
    metric = EvidenceCoverage([chunk])
    assert metric.score("Что означает ImagePullBackOff?", [Hit(chunk, 1)]) > 0
    assert metric.score("Как настроить Helm?", [Hit(chunk, 1)]) == 0
    assert metric.score("ImagePullBackOff", []) == 0


def test_triage_splits_are_disjoint_and_grounded():
    chunks = build_corpus(ROOT)
    valid = {chunk["section_id"] for chunk in chunks}
    seen = set()
    for name in ("dev", "test"):
        rows = read_jsonl(ROOT / "evaluation" / f"triage-{name}.jsonl")
        assert len(rows) == 24
        assert sum(bool(row["relevant_section_ids"]) for row in rows) == 16
        for row in rows:
            assert row["group"] not in seen
            seen.add(row["group"])
            assert set(row["relevant_section_ids"]) <= valid
            assert bool(row["relevant_section_ids"]) == (row["type"] != "unanswerable")


def test_workspace_and_evidence_contract():
    app = create_app(
        Settings(modes=("bm25",)),
        engine_factory=lambda settings: SearchEngine(build_corpus(ROOT), settings.modes),
    )
    with TestClient(app) as client:
        assert "Kube Triage" in client.get("/").text
        assert client.get("/static/workspace.js").status_code == 200
        response = client.post("/search", json={"question": "Что означает ImagePullBackOff?"})
        assert response.status_code == 200
        data = response.json()
        assert data["ranking"] is None
        assert data["evidence"] == "candidates"
        assert data["results"][0]["section_id"] == "images:89"
        assert len(data["results"][0]["version"]) == 40
        outside = client.post("/search", json={"question": "Как откатить релиз Helm на предыдущую ревизию?"})
        assert outside.json()["evidence"] == "insufficient_evidence"
