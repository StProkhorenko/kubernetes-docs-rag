import httpx
import pytest

from scripts.evaluate_answers import collect_answers


def test_collection_resumes_after_provider_error(tmp_path):
    questions = [
        {"id": "q1", "question": "Первый вопрос", "relevant_section_ids": ["section:1"]},
        {"id": "q2", "question": "Второй вопрос", "relevant_section_ids": []},
    ]
    output = tmp_path / "answers.jsonl"
    attempts = []

    def handler(request):
        attempts.append(request.read())
        if len(attempts) == 2:
            return httpx.Response(504)
        return httpx.Response(200, json={
            "mode": "qdrant", "status": "insufficient_evidence", "answer": "Недостаточно данных", "sources": []
        })

    with httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            collect_answers(client, questions, "qdrant", output)
        assert len(output.read_text(encoding="utf-8").splitlines()) == 1
        rows = collect_answers(client, questions, "qdrant", output)

    assert [row["id"] for row in rows] == ["q1", "q2"]
    assert attempts[0] != attempts[1] == attempts[2]
