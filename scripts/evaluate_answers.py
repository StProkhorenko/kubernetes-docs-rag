"""Собирает ответы /ask на отложенных вопросах для ручной проверки."""

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import httpx

from app.retrieval import CORPUS, DEFAULT_MODE, MODES, ROOT, read_jsonl


def collect_answers(client, questions, mode, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(output) if output.exists() else []
    if len(rows) > len(questions) or any(
        row["id"] != question["id"] or row["question"] != question["question"] or row["mode"] != mode
        for row, question in zip(rows, questions)
    ):
        raise ValueError("Файл содержит результаты другого набора или режима. Укажи новый --output.")

    with output.open("a", encoding="utf-8", newline="\n") as file:
        for question in questions[len(rows):]:
            started = perf_counter()
            response = client.post("/ask", json={"question": question["question"], "mode": mode})
            response.raise_for_status()
            row = {
                **question,
                **response.json(),
                "latency_ms": round((perf_counter() - started) * 1000, 3),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument("--output", type=Path, default=CORPUS.parent / "test-answer-results.jsonl")
    args = parser.parse_args()
    questions = read_jsonl(ROOT / "evaluation/test.jsonl")
    try:
        with httpx.Client(base_url=args.base_url, timeout=120) as client:
            health = client.get("/health")
            health.raise_for_status()
            if not health.json()["llm_configured"]:
                parser.exit(1, "Заполни LLM_API_KEY и LLM_FOLDER_ID и перезапусти приложение.\n")
            rows = collect_answers(client, questions, args.mode, args.output)
    except (httpx.HTTPError, ValueError) as exc:
        parser.exit(1, f"Оценка прервана: {exc}\nУспешные строки сохранены в {args.output}.\n")

    refused = [row for row in rows if row["status"] == "insufficient_evidence"]
    correct_refusals = sum(not row["relevant_section_ids"] for row in refused)
    false_refusals = len(refused) - correct_refusals
    unanswerable = sum(not row["relevant_section_ids"] for row in rows)
    print(f"Корректные отказы: {correct_refusals}/{unanswerable}.")
    print(f"Ложные отказы: {false_refusals}/{len(rows) - unanswerable}.")
    print(f"Средняя задержка /ask: {mean(row['latency_ms'] for row in rows):.1f} мс.")
    print(f"Ответы и источники для ручной проверки: {args.output}")


if __name__ == "__main__":
    main()
