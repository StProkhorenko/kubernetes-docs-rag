"""Сравнивает методы поиска на размеченных вопросах."""

import argparse
import json
from statistics import mean
from time import perf_counter
from pathlib import Path

from app.retrieval import CORPUS, MODES, QDRANT_URL, ROOT, SearchEngine, read_jsonl

TOP_K = 5
CANDIDATE_LIMIT = 20


def score_sections(hits, expected):
    sections = list(dict.fromkeys(hit.chunk["section_id"] for hit in hits))[:TOP_K]
    expected = set(expected)
    if not expected:
        return sections, None, None
    recall = len(expected.intersection(sections)) / len(expected)
    reciprocal_rank = next((1 / rank for rank, section in enumerate(sections, 1) if section in expected), 0.0)
    return sections, recall, reciprocal_rank


def evaluate(engine, questions, mode):
    # Прогрев исключает инициализацию библиотек из первого измерения.
    engine.search("Как работают пробы Kubernetes?", mode, CANDIDATE_LIMIT, ranking="chunk_rrf")
    rows = []
    for question in questions:
        start = perf_counter()
        hits = engine.search(question["question"], mode, CANDIDATE_LIMIT, ranking="chunk_rrf")
        elapsed = (perf_counter() - start) * 1000
        sections, recall, reciprocal_rank = score_sections(hits, question["relevant_section_ids"])
        rows.append({
            "id": question["id"],
            "type": question["type"],
            "expected_sections": question["relevant_section_ids"],
            "retrieved_sections": sections,
            "recall_at_5": recall,
            "mrr_at_5": reciprocal_rank,
            "latency_ms": round(elapsed, 3),
        })

    answerable = [row for row in rows if row["recall_at_5"] is not None]
    return {
        "answerable_count": len(answerable),
        "unanswerable_count": len(rows) - len(answerable),
        "recall_at_5": mean(row["recall_at_5"] for row in answerable),
        "mrr_at_5": mean(row["mrr_at_5"] for row in answerable),
        "mean_latency_ms": mean(row["latency_ms"] for row in rows),
        "questions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    questions = read_jsonl(ROOT / "evaluation" / f"{args.split}.jsonl")
    engine = SearchEngine(read_jsonl(CORPUS), args.modes, args.qdrant_url)
    results = {mode: evaluate(engine, questions, mode) for mode in args.modes}

    for mode, result in results.items():
        print(f"{mode:<7} Recall@5 {result['recall_at_5']:.3f}  MRR@5 {result['mrr_at_5']:.3f}"
              f"  mean {result['mean_latency_ms']:.1f} ms")
    output = args.output or (CORPUS.parent / f"{args.split}-search-results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"split": args.split, "candidate_limit": CANDIDATE_LIMIT, "results": results}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
