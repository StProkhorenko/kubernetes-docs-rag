"""Проверяет первые пять результатов HTTP API на новом наборе Kube Triage."""

import argparse
import json
import math
from statistics import mean, median
from time import perf_counter

import httpx

from app.retrieval import RANKINGS, ROOT, read_jsonl

METHODS = {"bm25": ("bm25", None), "e5": ("qdrant", None)}
METHODS.update({ranking: ("hybrid", ranking) for ranking in RANKINGS})


def fetch_rows(client, questions, mode, ranking):
    params = {"mode": mode, "limit": 5, "ranking": ranking}
    client.post("/search", json={**params, "question": "Что такое Pod?"}).raise_for_status()
    rows = []
    for question in questions:
        started = perf_counter()
        response = client.post("/search", json={**params, "question": question["question"]})
        response.raise_for_status()
        elapsed = (perf_counter() - started) * 1000
        data = response.json()
        sections = [hit["section_id"] for hit in data["results"]]
        expected = question["relevant_section_ids"]
        rank = next((index for index, section in enumerate(sections, 1) if section in expected), None)
        rows.append({
            "id": question["id"], "type": question["type"], "answerable": bool(expected),
            "rank": rank, "coverage": data["coverage"], "latency_ms": round(elapsed, 2),
            "duplicate_sections": len(sections) != len(set(sections)),
        })
    return rows


def summarize(rows):
    positives = [row for row in rows if row["answerable"]]
    latency = sorted(row["latency_ms"] for row in rows)
    return {
        "hits_at_5": sum(row["rank"] is not None for row in positives),
        "answerable": len(positives),
        "mrr_at_5": round(mean(1 / row["rank"] if row["rank"] else 0 for row in positives), 4),
        "duplicates": sum(row["duplicate_sections"] for row in rows),
        "p50_ms": round(median(latency), 2),
        "p95_ms": latency[math.ceil(.95 * len(latency)) - 1],
    }


def abstention(rows, threshold):
    positive = [row for row in rows if row["answerable"]]
    negative = [row for row in rows if not row["answerable"]]
    return {
        "threshold": threshold,
        "correct_abstentions": sum(row["coverage"] < threshold for row in negative),
        "unanswerable": len(negative),
        "false_abstentions": sum(row["coverage"] < threshold for row in positive),
        "answerable": len(positive),
    }


def threshold_on_dev(rows):
    options = [abstention(rows, point / 100) for point in range(10, 96, 5)]

    def quality(result):
        specificity = result["correct_abstentions"] / result["unanswerable"]
        sensitivity = 1 - result["false_abstentions"] / result["answerable"]
        return (specificity + sensitivity, sensitivity, -result["threshold"])

    return max(options, key=quality)["threshold"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--selected", choices=RANKINGS)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.split == "test" and (args.selected is None or args.threshold is None):
        parser.error("Для test заранее укажи --selected и --threshold, выбранные на dev.")
    questions = read_jsonl(ROOT / "evaluation" / f"triage-{args.split}.jsonl")
    with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False) as client:
        rows = {name: fetch_rows(client, questions, mode, ranking) for name, (mode, ranking) in METHODS.items()}
    summaries = {name: summarize(results) for name, results in rows.items()}
    selected = args.selected or max(RANKINGS, key=lambda name: (
        summaries[name]["hits_at_5"], summaries[name]["mrr_at_5"], -summaries[name]["duplicates"]
    ))
    threshold = args.threshold if args.threshold is not None else threshold_on_dev(rows[selected])
    report = {"split": args.split, "rankings": summaries, "selected": selected,
              "abstention": abstention(rows[selected], threshold), "rows": rows}
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
