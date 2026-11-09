"""Показывает найденные фрагменты и ссылки на источники."""

import argparse
import json

from app.retrieval import CORPUS, DEFAULT_MODE, MODES, QDRANT_URL, SearchEngine, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--document-id")
    parser.add_argument("--version")
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    args = parser.parse_args()
    engine = SearchEngine(read_jsonl(CORPUS), [args.mode], args.qdrant_url)
    hits = engine.search(args.question, args.mode, args.limit, document_id=args.document_id, version=args.version)
    print(json.dumps([{"score": hit.score, **hit.chunk} for hit in hits], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
