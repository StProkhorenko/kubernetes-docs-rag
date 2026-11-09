import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_corpus import build_corpus

ROOT = Path(__file__).resolve().parents[1]


class StageOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = build_corpus(ROOT)

    def test_corpus_is_reproducible_and_has_valid_source_lines(self):
        self.assertEqual(self.chunks, build_corpus(ROOT))
        self.assertEqual(len({chunk["chunk_id"] for chunk in self.chunks}), len(self.chunks))
        self.assertEqual(len({chunk["document_id"] for chunk in self.chunks}), 10)
        for chunk in self.chunks:
            raw_path = ROOT / "data/raw" / f"{chunk['document_id']}.md"
            line_count = len(raw_path.read_text(encoding="utf-8").splitlines())
            start, end = chunk["line_start"], chunk["line_end"]
            self.assertTrue(1 <= start <= end <= line_count, chunk)
            self.assertTrue(chunk["source_url"].endswith(f"#L{start}-L{end}"))
            self.assertIn(chunk["version"], chunk["source_url"])
            self.assertTrue(chunk["text"].strip())

    def test_cron_code_comments_do_not_create_sections(self):
        cron_chunks = [chunk for chunk in self.chunks if chunk["document_id"] == "cronjobs"]
        schedule = next(chunk for chunk in cron_chunks if "# ┌" in chunk["text"])
        self.assertEqual(schedule["section_id"], "cronjobs:43")
        self.assertIn("# * * * * *", schedule["text"])

    def test_question_splits_and_relevance_labels(self):
        section_ids = {chunk["section_id"] for chunk in self.chunks}
        groups = {}
        question_ids = set()
        questions = set()
        for split, total, unanswered in [("dev", 34, 9), ("test", 26, 6)]:
            path = ROOT / "evaluation" / f"{split}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), total)
            counts = Counter(row["type"] for row in rows)
            self.assertEqual(counts["unanswerable"], unanswered)
            self.assertTrue(counts["exact"] and counts["semantic"])
            self.assertEqual(set(counts), {"exact", "semantic", "unanswerable"})
            for row in rows:
                with self.subTest(question=row["id"]):
                    self.assertNotIn(row["id"], question_ids)
                    question_ids.add(row["id"])
                    normalized = " ".join(row["question"].casefold().split())
                    self.assertTrue(normalized)
                    self.assertNotIn(normalized, questions)
                    questions.add(normalized)
                    self.assertEqual(groups.setdefault(row["group"], split), split)
                    labels = set(row["relevant_section_ids"])
                    if row["type"] == "unanswerable":
                        self.assertFalse(labels)
                        self.assertIsNone(row["reference_answer"])
                    else:
                        self.assertTrue(labels)
                        self.assertTrue(labels <= section_ids, labels - section_ids)
                        self.assertTrue(row["reference_answer"])


if __name__ == "__main__":
    unittest.main()
