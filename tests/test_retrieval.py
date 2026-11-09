import unittest

from app.retrieval import CORPUS, Hit, SearchEngine, read_jsonl, reciprocal_rank_fusion, tokenize
from scripts.build_corpus import ROOT, build_corpus
from scripts.evaluate_search import score_sections


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = read_jsonl(CORPUS) if CORPUS.exists() else build_corpus(ROOT)
        cls.engine = SearchEngine(chunks, ["bm25"])

    def test_tokenizer_keeps_exact_terms_and_their_components(self):
        self.assertEqual(
            tokenize("--field-selector spec.restartPolicy"),
            ["--field-selector", "spec.restartpolicy", "field", "selector", "spec", "restartpolicy"],
        )

    def test_tokenizer_matches_russian_inflections(self):
        self.assertEqual(tokenize("контейнер контейнера контейнером Ёлка"), ["контейнер"] * 3 + ["елк"])

    def test_bm25_returns_source_and_respects_filters(self):
        question = "Что означает ImagePullBackOff?"
        hits = self.engine.search(question, "bm25", document_id="images")
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk["section_id"], "images:89")
        self.assertIn("#L", hits[0].chunk["source_url"])
        self.assertEqual(self.engine.search(question, "bm25", version="unknown"), [])

    def test_fusion_promotes_chunks_found_by_both_methods(self):
        a, b, c = [Hit({"chunk_id": key}, 42) for key in ("a", "b", "c")]
        result = reciprocal_rank_fusion(([a, b], [c, b]), 2)
        self.assertEqual([hit.chunk["chunk_id"] for hit in result], ["b", "a"])

    def test_evaluation_collapses_duplicate_sections_and_skips_unanswerable(self):
        hits = [Hit({"section_id": section}, 1) for section in ("a", "a", "b", "c")]
        self.assertEqual(score_sections(hits, ["b", "missing"]), (["a", "b", "c"], 0.5, 0.5))
        self.assertEqual(score_sections(hits, []), (["a", "b", "c"], None, None))


if __name__ == "__main__":
    unittest.main()
