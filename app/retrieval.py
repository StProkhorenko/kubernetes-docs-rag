"""Лексический, векторный и гибридный поиск по одному корпусу."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/chunks.jsonl"
QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_MODE = "hybrid"
MODEL = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_FILES = (
    "config.json", "model.safetensors", "modules.json", "sentence_bert_config.json",
    "1_Pooling/config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "sentencepiece.bpe.model",
)
MODES = ("bm25", "numpy", "qdrant", "hybrid")
RANKINGS = ("chunk_rrf", "section_rrf", "rerank")
DEFAULT_RANKING = "section_rrf"
TOKEN = re.compile(r"(?:--)?\w+(?:[.-]\w+)*")
RUSSIAN_WORD = re.compile(r"[а-я]+")
QUERY_STOP_WORDS = set("а и в на из у по до за для или что как какой какая какие можно если когда ли это где чем почему при к с о от его их нужно".split())


@dataclass(frozen=True)
class Hit:
    chunk: dict
    score: float


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text):
    tokens = TOKEN.findall(text.casefold().replace("ё", "е"))
    # Сохраняем точные совпадения для флагов и полей, добавляя отдельные компоненты.
    parts = [part for token in tokens if "." in token or "-" in token
             for part in re.split(r"[.-]+", token) if part]
    # Стеммер меняет своё состояние. Каждый вызов получает отдельный экземпляр.
    stemmer = snowballstemmer.stemmer("russian")
    return [stemmer.stemWord(token) if RUSSIAN_WORD.fullmatch(token) else token for token in tokens + parts]


def passage(chunk):
    headings = " > ".join(chunk["heading_path"])
    return f"passage: {headings}\n{chunk['text']}"


def model_snapshot():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        cached = Path(snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True))
    except LocalEntryNotFoundError:
        pass
    else:
        if all((cached / name).is_file() for name in MODEL_FILES):
            return cached

    downloaded = Path(snapshot_download(MODEL, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES)))
    missing = [name for name in MODEL_FILES if not (downloaded / name).is_file()]
    if missing:
        raise OSError(f"В снимке E5 отсутствуют файлы: {', '.join(missing)}")
    return downloaded


def reciprocal_rank_fusion(rankings, limit, rank_constant=60):
    scores = defaultdict(float)
    chunks = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk["chunk_id"]
            chunks[chunk_id] = hit.chunk
            scores[chunk_id] += 1 / (rank_constant + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [Hit(chunks[key], scores[key]) for key in ordered[:limit]]


def first_per_section(hits):
    sections = {}
    for hit in hits:
        sections.setdefault(hit.chunk["section_id"], hit)
    return list(sections.values())


def section_fusion(rankings, limit, rank_constant=60):
    scores = defaultdict(float)
    representatives = {}
    chunk_order = reciprocal_rank_fusion(rankings, sum(map(len, rankings)), rank_constant)
    for hit in first_per_section(chunk_order):
        representatives[hit.chunk["section_id"]] = hit.chunk
    for ranking in rankings:
        for rank, hit in enumerate(first_per_section(ranking), start=1):
            scores[hit.chunk["section_id"]] += 1 / (rank_constant + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [Hit(representatives[key], scores[key]) for key in ordered[:limit]]


def rerank_sections(hits, lexical, dense):
    # Повторно сортируем кандидатов по сумме обратных позиций.
    # При равенстве выше раздел с лучшей позицией BM25.
    lexical_ranks = {hit.chunk["section_id"]: i for i, hit in enumerate(first_per_section(lexical), 1)}
    dense_ranks = {hit.chunk["section_id"]: i for i, hit in enumerate(first_per_section(dense), 1)}

    def score(hit):
        section = hit.chunk["section_id"]
        return (
            (1 / lexical_ranks[section] if section in lexical_ranks else 0)
            + (1 / dense_ranks[section] if section in dense_ranks else 0)
        )

    ordered = sorted(hits, key=lambda hit: (
        -score(hit), lexical_ranks.get(hit.chunk["section_id"], float("inf"))
    ))
    return [Hit(hit.chunk, score(hit)) for hit in ordered]


QUERY_STOP_STEMS = set(tokenize(" ".join(sorted(QUERY_STOP_WORDS))))


def query_terms(question):
    return {token for token in tokenize(question) if len(token) > 1 and token not in QUERY_STOP_STEMS}


class EvidenceCoverage:
    """Доля взвешенных терминов запроса в одном из возвращённых фрагментов."""

    def __init__(self, chunks):
        self.chunk_terms = {chunk["chunk_id"]: set(tokenize(passage(chunk))) for chunk in chunks}
        self.frequency = Counter(token for terms in self.chunk_terms.values() for token in terms)

    def score(self, question, hits):
        terms = query_terms(question)
        if not hits or not terms:
            return 0.0
        weights = {term: 1 / (1 + self.frequency[term]) ** 0.5 for term in terms}
        return max(
            sum(weights[term] for term in terms & self.chunk_terms[hit.chunk["chunk_id"]])
            for hit in hits
        ) / sum(weights.values())


class QdrantIndex:
    def __init__(self, url, chunks, dimension, encode_passages):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.client = QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url, timeout=15)
        self.chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
        fingerprint = json.dumps(
            [MODEL_REVISION, [passage(chunk) for chunk in chunks], chunks],
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]
        self.collection = f"kubernetes_docs_{digest}"
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        if self.client.count(self.collection, exact=True).count != len(chunks):
            vectors = encode_passages()
            for start in range(0, len(chunks), 64):
                points = [
                    models.PointStruct(
                        id=chunk["chunk_id"],
                        vector=vector.tolist(),
                        payload={key: chunk[key] for key in ("document_id", "section_id", "version", "source_url")},
                    )
                    for chunk, vector in zip(chunks[start:start + 64], vectors[start:start + 64])
                ]
                self.client.upsert(self.collection, points, wait=True)

    def search(self, vector, limit, filters):
        models = self.models
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
        points = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            query_filter=models.Filter(must=conditions) if conditions else None,
            search_params=models.SearchParams(exact=True),
            with_payload=False,
        ).points
        return [Hit(self.chunks[str(point.id)], point.score) for point in points]


class SearchEngine:
    def __init__(self, chunks, modes, qdrant_url=QDRANT_URL):
        if not chunks:
            raise ValueError("Корпус пуст")
        self.chunks = chunks
        self.modes = frozenset(modes)
        self.evidence = EvidenceCoverage(chunks)
        if self.modes & {"bm25", "hybrid"}:
            documents = [tokenize(passage(chunk).removeprefix("passage: ")) for chunk in chunks]
            self.bm25 = BM25Okapi(documents)
        if self.modes & {"numpy", "qdrant", "hybrid"}:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(str(model_snapshot()), device="cpu", local_files_only=True)
        self.vectors = None
        if "numpy" in self.modes:
            self._encode_passages()
        if self.modes & {"qdrant", "hybrid"}:
            self.qdrant = QdrantIndex(
                qdrant_url, chunks, self.encoder.get_embedding_dimension(), self._encode_passages
            )

    def _encode_passages(self):
        if self.vectors is None:
            self.vectors = self.encoder.encode(
                [passage(chunk) for chunk in self.chunks], normalize_embeddings=True, show_progress_bar=False
            )
        return self.vectors

    def is_ready(self):
        if self.modes & {"qdrant", "hybrid"}:
            return self.qdrant.client.count(self.qdrant.collection, exact=True).count == len(self.chunks)
        return True

    def close(self):
        if self.modes & {"qdrant", "hybrid"}:
            self.qdrant.client.close()

    def _rank(self, scores, limit, filters, positive_only=False):
        ordered = np.argsort(-scores, kind="stable")
        hits = []
        for idx in ordered:
            chunk = self.chunks[idx]
            score = float(scores[idx])
            if positive_only and score <= 0:
                break
            if all(chunk[key] == value for key, value in filters.items()):
                hits.append(Hit(chunk, score))
                if len(hits) == limit:
                    break
        return hits

    def search(self, question, mode, limit=5, *, document_id=None, version=None, ranking=DEFAULT_RANKING):
        question = question.strip()
        if not question or limit < 1:
            raise ValueError("Нужен непустой вопрос и положительный лимит")
        if mode not in self.modes:
            raise ValueError(f"Режим {mode} не инициализирован")
        filters = {key: value for key, value in {"document_id": document_id, "version": version}.items()
                   if value is not None}
        if mode == "bm25":
            return self._rank(self.bm25.get_scores(tokenize(question)), limit, filters, positive_only=True)

        vector = self.encoder.encode(f"query: {question}", normalize_embeddings=True, show_progress_bar=False)
        if mode == "numpy":
            return self._rank(self.vectors @ vector, limit, filters)
        if mode == "qdrant":
            return self.qdrant.search(vector, limit, filters)

        candidate_limit = min(len(self.chunks), max(20, limit))
        lexical = self._rank(
            self.bm25.get_scores(tokenize(question)), candidate_limit, filters, positive_only=True
        )
        dense = self.qdrant.search(vector, candidate_limit, filters)
        if ranking == "chunk_rrf":
            return reciprocal_rank_fusion((lexical, dense), limit)
        sections = section_fusion((lexical, dense), candidate_limit * 2)
        if ranking == "rerank":
            sections = rerank_sections(sections, lexical, dense)
        return sections[:limit]
