"""HTTP API поиска и ответов по корпусу Kubernetes."""

import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, StringConstraints
from qdrant_client.http.exceptions import ApiException

from app.generation import GenerationError, YandexGenerator
from app.retrieval import CORPUS, DEFAULT_MODE, MODES, QDRANT_URL, ROOT, SearchEngine, read_jsonl

Mode = Literal["bm25", "numpy", "qdrant", "hybrid"]


@dataclass(frozen=True)
class Settings:
    modes: tuple[str, ...] = ("bm25", "qdrant", "hybrid")
    qdrant_url: str = QDRANT_URL
    llm_api_key: str = ""
    llm_folder_id: str = ""
    llm_model: str = "yandexgpt/rc"
    llm_timeout: float = 20.0

    @property
    def default_mode(self):
        if DEFAULT_MODE in self.modes:
            return DEFAULT_MODE
        return "qdrant" if "qdrant" in self.modes else self.modes[0]

    @classmethod
    def from_env(cls):
        values = {**dotenv_values(ROOT / ".env"), **os.environ}
        modes = tuple(dict.fromkeys(
            value.strip() for value in (values.get("SEARCH_MODES") or "bm25,qdrant,hybrid").split(",")
            if value.strip()
        ))
        if not modes or any(mode not in MODES for mode in modes):
            raise ValueError("SEARCH_MODES должен содержать один или несколько поддерживаемых режимов.")
        timeout = float(values.get("LLM_TIMEOUT_SECONDS") or "20")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS должен быть положительным числом.")
        return cls(
            modes=modes,
            qdrant_url=values.get("QDRANT_URL") or QDRANT_URL,
            llm_api_key=(values.get("LLM_API_KEY") or "").strip(),
            llm_folder_id=(values.get("LLM_FOLDER_ID") or "").strip(),
            llm_model=(values.get("LLM_MODEL") or "yandexgpt/rc").strip(),
            llm_timeout=timeout,
        )


class Query(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)] = Field(
        examples=["Что означает ImagePullBackOff?"]
    )
    mode: Mode | None = None
    limit: int = Field(default=5, ge=1, le=10)
    document_id: str | None = None
    version: str | None = None


class Source(BaseModel):
    chunk_id: str
    section_id: str
    document_id: str
    heading: str
    source_url: str
    text: str


class SearchResult(Source):
    score: float


class SearchResponse(BaseModel):
    mode: Mode
    results: list[SearchResult]


class AskResponse(BaseModel):
    mode: Mode
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_ready: bool
    llm_configured: bool
    default_mode: Mode
    search_modes: list[Mode]


def load_engine(settings):
    if not CORPUS.is_file():
        raise RuntimeError("Корпус не найден. Сначала запусти python scripts/build_corpus.py.")
    return SearchEngine(read_jsonl(CORPUS), settings.modes, settings.qdrant_url)


def api_error(status_code, code, message):
    return HTTPException(status_code, detail={"code": code, "message": message})


def get_mode(query, settings):
    mode = query.mode or settings.default_mode
    if mode not in settings.modes:
        raise api_error(422, "mode_disabled", "Режим поиска не включён в настройках сервиса.")
    return mode


def search_hits(request, query, mode):
    try:
        return request.app.state.engine.search(
            query.question, mode, query.limit, document_id=query.document_id, version=query.version
        )
    except ApiException as exc:
        raise api_error(503, "index_unavailable", "Векторный индекс недоступен.") from exc


def create_app(settings=None, *, engine_factory=load_engine, llm_transport=None):
    @asynccontextmanager
    async def lifespan(app):
        config = settings or Settings.from_env()
        app.state.config = config
        app.state.engine = engine_factory(config)
        try:
            with httpx.Client(timeout=config.llm_timeout, transport=llm_transport) as client:
                app.state.generator = (
                    YandexGenerator(client, config.llm_api_key, config.llm_folder_id, config.llm_model)
                    if config.llm_api_key and config.llm_folder_id else None
                )
                yield
        finally:
            app.state.engine.close()

    app = FastAPI(title="Kubernetes Docs RAG", version="0.3.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    def health(request: Request, response: Response):
        """Проверить готовность индекса и наличие настроек LLM."""
        config = request.app.state.config
        try:
            index_ready = request.app.state.engine.is_ready()
        except ApiException:
            index_ready = False
        if not index_ready:
            response.status_code = 503
        return HealthResponse(
            status="ok" if index_ready else "degraded",
            index_ready=index_ready,
            llm_configured=request.app.state.generator is not None,
            default_mode=config.default_mode,
            search_modes=list(config.modes),
        )

    @app.post("/search", response_model=SearchResponse, responses={503: {"description": "Индекс недоступен"}})
    def search(query: Query, request: Request):
        """Найти фрагменты документации и ссылки на оригинал."""
        mode = get_mode(query, request.app.state.config)
        hits = search_hits(request, query, mode)
        return SearchResponse(mode=mode, results=[SearchResult(score=hit.score, **hit.chunk) for hit in hits])

    @app.post("/ask", response_model=AskResponse, responses={
        502: {"description": "Ошибка LLM или некорректные ссылки"},
        503: {"description": "LLM не настроена или индекс недоступен"},
        504: {"description": "Таймаут LLM"},
    })
    def ask(query: Query, request: Request):
        """Ответить по найденным фрагментам или сообщить о недостатке данных."""
        mode = get_mode(query, request.app.state.config)
        generator = request.app.state.generator
        if generator is None:
            raise api_error(503, "llm_not_configured", "Заполни LLM_API_KEY и LLM_FOLDER_ID.")
        hits = search_hits(request, query, mode)
        try:
            result = generator.answer(query.question, hits)
        except GenerationError as exc:
            raise api_error(exc.status_code, exc.code, str(exc)) from exc
        return AskResponse(
            mode=mode,
            status=result.status,
            answer=result.answer,
            sources=[Source.model_validate(hit.chunk) for hit in result.sources],
        )

    return app


app = create_app()
