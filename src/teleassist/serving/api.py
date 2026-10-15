"""FastAPI service: retrieval-only and RAG answers with cited sources."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("teleassist.api")

Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class Settings:
    """Paths come from environment variables so the service runs anywhere."""

    def __init__(self) -> None:
        self.config = os.environ.get("TELEASSIST_CONFIG", "configs/default.yaml")
        self.chunks = os.environ.get("TELEASSIST_CHUNKS", "data/processed/chunks_official.jsonl")
        self.index = os.environ.get("TELEASSIST_INDEX", "data/processed/dense_index_official.npz")
        self.adapter = os.environ.get("TELEASSIST_ADAPTER", "models/teleassist-lora")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Mode = "hybrid_rerank"
    generate: bool = True
    use_lora: bool = True


class AskResponse(BaseModel):
    question: str
    mode: str
    generated: bool
    used_lora: bool
    answer: str | None
    abstained: bool
    citations: list[dict[str, Any]]
    invalid_citation_numbers: list[int]
    retrieved: list[dict[str, Any]]
    latency_ms: dict[str, float]


def public_chunk(chunk: dict[str, Any], rank: int | None = None, chars: int = 300) -> dict[str, Any]:
    meta = chunk.get("metadata", {})
    text = " ".join(str(chunk.get("text", "")).split())
    out = {
        "id": str(chunk.get("id")),
        "doc_title": meta.get("doc_title"),
        "year": meta.get("year"),
        "family": meta.get("family"),
        "source": meta.get("source"),
        "snippet": text[:chars],
    }
    if rank is not None:
        out["rank"] = rank
    return out


class Service:
    """Holds the retriever, the lazily created answerer and the chunk table."""

    def __init__(
        self,
        settings: Settings,
        retriever: Any | None = None,
        rag: Any | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.rag = rag
        self._chunks = chunks
        self.chunk_map: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    @property
    def adapter_available(self) -> bool:
        return Path(self.settings.adapter).exists()

    def load(self) -> None:
        if self._chunks is None:
            with open(self.settings.chunks, encoding="utf-8") as fh:
                self._chunks = [json.loads(line) for line in fh if line.strip()]
        self.chunk_map = {str(c["id"]): c for c in self._chunks}
        if self.retriever is None:
            from teleassist.retrieval.pipeline import Retriever

            self.retriever = Retriever.from_files(
                self.settings.chunks,
                dense_index_path=self.settings.index,
                config_path=self.settings.config,
            )
        logger.info("loaded %d chunks", len(self.chunk_map))

    def get_rag(self) -> Any:
        if self.rag is None:
            from teleassist.generation.rag import RAGAnswerer

            self.rag = RAGAnswerer(
                retriever=self.retriever,
                config_path=self.settings.config,
                chunks_path=self.settings.chunks,
                index_path=self.settings.index,
                adapter_path=self.settings.adapter if self.adapter_available else None,
            )
        return self.rag


def create_app(service: Service | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = service or Service(Settings())
        svc.load()
        app.state.service = svc
        yield

    app = FastAPI(title="TeleAssist", version="0.1.0", lifespan=lifespan)

    def svc_of(request: Request) -> Service:
        return request.app.state.service

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        svc = svc_of(request)
        return {
            "status": "ok",
            "chunks": len(svc.chunk_map),
            "adapter_available": svc.adapter_available,
            "modes": ["bm25", "dense", "hybrid", "hybrid_rerank"],
        }

    @app.get("/sources/{chunk_id}")
    def source(chunk_id: str, request: Request) -> dict[str, Any]:
        chunk = svc_of(request).chunk_map.get(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Unknown chunk id")
        out = public_chunk(chunk, chars=10_000)
        out["text"] = chunk.get("text", "")
        return out

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request) -> AskResponse:
        svc = svc_of(request)
        if req.generate and req.use_lora and not svc.adapter_available:
            raise HTTPException(
                status_code=400,
                detail=f"LoRA adapter not found at {svc.settings.adapter}; send use_lora=false.",
            )
        try:
            with svc.lock:
                if not req.generate:
                    start = time.perf_counter()
                    chunks = svc.retriever.search(req.question, mode=req.mode, top_k=5)
                    ms = (time.perf_counter() - start) * 1000
                    return AskResponse(
                        question=req.question,
                        mode=req.mode,
                        generated=False,
                        used_lora=False,
                        answer=None,
                        abstained=False,
                        citations=[],
                        invalid_citation_numbers=[],
                        retrieved=[public_chunk(c, i + 1) for i, c in enumerate(chunks)],
                        latency_ms={"retrieval": round(ms, 1), "generation": 0.0, "total": round(ms, 1)},
                    )
                result = svc.get_rag().answer(req.question, mode=req.mode, use_lora=req.use_lora)
        except HTTPException:
            raise
        except Exception as exc:  # surfaced to the client instead of a bare 500
            logger.exception("ask failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return AskResponse(
            question=req.question,
            mode=req.mode,
            generated=True,
            used_lora=req.use_lora,
            answer=result["answer"],
            abstained=bool(result["abstained"]),
            citations=result["citations"],
            invalid_citation_numbers=result["invalid_citation_numbers"],
            retrieved=[public_chunk(c, i + 1) for i, c in enumerate(result["retrieved_chunks"])],
            latency_ms={k: float(v) for k, v in result["latency_ms"].items()},
        )

    return app


app = create_app()
