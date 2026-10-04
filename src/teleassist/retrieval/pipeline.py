"""Retriever facade and command line entry points for index and search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from teleassist.config import load_config
from teleassist.retrieval.bm25 import BM25Index
from teleassist.retrieval.dense import DenseIndex, build_dense_index
from teleassist.retrieval.fusion import reciprocal_rank_fusion
from teleassist.retrieval.reranker import CrossEncoderReranker


class Retriever:
    """Search all supplied chunks using BM25, dense, hybrid, or reranked hybrid mode."""

    MODES = {"bm25", "dense", "hybrid", "hybrid_rerank"}

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        dense_index: DenseIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        dense_index_path: str | Path = "data/processed/dense_index.npz",
        config_path: str | Path = "configs/default.yaml",
    ):
        self.chunks = chunks
        self.chunks_by_id = {str(chunk["id"]): chunk for chunk in chunks}
        self.bm25 = BM25Index(chunks)
        self.dense_index = dense_index
        self.reranker = reranker
        self.dense_index_path = Path(dense_index_path)
        self.config = load_config(config_path)

    @classmethod
    def from_files(
        cls,
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        *,
        dense_index_path: str | Path = "data/processed/dense_index.npz",
        config_path: str | Path = "configs/default.yaml",
    ) -> Retriever:
        chunks = [
            json.loads(line)
            for line in Path(chunks_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dense_index = DenseIndex.load(dense_index_path) if Path(dense_index_path).exists() else None
        return cls(
            chunks,
            dense_index=dense_index,
            dense_index_path=dense_index_path,
            config_path=config_path,
        )

    def _dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.dense_index is None:
            if not self.dense_index_path.exists():
                raise FileNotFoundError(
                    f"Dense index not found at {self.dense_index_path}; run build-index first"
                )
            self.dense_index = DenseIndex.load(self.dense_index_path)
        return self.dense_index.search(query, self.chunks_by_id, top_k=top_k)

    def search(
        self, query: str, *, mode: str = "hybrid_rerank", top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return up to top_k chunk records, with retrieval scores attached."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {sorted(self.MODES)}")
        retrieval = self.config.retrieval
        if mode == "bm25":
            return self.bm25.search(query, top_k=top_k)
        if mode == "dense":
            return self._dense_search(query, top_k)

        candidates = reciprocal_rank_fusion(
            self.bm25.search(query, top_k=retrieval.candidate_k),
            self._dense_search(query, retrieval.candidate_k),
            k=retrieval.rrf_k,
            top_k=20,
        )
        if mode == "hybrid":
            return candidates[:top_k]
        if self.reranker is None:
            self.reranker = CrossEncoderReranker(self.config.models.reranker)
        return self.reranker.rerank(query, candidates, top_k=top_k)


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-index", help="embed every corpus chunk and save the dense index"
    )
    build.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    build.add_argument("--index", type=Path, default=Path("data/processed/dense_index.npz"))
    build.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    search = subparsers.add_parser("search", help="search the full corpus")
    search.add_argument("query")
    search.add_argument("--mode", choices=sorted(Retriever.MODES), default="hybrid_rerank")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    search.add_argument("--index", type=Path, default=Path("data/processed/dense_index.npz"))
    search.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config = load_config(args.config)
    if args.command == "build-index":
        result = build_dense_index(args.chunks, args.index, model_name=config.models.embedding)
        print(f"Indexed {len(result.chunk_ids)} chunks to {args.index}")
        return
    retriever = Retriever.from_files(
        args.chunks, dense_index_path=args.index, config_path=args.config
    )
    results = retriever.search(args.query, mode=args.mode, top_k=args.top_k)
    for rank, chunk in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        snippet = " ".join(str(chunk.get("text", "")).split())[:200]
        print(
            f"{rank}. {metadata.get('doc_title', 'Unknown')} | year={metadata.get('year')} "
            f"| family={metadata.get('family')} | score={chunk.get('score', 0):.4f}\n{snippet}\n"
        )


if __name__ == "__main__":
    main()
