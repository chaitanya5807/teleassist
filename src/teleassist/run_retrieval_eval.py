"""Evaluate retrieval against any JSONL file of questions and gold chunk IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from teleassist.retrieval.metrics import evaluate_retrieval
from teleassist.retrieval.pipeline import Retriever


def run_eval(
    eval_set_path: str | Path,
    *,
    chunks_path: str | Path = "data/processed/chunks.jsonl",
    index_path: str | Path = "data/processed/dense_index.npz",
    config_path: str | Path = "configs/default.yaml",
    mode: str = "hybrid_rerank",
) -> dict[str, float]:
    """Run a named retrieval mode against records with question and gold_chunk_ids."""
    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in Path(eval_set_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retriever = Retriever.from_files(
        chunks_path, dense_index_path=index_path, config_path=config_path
    )
    ranked = [retriever.search(row["question"], mode=mode, top_k=5) for row in rows]
    return evaluate_retrieval(
        [[str(chunk["id"]) for chunk in results] for results in ranked],
        [row["gold_chunk_ids"] for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_set", type=Path, help="JSONL rows: question and gold_chunk_ids")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/dense_index.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--mode", choices=sorted(Retriever.MODES), default="hybrid_rerank")
    args = parser.parse_args()
    metrics = run_eval(
        args.eval_set,
        chunks_path=args.chunks,
        index_path=args.index,
        config_path=args.config,
        mode=args.mode,
    )
    print("SMOKE TEST ONLY, not results")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
