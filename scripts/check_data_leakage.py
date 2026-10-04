"""Check SFT train data against held-out eval sources and questions."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def check_leakage(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    split: dict[str, Any],
) -> float:
    eval_sources = {str(item["source"]) for item in split["eval_documents"]}
    eval_families = {str(item["family"]) for item in split["eval_documents"]}
    train_chunk_ids = {str(chunk["id"]) for chunk in chunks}
    chunks_by_id = {str(chunk["id"]): chunk for chunk in chunks}
    for record in train_records:
        for chunk_id in record.get("gold_chunk_ids", []):
            if str(chunk_id) not in train_chunk_ids:
                raise AssertionError(f"Unknown train gold chunk: {chunk_id}")
            chunk = chunks_by_id[str(chunk_id)]
            metadata = chunk.get("metadata", {})
            if metadata.get("source") in eval_sources or metadata.get("family") in eval_families:
                raise AssertionError(f"Train gold chunk {chunk_id} belongs to eval data")
        for passage in record.get("context", []):
            chunk = chunks_by_id.get(str(passage.get("id")))
            if chunk is None:
                raise AssertionError(f"Unknown train context chunk: {passage.get('id')}")
            metadata = chunk.get("metadata", {})
            if metadata.get("source") in eval_sources or metadata.get("family") in eval_families:
                raise AssertionError(
                    f"Train context chunk {passage.get('id')} belongs to eval data"
                )
    eval_questions = [str(item.get("question", "")) for item in eval_records]
    train_questions = [str(item.get("question", "")) for item in train_records]
    maximum = max(
        (
            SequenceMatcher(None, eval_question.casefold(), train_question.casefold()).ratio()
            for eval_question in eval_questions
            for train_question in train_questions
        ),
        default=0.0,
    )
    print(f"Maximum eval/train question similarity: {maximum:.4f}")
    print(f"Similarity flag (> 0.9): {maximum > 0.9}")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train/train.jsonl"))
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/split.json"))
    args = parser.parse_args()
    check_leakage(
        read_jsonl(args.train),
        read_jsonl(args.eval_set),
        read_jsonl(args.chunks),
        json.loads(args.split.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    main()
