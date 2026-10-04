"""Review eval questions interactively, saving every decision to the JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )


def review(path: Path, chunks_path: Path, *, input_fn=input, print_fn=print) -> None:
    items = read_jsonl(path)
    chunks = {str(chunk["id"]): chunk for chunk in read_jsonl(chunks_path)}
    position = 0
    while position < len(items):
        item = items[position]
        print_fn(f"\n[{position + 1}/{len(items)}] {item.get('id', '')}")
        print_fn(f"Question: {item.get('question', '')}")
        print_fn(f"Reference answer: {item.get('reference_answer', '')}")
        print_fn(f"Evidence: {item.get('evidence', '')}")
        print_fn("Gold chunk text:")
        for chunk_id in item.get("gold_chunk_ids", []):
            chunk = chunks.get(str(chunk_id))
            text = chunk.get("text", "[missing chunk]") if chunk else "[missing chunk]"
            print_fn(f"[{chunk_id}] {text}")
        choice = input_fn("y=verify, n=delete, e=edit question/answer, s=skip: ").strip().lower()
        if choice == "y":
            item["verified"] = True
            position += 1
        elif choice == "n":
            items.pop(position)
        elif choice == "e":
            field = input_fn("Edit which field? (q)uestion or (a)nswer: ").strip().lower()
            if field in {"q", "question"}:
                item["question"] = input_fn("New question: ").strip()
            elif field in {"a", "answer"}:
                item["reference_answer"] = input_fn("New reference answer: ").strip()
            else:
                print_fn("No edit made; choose q or a next time.")
        elif choice == "s":
            position += 1
        else:
            print_fn("Choose y, n, e, or s.")
        save_jsonl(path, items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    args = parser.parse_args()
    review(args.eval_set, args.chunks)


if __name__ == "__main__":
    main()
