"""Append generated JSONL rows with a chunk progress ledger."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def append_resumable(
    chunks: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    out_dir: Path,
    max_chunks: int | None,
    chunk_ids: Callable[[dict[str, Any]], list[str]],
    filename: str | Callable[[dict[str, Any]], str],
) -> tuple[int, int]:
    """Append records grouped by source chunk and mark each chunk only after flush."""
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    rejects_path = out_dir / "rejects.jsonl"

    def path_for(record: dict[str, Any]) -> Path:
        return out_dir / (filename(record) if callable(filename) else filename)

    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"processed_chunk_ids": []}
    )
    processed = set(progress["processed_chunk_ids"])
    existing = set()
    for output_path in out_dir.glob("*.jsonl"):
        if output_path == rejects_path:
            continue
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.add(json.loads(line).get("id"))
    groups: dict[str, list[dict[str, Any]]] = {str(c["id"]): [] for c in chunks}
    for record in records:
        for identifier in chunk_ids(record):
            if identifier in groups:
                groups[identifier].append(record)
                break
    accepted = rejected = newly = 0
    started = time.monotonic()
    total = len(groups)
    for identifier, group in groups.items():
        if identifier in processed:
            continue
        if max_chunks is not None and newly >= max_chunks:
            break
        newly += 1
        for record in group:
            if record.get("id") in existing:
                continue
            with path_for(record).open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {k: v for k, v in record.items() if k != "_output_file"}, ensure_ascii=False
                    )
                    + "\n"
                )
                stream.flush()
            existing.add(record.get("id"))
            accepted += 1
        if not group:
            with rejects_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"chunk_id": identifier, "reason": "no accepted generated item"})
                    + "\n"
                )
                stream.flush()
            rejected += 1
        processed.add(identifier)
        progress_path.write_text(
            json.dumps({"processed_chunk_ids": sorted(processed)}), encoding="utf-8"
        )
        if newly % 25 == 0:
            elapsed = time.monotonic() - started
            eta = (elapsed / newly) * max(0, total - len(processed))
            print(
                f"chunks={len(processed)}/{total} accepted={accepted} "
                f"rejected={rejected} ETA={eta:.0f}s"
            )
    return accepted, rejected
