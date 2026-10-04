"""Deterministic document-level train/eval splits stratified by document type."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def split_corpus_documents(
    chunks: list[dict[str, Any]], *, seed: int = 42, eval_fraction: float = 0.2
) -> dict[str, Any]:
    """Split distinct chunk sources by doc_type, with each type represented in eval."""
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")

    documents: dict[str, dict[str, Any]] = {}
    chunk_counts: Counter[str] = Counter()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = str(metadata.get("source", "")).strip()
        if not source:
            continue
        chunk_counts[source] += 1
        documents.setdefault(
            source,
            {
                "source": source,
                "source_type": metadata.get("source_type", "unknown"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "doc_title": metadata.get("doc_title", source),
                "year": metadata.get("year"),
            },
        )

    total_chunks = sum(chunk_counts.values())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, document in documents.items():
        grouped[str(document["doc_type"])].append({**document, "chunk_count": chunk_counts[source]})

    rng = random.Random(seed)
    train_documents: list[dict[str, Any]] = []
    eval_documents: list[dict[str, Any]] = []
    for doc_type in sorted(grouped):
        group = sorted(grouped[doc_type], key=lambda document: document["source"])
        rng.shuffle(group)
        eval_count = max(1, int(len(group) * eval_fraction + 0.5))
        eval_documents.extend(group[:eval_count])
        train_documents.extend(group[eval_count:])

    if total_chunks and len(eval_documents) == 1:
        high_share_in_eval = eval_documents[0]["chunk_count"] / total_chunks > 0.20
        if high_share_in_eval and train_documents:
            candidate = train_documents.pop(0)
            eval_documents.append(candidate)
        elif high_share_in_eval:
            raise ValueError("Cannot keep a >20% document from being the only eval document")

    return {
        "seed": seed,
        "eval_fraction": eval_fraction,
        "train_documents": sorted(train_documents, key=lambda document: document["source"]),
        "eval_documents": sorted(eval_documents, key=lambda document: document["source"]),
    }


def write_document_split(path: str | Path, split: dict[str, Any]) -> None:
    """Write the selected train and eval document lists as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
