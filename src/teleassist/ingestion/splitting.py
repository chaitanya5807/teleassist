"""Deterministic train/eval splits that keep related document families together."""

from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REGULATION_FAMILIES = frozenset({"TCCCPR", "MNP", "TCPR", "QOS", "COMPLAINT"})
OPERATOR_FAMILIES = frozenset({"AIRTEL", "JIO"})


def split_corpus_families(
    chunks: list[dict[str, Any]],
    *,
    seed: int = 42,
    eval_fraction: float = 0.2,
    eval_family_names: set[str] | None = None,
) -> dict[str, Any]:
    """Choose eval families, or preserve a supplied family assignment."""
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")

    source_docs: dict[str, dict[str, Any]] = {}
    family_docs: dict[str, dict[str, Any]] = defaultdict(dict)
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = str(metadata.get("source", "")).strip()
        if not source:
            continue
        family = str(metadata.get("family", "")).strip()
        if not family:
            raise ValueError(f"Chunk source {source!r} has no family metadata")
        document = source_docs.setdefault(
            source,
            {
                "source": source,
                "source_type": metadata.get("source_type", "unknown"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "doc_title": metadata.get("doc_title", source),
                "year": None
                if metadata.get("source_type") == "wikipedia"
                else metadata.get("year"),
                "family": family,
                "chunk_count": 0,
            },
        )
        if document["family"] != family:
            raise ValueError(f"Document {source!r} has multiple family values")
        document["chunk_count"] += 1
        family_docs[family][source] = document

    if not source_docs:
        raise ValueError("Cannot split a corpus with no chunked documents")

    families: dict[str, dict[str, Any]] = {}
    for family, sources in family_docs.items():
        documents = sorted(sources.values(), key=lambda document: document["source"])
        families[family] = {
            "family": family,
            "chunks": sum(document["chunk_count"] for document in documents),
            "documents": documents,
            "wikipedia_sources": {
                document["source"]
                for document in documents
                if document["source_type"] == "wikipedia"
            },
            "wikipedia_chunks": sum(
                document["chunk_count"]
                for document in documents
                if document["source_type"] == "wikipedia"
            ),
        }

    total_chunks = sum(document["chunk_count"] for document in source_docs.values())
    present = set(families)
    mandatory_regulations = {family for family in ("MNP", "TCPR") if family in present}
    if len(mandatory_regulations) < min(2, len(present & REGULATION_FAMILIES)):
        mandatory_regulations = present & {"MNP", "TCPR"}

    non_wikipedia = sorted(
        family for family, stats in families.items() if not stats["wikipedia_sources"]
    )
    wikipedia = sorted(set(families) - set(non_wikipedia))
    wiki_subsets = [()]
    wiki_subsets.extend((family,) for family in wikipedia)
    wiki_subsets.extend(itertools.combinations(wikipedia, 2))

    rng = random.Random(seed)
    best_score: tuple[float, float, int] | None = None
    if eval_family_names is not None:
        if not eval_family_names:
            raise ValueError("A preserved eval family assignment cannot be empty")
        missing = eval_family_names - present
        if missing:
            raise ValueError(
                "Preserved eval families have no chunks after cleaning: " f"{sorted(missing)}"
            )
        best_eval: tuple[str, ...] | None = tuple(sorted(eval_family_names))
    else:
        best_eval = None
    tied_candidates = 0
    candidate_masks = () if best_eval is not None else range(1 << len(non_wikipedia))
    for mask in candidate_masks:
        selected_base = {family for bit, family in enumerate(non_wikipedia) if mask & (1 << bit)}
        if not mandatory_regulations <= selected_base:
            continue
        if len(selected_base & REGULATION_FAMILIES) < min(3, len(present & REGULATION_FAMILIES)):
            continue
        if not selected_base & OPERATOR_FAMILIES:
            continue
        for wiki_subset in wiki_subsets:
            selected = selected_base | set(wiki_subset)
            wiki_page_count = sum(len(families[family]["wikipedia_sources"]) for family in selected)
            if wiki_page_count > 2:
                continue
            eval_chunks = sum(families[family]["chunks"] for family in selected)
            eval_wikipedia_chunks = sum(families[family]["wikipedia_chunks"] for family in selected)
            if eval_wikipedia_chunks * 100 > eval_chunks * 15:
                continue
            largest_family = max(families[family]["chunks"] for family in selected)
            if largest_family * 100 > eval_chunks * 30:
                continue

            score = (
                abs(eval_chunks - total_chunks * eval_fraction),
                eval_wikipedia_chunks / eval_chunks,
                len(selected),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_eval = tuple(sorted(selected))
                tied_candidates = 1
            elif score == best_score:
                tied_candidates += 1
                if rng.randrange(tied_candidates) == 0:
                    best_eval = tuple(sorted(selected))

    if best_eval is None:
        raise ValueError("No family split satisfies the requested eval constraints")

    eval_family_set = set(best_eval)
    eval_documents = sorted(
        (document for family in best_eval for document in families[family]["documents"]),
        key=lambda document: document["source"],
    )
    train_documents = sorted(
        (
            document
            for family, stats in families.items()
            if family not in eval_family_set
            for document in stats["documents"]
        ),
        key=lambda document: document["source"],
    )
    eval_chunks = sum(families[family]["chunks"] for family in best_eval)
    train_chunks = total_chunks - eval_chunks
    eval_families = [
        {
            "family": family,
            "chunks": families[family]["chunks"],
            "share": families[family]["chunks"] / eval_chunks,
            "wikipedia_pages": len(families[family]["wikipedia_sources"]),
            "wikipedia_chunks": families[family]["wikipedia_chunks"],
            "documents": [document["source"] for document in families[family]["documents"]],
        }
        for family in best_eval
    ]
    return {
        "seed": seed,
        "target_eval_fraction": eval_fraction,
        "total_chunks": total_chunks,
        "eval_chunks": eval_chunks,
        "eval_share": eval_chunks / total_chunks,
        "train_chunks": train_chunks,
        "train_share": train_chunks / total_chunks,
        "eval_families": eval_families,
        "train_families": sorted(present - eval_family_set),
        "train_documents": train_documents,
        "eval_documents": eval_documents,
    }


def write_document_split(path: str | Path, split: dict[str, Any]) -> None:
    """Write the selected family and document lists as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
