"""Reciprocal rank fusion for combining ranked retrieval results."""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]], k: int = 60, top_k: int = 20
) -> list[dict[str, Any]]:
    """Fuse ranked lists using RRF score 1 / (k + rank), with ranks starting at one."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    for results in ranked_lists:
        seen: set[str] = set()
        rank = 0
        for result in results:
            chunk_id = str(result.get("id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            records[chunk_id] = result
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [{**records[chunk_id], "score": scores[chunk_id]} for chunk_id in ordered]
