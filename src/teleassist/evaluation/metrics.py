"""Simple retrieval quality metrics at useful cutoffs."""

from __future__ import annotations

from collections.abc import Iterable


def evaluate_retrieval(
    ranked_ids: Iterable[Iterable[str]],
    gold_ids: Iterable[Iterable[str]],
    *,
    recall_k: int = 5,
) -> dict[str, float]:
    """Compute mean Recall@recall_k, MRR@10, and hit-rate@5 across questions."""
    if recall_k <= 0:
        raise ValueError("recall_k must be positive")
    ranked = [list(ids) for ids in ranked_ids]
    gold = [set(ids) for ids in gold_ids]
    if len(ranked) != len(gold):
        raise ValueError("ranked results and gold IDs must have the same question count")
    if not ranked:
        return {f"recall@{recall_k}": 0.0, "mrr@10": 0.0, "hit_rate@5": 0.0}
    recall = []
    reciprocal_rank = []
    hit_rate = []
    for results, relevant in zip(ranked, gold, strict=True):
        if not relevant:
            recall.append(0.0)
            reciprocal_rank.append(0.0)
            hit_rate.append(0.0)
            continue
        top_recall = results[:recall_k]
        top_five = results[:5]
        recall.append(len(set(top_recall) & relevant) / len(relevant))
        hit_rate.append(float(bool(set(top_five) & relevant)))
        rank = next(
            (index for index, chunk_id in enumerate(results[:10], start=1) if chunk_id in relevant),
            None,
        )
        reciprocal_rank.append(0.0 if rank is None else 1 / rank)
    count = len(ranked)
    return {
        f"recall@{recall_k}": sum(recall) / count,
        "mrr@10": sum(reciprocal_rank) / count,
        "hit_rate@5": sum(hit_rate) / count,
    }
