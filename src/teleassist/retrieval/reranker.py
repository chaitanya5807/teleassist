"""Batched cross-encoder scoring for a small candidate list."""

from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    """Rerank query/chunk pairs with the configured sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 16,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], *, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Score candidate text in batches and return the strongest results first."""
        if not candidates or top_k <= 0:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        pairs = [(query, str(candidate.get("text", ""))) for candidate in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: -float(pair[1]),
        )[:top_k]
        return [{**candidate, "score": float(score)} for candidate, score in ordered]
