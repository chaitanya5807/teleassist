"""Normalized embedding retrieval backed by a NumPy flat cosine index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


def _load_encoder(model_name: str) -> Encoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class DenseIndex:
    """Exact cosine search over normalized vectors, persisted as a compressed NPZ."""

    def __init__(
        self,
        vectors: np.ndarray,
        chunk_ids: list[str],
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        encoder: Encoder | None = None,
    ):
        normalized = _normalize(vectors)
        if len(normalized) != len(chunk_ids):
            raise ValueError("Each vector must have exactly one chunk ID")
        self.vectors = normalized
        self.chunk_ids = list(chunk_ids)
        self.model_name = model_name
        self._encoder = encoder

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        encoder: Encoder | None = None,
    ) -> DenseIndex:
        """Encode every chunk in the supplied corpus and make an in-memory index."""
        model = encoder or _load_encoder(model_name)
        vectors = model.encode(
            [str(chunk.get("text", "")) for chunk in chunks],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return cls(
            np.asarray(vectors, dtype=np.float32),
            [str(chunk["id"]) for chunk in chunks],
            model_name=model_name,
            encoder=model,
        )

    def save(self, path: str | Path) -> Path:
        """Save vectors, IDs, and model name in a single compressed file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            vectors=self.vectors,
            chunk_ids=np.asarray(self.chunk_ids, dtype=np.str_),
            model_name=np.asarray(self.model_name, dtype=np.str_),
        )
        return output

    @classmethod
    def load(cls, path: str | Path, *, encoder: Encoder | None = None) -> DenseIndex:
        """Load the saved flat index without loading the embedding model yet."""
        with np.load(Path(path), allow_pickle=False) as saved:
            return cls(
                saved["vectors"],
                saved["chunk_ids"].astype(str).tolist(),
                model_name=str(saved["model_name"].item()),
                encoder=encoder,
            )

    def search(
        self, query: str, chunks_by_id: dict[str, dict[str, Any]], *, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Encode an instructed query and return cosine-ranked chunks."""
        if top_k <= 0 or not self.chunk_ids:
            return []
        if self._encoder is None:
            self._encoder = _load_encoder(self.model_name)
        vector = self._encoder.encode(
            [QUERY_PREFIX + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = self.vectors @ _normalize(np.asarray(vector))[0]
        order = np.argsort(-similarities, kind="stable")[:top_k]
        return [
            {**chunks_by_id[self.chunk_ids[index]], "score": float(similarities[index])}
            for index in order
            if self.chunk_ids[index] in chunks_by_id
        ]


def build_dense_index(
    chunks_path: str | Path,
    index_path: str | Path,
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
) -> DenseIndex:
    """Build and save an index from chunks.jsonl."""
    import json

    chunks = [
        json.loads(line)
        for line in Path(chunks_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = DenseIndex.build(chunks, model_name=model_name, batch_size=batch_size)
    index.save(index_path)
    return index
