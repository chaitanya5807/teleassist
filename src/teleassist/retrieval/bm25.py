"""A small BM25 implementation that ranks the existing chunk records."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Return lowercase word and number tokens."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """BM25Okapi-style scoring over chunk text, implemented without a BM25 package."""

    def __init__(self, chunks: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be between zero and one")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(str(chunk.get("text", ""))) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if chunks else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_frequencies: Counter[str] = Counter(
            term for terms in self.term_frequencies for term in terms
        )

    def score(self, query: str, document_index: int) -> float:
        """Calculate one chunk's BM25 score for a query."""
        if not 0 <= document_index < len(self.chunks):
            raise IndexError(document_index)
        if not self.chunks or not self.average_length:
            return 0.0
        score = 0.0
        frequencies = self.term_frequencies[document_index]
        length = self.lengths[document_index]
        for term in tokenize(query):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = self.document_frequencies[term]
            idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
            score += idf * frequency * (self.k1 + 1) / denominator
        return score

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the highest scoring chunks with a numeric score field."""
        if top_k <= 0:
            return []
        scored = [(self.score(query, index), index) for index in range(len(self.chunks))]
        scored.sort(key=lambda pair: (-pair[0], str(self.chunks[pair[1]].get("id", ""))))
        return [
            {**self.chunks[index], "score": score} for score, index in scored[:top_k] if score > 0
        ]
