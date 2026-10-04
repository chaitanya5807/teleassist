import math

import pytest

from teleassist.retrieval.bm25 import BM25Index


def test_bm25_matches_hand_calculation_on_tiny_corpus() -> None:
    chunks = [
        {"id": "d1", "text": "alpha beta"},
        {"id": "d2", "text": "alpha"},
        {"id": "d3", "text": "beta"},
    ]
    index = BM25Index(chunks, k1=1.5, b=0.75)

    # For query "alpha", N=3 and df(alpha)=2, so IDF=ln(1+(3-2+0.5)/(2+0.5))=ln(1.6).
    # Average length is 4/3. For d1 (f=1, length=2), denominator is
    # 1 + 1.5*(0.25 + 0.75*2/(4/3)) = 3.0625 and TF factor is 2.5/3.0625.
    # Thus d1 scores ln(1.6)*2.5/3.0625 ~= 0.38368.
    idf = math.log(1.6)
    expected_d1 = idf * 2.5 / 3.0625
    expected_d2 = idf * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 1 / (4 / 3)))
    assert index.score("alpha", 0) == pytest.approx(expected_d1)
    assert index.score("alpha", 1) == pytest.approx(expected_d2)
    assert index.score("alpha", 2) == 0
    assert [item["id"] for item in index.search("alpha")] == ["d2", "d1"]


def test_bm25_empty_query_has_no_hits() -> None:
    assert BM25Index([{"id": "a", "text": "some words"}]).search("absent") == []
