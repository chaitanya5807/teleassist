import pytest

from teleassist.evaluation.metrics import evaluate_retrieval


def test_recall_mrr_and_hit_rate() -> None:
    result = evaluate_retrieval(
        [["a", "b", "c"], ["d", "x", "e"]],
        [["b", "c"], ["e", "f"]],
    )
    assert result["recall@5"] == pytest.approx(0.75)
    assert result["mrr@10"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert result["hit_rate@5"] == 1.0
