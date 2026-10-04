from teleassist.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_combines_ranks_with_k_60() -> None:
    first = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    second = [{"id": "b", "text": "B"}, {"id": "c", "text": "C"}]
    fused = reciprocal_rank_fusion(first, second, k=60, top_k=3)
    assert [item["id"] for item in fused] == ["b", "a", "c"]
    assert fused[0]["score"] == 1 / 62 + 1 / 61
    assert fused[1]["score"] == 1 / 61
