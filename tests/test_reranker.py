from teleassist.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self):
        self.batch_size = None

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.batch_size = batch_size
        assert show_progress_bar is False
        return [len(text) for _, text in pairs]


def test_reranker_batches_and_orders_candidates() -> None:
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model, batch_size=2)
    results = reranker.rerank(
        "question", [{"id": "short", "text": "x"}, {"id": "long", "text": "long text"}]
    )
    assert model.batch_size == 2
    assert [item["id"] for item in results] == ["long", "short"]
