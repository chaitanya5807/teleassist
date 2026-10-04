import json

import numpy as np

from teleassist.retrieval.dense import DenseIndex
from teleassist.retrieval.pipeline import Retriever
from teleassist.retrieval.reranker import CrossEncoderReranker
from teleassist.run_retrieval_eval import run_eval


class FixedRetriever:
    def search(self, question, *, mode, top_k):
        return [{"id": question, "score": 1.0}][:top_k]


class QueryEncoder:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts])


class FixedReranker:
    def predict(self, pairs, *, batch_size, show_progress_bar):
        return [float("yes" in passage) for _, passage in pairs]


def test_eval_runner_accepts_arbitrary_eval_jsonl_path(tmp_path, monkeypatch) -> None:
    eval_file = tmp_path / "elsewhere" / "my_eval_set.jsonl"
    eval_file.parent.mkdir()
    eval_file.write_text(
        json.dumps({"question": "first", "gold_chunk_ids": ["first"]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        Retriever,
        "from_files",
        classmethod(lambda cls, *args, **kwargs: FixedRetriever()),
    )
    metrics = run_eval(eval_file)
    assert metrics == {"recall@5": 1.0, "mrr@10": 1.0, "hit_rate@5": 1.0}


def test_retriever_bm25_mode_works_without_dense_index() -> None:
    chunks = [
        {"id": "a", "text": "number porting mobile number portability", "metadata": {}},
        {"id": "b", "text": "customer complaints", "metadata": {}},
    ]
    retriever = Retriever(chunks)
    assert retriever.search("number portability", mode="bm25")[0]["id"] == "a"


def test_dense_hybrid_and_reranked_modes() -> None:
    chunks = [
        {"id": "a", "text": "number portability yes", "metadata": {}},
        {"id": "b", "text": "customer complaint no", "metadata": {}},
    ]
    dense = DenseIndex(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ["a", "b"],
        encoder=QueryEncoder(),
    )
    reranker = CrossEncoderReranker(model=FixedReranker())
    retriever = Retriever(chunks, dense_index=dense, reranker=reranker)

    assert retriever.search("number portability", mode="dense")[0]["id"] == "a"
    assert {item["id"] for item in retriever.search("number portability", mode="hybrid")} == {
        "a",
        "b",
    }
    assert retriever.search("question", mode="hybrid_rerank")[0]["id"] == "a"
