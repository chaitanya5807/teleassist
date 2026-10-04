import numpy as np

from teleassist.retrieval.dense import QUERY_PREFIX, DenseIndex


class TinyEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.asarray([[1, 0] if "alpha" in text.lower() else [0, 1] for text in texts])


def test_dense_index_normalizes_saves_loads_and_instructs_query(tmp_path) -> None:
    encoder = TinyEncoder()
    chunks = [
        {"id": "a", "text": "alpha words", "metadata": {}},
        {"id": "b", "text": "beta words", "metadata": {}},
    ]
    index = DenseIndex.build(chunks, encoder=encoder)
    assert np.linalg.norm(index.vectors, axis=1).tolist() == [1.0, 1.0]
    path = tmp_path / "vectors.npz"
    index.save(path)
    loaded = DenseIndex.load(path, encoder=encoder)
    results = loaded.search("alpha query", {item["id"]: item for item in chunks}, top_k=1)
    assert results[0]["id"] == "a"
    assert encoder.calls[-1][0] == [QUERY_PREFIX + "alpha query"]
