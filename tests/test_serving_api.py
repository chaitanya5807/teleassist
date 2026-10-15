from fastapi.testclient import TestClient

from teleassist.serving.api import Service, Settings, create_app

CHUNK = {
    "id": "c1",
    "text": "Porting takes three working days.",
    "metadata": {"doc_title": "Test Doc", "year": 2024, "family": "MNP", "source": "manual/test.pdf"},
}


class FakeRetriever:
    def search(self, question, mode="hybrid_rerank", top_k=5):
        return [CHUNK]


class FakeRag:
    def answer(self, question, mode, use_lora=False):
        return {
            "answer": "Three working days. [1]",
            "citations": [{"id": "c1", "doc_title": "Test Doc", "year": 2024, "family": "MNP", "snippet": "x"}],
            "retrieved_chunks": [CHUNK],
            "invalid_citation_numbers": [],
            "abstained": False,
            "latency_ms": {"retrieval": 1.0, "generation": 2.0, "total": 3.0},
        }


def make_client(adapter="nonexistent/adapter"):
    settings = Settings()
    settings.adapter = adapter
    service = Service(settings, retriever=FakeRetriever(), rag=FakeRag(), chunks=[CHUNK])
    return TestClient(create_app(service))


def test_health_and_sources():
    with make_client() as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["chunks"] == 1
        assert client.get("/sources/c1").json()["text"].startswith("Porting")
        assert client.get("/sources/nope").status_code == 404


def test_retrieval_only():
    with make_client() as client:
        body = client.post("/ask", json={"question": "How long does porting take?", "generate": False}).json()
        assert body["answer"] is None and body["generated"] is False
        assert body["retrieved"][0]["rank"] == 1


def test_generate_with_citation():
    with make_client() as client:
        body = client.post(
            "/ask", json={"question": "How long does porting take?", "use_lora": False}
        ).json()
        assert body["answer"].endswith("[1]") and body["citations"][0]["id"] == "c1"


def test_validation_and_missing_adapter():
    with make_client() as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
        assert client.post("/ask", json={"question": "How long does porting take?"}).status_code == 400
