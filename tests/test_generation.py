from teleassist.generation.prompts import ABSTENTION, build_messages
from teleassist.generation.rag import RAGAnswerer


class MockLLM:
    def __init__(self, answer):
        self.answer_text = answer
        self.messages = None

    def generate(self, messages, *, use_lora=False):
        self.messages = messages
        return self.answer_text


class MockRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, question, *, mode, top_k):
        self.calls.append((question, mode, top_k))
        return self.chunks[:top_k]


def chunk(identifier="chunk-1"):
    return {
        "id": identifier,
        "text": "A subscriber may port their number after 90 days.",
        "metadata": {"doc_title": "Airtel Charter", "year": 2024, "family": "AIRTEL"},
    }


def make_answerer(answer, tmp_path, *, retrieved=None):
    config = tmp_path / "config.yaml"
    config.write_text("models:\n  generator: test/mock\n", encoding="utf-8")
    llm = MockLLM(answer)
    retriever = MockRetriever(retrieved or [chunk()])
    return RAGAnswerer(retriever, llm, config_path=str(config)), retriever, llm


def test_prompt_numbers_passages_with_title_and_year():
    messages = build_messages("How long?", [chunk(), chunk("chunk-2")])

    assert messages[0]["role"] == "system"
    assert "Answer ONLY from the numbered context passages" in messages[0]["content"]
    assert "cite each factual" in messages[0]["content"].lower()
    assert "[1] Airtel Charter (2024)" in messages[1]["content"]
    assert "[2] Airtel Charter (2024)" in messages[1]["content"]
    assert "Question: How long?" in messages[1]["content"]


def test_valid_citation_maps_to_chunk_and_returns_latency(tmp_path):
    answerer, retriever, llm = make_answerer("It takes 90 days [1].", tmp_path)

    result = answerer.answer("How long?", mode="bm25")

    assert result["citations"] == [
        {
            "id": "chunk-1",
            "doc_title": "Airtel Charter",
            "year": 2024,
            "family": "AIRTEL",
            "snippet": "A subscriber may port their number after 90 days.",
        }
    ]
    assert result["retrieved_chunks"][0]["id"] == "chunk-1"
    assert retriever.calls == [("How long?", "bm25", 5)]
    assert llm.messages[1]["content"].startswith("Context passages:")
    assert set(result["latency_ms"]) == {"retrieval", "generation", "total"}
    assert result["latency_ms"]["total"] == round(
        result["latency_ms"]["retrieval"] + result["latency_ms"]["generation"], 3
    )
    assert result["abstained"] is False


def test_invalid_citation_number_is_reported(tmp_path):
    answerer, _, _ = make_answerer("See [3].", tmp_path)

    result = answerer.answer("Question")

    assert result["citations"] == []
    assert result["invalid_citation_numbers"] == [3]


def test_answer_without_citation_has_empty_citations(tmp_path):
    answerer, _, _ = make_answerer("A concise answer.", tmp_path)

    result = answerer.answer("Question")

    assert result["citations"] == []
    assert result["invalid_citation_numbers"] == []


def test_exact_refusal_marks_abstention(tmp_path):
    answerer, _, _ = make_answerer(ABSTENTION, tmp_path)

    result = answerer.answer("Question")

    assert result["abstained"] is True
    assert result["answer"] == ABSTENTION


def test_no_retrieval_baseline_skips_retriever(tmp_path):
    answerer, retriever, llm = make_answerer("A general answer.", tmp_path)

    result = answerer.answer("Question", mode="no_retrieval")

    assert retriever.calls == []
    assert result["retrieved_chunks"] == []
    assert result["latency_ms"]["retrieval"] == 0
    assert "no context" not in llm.messages[1]["content"].lower()
    assert "provided documents" not in llm.messages[0]["content"].lower()
