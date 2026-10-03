"""Behavior checks for token-aware chunks and stable metadata."""

from teleassist.ingestion.chunking import _tokens, chunk_document


def _unit(text: str) -> dict:
    return {"text": text, "metadata": {"source": "demo", "page": 3, "section": "Basics"}}


def test_chunks_have_configured_overlap() -> None:
    chunks = chunk_document([_unit("telecom " * 80)], chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    for left, right in zip(chunks[:-1], chunks[1:], strict=True):
        overlap_chars = max(
            size
            for size in range(1, min(len(left["text"]), len(right["text"])) + 1)
            if left["text"][-size:] == right["text"][:size]
        )
        assert overlap_chars >= 10


def test_every_chunk_respects_size_limit() -> None:
    text = "Mobile networks connect subscribers. " * 40
    chunks = chunk_document([_unit(text)], chunk_size=32, overlap=8)
    assert chunks
    assert all(len(_tokens(chunk["text"])) <= 32 for chunk in chunks)


def test_chunk_ids_are_stable_for_identical_input() -> None:
    text = "A stable sample document. " * 12
    first = chunk_document([_unit(text)], chunk_size=16, overlap=3)
    second = chunk_document([_unit(text)], chunk_size=16, overlap=3)
    assert [item["id"] for item in first] == [item["id"] for item in second]


def test_chunk_metadata_is_preserved() -> None:
    chunks = chunk_document([_unit("A short sample.")], chunk_size=16, overlap=2)
    assert chunks[0]["metadata"] == {"source": "demo", "page": 3, "section": "Basics"}
