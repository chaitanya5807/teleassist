"""Behavior checks for token-aware chunks and stable metadata."""

import json
from unittest.mock import Mock

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from teleassist.ingestion.chunking import _tokens, build_chunks, chunk_document
from teleassist.ingestion.download import download_sources


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


def test_manual_text_file_is_chunked_and_manifested(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    manual_dir = raw_dir / "manual"
    manual_dir.mkdir(parents=True)
    (manual_dir / "consumer_guide.txt").write_text(
        "Telecom consumers can contact support about billing and service issues. " * 8,
        encoding="utf-8",
    )
    (manual_dir / "kyc_guide.html").write_text(
        "<html><h1>KYC guide</h1><p>Identity checks help protect mobile accounts. "
        "Keep your account details current with your provider.</p></html>",
        encoding="utf-8",
    )
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=150)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 20 100 Td (PDF telecom manual text.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(manual_dir / "terms.pdf")

    download_sources(raw_dir, titles=(), bundle_path=None)
    chunks = build_chunks(
        tmp_path / "empty_corpus",
        tmp_path / "chunks.jsonl",
        chunk_size=32,
        overlap=4,
        manual_dir=manual_dir,
    )
    manifest = json.loads((raw_dir / "MANIFEST.json").read_text(encoding="utf-8"))

    assert chunks
    assert all(chunk["metadata"]["source_type"] == "manual" for chunk in chunks)
    assert {chunk["metadata"]["source"] for chunk in chunks} == {
        "manual/consumer_guide.txt",
        "manual/kyc_guide.html",
        "manual/terms.pdf",
    }
    assert all(source["source_type"] == "manual" for source in manifest["sources"])
    assert all(source["status"] == "available" for source in manifest["sources"])


def test_wikipedia_downloader_retries_with_retry_after_and_backoff(tmp_path) -> None:
    responses = [
        Mock(status_code=429, headers={"Retry-After": "3"}),
        Mock(status_code=503, headers={}),
        Mock(status_code=200, headers={}),
        Mock(status_code=200, headers={}),
    ]
    responses[2].json.return_value = {
        "query": {"pages": {"1": {"title": "5G", "extract": "Fifth generation mobile networks."}}}
    }
    responses[3].json.return_value = {
        "query": {"pages": {"2": {"title": "4G", "extract": "Fourth generation mobile networks."}}}
    }
    session = Mock()
    session.headers = {}
    session.get.side_effect = responses
    delays: list[float] = []

    records = download_sources(
        tmp_path,
        titles=("5G", "4G"),
        session=session,
        bundle_path=None,
        sleep=delays.append,
        monotonic=lambda: 0.0,
    )

    assert [source["status"] for source in records] == ["downloaded", "downloaded"]
    assert delays == [3.0, 2.0, 1.0]
    assert session.headers["User-Agent"].startswith("TeleAssist/0.1 (student project;")
    assert session.get.call_count == 4


def test_wikipedia_downloader_uses_cached_successful_pages(tmp_path) -> None:
    response = Mock(status_code=200, headers={})
    response.json.return_value = {
        "query": {"pages": {"1": {"title": "5G", "extract": "Cached mobile network article."}}}
    }
    first_session = Mock()
    first_session.headers = {}
    first_session.get.return_value = response
    download_sources(
        tmp_path,
        titles=("5G",),
        session=first_session,
        bundle_path=None,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    second_session = Mock()
    second_session.headers = {}
    records = download_sources(
        tmp_path,
        titles=("5G",),
        session=second_session,
        bundle_path=None,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert records[0]["status"] == "cached"
    second_session.get.assert_not_called()
