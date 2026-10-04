"""Behavior checks for token-aware chunks and stable metadata."""

import json
from unittest.mock import Mock

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from teleassist.ingestion.chunking import _tokens, build_chunks, chunk_document
from teleassist.ingestion.download import (
    DROPPED_WIKIPEDIA_TITLES,
    WIKIPEDIA_TITLES,
    download_sources,
)
from teleassist.ingestion.parse import (
    english_filter_reason,
    filter_english_units,
    infer_document_metadata,
)
from teleassist.ingestion.splitting import split_corpus_documents


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


def test_chunk_boundaries_do_not_create_unicode_replacement_characters() -> None:
    chunks = chunk_document(
        [_unit("Legal text includes mathematical symbols 𝐍𝐮𝐦𝐛𝐞𝐫 and currency ₹. " * 5)],
        chunk_size=8,
        overlap=2,
    )
    assert chunks
    assert all("\ufffd" not in chunk["text"] for chunk in chunks)


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
    pdf_text = "PDF telecom manual text for subscribers and their account support. " * 8
    content.set_data(f"BT /F1 12 Tf 20 100 Td ({pdf_text}) Tj ET".encode())
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
    assert all(
        "doc_type" in source and "year" in source and "doc_title" in source
        for source in manifest["sources"]
    )
    assert all(
        "doc_type" in chunk["metadata"] and "doc_title" in chunk["metadata"] for chunk in chunks
    )


def test_manual_doc_type_inference_rules() -> None:
    assert (
        infer_document_metadata("trai_consumer_handbook_2018_english.pdf")["doc_type"]
        == "faq_or_guide"
    )
    assert (
        infer_document_metadata("trai_ekyc_recommendations_2017.pdf")["doc_type"]
        == "recommendation"
    )
    assert (
        infer_document_metadata("Consultation_Paper_2024.pdf")["doc_type"]
        == "draft_or_consultation"
    )
    assert (
        infer_document_metadata("trai_mnp_9th_amendment_2024.pdf")["doc_type"] == "final_regulation"
    )
    assert infer_document_metadata("airtel_consumer_charter.pdf")["doc_type"] == "faq_or_guide"
    assert (
        infer_document_metadata("airtel_telecom_consumers_charter.pdf")["doc_type"]
        == "faq_or_guide"
    )
    assert infer_document_metadata("jio_telecom_consumer_chart.pdf")["doc_type"] == "faq_or_guide"
    assert infer_document_metadata("pib_press_release.html")["doc_type"] == "faq_or_guide"
    assert infer_document_metadata("dot_alternate_digital_kyc_2026.pdf")["year"] == 2026
    assert infer_document_metadata("trai_mnp_2009_consolidated_2024.pdf")["year"] == 2024
    assert (
        infer_document_metadata("SIM card", source_type="wikipedia")["doc_type"] == "encyclopedia"
    )


def test_wikipedia_fallback_is_trimmed_but_keeps_at_least_40_titles() -> None:
    assert len(WIKIPEDIA_TITLES) >= 40
    assert DROPPED_WIKIPEDIA_TITLES.isdisjoint(WIKIPEDIA_TITLES)


def test_english_filter_rejects_hindi_garbled_and_keeps_english() -> None:
    hindi = "यह दूरसंचार सेवा ग्राहक के लिए उपलब्ध है।"
    garbled = "The customer must submit the form � before service activation."
    english = "The customer can contact the service provider to update a mobile account."
    gibberish = "zxqv rblp thmkw pqrn vxxz"

    assert english_filter_reason(hindi) == "more than 15% Devanagari letters"
    assert english_filter_reason(garbled) == "replacement character"
    assert english_filter_reason(english) is None
    assert english_filter_reason(gibberish) == "low recognizable-English word share"


def test_english_filter_drops_blocks_and_logs_document_totals(caplog) -> None:
    caplog.set_level("INFO")
    units = [
        _unit("The customer can contact the service provider for help."),
        _unit("यह दूरसंचार सेवा ग्राहक के लिए उपलब्ध है।"),
        _unit("The customer must submit this form � before activation."),
    ]
    kept = filter_english_units(units, document_name="mixed.pdf")

    assert [unit["text"] for unit in kept] == [units[0]["text"]]
    assert "English filter dropped 2 pages/blocks" in caplog.text
    assert "from mixed.pdf" in caplog.text


def test_english_filter_removes_small_devanagari_header_from_english_page() -> None:
    text = (
        "The regulation explains the service quality requirements and reporting process. "
        "Providers must submit their information within the stated period.\n"
        "भारत का राजपत्र : असाधारण\n"
        "The authority will publish the results for consumers and service providers."
    )
    kept = filter_english_units([_unit(text)], document_name="bilingual-page.pdf")

    assert len(kept) == 1
    assert "भारत" not in kept[0]["text"]
    assert "The regulation explains" in kept[0]["text"]


def test_document_split_is_stratified_and_large_document_is_not_only_eval() -> None:
    chunks = []
    for source, doc_type, count in (
        ("manual/large.pdf", "final_regulation", 80),
        ("manual/small.pdf", "final_regulation", 1),
        ("manual/guide.pdf", "faq_or_guide", 4),
        ("wikipedia/SIM card", "encyclopedia", 4),
    ):
        chunks.extend(
            {
                "metadata": {
                    "source": source,
                    "source_type": "manual" if source.startswith("manual/") else "wikipedia",
                    "doc_type": doc_type,
                    "doc_title": source,
                    "year": None,
                }
            }
            for _ in range(count)
        )

    split = split_corpus_documents(chunks, seed=42)
    eval_types = {document["doc_type"] for document in split["eval_documents"]}
    assert eval_types == {"final_regulation", "faq_or_guide", "encyclopedia"}
    if any(document["source"] == "manual/large.pdf" for document in split["eval_documents"]):
        assert len(split["eval_documents"]) > 1


def test_drafts_are_excluded_by_default(tmp_path) -> None:
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / "Draft_Regulation_2024.txt").write_text("Draft text. " * 100, encoding="utf-8")
    chunks = build_chunks(tmp_path / "empty", tmp_path / "chunks.jsonl", manual_dir=manual_dir)
    assert chunks == []


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
