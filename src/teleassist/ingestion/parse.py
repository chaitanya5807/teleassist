"""Parse PDF, HTML, Markdown, text, and bundled JSONL into metadata-bearing pages."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

from teleassist.config import DEFAULT_DOCUMENT_FAMILIES

LOGGER = logging.getLogger(__name__)
YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
MIN_ENGLISH_WORD_SHARE = 0.05
RECOGNIZABLE_ENGLISH_WORDS = frozenset(
    """
    a able about above according account across act action activity actually add additional address
    after again against all also always am an and another any are as ask at available back be
    because
    become been before being below between both but by call can care case cause change charge check
    choice choose claim clear clearly code come common communication company complaint complete
    condition connect connection consent consumer contact continue contract control copy correct
    could country customer data date day days decision delay department describe details determine
    different
    do does document during each early either email end enough enter error even ever every example
    except exchange explain fail failure family fee field file final find first following for form
    from full further get give good government group had has have he help her here high him his home
    how if important in include including increase individual information initial input install
    instead into is issue it its itself job join just keep kind know known language last later law
    learn legal less
    liable like limit list local long made make many may me mean means measure member message might
    mobile more most must my name national necessary need network never new next no not note notice
    number of off offer office often on once one only open operate operator option or other our out
    over own page paper part particular party pass pay payment per period person personal phone
    place
    point policy possible power present press previous price provide provider public purpose quality
    question rate reach read receive record redressal regulation regulatory relevant remain remove
    request require requirement response result return right rule same say section see service set
    shall she should show similar since so source specific standard state statement status still
    subject such support system take tell term than that the their them then there these they thing
    this those through time to together too total transfer try type under understand until up update
    upon use used user using valid value verify very via view was way we well were what when where
    which while who why will with
    within without work would write year years you your
    act amendment authority annexure address broadband building call centre carrier cellular
    charter citizen city clause code contact centers circle east floor house limited mobile numbers
    north office park phase postpaid prepaid registered road south street tower west
    consultation consumer protection court customer care department digital dispute do not disturb
    eligibility equipment fair fee filing handset identity instruction internet invoice kyc licence
    license line mobile number portability provider quality of service recharge regulation roaming
    service provider sim spam spectrum subscriber tariff telecom telecommunication telephone terms
    trai tribunal validity verification voice wireless
    """.split()
)
DEVANAGARI_LETTER_PATTERN = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF]")


def infer_document_metadata(
    path: str | Path,
    first_page_text: str = "",
    *,
    source_type: str = "manual",
    title: str | None = None,
    family_mapping: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Infer stable corpus tags from a source filename and its first page."""
    file_path = Path(path)
    stem = file_path.stem
    key = re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()
    first_page = clean_text(first_page_text)
    combined = f"{key} {first_page.lower()}"
    if source_type == "wikipedia":
        doc_type = "encyclopedia"
    elif (
        "trai ekyc recommendations 2017" in key or "trai_ekyc_recommendations_2017" in stem.lower()
    ):
        doc_type = "recommendation"
    elif (
        "trai consumer handbook 2018 english" in key
        or (
            any(key.startswith(operator + " ") for operator in ("airtel", "jio", "vi"))
            and any(word in key for word in ("charter", "chart"))
        )
        or (stem.lower().startswith("pib_") and file_path.suffix.lower() in {".html", ".htm"})
    ):
        doc_type = "faq_or_guide"
    elif re.search(r"\b(draft|consultation|recommendation)\b", combined):
        doc_type = "draft_or_consultation"
    else:
        doc_type = "final_regulation"

    years_in_name = YEAR_PATTERN.findall(stem) if source_type != "wikipedia" else []
    years_in_page = YEAR_PATTERN.findall(first_page)
    year = (
        int(years_in_name[-1])
        if years_in_name
        else int(years_in_page[-1])
        if years_in_page
        else None
    )
    if source_type == "wikipedia":
        year = None
    readable_title = title or stem.replace("_", " ").replace("-", " ").strip().title()
    markers = family_mapping or DEFAULT_DOCUMENT_FAMILIES
    family_key = re.sub(r"[^a-z0-9]+", " ", f"{stem} {readable_title}".lower()).strip()
    family = next(
        (
            name
            for name, values in markers.items()
            if any(
                re.sub(r"[^a-z0-9]+", " ", marker.lower()).strip() in family_key
                for marker in values
            )
        ),
        None,
    )
    if family is None:
        slug = re.sub(r"[^A-Z0-9]+", "_", readable_title.upper()).strip("_") or "UNKNOWN"
        family = f"WIKIPEDIA_{slug}" if source_type == "wikipedia" else f"MANUAL_{slug}"
    return {
        "doc_type": doc_type,
        "year": year,
        "doc_title": readable_title,
        "family": family,
    }


def pdf_text_stats(path: str | Path) -> dict[str, int | float]:
    """Return page, character, and average extracted-text statistics for a PDF."""
    reader = PdfReader(str(path))
    texts = [clean_text(page.extract_text() or "") for page in reader.pages]
    pages = len(texts)
    characters = sum(map(len, texts))
    return {
        "pages": pages,
        "characters": characters,
        "average_chars_per_page": characters / pages if pages else 0.0,
    }


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def english_filter_reason(text: str) -> str | None:
    """Return why a text block is excluded from this English-only corpus, if any."""
    if "\ufffd" in text:
        return "replacement character"
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    devanagari = sum(bool(DEVANAGARI_LETTER_PATTERN.fullmatch(character)) for character in letters)
    if letters and devanagari / len(letters) > 0.15:
        return "more than 15% Devanagari letters"
    words = ENGLISH_WORD_PATTERN.findall(text.lower())
    recognizable = sum(word in RECOGNIZABLE_ENGLISH_WORDS for word in words)
    if not words or recognizable / len(words) < MIN_ENGLISH_WORD_SHARE:
        return "low recognizable-English word share"
    return None


def filter_english_units(
    units: list[dict[str, Any]], *, document_name: str
) -> list[dict[str, Any]]:
    """Drop non-English or corrupted page/text units and log per-document losses."""
    kept: list[dict[str, Any]] = []
    dropped_pages = 0
    dropped_characters = 0
    stripped_blocks = 0
    stripped_characters = 0
    reasons: dict[str, int] = {}
    for unit in units:
        text = str(unit.get("text", ""))
        reason = english_filter_reason(text)
        if reason is None:
            lines = text.splitlines(keepends=True)
            english_lines = [line for line in lines if not DEVANAGARI_LETTER_PATTERN.search(line)]
            cleaned_text = "".join(english_lines).strip()
            removed_characters = len(text) - len("".join(english_lines))
            if cleaned_text and english_filter_reason(cleaned_text) is None:
                if removed_characters:
                    stripped_blocks += 1
                    stripped_characters += removed_characters
                    unit = {**unit, "text": cleaned_text}
                kept.append(unit)
            else:
                dropped_pages += 1
                dropped_characters += len(text)
                reasons["low recognizable-English word share after script cleanup"] = (
                    reasons.get("low recognizable-English word share after script cleanup", 0) + 1
                )
        else:
            dropped_pages += 1
            dropped_characters += len(text)
            reasons[reason] = reasons.get(reason, 0) + 1
    if dropped_pages or stripped_blocks:
        LOGGER.info(
            "English filter dropped %d pages/blocks (%d characters) and stripped %d "
            "Devanagari line characters across %d blocks from %s: %s",
            dropped_pages,
            dropped_characters,
            stripped_characters,
            stripped_blocks,
            document_name,
            ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items())),
        )
    return kept


def _markdown_sections(text: str, source: str, source_type: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    title = Path(source).stem.replace("_", " ")
    heading = title
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            section_text = clean_text("\n".join(body))
            if section_text:
                sections.append(
                    {
                        "text": section_text,
                        "metadata": {
                            "source": source,
                            "source_type": source_type,
                            "page": None,
                            "section": heading,
                        },
                    }
                )
            heading = match.group(1).strip()
            body = []
        else:
            body.append(line)
    section_text = clean_text("\n".join(body))
    if section_text:
        sections.append(
            {
                "text": section_text,
                "metadata": {
                    "source": source,
                    "source_type": source_type,
                    "page": None,
                    "section": heading,
                },
            }
        )
    return sections


def parse_document(
    path: str | Path,
    *,
    source: str | None = None,
    source_type: str | None = None,
    family_mapping: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Parse one supported source file into text units with source/page/section metadata."""
    file_path = Path(path)
    source_name = source or file_path.name
    if source_type is not None:
        document_source_type = source_type
    elif "manual" in file_path.parts:
        document_source_type = "manual"
    elif file_path.parent.name.lower() == "raw":
        document_source_type = "wikipedia"
    else:
        document_source_type = "downloaded"
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        page_texts = [clean_text(page.extract_text() or "") for page in reader.pages]
        if document_source_type == "manual" and (
            not page_texts or sum(map(len, page_texts)) / len(page_texts) < 200
        ):
            average = sum(map(len, page_texts)) / len(page_texts) if page_texts else 0.0
            LOGGER.warning(
                "Manual document %s likely scanned, skipped (%.1f chars/page)", source_name, average
            )
            return []
        doc_tags = infer_document_metadata(
            file_path,
            page_texts[0] if page_texts else "",
            source_type=document_source_type,
            family_mapping=family_mapping,
        )
        return [
            {
                "text": clean_text(page.extract_text() or ""),
                "metadata": {
                    "source": source_name,
                    "source_type": document_source_type,
                    "page": index + 1,
                    "section": None,
                    **doc_tags,
                },
            }
            for index, page in enumerate(reader.pages)
            if clean_text(page.extract_text() or "")
        ]
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(file_path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for element in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            element.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else file_path.stem
        headings = [item.get_text(" ", strip=True) for item in soup.find_all(["h1", "h2", "h3"])]
        text = clean_text(soup.get_text("\n", strip=True))
        if not text:
            return []
        doc_tags = infer_document_metadata(
            file_path,
            text,
            source_type=document_source_type,
            title=title,
            family_mapping=family_mapping,
        )
        return [
            {
                "text": text,
                "metadata": {
                    "source": source_name,
                    "source_type": document_source_type,
                    "page": None,
                    "section": headings or title,
                    **doc_tags,
                },
            }
        ]
    if suffix in {".md", ".markdown"}:
        markdown = file_path.read_text(encoding="utf-8", errors="replace")
        return _markdown_sections(markdown, source_name, document_source_type)
    if suffix in {".txt", ".text"}:
        text = clean_text(file_path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            return []
        doc_tags = infer_document_metadata(
            file_path, text, source_type=document_source_type, family_mapping=family_mapping
        )
        return [
            {
                "text": text,
                "metadata": {
                    "source": source_name,
                    "source_type": document_source_type,
                    "page": None,
                    "section": None,
                    **doc_tags,
                },
            }
        ]
    if suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = clean_text(str(item.get("text", "")))
            if text:
                item_source_type = item.get("source_type", "wikipedia")
                item_title = item.get("doc_title", item.get("title", f"record {line_number}"))
                family_tags = infer_document_metadata(
                    item_title,
                    text,
                    source_type=item_source_type,
                    title=item_title,
                    family_mapping=family_mapping,
                )
                records.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": item.get("source", source_name),
                            "source_type": item_source_type,
                            "page": None,
                            "section": item.get("title", f"record {line_number}"),
                            "license_note": item.get("license_note"),
                            "retrieved_at": item.get("retrieved_at"),
                            "doc_type": item.get("doc_type", "encyclopedia"),
                            "year": None if item_source_type == "wikipedia" else item.get("year"),
                            "doc_title": item_title,
                            "family": family_tags["family"],
                        },
                    }
                )
        return records
    raise ValueError(f"Unsupported document type: {suffix or '(no extension)'}")
