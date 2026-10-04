"""Parse PDF, HTML, Markdown, text, and bundled JSONL into metadata-bearing pages."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)
YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def infer_document_metadata(
    path: str | Path,
    first_page_text: str = "",
    *,
    source_type: str = "manual",
    title: str | None = None,
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
    readable_title = title or stem.replace("_", " ").replace("-", " ").strip().title()
    return {
        "doc_type": doc_type,
        "year": year,
        "doc_title": readable_title,
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
            file_path, page_texts[0] if page_texts else "", source_type=document_source_type
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
            file_path, text, source_type=document_source_type, title=title
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
        doc_tags = infer_document_metadata(file_path, text, source_type=document_source_type)
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
                records.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": item.get("source", source_name),
                            "source_type": item.get("source_type", "wikipedia"),
                            "page": None,
                            "section": item.get("title", f"record {line_number}"),
                            "license_note": item.get("license_note"),
                            "retrieved_at": item.get("retrieved_at"),
                            "doc_type": item.get("doc_type", "encyclopedia"),
                            "year": item.get("year"),
                            "doc_title": item.get("doc_title", item.get("title", "")),
                        },
                    }
                )
        return records
    raise ValueError(f"Unsupported document type: {suffix or '(no extension)'}")
