"""Parse PDF, HTML, Markdown, text, and bundled JSONL into metadata-bearing pages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _markdown_sections(text: str, source: str) -> list[dict[str, Any]]:
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
                        "metadata": {"source": source, "page": None, "section": heading},
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
                "metadata": {"source": source, "page": None, "section": heading},
            }
        )
    return sections


def parse_document(path: str | Path, *, source: str | None = None) -> list[dict[str, Any]]:
    """Parse one supported source file into text units with source/page/section metadata."""
    file_path = Path(path)
    source_name = source or file_path.name
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return [
            {
                "text": clean_text(page.extract_text() or ""),
                "metadata": {"source": source_name, "page": index + 1, "section": None},
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
        return [
            {
                "text": text,
                "metadata": {"source": source_name, "page": None, "section": headings or title},
            }
        ]
    if suffix in {".md", ".markdown"}:
        markdown = file_path.read_text(encoding="utf-8", errors="replace")
        return _markdown_sections(markdown, source_name)
    if suffix in {".txt", ".text"}:
        text = clean_text(file_path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            return []
        return [{"text": text, "metadata": {"source": source_name, "page": None, "section": None}}]
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
                            "page": None,
                            "section": item.get("title", f"record {line_number}"),
                            "license_note": item.get("license_note"),
                            "retrieved_at": item.get("retrieved_at"),
                        },
                    }
                )
        return records
    raise ValueError(f"Unsupported document type: {suffix or '(no extension)'}")
