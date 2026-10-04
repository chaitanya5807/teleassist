"""Token-aware recursive text chunking with stable identifiers and a CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
import tiktoken
from pypdf.errors import PdfReadError

from teleassist.config import load_config
from teleassist.ingestion.parse import filter_english_units, parse_document
from teleassist.ingestion.splitting import split_corpus_families, write_document_split

LOGGER = logging.getLogger(__name__)
try:
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except (OSError, ValueError, ImportError, requests.RequestException):
    LOGGER.warning(
        "Could not load cl100k_base tokenizer data; using a deterministic lexical fallback"
    )
    _ENCODING = None


def _tokens(text: str) -> list[int] | list[str]:
    if _ENCODING is not None:
        return _ENCODING.encode(text)
    return re.findall(r"\s+|[\w]+|[^\w\s]", text, flags=re.UNICODE)


def _decode(tokens: list[int] | list[str]) -> str:
    if tokens and isinstance(tokens[0], str):
        return "".join(tokens)
    if _ENCODING is None:
        return ""
    return _ENCODING.decode(tokens, errors="ignore")


def _recursive_split(text: str, limit: int) -> list[str]:
    """Split by paragraphs, sentences, then tokenizer windows until every part fits."""
    if len(_tokens(text)) <= limit:
        return [text]
    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+"):
        parts: list[str] = []
        start = 0
        for match in re.finditer(pattern, text):
            parts.append(text[start : match.end()])
            start = match.end()
        parts.append(text[start:])
        parts = [part for part in parts if part.strip()]
        if len(parts) > 1:
            result: list[str] = []
            for part in parts:
                if len(_tokens(part)) < len(_tokens(text)):
                    result.extend(_recursive_split(part, limit))
                else:
                    result.append(part)
            if len(result) > 1:
                return result
    ids = _tokens(text)
    return [_decode(ids[start : start + limit]) for start in range(0, len(ids), limit)]


def _stable_id(metadata: dict[str, Any], text: str, occurrence: int) -> str:
    identity = (
        json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str)
        + "\n"
        + text
        + f"\n{occurrence}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def chunk_document(
    units: Iterable[dict[str, Any]], *, chunk_size: int = 512, overlap: int = 64
) -> list[dict[str, Any]]:
    """Create overlapping token-bounded chunks while retaining each unit's metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}
    for unit in units:
        text = str(unit.get("text", "")).strip()
        if not text:
            continue
        metadata = dict(unit.get("metadata", {}))
        pieces = _recursive_split(text, chunk_size)
        current: list[int] | list[str] = []

        def emit(tokens: list[int] | list[str], chunk_metadata: dict[str, Any] = metadata) -> None:
            content = _decode(tokens)
            if content.strip():
                identity = (
                    json.dumps(chunk_metadata, sort_keys=True, ensure_ascii=False, default=str)
                    + "\n"
                    + content
                )
                occurrence = occurrences.get(identity, 0)
                occurrences[identity] = occurrence + 1
                chunks.append(
                    {
                        "id": _stable_id(chunk_metadata, content, occurrence),
                        "text": content,
                        "metadata": chunk_metadata.copy(),
                    }
                )

        for piece in pieces:
            piece_tokens = _tokens(piece)
            if current and len(current) + len(piece_tokens) > chunk_size:
                emit(current)
                current = current[-overlap:] if overlap else []
            while piece_tokens:
                capacity = chunk_size - len(current)
                current.extend(piece_tokens[:capacity])
                piece_tokens = piece_tokens[capacity:]
                if piece_tokens:
                    emit(current)
                    current = current[-overlap:] if overlap else []
        emit(current)
    return chunks


def build_chunks(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    manual_dir: str | Path | None = None,
    include_drafts: bool = False,
    family_mapping: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Parse the corpus and optional manual files, then write chunks as JSON Lines."""
    source_dir = Path(input_dir)
    documents: list[dict[str, Any]] = []
    paths = [source_dir] if source_dir.is_file() else sorted(source_dir.rglob("*"))
    for path in paths:
        if (
            not path.is_file()
            or path.name.upper() == "MANIFEST.JSON"
            or "excluded" in {part.lower() for part in path.parts}
        ):
            continue
        try:
            parsed = filter_english_units(
                parse_document(path, family_mapping=family_mapping), document_name=str(path)
            )
            documents.extend(
                unit
                for unit in parsed
                if include_drafts
                or unit.get("metadata", {}).get("doc_type") != "draft_or_consultation"
            )
        except (OSError, ValueError, json.JSONDecodeError, PdfReadError) as exc:
            LOGGER.warning("Skipping source %s during parsing: %s", path, exc)
    if manual_dir is not None:
        manual_root = Path(manual_dir)
        if manual_root.exists():
            for path in sorted(manual_root.rglob("*")):
                if not path.is_file() or "excluded" in {part.lower() for part in path.parts}:
                    continue
                if path.suffix.lower() not in {".pdf", ".html", ".htm", ".txt"}:
                    LOGGER.warning("Skipping unsupported manual document %s", path)
                    continue
                source = f"manual/{path.relative_to(manual_root).as_posix()}"
                try:
                    parsed = filter_english_units(
                        parse_document(
                            path,
                            source=source,
                            source_type="manual",
                            family_mapping=family_mapping,
                        ),
                        document_name=source,
                    )
                    documents.extend(
                        unit
                        for unit in parsed
                        if include_drafts
                        or unit.get("metadata", {}).get("doc_type") != "draft_or_consultation"
                    )
                except (OSError, ValueError, json.JSONDecodeError, PdfReadError) as exc:
                    LOGGER.warning("Skipping manual source %s during parsing: %s", path, exc)
    chunks = chunk_document(documents, chunk_size=chunk_size, overlap=overlap)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
    out.write_text(serialized, encoding="utf-8")
    LOGGER.info(
        "Wrote %d chunks from %d parsed document units to %s", len(chunks), len(documents), out
    )
    return chunks


def format_source_table(manifest_path: str | Path, chunks: list[dict[str, Any]]) -> str:
    """Format per-source status, extracted character count, and chunk count."""
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        return f"Source manifest not found: {manifest_file}"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    chunk_counts = Counter(str(chunk.get("metadata", {}).get("source", "")) for chunk in chunks)
    rows: list[tuple[str, str, int, int]] = []
    for source in manifest.get("sources", []):
        source_type = source.get("source_type", "wikipedia")
        source_key = source.get("source") if source_type == "manual" else source.get("url", "")
        count = 0 if source.get("status") == "duplicate_redirect" else chunk_counts[str(source_key)]
        rows.append(
            (
                str(source.get("requested_title", source.get("title", ""))),
                str(source.get("status", "unknown")),
                int(source.get("char_count", 0)),
                count,
            )
        )

    title_width = max(5, min(56, max((len(row[0]) for row in rows), default=5)))
    status_width = max(6, min(20, max((len(row[1]) for row in rows), default=6)))
    lines = [
        (
            f"| {'Title':<{title_width}} | {'Status':<{status_width}} | "
            f"{'Characters':>10} | {'Chunks':>6} |"
        ),
        f"|{'-' * (title_width + 2)}|{'-' * (status_width + 2)}|------------:|-------:|",
    ]
    lines.extend(
        f"| {title:<{title_width}} | {status:<{status_width}} | {characters:>10,} | {count:>6,} |"
        for title, status, characters, count in rows
    )
    return "\n".join(lines)


def main() -> None:
    """Build processed chunks from the configured raw corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--manual-dir", type=Path, default=Path("data/raw/manual"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/MANIFEST.json"))
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--split-output", type=Path, default=Path("data/processed/split.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    chunks = build_chunks(
        args.input_dir,
        args.output,
        chunk_size=args.chunk_size or config.chunking.chunk_size,
        overlap=config.chunking.overlap if args.overlap is None else args.overlap,
        manual_dir=args.manual_dir,
        include_drafts=config.include_drafts,
        family_mapping=config.document_families,
    )
    document_split = split_corpus_families(chunks, seed=config.seed)
    write_document_split(args.split_output, document_split)
    LOGGER.info(
        "Wrote family split: %d train, %d eval to %s",
        len(document_split["train_documents"]),
        len(document_split["eval_documents"]),
        args.split_output,
    )
    print(format_source_table(args.manifest, chunks))


if __name__ == "__main__":
    main()
