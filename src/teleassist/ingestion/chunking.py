"""Token-aware recursive text chunking with stable identifiers and a CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
import tiktoken

from teleassist.config import load_config
from teleassist.ingestion.parse import parse_document

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
    return _ENCODING.decode(tokens)


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

        def emit(
            tokens: list[int] | list[str], chunk_metadata: dict[str, Any] = metadata
        ) -> None:
            content = _decode(tokens)
            if content.strip():
                identity = json.dumps(
                    chunk_metadata, sort_keys=True, ensure_ascii=False, default=str
                ) + "\n" + content
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
) -> list[dict[str, Any]]:
    """Parse supported files under a directory and write chunks as JSON Lines."""
    source_dir = Path(input_dir)
    documents: list[dict[str, Any]] = []
    paths = [source_dir] if source_dir.is_file() else sorted(source_dir.rglob("*"))
    for path in paths:
        if not path.is_file() or path.name.upper() == "MANIFEST.JSON":
            continue
        try:
            documents.extend(parse_document(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping source %s during parsing: %s", path, exc)
    chunks = chunk_document(documents, chunk_size=chunk_size, overlap=overlap)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
    out.write_text(serialized, encoding="utf-8")
    LOGGER.info(
        "Wrote %d chunks from %d parsed document units to %s", len(chunks), len(documents), out
    )
    return chunks


def main() -> None:
    """Build processed chunks from the configured raw corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_chunks(
        args.input_dir,
        args.output,
        chunk_size=args.chunk_size or config.chunking.chunk_size,
        overlap=config.chunking.overlap if args.overlap is None else args.overlap,
    )


if __name__ == "__main__":
    main()
