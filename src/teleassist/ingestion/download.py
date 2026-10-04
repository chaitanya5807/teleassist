"""Download a small public Wikipedia telecom corpus and record provenance."""

from __future__ import annotations

import argparse
import email.utils
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import requests
from pypdf.errors import PdfReadError

from teleassist.config import load_config
from teleassist.ingestion.parse import infer_document_metadata, parse_document, pdf_text_stats

LOGGER = logging.getLogger(__name__)
USER_AGENT = "TeleAssist/0.1 (student project; contact: chaitanyatallapudi58@gmail.com)"
WIKIPEDIA_LICENSE = "Wikipedia content is available under CC BY-SA 4.0; see page for attribution."
WIKIPEDIA_TITLES = (
    "Telecommunications",
    "Telecommunications in India",
    "Telecom Regulatory Authority of India",
    "Department of Telecommunications (India)",
    "Mobile phone",
    "Mobile network operator",
    "Cellular network",
    "Mobile telecommunications network",
    "SIM card",
    "Mobile data",
    "Mobile number portability",
    "5G",
    "4G",
    "3G",
    "Broadband",
    "Short Message Service",
    "Prepaid mobile phone",
    "Postpaid mobile phone",
    "Know your customer",
    "Telecommunications satellites",
    "Telephone numbering plan",
    "Radio spectrum",
    "Spectrum auction",
    "Telephone solicitation",
    "Call detail record",
    "Internet service provider",
    "Mobile virtual network operator",
    "International Mobile Subscriber Identity",
    "International Mobile Equipment Identity",
    "Embedded SIM",
    "Mobile broadband",
    "Internet in India",
    "Data roaming",
    "Roaming",
    "Wireless Internet service provider",
    "LTE (telecommunication)",
    "Telecommunications policy of India",
    "Consumer protection",
    "Do Not Call Registry",
    "Do Not Disturb (telecommunications)",
    "Unsolicited commercial communication",
)

DROPPED_WIKIPEDIA_TITLES = frozenset(
    {
        "Optical fiber",
        "Communications satellite",
        "Telephone exchange",
        "Electromagnetic spectrum",
        "Radio communication",
        "Wi-Fi",
        "Voice over IP",
        "Landline",
        "Public switched telephone network",
        "Telephone",
        "Telephone call",
        "Customer service",
        "Wireless communication",
        "Internet access",
        "Telecommunications equipment",
    }
)


def _safe_filename(title: str) -> str:
    """Convert a page title to a portable filename."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".txt"


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After as seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _read_download_cache(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Index previously downloaded Wikipedia files by their requested title."""
    manifest_path = raw_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read download cache manifest %s: %s", manifest_path, exc)
        return {}

    cache: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("sources", []):
        if entry.get("status") not in {"downloaded", "cached", "duplicate_redirect"}:
            continue
        if entry.get("source_type", "wikipedia") != "wikipedia":
            continue
        file_name = entry.get("file")
        if not file_name or not (raw_dir / file_name).is_file():
            continue
        cache[entry.get("requested_title", entry.get("title", ""))] = entry
        cache.setdefault(entry.get("title", ""), entry)
        source_url = entry.get("url", "")
        if source_url:
            url_title = unquote(urlsplit(source_url).path.rsplit("/", 1)[-1]).replace("_", " ")
            cache.setdefault(url_title, entry)
    return cache


def _get_wikipedia_page(
    client: requests.Session,
    title: str,
    *,
    timeout: float,
    max_retries: int,
    min_request_interval: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    last_request_started: float | None,
) -> tuple[requests.Response, float]:
    """Fetch one page, retrying rate limits and server errors with backoff."""
    retry_delay = 0.0
    for attempt in range(max_retries + 1):
        now = monotonic()
        interval_delay = (
            max(0.0, min_request_interval - (now - last_request_started))
            if last_request_started is not None
            else 0.0
        )
        wait = max(retry_delay, interval_delay)
        if wait:
            sleep(wait)
        request_started = monotonic()
        response = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
            },
            timeout=timeout,
        )
        status = response.status_code
        if status != 429 and not 500 <= status <= 599:
            response.raise_for_status()
            return response, request_started
        if attempt >= max_retries:
            response.raise_for_status()

        retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
        backoff = float(2**attempt)
        retry_delay = max(min_request_interval, retry_after or backoff)
        LOGGER.warning(
            "Wikipedia returned HTTP %d for %s; retry %d/%d in %.1f seconds",
            status,
            title,
            attempt + 1,
            max_retries,
            retry_delay,
        )
        last_request_started = request_started

    raise RuntimeError(f"No response received for Wikipedia title {title}")


def download_sources(
    output_dir: str | Path,
    *,
    titles: tuple[str, ...] = WIKIPEDIA_TITLES,
    timeout: float = 20.0,
    session: requests.Session | None = None,
    bundle_path: str | Path | None = None,
    max_retries: int = 5,
    min_request_interval: float = 1.0,
    include_drafts: bool = False,
    family_mapping: dict[str, list[str]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """Fetch and cache Wikipedia extracts, logging and recording individual failures."""
    if max_retries < 0:
        raise ValueError("max_retries must not be negative")
    if min_request_interval < 1.0:
        raise ValueError("min_request_interval must be at least one second")
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for dropped_title in DROPPED_WIKIPEDIA_TITLES:
        (raw_dir / _safe_filename(dropped_title)).unlink(missing_ok=True)
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    records: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    cache = _read_download_cache(raw_dir)
    last_request_started: float | None = None

    for title in titles:
        page_url = "https://en.wikipedia.org/wiki/" + requests.utils.quote(title.replace(" ", "_"))
        retrieved_at = datetime.now(UTC).isoformat()
        record: dict[str, Any] = {
            "title": title,
            "requested_title": title,
            "url": page_url,
            "retrieved_at": retrieved_at,
            "license_note": WIKIPEDIA_LICENSE,
            "source_type": "wikipedia",
            "status": "failed",
            "char_count": 0,
            **infer_document_metadata(
                title,
                source_type="wikipedia",
                title=title,
                family_mapping=family_mapping,
            ),
        }
        try:
            cached = cache.get(title)
            if cached is not None:
                file_name = cached["file"]
                text = (raw_dir / file_name).read_text(encoding="utf-8").strip()
                actual_title = cached.get("title", title)
                page_url = cached.get("url", page_url)
                is_duplicate = any(doc["title"] == actual_title for doc in docs)
                record.update(
                    {
                        "title": actual_title,
                        "url": page_url,
                        "file": file_name,
                        "status": "duplicate_redirect" if is_duplicate else "cached",
                        "char_count": len(text),
                        **infer_document_metadata(
                            actual_title,
                            text,
                            source_type="wikipedia",
                            title=actual_title,
                            family_mapping=family_mapping,
                        ),
                    }
                )
                if not is_duplicate:
                    docs.append(
                        {
                            "title": actual_title,
                            "source": page_url,
                            "retrieved_at": cached.get("retrieved_at", retrieved_at),
                            "license_note": cached.get("license_note", WIKIPEDIA_LICENSE),
                            "source_type": "wikipedia",
                            "text": text,
                            **infer_document_metadata(
                                actual_title,
                                text,
                                source_type="wikipedia",
                                title=actual_title,
                                family_mapping=family_mapping,
                            ),
                        }
                    )
                records.append(record)
                LOGGER.info("Using cached Wikipedia page %s", title)
                continue

            response, last_request_started = _get_wikipedia_page(
                client,
                title,
                timeout=timeout,
                max_retries=max_retries,
                min_request_interval=min_request_interval,
                sleep=sleep,
                monotonic=monotonic,
                last_request_started=last_request_started,
            )
            payload = response.json()
            pages = payload.get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            text = page.get("extract", "").strip()
            if not text:
                raise ValueError("Wikipedia returned an empty extract")
            actual_title = page.get("title", title)
            page_url = "https://en.wikipedia.org/wiki/" + requests.utils.quote(
                actual_title.replace(" ", "_")
            )
            file_name = _safe_filename(actual_title)
            if any(doc["title"] == actual_title for doc in docs):
                record.update(
                    {
                        "title": actual_title,
                        "url": page_url,
                        "file": file_name,
                        "status": "duplicate_redirect",
                        "char_count": len(text),
                        **infer_document_metadata(
                            actual_title,
                            text,
                            source_type="wikipedia",
                            title=actual_title,
                            family_mapping=family_mapping,
                        ),
                    }
                )
                records.append(record)
                LOGGER.info("Skipping duplicate Wikipedia redirect %s -> %s", title, actual_title)
                continue
            (raw_dir / file_name).write_text(text + "\n", encoding="utf-8")
            record.update(
                {
                    "title": actual_title,
                    "url": page_url,
                    "status": "downloaded",
                    "file": file_name,
                    "char_count": len(text),
                    **infer_document_metadata(
                        actual_title,
                        text,
                        source_type="wikipedia",
                        title=actual_title,
                        family_mapping=family_mapping,
                    ),
                }
            )
            docs.append(
                {
                    "title": actual_title,
                    "source": page_url,
                    "retrieved_at": retrieved_at,
                    "license_note": WIKIPEDIA_LICENSE,
                    "source_type": "wikipedia",
                    "text": text,
                    **infer_document_metadata(
                        actual_title,
                        text,
                        source_type="wikipedia",
                        title=actual_title,
                        family_mapping=family_mapping,
                    ),
                }
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            last_request_started = monotonic()
            record["error"] = str(exc)
            LOGGER.warning("Skipping Wikipedia source %s: %s", title, exc)
        records.append(record)

    manual_dir = raw_dir / "manual"
    manual_paths = sorted(manual_dir.rglob("*")) if manual_dir.exists() else []
    for path in manual_paths:
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pdf", ".html", ".htm", ".txt"}:
            LOGGER.warning("Skipping unsupported manual document %s", path)
            continue
        relative_path = path.relative_to(raw_dir).as_posix()
        manual_status = "available"
        char_count = 0
        pages = 1
        doc_tags: dict[str, Any] = infer_document_metadata(path, family_mapping=family_mapping)
        try:
            if path.suffix.lower() == ".pdf":
                stats = pdf_text_stats(path)
                pages = int(stats["pages"])
                char_count = int(stats["characters"])
                first_text_units = (
                    parse_document(
                        path,
                        source=relative_path,
                        source_type="manual",
                        family_mapping=family_mapping,
                    )
                    if float(stats["average_chars_per_page"]) >= 200
                    else []
                )
                if float(stats["average_chars_per_page"]) < 200:
                    manual_status = "likely_scanned_skipped"
                    LOGGER.warning(
                        "Manual document %s likely scanned, skipped (%.1f chars/page)",
                        relative_path,
                        float(stats["average_chars_per_page"]),
                    )
                    first_page = ""
                else:
                    first_page = first_text_units[0]["text"] if first_text_units else ""
                doc_tags = infer_document_metadata(
                    path, first_page, source_type="manual", family_mapping=family_mapping
                )
            else:
                parsed = parse_document(
                    path,
                    source=relative_path,
                    source_type="manual",
                    family_mapping=family_mapping,
                )
                char_count = sum(len(unit["text"]) for unit in parsed)
                first_page = parsed[0]["text"] if parsed else ""
                doc_tags = infer_document_metadata(
                    path, first_page, source_type="manual", family_mapping=family_mapping
                )
            if doc_tags["doc_type"] == "draft_or_consultation" and not include_drafts:
                manual_status = "excluded_draft"
        except (OSError, ValueError, PdfReadError) as exc:
            manual_status = "parse_error"
            LOGGER.warning("Could not parse manual source %s: %s", relative_path, exc)
        records.append(
            {
                "title": path.stem,
                "source": relative_path,
                "file": relative_path,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "license_note": "User-provided local document",
                "source_type": "manual",
                "status": manual_status,
                "char_count": char_count,
                "pages": pages,
                **doc_tags,
            }
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": records,
    }
    (raw_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if bundle_path is not None and docs:
        bundle = Path(bundle_path)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        current_docs = {doc["title"]: doc for doc in docs}
        for doc in current_docs.values():
            doc.update(
                infer_document_metadata(
                    doc["title"],
                    doc.get("text", ""),
                    source_type="wikipedia",
                    title=doc["title"],
                    family_mapping=family_mapping,
                )
            )
        bundle.write_text(
            "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in current_docs.values()),
            encoding="utf-8",
        )
        LOGGER.info("Bundled %d distinct documents in %s", len(current_docs), bundle)
    LOGGER.info("Downloaded %d of %d sources", len(docs), len(titles))
    return records


def main() -> None:
    """Run the Phase 2 corpus downloader from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=Path("data/fallback/wikipedia_telecom.jsonl"),
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    download_sources(
        args.output_dir,
        timeout=args.timeout,
        bundle_path=args.bundle_path,
        include_drafts=config.include_drafts,
        family_mapping=config.document_families,
    )


if __name__ == "__main__":
    main()
