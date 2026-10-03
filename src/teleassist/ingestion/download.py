"""Download a small public Wikipedia telecom corpus and record provenance."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)
USER_AGENT = "TeleAssistResearch/0.1 (public educational corpus; contact: project maintainer)"
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
    "Internet access",
    "Optical fiber",
    "Voice over IP",
    "Short Message Service",
    "Prepaid mobile phone",
    "Postpaid mobile phone",
    "Electronic identification",
    "Telecommunications satellites",
    "Telephone numbering plan",
    "Wireless communication",
)


def _safe_filename(title: str) -> str:
    """Convert a page title to a portable filename."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".txt"


def download_sources(
    output_dir: str | Path,
    *,
    titles: tuple[str, ...] = WIKIPEDIA_TITLES,
    timeout: float = 20.0,
    session: requests.Session | None = None,
    bundle_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch public Wikipedia extracts, logging and recording individual failures."""
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    records: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []

    for title in titles:
        page_url = "https://en.wikipedia.org/wiki/" + requests.utils.quote(title.replace(" ", "_"))
        retrieved_at = datetime.now(UTC).isoformat()
        record: dict[str, Any] = {
            "title": title,
            "url": page_url,
            "retrieved_at": retrieved_at,
            "license_note": WIKIPEDIA_LICENSE,
            "status": "failed",
        }
        try:
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
            response.raise_for_status()
            payload = response.json()
            pages = payload.get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            text = page.get("extract", "").strip()
            if not text:
                raise ValueError("Wikipedia returned an empty extract")
            actual_title = page.get("title", title)
            if any(doc["title"] == actual_title for doc in docs):
                record.update({"title": actual_title, "status": "duplicate_redirect"})
                records.append(record)
                LOGGER.info("Skipping duplicate Wikipedia redirect %s -> %s", title, actual_title)
                continue
            page_url = "https://en.wikipedia.org/wiki/" + requests.utils.quote(
                actual_title.replace(" ", "_")
            )
            file_name = _safe_filename(actual_title)
            (raw_dir / file_name).write_text(text + "\n", encoding="utf-8")
            record.update({"title": actual_title, "status": "downloaded", "file": file_name})
            docs.append(
                {
                    "title": actual_title,
                    "source": page_url,
                    "retrieved_at": retrieved_at,
                    "license_note": WIKIPEDIA_LICENSE,
                    "text": text,
                }
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            record["error"] = str(exc)
            LOGGER.warning("Skipping Wikipedia source %s: %s", title, exc)
        records.append(record)

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
        previous_docs: dict[str, dict[str, Any]] = {}
        if bundle.exists():
            for line in bundle.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    previous = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping malformed existing fallback entry in %s", bundle)
                    continue
                previous_docs[previous["title"]] = previous
        previous_docs.update({doc["title"]: doc for doc in docs})
        bundle.write_text(
            "".join(
                json.dumps(doc, ensure_ascii=False) + "\n" for doc in previous_docs.values()
            ),
            encoding="utf-8",
        )
        LOGGER.info("Bundled %d distinct documents in %s", len(previous_docs), bundle)
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_sources(args.output_dir, timeout=args.timeout, bundle_path=args.bundle_path)


if __name__ == "__main__":
    main()
