"""Chunk QA quality tagging and SFT data generation (no model training)."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from teleassist.config import load_config
from teleassist.generation.question_generator import QuestionGenerator, create_generator
from teleassist.retrieval.bm25 import BM25Index

NUMBER_PATTERN = re.compile(r"\d+(?:[.,/%:-]\d+)*")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
TOC_PATTERN = re.compile(r"^(contents|table of contents|index)$", re.I)
QOS_DISCUSSION_PATTERN = re.compile(
    r"input of stakeholders|analysis and conclusion|authority['’]s analysis|"
    r"stakeholders submitted|stakeholders suggested",
    re.I,
)


def chunk_rejection_reason(chunk: dict[str, Any]) -> str | None:
    """Classify a low-value chunk for QA while preserving it in retrieval."""
    text = str(chunk.get("text", ""))
    metadata = chunk.get("metadata", {})
    words = WORD_PATTERN.findall(text)
    family = str(metadata.get("family", "")).upper()
    if family == "QOS" and QOS_DISCUSSION_PATTERN.search(text):
        return "qos_stakeholder_discussion"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    if TOC_PATTERN.fullmatch(first_line) and (
        sum(bool(re.search(r"\.{2,}|\s\d{1,3}\s*$", line)) for line in lines) >= 2
        or len(words) < 100
    ):
        return "table_of_contents_or_index"
    contact_heading = any(
        re.search(
            r"\b(address|contact centre|contact center|contact numbers|appellate authority|"
            r"registered office|customer care|telephone|fax|email)\b",
            line,
            re.I,
        )
        for line in lines
    )
    phone_or_email_lines = sum(
        bool(EMAIL_PATTERN.search(line))
        or any(
            len(re.sub(r"\D", "", match.group())) >= 10 for match in PHONE_PATTERN.finditer(line)
        )
        for line in lines
    )
    address_lines = sum(
        bool(re.search(r"\b(address|road|street|floor|complex|avenue|building)\b", line, re.I))
        for line in lines
    )
    if len(lines) >= 3 and contact_heading and (phone_or_email_lines >= 4 or address_lines >= 4):
        return "address_or_contact_list"
    numbers = NUMBER_PATTERN.findall(text)
    numeric_rows = sum(len(NUMBER_PATTERN.findall(line)) >= 4 for line in lines)
    number_share = len(numbers) / max(1, len(words) + len(numbers))
    if (len(numbers) >= 20 and number_share >= 0.60) or (
        numeric_rows >= 4 and numeric_rows / max(1, len(lines)) >= 0.25
    ):
        return "mostly_numbers_or_tables"
    if len(words) < 40:
        return "under_40_words"
    return None


def annotate_qa_quality(chunks: list[dict[str, Any]]) -> Counter[str]:
    """Add usable_for_qa and an optional rejection reason without deleting chunks."""
    counts: Counter[str] = Counter()
    for chunk in chunks:
        reason = chunk_rejection_reason(chunk)
        chunk["usable_for_qa"] = reason is None
        if reason:
            chunk["qa_rejection_reason"] = reason
            counts[reason] += 1
        else:
            chunk.pop("qa_rejection_reason", None)
    return counts


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    """Load the full chunk set and persist QA quality flags alongside existing records."""
    chunks = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    annotate_qa_quality(chunks)
    Path(path).write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    return chunks


def support_overlap(answer: str, text: str) -> float:
    """Measure meaningful answer-token overlap with one or more source passages."""
    stop = {"the", "and", "that", "this", "with", "from", "have", "has", "are", "for", "you"}
    answer_words = set(w.casefold() for w in WORD_PATTERN.findall(answer)) - stop
    text_words = set(w.casefold() for w in WORD_PATTERN.findall(text)) - stop
    return len(answer_words & text_words) / max(1, len(answer_words))


def evidence_is_supported(evidence: Any, text: str) -> bool:
    """Check every evidence quote against source text, ignoring whitespace and case."""
    quotes = [evidence] if isinstance(evidence, str) else evidence
    normalized = re.sub(r"\s+", " ", text).casefold()
    return bool(quotes) and all(
        re.sub(r"\s+", " ", str(quote)).strip().casefold() in normalized for quote in quotes
    )


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _too_similar(question: str, previous: list[str], threshold: float = 0.88) -> bool:
    normalized = " ".join(question.casefold().split())
    return any(
        SequenceMatcher(None, normalized, " ".join(old.casefold().split())).ratio() >= threshold
        for old in previous
    )


def _get_context_candidates(
    bm25: BM25Index,
    question: str,
    train_ids: set[str],
    excluded_ids: set[str],
    wanted: int,
) -> list[dict[str, Any]]:
    candidates = bm25.search(question, top_k=100)
    selected = []
    for candidate in candidates:
        identifier = str(candidate.get("id"))
        if identifier not in train_ids or identifier in excluded_ids:
            continue
        if chunk_rejection_reason(candidate) is not None:
            continue
        selected.append(candidate)
        if len(selected) == wanted:
            break
    return selected


def _unsupported_question_candidates() -> list[tuple[str, str]]:
    """Create a varied pool of plausible but unsupported feature questions."""
    features = [
        "satellite emergency texting", "recharge refunds to a crypto wallet",
        "mobile data exchanged for electricity credits", "screen damage insurance",
        "rollover data shared with family members", "bank transfers from prepaid balance",
        "automatic bill payments with reward points", "guaranteed indoor signal strength",
        "international satellite calls on prepaid plans",
        "unused roaming minutes converted to cash",
        "one year number suspension without a fee", "battery failure data reimbursement",
        "combined bills across different operators", "free satellite phone handsets",
        "SIM ownership transfer through a blockchain wallet", "home broadband outage insurance",
        "mobile recharge paid in cryptocurrency", "phone theft replacement by TRAI",
        "a fixed tariff for calls to the Moon", "unlimited data rollover across providers",
        "free handset repairs for network outages", "electricity payments from phone balances",
        "operator funded travel insurance", "guaranteed coverage inside every building",
        "cash rewards for porting between networks",
    ]
    templates = [
        "Can I get {feature} with my mobile plan?",
        "Does my provider offer {feature}?",
        "Will TRAI require every operator to provide {feature}?",
        "Can a prepaid customer request {feature}?",
        "Is {feature} included when I activate a new SIM?",
        "Can I ask customer care to arrange {feature}?",
        "Do telecom rules guarantee subscribers {feature}?",
        "Can I use an app to claim {feature} from my operator?",
        "Would I qualify for {feature} after porting my number?",
        "Does the law say providers must give customers {feature}?",
        "Can I add {feature} to an existing account?",
        "Are operators required to refund me through {feature}?",
    ]
    return [
        (template.format(feature=feature), feature)
        for feature in features
        for template in templates
    ]


def make_sft_record(
    question: str,
    answer: str,
    evidence: Any,
    gold: dict[str, Any] | None,
    hard_negatives: list[dict[str, Any]],
    *,
    answerable: bool,
    rng: random.Random,
    five_context: bool = False,
    gold_last: bool = False,
) -> dict[str, Any] | None:
    """Build one shuffled SFT record and rewrite answer citations to final context positions."""
    gold_text = str(gold.get("text", "")) if gold else ""
    if answerable:
        if not gold or not evidence_is_supported(evidence, gold_text):
            return None
        if support_overlap(re.sub(r"\[\d+\]", "", answer), gold_text) < 0.20:
            return None
    elif answer != "I could not find this in the provided documents.":
        return None
    needed_negatives = (4 if five_context else 3) if answerable else (5 if five_context else 4)
    hard_negatives = hard_negatives[:needed_negatives]
    context_records = []
    if gold:
        context_records.append(gold)
    context_records.extend(hard_negatives)
    if gold and gold_last:
        context_records.remove(gold)
        context_records.append(gold)
    else:
        rng.shuffle(context_records)
    context = [
        {
            "id": item["id"],
            "text": item["text"],
            "metadata": item.get("metadata", {}),
        }
        for item in context_records
    ]
    if answerable:
        gold_position = next(i for i, item in enumerate(context) if item["id"] == gold["id"]) + 1
        answer = re.sub(r"\s*\[\d+\]", "", answer).strip() + f" [{gold_position}]"
    return {
        "question": question,
        "context": context,
        "answer": answer,
        "evidence": evidence,
        "gold_chunk_ids": [gold["id"]] if gold else [],
        "answerable": answerable,
        "category": (gold or {}).get("metadata", {}).get("family", "UNANSWERABLE"),
    }


def _family_val_split(
    records: list[dict[str, Any]], *, seed: int, fraction: float = 0.15
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = sorted({item["category"] for item in records})
    rng = random.Random(seed)
    rng.shuffle(families)
    count = max(1, round(len(families) * fraction)) if len(families) > 1 else 0
    val_families = set(families[:count])
    train = [item for item in records if item["category"] not in val_families]
    val = [item for item in records if item["category"] in val_families]
    return train, val


def build_sft_data(
    chunks: list[dict[str, Any]],
    split: dict[str, Any],
    generator: QuestionGenerator,
    *,
    target_count: int = 2000,
    seed: int = 42,
    max_doc_share: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create SFT examples from train documents with full-index BM25 negatives."""
    train_sources = {d["source"] for d in split["train_documents"]}
    train_chunks = [
        c
        for c in chunks
        if c.get("metadata", {}).get("source") in train_sources
        and c.get("usable_for_qa", chunk_rejection_reason(c) is None)
    ]
    if not train_chunks:
        raise ValueError("No usable QA chunks in train documents")
    train_ids = {str(c["id"]) for c in train_chunks}
    bm25 = BM25Index(chunks)
    rng = random.Random(seed)
    desired_unanswerable = round(target_count * 0.15)
    desired_answerable = target_count - desired_unanswerable
    cap = max(1, int(target_count * max_doc_share))
    doc_counts: Counter[str] = Counter()
    used_questions: list[str] = []
    records = []

    # Round-robin over documents keeps source counts below the configured cap.
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in train_chunks:
        by_doc[str(chunk["metadata"]["source"])].append(chunk)
    documents = list(by_doc)
    rng.shuffle(documents)
    for attempt in range(max(desired_answerable * 2, len(train_chunks))):
        if sum(doc_counts.values()) >= desired_answerable:
            break
        source = documents[attempt % len(documents)]
        if doc_counts[source] >= cap:
            continue
        gold = by_doc[source][(attempt // len(documents)) % len(by_doc[source])]
        generated = generator.generate(gold["text"], 3)
        for item in generated:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            evidence = item.get("evidence", [])
            if (
                not question
                or _too_similar(question, used_questions)
                or not evidence_is_supported(evidence, gold["text"])
                or support_overlap(re.sub(r"\[\d+\]", "", answer), gold["text"]) < 0.20
                or len(question.split()) > 40
                or len(answer.split()) > 120
            ):
                continue
            negatives = _get_context_candidates(bm25, question, train_ids, {str(gold["id"])}, 4)
            record = make_sft_record(
                question,
                answer,
                evidence,
                gold,
                negatives,
                answerable=True,
                rng=rng,
                five_context=(len(records) % 10 == 0),
                gold_last=(len(records) % 10 == 1),
            )
            if record:
                records.append(record)
                used_questions.append(question)
                doc_counts[source] += 1
                break

    # Unanswerable examples have only hard-negative context and the exact refusal.
    unsupported = _unsupported_question_candidates()
    for question, key_phrase in unsupported:
        if sum(1 for item in records if not item["answerable"]) >= desired_unanswerable:
            break
        hits = bm25.search(question, top_k=10)
        normalized_hits = " ".join(str(hit["text"]).casefold() for hit in hits)
        if key_phrase.casefold() in normalized_hits:
            continue
        negatives = _get_context_candidates(bm25, question, train_ids, set(), 5)
        record = make_sft_record(
            question,
            "I could not find this in the provided documents.",
            [],
            None,
            negatives,
            answerable=False,
            rng=rng,
            five_context=(len(records) % 10 == 0),
        )
        if record and not _too_similar(question, used_questions):
            records.append(record)
            used_questions.append(question)
    if len(records) < target_count:
        raise ValueError(
            f"Only generated {len(records)} valid samples (target {target_count}); "
            "run with an API/local generator or adjust the target."
        )
    records = records[:target_count]
    train, val = _family_val_split(records, seed=seed)
    return train, val


def print_quality_report(chunks: list[dict[str, Any]]) -> None:
    """Print QA usable counts and representative rejected examples by reason."""
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    rejected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = str(metadata.get("source", "unknown"))
        status = "usable" if chunk.get("usable_for_qa") else "unusable"
        totals[source][status] += 1
        if not chunk.get("usable_for_qa"):
            rejected[str(chunk["qa_rejection_reason"])].append(chunk)
    print("QA quality by document:")
    for source, counts in sorted(totals.items()):
        print(f"{source}: usable={counts['usable']} unusable={counts['unusable']}")
    for reason, items in sorted(rejected.items()):
        print(f"\nRejected type: {reason} ({len(items)} chunks)")
        for chunk in items[:5]:
            title = chunk.get("metadata", {}).get("doc_title")
            excerpt = chunk["text"][:280]
            print(f"- {title} [{chunk['id']}]: {excerpt!r}")


def _sample_only(
    chunks: list[dict[str, Any]],
    split: dict[str, Any],
    generator: QuestionGenerator,
    count: int,
    seed: int,
) -> None:
    """Exercise generation, BM25 negatives, and SFT formatting without writing full data."""
    train_sources = {doc["source"] for doc in split["train_documents"]}
    train_chunks = [
        c
        for c in chunks
        if c.get("usable_for_qa") and c.get("metadata", {}).get("source") in train_sources
    ]
    rng = random.Random(seed)
    if not train_chunks:
        raise ValueError("No usable chunks for a local plumbing sample")
    bm25 = BM25Index(chunks)
    ids = {str(c["id"]) for c in train_chunks}
    for kind in ("answerable", "unanswerable", "five_context", "gold_last"):
        print(f"\nSFT mock sample kind: {kind}")
        printed = 0
        if kind == "unanswerable":
            candidates = [
                ("Can I send my mobile balance to a bank account?", "bank account transfer"),
                ("Does my SIM provide satellite SOS messaging?", "satellite SOS"),
                ("Can roaming charges be paid using cryptocurrency?", "cryptocurrency roaming"),
                ("Will a failed recharge be refunded to my crypto wallet?", "crypto wallet"),
                ("Can family members share a rollover data wallet?", "rollover data wallet"),
                ("Will my operator insure my phone screen?", "screen damage insurance"),
                ("Can unused data cover my electricity bill?", "electricity bill credits"),
                ("Is there a subsidy for satellite phone handsets?", "satellite handset subsidy"),
                ("Can SIM ownership be transferred with a blockchain wallet?", "blockchain wallet"),
                (
                    "Can recharge credit pay my electricity provider?",
                    "recharge electricity provider",
                ),
            ]
            for question, phrase in candidates:
                hits = bm25.search(question, top_k=10)
                hit_text = " ".join(str(h["text"]).casefold() for h in hits)
                if phrase in hit_text:
                    continue
                context = _get_context_candidates(bm25, question, ids, set(), 5)
                record = make_sft_record(
                    question,
                    "I could not find this in the provided documents.",
                    [],
                    None,
                    context,
                    answerable=False,
                    rng=rng,
                    five_context=(kind == "five_context"),
                )
                if record:
                    _print_sample_record(record)
                    printed += 1
                    if printed == count:
                        break
            continue
        for gold in train_chunks:
            if printed >= count:
                break
            generated = generator.generate(gold["text"], 1)
            if not generated:
                continue
            item = generated[0]
            if not evidence_is_supported(item["evidence"], gold["text"]):
                continue
            question = item["question"] + f" ({printed + 1})"
            negatives = _get_context_candidates(bm25, question, ids, {str(gold["id"])}, 4)
            record = make_sft_record(
                question,
                item["answer"],
                item["evidence"],
                gold,
                negatives,
                answerable=True,
                rng=rng,
                five_context=(kind == "five_context"),
                gold_last=(kind == "gold_last"),
            )
            if record:
                _print_sample_record(record)
                printed += 1


def _print_sample_record(record: dict[str, Any]) -> None:
    """Print a compact data sample with IDs and short passage previews."""
    summary = {
        **record,
        "context": [
            {"id": passage["id"], "snippet": " ".join(passage["text"].split())[:100]}
            for passage in record["context"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/split.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--backend", choices=["openai", "local_hf", "mock"], default=None)
    parser.add_argument("--target-count", type=int, default=2000)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/train"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config = load_config(args.config)
    chunks = load_chunks(args.chunks)
    print_quality_report(chunks)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    generator = create_generator(
        args.backend or config.question_generator_backend,
        model_name=config.models.question_generator,
    )
    if args.sample_only:
        _sample_only(chunks, split, generator, args.sample_count, config.seed)
        return
    train, val = build_sft_data(
        chunks,
        split,
        generator,
        target_count=args.target_count,
        seed=config.seed,
        max_doc_share=config.max_doc_share_for_sampling,
    )
    _write_jsonl(args.output_dir / "train.jsonl", train)
    _write_jsonl(args.output_dir / "val.jsonl", val)
    print(f"Wrote {len(train)} train and {len(val)} val samples to {args.output_dir}")


if __name__ == "__main__":
    main()
