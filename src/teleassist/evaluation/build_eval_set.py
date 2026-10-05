"""Build a held-out QA evaluation set from eval-family documents only."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from teleassist.config import load_config
from teleassist.generation.prompts import ABSTENTION
from teleassist.generation.question_generator import QuestionGenerator, create_generator
from teleassist.resumable import append_resumable
from teleassist.retrieval.bm25 import BM25Index
from teleassist.training.build_sft_data import (
    _too_similar,
    annotate_qa_quality,
    evidence_is_supported,
    load_chunks,
    support_overlap,
)

UNANSWERABLE_QUESTIONS = [
    (
        "Can I transfer leftover prepaid credit directly into my bank account?",
        "bank account transfer",
        "AIRTEL",
    ),
    ("Does my SIM include satellite SOS messaging at no extra cost?", "satellite SOS", "KYC"),
    ("Can I pay roaming charges using cryptocurrency?", "cryptocurrency roaming", "MNP"),
    (
        "Will the operator reimburse a failed recharge to my crypto wallet?",
        "crypto wallet",
        "AIRTEL",
    ),
    (
        "Can I add my family members to a shared rollover data wallet?",
        "rollover data wallet",
        "TCPR",
    ),
    (
        "Does TRAI require operators to insure phones against screen damage?",
        "screen damage insurance",
        "COMPLAINT",
    ),
    (
        "Can unused mobile data be exchanged for electricity bill credits?",
        "electricity bill credits",
        "QOS",
    ),
    (
        "Is there a government subsidy for satellite phone handsets?",
        "satellite handset subsidy",
        "KYC",
    ),
    ("Can I transfer SIM ownership using a blockchain wallet?", "blockchain wallet", "KYC"),
    (
        "Do operators offer automatic bill payment with reward points?",
        "bill payment reward points",
        "AIRTEL",
    ),
    (
        "Can a telecom plan guarantee the same signal strength in every room?",
        "guaranteed indoor signal strength",
        "QOS",
    ),
    (
        "Will my provider compensate me for a missed online meeting?",
        "missed meeting compensation",
        "COMPLAINT",
    ),
    ("Can unused roaming minutes be converted into cash?", "roaming minutes cash", "TCPR"),
    (
        "Does a prepaid plan include free international satellite calls?",
        "international satellite calls",
        "MNP",
    ),
    ("Can I pause my mobile number for a year without paying?", "pause number one year", "AIRTEL"),
    (
        "Will my provider reimburse data lost because my phone battery failed?",
        "battery failure reimbursement",
        "QOS",
    ),
    (
        "Can two operators combine their plans into a single bill?",
        "combine operator plans bill",
        "TCPR",
    ),
    (
        "Does TRAI guarantee minimum mobile speed inside every home?",
        "guaranteed indoor speed every room",
        "QOS",
    ),
    (
        "Can mobile recharge credit pay my electricity provider?",
        "recharge electricity provider",
        "AIRTEL",
    ),
    (
        "Is there a standard tariff for calls from Earth to the Moon?",
        "earth moon call tariff",
        "MNP",
    ),
]


def verify_unanswerable(question: str, key_phrase: str, bm25: BM25Index) -> bool:
    """Keep an unanswerable only when its defining phrase is absent from top BM25 hits."""
    hits = bm25.search(question, top_k=10)
    text = " ".join(str(hit.get("text", "")).casefold() for hit in hits)
    return key_phrase.casefold() not in text


def _valid_answer(item: dict[str, Any], context: str, *, multi: bool) -> bool:
    answer = str(item.get("answer", ""))
    citations = set(int(number) for number in re.findall(r"\[(\d+)\]", answer))
    expected = {1, 2} if multi else {1}
    return (
        expected <= citations
        and evidence_is_supported(item.get("evidence", []), context)
        and support_overlap(re.sub(r"\[\d+\]", "", answer), context) >= 0.20
        and 2 <= len(str(item.get("question", "")).split()) <= 40
        and len(answer.split()) <= 120
    )


def _record(
    index: int,
    item: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    answerable: bool,
    category: str,
) -> dict[str, Any]:
    return {
        "id": f"eval-{index:04d}",
        "question": item["question"],
        "reference_answer": item["answer"],
        "evidence": item.get("evidence", []),
        "gold_chunk_ids": [str(chunk["id"]) for chunk in chunks],
        "answerable": answerable,
        "category": category,
        "verified": False,
    }


def build_eval_set(
    chunks: list[dict[str, Any]],
    split: dict[str, Any],
    generator: QuestionGenerator,
    *,
    bm25: BM25Index | None = None,
    target_count: int = 100,
    seed: int = 42,
    max_doc_share: float = 0.15,
) -> list[dict[str, Any]]:
    """Create 80 answerable (including 15 multi-chunk) and 20 verified unknown items."""
    annotate_qa_quality(chunks)
    eval_sources = {entry["source"] for entry in split["eval_documents"]}
    candidates = [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("source") in eval_sources
        and chunk.get("usable_for_qa", False)
    ]
    if not candidates:
        raise ValueError("No usable QA chunks in eval documents")
    bm25_index = bm25 or BM25Index(chunks)
    rng = random.Random(seed)
    wanted_unanswerable = round(target_count * 0.20)
    wanted_multi = round(target_count * 0.15)
    wanted_answerable_single = target_count - wanted_unanswerable - wanted_multi
    document_cap = max(1, int(target_count * max_doc_share))
    document_counts: Counter[str] = Counter()
    questions: list[str] = []
    records: list[dict[str, Any]] = []

    # Multi-chunk pairs come from the same source when possible, otherwise one family.
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in candidates:
        by_family[str(chunk.get("metadata", {}).get("family", "UNKNOWN"))].append(chunk)
    pairs = []
    for family_chunks in by_family.values():
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in family_chunks:
            by_source[str(chunk["metadata"]["source"])].append(chunk)
        for source_chunks in by_source.values():
            if len(source_chunks) >= 2:
                pairs.extend(zip(source_chunks[::2], source_chunks[1::2], strict=False))
        if len(pairs) < wanted_multi:
            other = family_chunks[1:]
            pairs.extend((family_chunks[0], chunk) for chunk in other)
    rng.shuffle(pairs)
    for first, second in pairs:
        if sum(1 for record in records if len(record["gold_chunk_ids"]) > 1) >= wanted_multi:
            break
        sources = {str(first["metadata"]["source"]), str(second["metadata"]["source"])}
        if any(document_counts[source] >= document_cap for source in sources):
            continue
        context = f"[1]\n{first['text']}\n\n[2]\n{second['text']}"
        for item in generator.generate(context, 3):
            question = str(item.get("question", "")).strip()
            if not _valid_answer(item, context, multi=True) or _too_similar(question, questions):
                continue
            record = _record(
                len(records) + 1,
                item,
                [first, second],
                answerable=True,
                category=str(first.get("metadata", {}).get("family", "UNKNOWN")),
            )
            records.append(record)
            questions.append(question)
            for source in sources:
                document_counts[source] += 1
            break

    # Single passage questions fill the remainder of the answerable quota.
    singles = candidates.copy()
    rng.shuffle(singles)
    for chunk in singles:
        if (
            sum(1 for record in records if record["answerable"])
            >= wanted_multi + wanted_answerable_single
        ):
            break
        source = str(chunk["metadata"]["source"])
        if document_counts[source] >= document_cap:
            continue
        for item in generator.generate(chunk["text"], 3):
            question = str(item.get("question", "")).strip()
            if not _valid_answer(item, chunk["text"], multi=False) or _too_similar(
                question, questions
            ):
                continue
            records.append(
                _record(
                    len(records) + 1,
                    item,
                    [chunk],
                    answerable=True,
                    category=str(chunk.get("metadata", {}).get("family", "UNKNOWN")),
                )
            )
            questions.append(question)
            document_counts[source] += 1
            break

    # Fill unavailable multi slots with singles only after trying all eligible pairs.
    if sum(1 for record in records if record["answerable"]) < target_count - wanted_unanswerable:
        for chunk in singles:
            if (
                sum(1 for record in records if record["answerable"])
                >= target_count - wanted_unanswerable
            ):
                break
            source = str(chunk["metadata"]["source"])
            if document_counts[source] >= document_cap or _too_similar(chunk["text"], questions):
                continue
            for item in generator.generate(chunk["text"], 1):
                question = str(item.get("question", "")).strip()
                if not _valid_answer(item, chunk["text"], multi=False) or _too_similar(
                    question, questions
                ):
                    continue
                records.append(
                    _record(
                        len(records) + 1,
                        item,
                        [chunk],
                        answerable=True,
                        category=str(chunk["metadata"].get("family", "UNKNOWN")),
                    )
                )
                questions.append(question)
                document_counts[source] += 1
                break

    for question, key_phrase, category in UNANSWERABLE_QUESTIONS:
        if sum(1 for record in records if not record["answerable"]) >= wanted_unanswerable:
            break
        if not verify_unanswerable(question, key_phrase, bm25_index) or _too_similar(
            question, questions
        ):
            continue
        item = {"question": question, "answer": ABSTENTION, "evidence": []}
        records.append(_record(len(records) + 1, item, [], answerable=False, category=category))
        questions.append(question)
    if len(records) != target_count:
        raise ValueError(
            f"Built {len(records)} of {target_count} eval items; insufficient eligible "
            "diverse questions or verified unanswerables."
        )
    multi_count = sum(len(record["gold_chunk_ids"]) > 1 for record in records)
    if multi_count < wanted_multi:
        raise ValueError(
            f"Built only {multi_count} multi-chunk eval questions; expected at least "
            f"{wanted_multi}. Add eligible same-document/family chunk pairs or review "
            "the generator."
        )
    records = [{**record, "id": f"eval-{index:04d}"} for index, record in enumerate(records, 1)]
    # Hard assertion: no source document exceeds the sampling cap.
    supplied = Counter(
        source
        for record in records
        for source in {
            str(chunk["metadata"]["source"])
            for chunk in chunks
            if str(chunk["id"]) in record["gold_chunk_ids"]
        }
    )
    if any(count / target_count > max_doc_share for count in supplied.values()):
        raise ValueError("Eval question document share exceeded max_doc_share")
    return records


def print_samples(records: list[dict[str, Any]], count: int) -> None:
    for label, predicate in (
        (
            "answerable single-chunk",
            lambda item: item["answerable"] and len(item["gold_chunk_ids"]) == 1,
        ),
        (
            "answerable multi-chunk",
            lambda item: item["answerable"] and len(item["gold_chunk_ids"]) > 1,
        ),
        ("unanswerable", lambda item: not item["answerable"]),
    ):
        print(f"\nEval mock sample kind: {label}")
        for item in [record for record in records if predicate(record)][:count]:
            print(json.dumps(item, ensure_ascii=False))


def build_mock_samples(
    chunks: list[dict[str, Any]],
    split: dict[str, Any],
    generator: QuestionGenerator,
    *,
    count: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create exactly count examples of each eval sample kind for plumbing checks."""
    annotate_qa_quality(chunks)
    eval_sources = {doc["source"] for doc in split["eval_documents"]}
    candidates = [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("source") in eval_sources
        and chunk.get("usable_for_qa", False)
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    records: list[dict[str, Any]] = []
    questions: list[str] = []
    singles = 0
    for chunk in candidates:
        if singles >= count:
            break
        for item in generator.generate(chunk["text"], 1):
            context = chunk["text"]
            if not _valid_answer(item, context, multi=False):
                continue
            item = {**item, "question": f"{item['question']} (mock sample {singles + 1})"}
            records.append(
                _record(
                    len(records) + 1,
                    item,
                    [chunk],
                    answerable=True,
                    category=str(chunk["metadata"].get("family", "UNKNOWN")),
                )
            )
            questions.append(item["question"])
            singles += 1
            break

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in candidates:
        by_family[str(chunk.get("metadata", {}).get("family", "UNKNOWN"))].append(chunk)
    pairs = []
    for family_chunks in by_family.values():
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in family_chunks:
            by_source[str(chunk["metadata"]["source"])].append(chunk)
        for source_chunks in by_source.values():
            pairs.extend(zip(source_chunks[::2], source_chunks[1::2], strict=False))
        if len(pairs) < count:
            pairs.extend((family_chunks[0], other) for other in family_chunks[1:])
    rng.shuffle(pairs)
    multi = 0
    for first, second in pairs:
        if multi >= count:
            break
        context = f"[1]\n{first['text']}\n\n[2]\n{second['text']}"
        for item in generator.generate(context, 1):
            if not _valid_answer(item, context, multi=True):
                continue
            item = {**item, "question": f"{item['question']} (mock multi {multi + 1})"}
            records.append(
                _record(
                    len(records) + 1,
                    item,
                    [first, second],
                    answerable=True,
                    category=str(first["metadata"].get("family", "UNKNOWN")),
                )
            )
            questions.append(item["question"])
            multi += 1
            break

    bm25 = BM25Index(chunks)
    unanswerable = 0
    for question, phrase, category in UNANSWERABLE_QUESTIONS:
        if unanswerable >= count:
            break
        if not verify_unanswerable(question, phrase, bm25):
            continue
        records.append(
            _record(
                len(records) + 1,
                {"question": question, "answer": ABSTENTION, "evidence": []},
                [],
                answerable=False,
                category=category,
            )
        )
        unanswerable += 1
    if (singles, multi, unanswerable) != (count, count, count):
        raise ValueError(
            f"Mock samples available: single={singles}, multi={multi}, "
            f"unanswerable={unanswerable}; expected {count} of each."
        )
    return records


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/split.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--backend", choices=["openai", "local_hf", "mock"], default=None)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/eval/eval_set.jsonl"))
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    chunks = load_chunks(args.chunks)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    backend = args.backend or config.question_generator_backend
    generator = create_generator(backend, model_name=config.models.question_generator)
    target_count = (
        min(args.sample_count * 7, args.target_count) if args.sample_only else args.target_count
    )
    if args.sample_only:
        records = build_mock_samples(
            chunks, split, generator, count=args.sample_count, seed=config.seed
        )
        print_samples(records, args.sample_count)
        return
    records = build_eval_set(
        chunks,
        split,
        generator,
        target_count=target_count,
        seed=config.seed,
        max_doc_share=config.max_doc_share_for_sampling,
    )
    if args.max_chunks is not None or args.out_dir is not None:
        out_dir = args.out_dir or args.output.parent
        append_resumable(
            chunks,
            records,
            out_dir=out_dir,
            max_chunks=args.max_chunks,
            chunk_ids=lambda record: record.get("gold_chunk_ids", []),
            filename=args.output.name,
        )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} eval items to {args.output}")


if __name__ == "__main__":
    main()
