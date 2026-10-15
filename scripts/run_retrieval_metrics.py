"""Retrieval-only metrics on the verified, answerable eval questions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from teleassist.retrieval.pipeline import Retriever

MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
KS = [1, 3, 5, 10]


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def frac(k, n):
    return f"{k}/{n} ({100 * k / max(1, n):.0f}%)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--index", default="data/processed/dense_index.npz")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--eval", default="data/eval/eval_set.jsonl")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunks = {str(c["id"]): c for c in read_jsonl(args.chunks)}
    items = [r for r in read_jsonl(args.eval) if r.get("verified") and r["answerable"]]
    n = len(items)
    print(f"{n} answerable verified questions | {len(chunks)} chunks | label={args.label}")

    def meta(cid, key):
        return str(chunks.get(str(cid), {}).get("metadata", {}).get(key, "?"))

    retriever = Retriever.from_files(
        args.chunks, dense_index_path=args.index, config_path=args.config
    )
    summary, per_question = [], []
    for mode in MODES:
        hits = {k: 0 for k in KS}
        rec5 = rec10 = rr = 0.0
        doc5 = fam5 = 0
        times = []
        for it in items:
            gold = [str(g) for g in it["gold_chunk_ids"]]
            start = time.perf_counter()
            res = retriever.search(it["question"], mode=mode, top_k=10)
            times.append((time.perf_counter() - start) * 1000)
            ids = [str(r["id"]) for r in res]
            ranks = [ids.index(g) + 1 for g in gold if g in ids]
            first = min(ranks) if ranks else None
            for k in KS:
                hits[k] += int(first is not None and first <= k)
            rec5 += len(set(gold) & set(ids[:5])) / len(gold)
            rec10 += len(set(gold) & set(ids[:10])) / len(gold)
            rr += 1 / first if first else 0.0
            gold_docs = {meta(g, "source") for g in gold}
            gold_fams = {meta(g, "family") for g in gold}
            doc5 += int(bool(gold_docs & {meta(i, "source") for i in ids[:5]}))
            fam5 += int(bool(gold_fams & {meta(i, "family") for i in ids[:5]}))
            per_question.append(
                {"mode": mode, "id": it["id"], "first_gold_rank": first or "",
                 "question": it["question"][:100]}
            )
        lo, hi = wilson(hits[5], n)
        summary.append({
            "mode": mode,
            "hit@1": frac(hits[1], n),
            "hit@3": frac(hits[3], n),
            "hit@5": frac(hits[5], n),
            "hit@5_95%_interval": f"{100 * lo:.0f}-{100 * hi:.0f}%",
            "hit@10": frac(hits[10], n),
            "recall@5": round(rec5 / n, 3),
            "recall@10": round(rec10 / n, 3),
            "mrr@10": round(rr / n, 3),
            "same_doc_hit@5": frac(doc5, n),
            "same_family_hit@5": frac(fam5, n),
            "mean_latency_ms": round(sum(times) / len(times)),
        })
        print(f"done: {mode}", flush=True)

    base = out / f"retrieval_{args.label}"
    with open(f"{base}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with open(f"{base}_per_question.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_question[0].keys()))
        writer.writeheader()
        writer.writerows(per_question)
    cols = list(summary[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in summary]
    Path(f"{base}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
