"""Run the six-system ablation on the verified eval set and write result tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from teleassist.generation.prompts import ABSTENTION
from teleassist.generation.rag import RAGAnswerer

SYSTEMS = [
    ("A_base_no_retrieval", "no_retrieval", False),
    ("B_rag_dense", "dense", False),
    ("C_rag_bm25", "bm25", False),
    ("D_rag_hybrid", "hybrid", False),
    ("E_rag_hybrid_rerank", "hybrid_rerank", False),
    ("F_rag_hybrid_rerank_lora", "hybrid_rerank", True),
]
STOP = set(
    "a an the of to in on for and or is are was were be been it this that with as by at "
    "from can you your i my me do does will if not no yes they their there".split()
)


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def toks(text):
    return re.findall(r"[a-z0-9]+", re.sub(r"\[\d+\]", " ", text.lower()))


def f1(pred, ref):
    p, r = toks(pred), toks(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def containment(pred, ref):
    ref_t = [t for t in toks(ref) if t not in STOP]
    if not ref_t:
        return 0.0
    pred_t = set(toks(pred))
    return sum(t in pred_t for t in ref_t) / len(ref_t)


def unsupported(pred, passages):
    pred_t = [t for t in toks(pred) if t not in STOP]
    if not pred_t:
        return 0.0
    ctx = set(toks(" ".join(passages)))
    return sum(t not in ctx for t in pred_t) / len(pred_t)


def is_abstain(answer):
    return ABSTENTION.lower() in answer.lower()


def frac(flags):
    return f"{sum(flags)}/{len(flags)} ({100 * sum(flags) / max(1, len(flags)):.0f}%)"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def p95(values):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))] if ordered else 0.0


def summarize(name, mode, items, done):
    recs = {it["id"]: done[(name, it["id"])] for it in items if (name, it["id"]) in done}
    answerable = [it for it in items if it["answerable"] and it["id"] in recs]
    unanswerable = [it for it in items if not it["answerable"] and it["id"] in recs]
    f1s, cons, cited, hits, false_ref, unsup = [], [], [], [], [], []
    for it in answerable:
        r = recs[it["id"]]
        gold = {str(g) for g in it["gold_chunk_ids"]}
        f1s.append(f1(r["answer"], it["reference_answer"]))
        cons.append(containment(r["answer"], it["reference_answer"]))
        cited.append(bool(gold & {str(c) for c in r["cited_ids"]}))
        hits.append(bool(gold & {str(c) for c in r["retrieved_ids"]}))
        false_ref.append(r["abstained"])
        if r["context"] and not r["abstained"]:
            unsup.append(unsupported(r["answer"], r["context"]))
    refusals = [recs[it["id"]]["abstained"] for it in unanswerable]
    lat = [r["latency_total_ms"] for r in recs.values()]
    none = mode == "no_retrieval"
    return {
        "system": name,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "token_f1": round(mean(f1s), 3),
        "containment": round(mean(cons), 3),
        "retrieval_hit_at_5": "n/a" if none else frac(hits),
        "cited_gold_chunk": "n/a" if none else frac(cited),
        "false_refusal": frac(false_ref),
        "correct_refusal": frac(refusals),
        "unsupported_proxy": "n/a" if none else round(mean(unsup), 3),
        "latency_mean_ms": round(mean(lat)),
        "latency_p95_ms": round(p95(lat)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", default="data/eval/eval_set.jsonl")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--index", default="data/processed/dense_index.npz")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use N questions")
    ap.add_argument("--systems", default=None, help="comma-separated system names")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    items = [r for r in read_jsonl(args.eval) if r.get("verified")]
    if args.limit:
        half = max(1, args.limit // 2)
        ans = [r for r in items if r["answerable"]][:half]
        una = [r for r in items if not r["answerable"]][: args.limit - half]
        items = ans + una
    print(f"{len(items)} verified questions")

    done = {}
    if pred_path.exists():
        for rec in read_jsonl(pred_path):
            done[(rec["system"], rec["id"])] = rec

    rag = RAGAnswerer(
        config_path=args.config,
        chunks_path=args.chunks,
        index_path=args.index,
        adapter_path=args.adapter,
    )
    rag.llm.load_in_4bit = True
    print("generator:", rag.llm.model_name, "| adapter:", args.adapter)
    chosen = set(args.systems.split(",")) if args.systems else None

    for name, mode, lora in SYSTEMS:
        if chosen and name not in chosen:
            continue
        if lora and not args.adapter:
            print("skipping", name, "(no --adapter given)")
            continue
        todo = [it for it in items if (name, it["id"]) not in done]
        for k, it in enumerate(todo, 1):
            res = rag.answer(it["question"], mode=mode, use_lora=lora)
            rec = {
                "system": name,
                "id": it["id"],
                "answerable": it["answerable"],
                "question": it["question"],
                "answer": res["answer"],
                "abstained": is_abstain(res["answer"]),
                "cited_ids": [c["id"] for c in res["citations"]],
                "invalid_citations": res["invalid_citation_numbers"],
                "retrieved_ids": [c.get("id") for c in res["retrieved_chunks"]],
                "context": [c.get("text", "") for c in res["retrieved_chunks"]],
                "latency_total_ms": res["latency_ms"]["total"],
            }
            with open(pred_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(name, it["id"])] = rec
            if k % 5 == 0 or k == len(todo):
                print(f"{name}: {k}/{len(todo)}", flush=True)

    rows = [summarize(n, m, items, done) for n, m, _ in SYSTEMS if any(k[0] == n for k in done)]
    if not rows:
        print("nothing to summarize")
        return
    with open(out / "ablation.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
