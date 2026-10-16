"""Test a reranker-score abstention rule: tune on train/val questions, test once on eval."""

import argparse
import json
import random
import re
import sys

from teleassist.retrieval.pipeline import Retriever


def read(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def norm(text):
    return re.sub(r"\s+", " ", text).casefold()


def top_score(results):
    first = results[0]
    for key in ("rerank_score", "score"):
        if first.get(key) is not None:
            return float(first[key])
    sys.exit(f"No score field found. Keys are: {sorted(first.keys())}")


def evidence_present(quotes, results):
    texts = [norm(r["text"]) for r in results]
    cleaned = [norm(q) for q in quotes if isinstance(q, str) and q.strip()]
    return any(q in t for q in cleaned for t in texts)


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(pos, neg):
    values = sorted(set(pos + neg))
    cands = [values[0] - 1] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)] + [values[-1] + 1]
    best = None
    for t in cands:
        tpr = sum(s >= t for s in pos) / len(pos)
        tnr = sum(s < t for s in neg) / len(neg)
        bal = (tpr + tnr) / 2
        if best is None or bal > best[0]:
            best = (bal, t, tpr, tnr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks_official.jsonl")
    ap.add_argument("--index", default="data/processed/dense_index_official.npz")
    ap.add_argument("--predictions", default="results/eval_v2/predictions.jsonl")
    ap.add_argument("--train-sample", type=int, default=200)
    args = ap.parse_args()

    retriever = Retriever.from_files(
        args.chunks, dense_index_path=args.index, config_path="configs/default.yaml"
    )

    def score_rows(rows):
        pos, neg = [], []
        for i, row in enumerate(rows, 1):
            res = retriever.search(row["question"], mode="hybrid_rerank", top_k=5)
            s = top_score(res)
            ok = row["answerable"] and evidence_present(row.get("evidence", []), res)
            (pos if ok else neg).append(s)
            if i % 50 == 0:
                print(f"  scored {i}/{len(rows)}", flush=True)
        return pos, neg

    val = read("data/train/val.jsonl")
    train = read("data/train/train.jsonl")
    random.Random(1).shuffle(train)
    tuning = val + train[: args.train_sample]
    print(f"tuning on {len(tuning)} train/val questions")
    pos, neg = score_rows(tuning)
    bal, thr, tpr, tnr = best_threshold(pos, neg)
    print(f"should answer: {len(pos)} | should refuse: {len(neg)}")
    print(f"tuning AUROC: {auroc(pos, neg):.3f}")
    print(f"chosen threshold: {thr:.3f} (answers kept {tpr:.0%}, refusals caught {tnr:.0%})")

    ev = [r for r in read("data/eval/eval_set.jsonl") if r.get("verified")]
    preds = {
        p["id"]: p
        for p in read(args.predictions)
        if p["system"] == "F_rag_hybrid_rerank_lora"
    }
    groups = {"unanswerable": [], "answerable_evidence_in": [], "answerable_evidence_absent": []}
    for row in ev:
        res = retriever.search(row["question"], mode="hybrid_rerank", top_k=5)
        s = top_score(res)
        gated = s < thr
        pred = preds.get(row["id"])
        refused = gated or bool(pred and pred["abstained"])
        if not row["answerable"]:
            groups["unanswerable"].append((s, gated, refused, False))
            continue
        present = evidence_present(row.get("evidence", []), res)
        cited = False
        if pred and present and not refused:
            ids = [str(i) for i in pred["retrieved_ids"]]
            quotes = [norm(q) for q in row["evidence"] if isinstance(q, str) and q.strip()]
            ctx = [norm(t) for t in pred["context"]]
            cited = any(
                str(c) in ids and any(q in ctx[ids.index(str(c))] for q in quotes)
                for c in pred["cited_ids"]
            )
        key = "answerable_evidence_in" if present else "answerable_evidence_absent"
        groups[key].append((s, gated, refused, cited))

    print("\n=== eval set, system F v1 with the score gate (threshold fixed beforehand) ===")
    for name, items in groups.items():
        n = len(items)
        gate = sum(g for _, g, _, _ in items)
        refused = sum(r for _, _, r, _ in items)
        extra = ""
        if name == "answerable_evidence_in":
            extra = f" | answered and cited the evidence: {sum(c for _, _, _, c in items)}"
        print(f"{name:28s} n={n:2d} | gate refused {gate:2d} | refused overall {refused:2d}{extra}")
    in_scores = [s for s, _, _, _ in groups["answerable_evidence_in"]]
    out_scores = [s for s, _, _, _ in groups["unanswerable"] + groups["answerable_evidence_absent"]]
    print(f"eval AUROC (evidence reached the model vs not): {auroc(in_scores, out_scores):.3f}")
    print("baseline F v1 without the gate: refused 6/10 unanswerable, 0/16 evidence-absent")


if __name__ == "__main__":
    main()

