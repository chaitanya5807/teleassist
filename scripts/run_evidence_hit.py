import argparse, json, re
from teleassist.retrieval.pipeline import Retriever

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
def norm(s): return re.sub(r"\s+", " ", s).casefold()

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
ap.add_argument("--index", default="data/processed/dense_index.npz")
args = ap.parse_args()
items = [r for r in read("data/eval/eval_set.jsonl")
         if r.get("verified") and r["answerable"] and r.get("evidence")]
ret = Retriever.from_files(args.chunks, dense_index_path=args.index, config_path="configs/default.yaml")
print(f"{len(items)} questions with evidence quotes | chunks: {args.chunks}")
print("mode | evidence in top-1 | top-3 | top-5 | top-10")
for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
    counts = {1: 0, 3: 0, 5: 0, 10: 0}
    for it in items:
        quotes = [norm(q) for q in it["evidence"] if isinstance(q, str) and q.strip()]
        res = ret.search(it["question"], mode=mode, top_k=10)
        texts = [norm(r["text"]) for r in res]
        for k in counts:
            if any(q in t for q in quotes for t in texts[:k]):
                counts[k] += 1
    n = len(items)
    print(mode, "|", " | ".join(f"{counts[k]}/{n}" for k in (1, 3, 5, 10)))
