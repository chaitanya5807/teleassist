"""Split generation results by whether the evidence quote reached the model."""

import collections
import json
import re
import sys


def read(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def norm(text):
    return re.sub(r"\s+", " ", text).casefold()


ev = {
    e["id"]: e
    for e in read("data/eval/eval_set.jsonl")
    if e.get("verified") and e["answerable"] and e.get("evidence")
}
for path in sys.argv[1:]:
    print("=====", path)
    preds = read(path)
    for system in sorted({p["system"] for p in preds}):
        if system.startswith("A_"):
            continue
        s = collections.Counter()
        for p in preds:
            if p["system"] != system or p["id"] not in ev:
                continue
            quotes = [norm(q) for q in ev[p["id"]]["evidence"] if isinstance(q, str) and q.strip()]
            ctx = [norm(t) for t in p["context"]]
            ids = [str(i) for i in p["retrieved_ids"]]
            present = any(q in t for q in quotes for t in ctx)
            cited_ok = any(
                str(c) in ids and any(q in ctx[ids.index(str(c))] for q in quotes)
                for c in p["cited_ids"]
            )
            key = "present" if present else "absent"
            s[key] += 1
            s[key + "_refused"] += int(p["abstained"])
            s[key + "_cited"] += int(cited_ok)
        print(system)
        print(
            f"  evidence IN passages : {s['present']:2d} | refused {s['present_refused']:2d}"
            f" | cited a passage containing the evidence {s['present_cited']:2d}"
        )
        print(
            f"  evidence NOT in them : {s['absent']:2d} | refused {s['absent_refused']:2d}"
            f" | answered anyway {s['absent'] - s['absent_refused']:2d}"
        )
