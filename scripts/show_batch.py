import json, sys
sys.stdout.reconfigure(errors="replace")
start, count = int(sys.argv[1]), int(sys.argv[2])
chunks = {}
for l in open("data/processed/chunks.jsonl", encoding="utf-8"):
    c = json.loads(l)
    chunks[str(c["id"])] = (c["metadata"]["source"], c["text"])
items = [json.loads(l) for l in open("data/eval/eval_set.jsonl", encoding="utf-8") if l.strip()]
todo = [i for i in items if not i.get("verified", False)]
print(f"{len(todo)} unreviewed in total; showing {start}-{start+count-1}\n")
for it in todo[start:start+count]:
    print("ID:", it["id"], "| answerable:", it["answerable"])
    print("Q:", it["question"])
    print("A:", it["reference_answer"])
    print("EVIDENCE:", it.get("evidence"))
    for g in it["gold_chunk_ids"]:
        src, text = chunks.get(str(g), ("?", ""))
        print("SOURCE:", src)
        print("PASSAGE:", " ".join(text.split())[:700])
    print("-" * 60)
