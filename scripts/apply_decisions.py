import json, sys
dec = json.load(open("scripts/decisions.json", encoding="utf-8-sig"))
path = "data/eval/eval_set.jsonl"
items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
by_id = {i["id"]: i for i in items}
deleted = []
for d in dec:
    it = by_id.get(d["id"])
    if it is None:
        print("NOT FOUND:", d["id"]); continue
    if d["action"] == "delete":
        deleted.append(it); items.remove(it)
    else:
        if d["action"] == "edit":
            it["question"] = d["question"]; it["reference_answer"] = d["answer"]
        it["verified"] = True
with open(path, "w", encoding="utf-8") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
with open("data/eval/eval_set_deleted.jsonl", "a", encoding="utf-8") as f:
    for it in deleted: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("verified:", sum(1 for i in items if i.get("verified")), "| unreviewed:", sum(1 for i in items if not i.get("verified")), "| total:", len(items))

