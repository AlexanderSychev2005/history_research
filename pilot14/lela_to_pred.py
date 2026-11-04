"""Converts LELA's output on pilot14/lela_in/paragraphs.jsonl (ids "<doc>|<para>") to pilot14/pred/<run>/<doc>.jsonl.

  python pilot14/lela_to_pred.py pilot14/pred_raw/lela_ext_t03_first.jsonl lela_ext_t03_first
"""
import json
import os
import sys

src, run = sys.argv[1], sys.argv[2]
rows = {}
for line in open(src, encoding="utf-8"):
    d = json.loads(line)
    doc, para = d["id"].split("|")
    rows.setdefault(doc, []).extend({**e, "para": para} for e in d["entities"])
os.makedirs(f"pilot14/pred/{run}", exist_ok=True)
for doc, ents in rows.items():
    ents.sort(key=lambda e: (e["para"], e["start"]))
    open(f"pilot14/pred/{run}/{doc}.jsonl", "w", encoding="utf-8").write(json.dumps({"id": doc, "entities": ents}, ensure_ascii=False) + "\n")
print(run, {doc: len(e) for doc, e in rows.items()}, "mentions")
