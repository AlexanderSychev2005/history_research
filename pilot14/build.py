"""Builds the 14-paragraph test set from avojarot's gold (eval/gold) and the YAGO ids in entity_map.json.

Run from the project root:  YAGO_JSONL=<LELA>/data/yago/yago-entities.jsonl python pilot14/build.py
  texts/<doc>.json + eval/gold/<doc>.json + pilot14/entity_map.json
    -> pilot14/texts/<doc>.json   (the 14 gold paragraphs)
       pilot14/gold.json          (mentions: doc, para, start, end, text, entity = yago id or null)

p004 of both papers is the pilot's pair of paragraphs, on which the pipeline was developed; the other 12 were not
used for any design choice. score.py reports the two groups separately.
"""
import json
import os

DOCS = ("alford2026", "nechaeva2017")
TUNED = [("alford2026", "p004"), ("nechaeva2017", "p004")]


def main():
    emap = json.load(open("pilot14/entity_map.json", encoding="utf-8"))
    mentions = []
    os.makedirs("pilot14/texts", exist_ok=True)
    used = set()
    for doc in DOCS:
        gold = json.load(open(f"eval/gold/{doc}.json", encoding="utf-8"))
        paras = {p["id"]: p["text"] for p in json.load(open(f"texts/{doc}.json", encoding="utf-8"))["paragraphs"]}
        scope = gold["scope"]["paragraphs"]
        json.dump({"doc_id": doc, "paragraphs": [{"id": i, "text": paras[i]} for i in scope]},
                  open(f"pilot14/texts/{doc}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        eids = {e["eid"] for e in gold["entities"]}
        assert eids == set(emap[doc]), f"{doc}: entity map differs from the gold: {eids ^ set(emap[doc])}"
        for m in gold["mentions"]:
            assert paras[m["para"]][m["start"]:m["end"]] == m["text"]
            title = emap[doc][m["entity"]][0]
            used.add(title)
            mentions.append({"doc": doc, "para": m["para"], "start": m["start"], "end": m["end"], "text": m["text"],
                             "entity": f"yago:{title}" if title else None})
    used.discard(None)
    if os.environ.get("YAGO_JSONL"):
        found = set()
        for line in open(os.environ["YAGO_JSONL"], encoding="utf-8"):
            i = line.find('"id": "')
            t = line[i + 12:line.find('"', i + 7)]
            if t in used:
                found.add(t)
        assert found == used, f"ids not in YAGO: {used - found}"
    mentions.sort(key=lambda m: (m["doc"], m["para"], m["start"]))
    json.dump({"tuned": TUNED, "mentions": mentions}, open("pilot14/gold.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n_ent = sum(1 for m in mentions if m["entity"])
    print(len(mentions), "mentions,", n_ent, "with a YAGO entity,", len(mentions) - n_ent, "without;",
          len({(m['doc'], m['para']) for m in mentions}), "paragraphs;", len(used), "distinct YAGO entities")


if __name__ == "__main__":
    main()
