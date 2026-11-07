"""Provenance of dataset/: how paragraphs/ and gold/ were assembled (run once; the JSON files in dataset/ are now the source).

  mentions, facts, entity types and aliases   archive/eval/gold/<doc>.json      (Max; built by archive/eval/make_gold.py)
  YAGO id (or null) for every entity          archive/pilot14/entity_map.json   (Alex/Claude, chosen from the YAGO 4.5 dump
                                              before any system output was looked at)
  text of the 14 paragraphs                   archive/texts/<doc>.json          (archive/prepare/prepare_texts.py, from the PDFs)
  relation vocabulary                         relations.json (copied from archive/schema/history_relations.json)

  from the repository root: YAGO_JSONL=<LELA>/data/yago/yago-entities.jsonl python archive/build_dataset.py

Checks: every offset spells its text, every entity and relation exists, every YAGO id exists in the YAGO 4.5 dump.
"""
import json
import os
from pathlib import Path

DOCS = ("alford2026", "nechaeva2017")
OUT = Path("dataset")
STATUS = ("draft: annotated by Claude (one annotator), not yet reviewed by a historian; YAGO ids chosen from the YAGO 4.5 "
          "dump by the same")


def main():
    emap = json.loads(Path("archive/pilot14/entity_map.json").read_text(encoding="utf-8"))
    relations = set(json.loads(Path("dataset/relations.json").read_text(encoding="utf-8")))
    (OUT / "paragraphs").mkdir(exist_ok=True)
    (OUT / "gold").mkdir(exist_ok=True)
    used, totals = set(), {"paragraphs": 0, "mentions": 0, "entities": 0, "linkable": 0, "facts": 0}
    for doc in DOCS:
        src = json.loads(Path(f"archive/eval/gold/{doc}.json").read_text(encoding="utf-8"))
        text = json.loads(Path(f"archive/texts/{doc}.json").read_text(encoding="utf-8"))
        scope = src["scope"]["paragraphs"]
        paras = {p["id"]: p for p in text["paragraphs"]}
        json.dump({"doc_id": doc, "title": text.get("title"), "source": text.get("source"),
                   "paragraphs": [{"id": i, "text": paras[i]["text"], "section": paras[i].get("section"),
                                   "page": paras[i].get("page")} for i in scope]},
                  (OUT / "paragraphs" / f"{doc}.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
        assert {e["eid"] for e in src["entities"]} == set(emap[doc]), f"{doc}: entity map and gold differ"
        yago = {eid: (f"yago:{v[0]}" if v[0] else None) for eid, v in emap[doc].items()}
        entities = [{"eid": e["eid"], "label": e["label"], "type": e["type"], "aliases": e.get("aliases", []),
                     "yago": yago[e["eid"]], "note": emap[doc][e["eid"]][1]} for e in src["entities"]]
        eids = {e["eid"] for e in entities}
        for m in src["mentions"]:
            assert paras[m["para"]]["text"][m["start"]:m["end"]] == m["text"], (doc, m)
            assert m["entity"] in eids and m["para"] in scope
        for f in src["facts"]:
            assert f["subj"] in eids and f["obj"] in eids and f["rel"] in relations and f["para"] in scope, (doc, f)
        used |= {y for y in yago.values() if y}
        gold = {"doc_id": doc, "annotator": STATUS, "scope": {"paragraphs": scope}, "entities": entities,
                "mentions": [{**m, "yago": yago[m["entity"]]} for m in src["mentions"]], "facts": src["facts"]}
        json.dump(gold, (OUT / "gold" / f"{doc}.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
        totals["paragraphs"] += len(scope)
        totals["mentions"] += len(gold["mentions"])
        totals["entities"] += len(entities)
        totals["linkable"] += sum(1 for e in entities if e["yago"])
        totals["facts"] += len(gold["facts"])
    if os.environ.get("YAGO_JSONL"):
        found = set()
        for line in open(os.environ["YAGO_JSONL"], encoding="utf-8"):
            i = line.find('"id": "')
            t = "yago:" + line[i + 12:line.find('"', i + 7)]
            if t in used:
                found.add(t)
        assert found == used, f"ids not in YAGO: {used - found}"
    print(totals, "| distinct YAGO ids:", len(used))


if __name__ == "__main__":
    main()
