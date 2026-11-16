"""The Jev pipeline, end to end: paragraphs in; mentions, entities with YAGO ids, and relations out.

    python jev/pipeline.py                     the 14 paragraphs of dataset/, written to jev/pred/
    python jev/pipeline.py --gold-mentions     the same from the gold mentions (stages 2 and 3 alone), to jev/pred/gold-mentions/
    python jev/pipeline.py --input texts/alford2026.json --out out/     any {doc_id, paragraphs: [{id, text}]} file
    python jev/score.py                        scores of jev/pred against dataset/gold

Every answer is cached in jev/cache/, so a second run costs nothing; delete the folder to ask again.
Needs JEV_API_KEY (.env). Python 3.9+, standard library only.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from client import Jev                                             # noqa: E402
from linking import cluster, link                                  # noqa: E402
from mentions import find_mentions, gold_mentions                  # noqa: E402
from questions import T_FACT, T_LINK, T_NAME                       # noqa: E402
from relations import decide, facts                                # noqa: E402
from sources import Wikipedia, Yago                                # noqa: E402

DATA = os.path.join(HERE, "..", "dataset")
DOCS = ("alford2026", "nechaeva2017")


def run(jev, wiki, yago, doc, gold=None):
    """-> (the prediction, every decision of the relation stage)"""
    paragraphs = [(p["id"], p["text"]) for p in doc["paragraphs"]]
    mentions = gold_mentions(jev, paragraphs, gold) if gold else find_mentions(jev, paragraphs)
    entities = cluster(jev, paragraphs, mentions)
    link(jev, wiki, yago, paragraphs, mentions, entities)
    by_id = {e["eid"]: e for e in entities}
    for m in mentions:                                             # the relation stage reads the entity's kinds
        m["kinds"], m["yago"] = by_id[m["entity"]]["kinds"], by_id[m["entity"]]["yago"]
    decisions = decide(jev, paragraphs, mentions)
    found = facts(decisions)
    records = yago.entities([e["wikipedia"] for e in entities if e["wikipedia"]])
    for f in found:
        s, o = by_id[f["subj"]], by_id[f["obj"]]
        f["subj_yago"], f["obj_yago"] = s["yago"], o["yago"]
        # does YAGO already state it? True / False, None if an end is unlinked or the relation is ours (ex:)
        f["in_yago"] = yago.has_fact(f["rel"], records.get(s["wikipedia"]), records.get(o["wikipedia"]))
    for e in entities:                                             # a candidate: its probability, or why it was dropped
        e["candidates"] = [{k: c[k] for k in ("title", "yago", "p", "out", "via") if c.get(k) is not None}
                           for c in sorted(e["candidates"], key=lambda c: -c["p"])]
    pred = {"doc_id": doc["doc_id"], "model": jev.model, "thresholds": {"name": T_NAME, "link": T_LINK, "fact": T_FACT},
            "mentions": [{k: m[k] for k in ("para", "start", "end", "text", "type", "p", "entity", "yago")}
                         for m in mentions],
            "entities": entities, "facts": found}
    return pred, decisions


def save(pred, decisions, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, pred["doc_id"])
    json.dump(pred, open(path + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(decisions, open(path + ".decisions.json", "w", encoding="utf-8"), ensure_ascii=False)
    candidates = {e["eid"]: [c["yago"] for c in e["candidates"]] for e in pred["entities"]}
    rows = [{"para": m["para"], "text": m["text"], "start": m["start"], "end": m["end"], "entity_id": m["yago"],
             "candidates": candidates[m["entity"]]} for m in pred["mentions"]]         # the format of hybrid/pred
    open(path + ".jsonl", "w", encoding="utf-8").write(
        json.dumps({"id": pred["doc_id"], "entities": rows}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-mentions", action="store_true", help="start from the gold mentions of dataset/gold")
    ap.add_argument("--input", nargs="*", help="paragraph files instead of dataset/paragraphs")
    ap.add_argument("--out", help="output folder (default jev/pred, or jev/pred/gold-mentions)")
    a = ap.parse_args()
    if a.gold_mentions and a.input:
        ap.error("--gold-mentions works on the dataset only")
    files = a.input or [os.path.join(DATA, "paragraphs", d + ".json") for d in DOCS]
    out = a.out or os.path.join(HERE, "pred", "gold-mentions" if a.gold_mentions else "")
    jev, wiki, yago, t = Jev(), Wikipedia(), Yago(), time.time()
    for path in files:
        doc = json.load(open(path, encoding="utf-8"))
        gold = (json.load(open(os.path.join(DATA, "gold", doc["doc_id"] + ".json"), encoding="utf-8"))
                if a.gold_mentions else None)
        pred, decisions = run(jev, wiki, yago, doc, gold)
        save(pred, decisions, out)
        linked = sum(1 for e in pred["entities"] if e["yago"])
        print(f"{doc['doc_id']}: {len(pred['mentions'])} mentions, {len(pred['entities'])} entities ({linked} linked), "
              f"{len(pred['facts'])} facts", flush=True)
    print(f"Jev: {jev.report()}; {time.time() - t:.0f} s; saved to {os.path.relpath(out)}")
