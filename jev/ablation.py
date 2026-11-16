"""Linking alone, on the gold mentions, with the parts of jev/linking.py switched on one after another, as Markdown:

    python jev/ablation.py

Link precision / recall / F1 are over strict spans, as hybrid/score.py counts them. jev/results/ holds the output.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from client import Jev                                   # noqa: E402
from linking import cluster, link                        # noqa: E402
from mentions import gold_mentions                       # noqa: E402
from score import DOCS, gold, hybrid_score               # noqa: E402
from sources import Wikipedia, Yago                      # noqa: E402

STEPS = [("search only", set()),
         ("+ dossier: Jev reads every sentence about the entity, not one paragraph", {"dossier"}),
         ("+ kind gate", {"dossier", "kind"}),
         ("+ period gate", {"dossier", "kind", "period"}),
         ("+ link graph: candidates from the articles the sure entities link to", {"dossier", "kind", "period", "graph"}),
         ("+ YAGO: membership, class Person and years", {"dossier", "kind", "period", "graph", "yago"}),
         ("+ YAGO's alternate names in the description (the pipeline)",
          {"dossier", "kind", "period", "graph", "yago", "aliases"})]


def run(jev, wiki, yago, parts, threshold):
    """mention, link and candidate scores of linking with these parts"""
    with tempfile.TemporaryDirectory() as tmp:
        for d in DOCS:
            doc = json.load(open(os.path.join(HERE, "..", "dataset", "paragraphs", d + ".json"), encoding="utf-8"))
            paragraphs = [(p["id"], p["text"]) for p in doc["paragraphs"]]
            mentions = gold_mentions(jev, paragraphs, gold(d))
            entities = {e["eid"]: e for e in cluster(jev, paragraphs, mentions)}
            link(jev, wiki, yago, paragraphs, mentions, list(entities.values()), threshold, parts)
            rows = [{"para": m["para"], "start": m["start"], "end": m["end"], "entity_id": entities[m["entity"]]["yago"],
                     "candidates": [c["yago"] for c in entities[m["entity"]]["candidates"]]} for m in mentions]
            open(os.path.join(tmp, d + ".jsonl"), "w", encoding="utf-8").write(json.dumps({"id": d, "entities": rows}) + "\n")
        return hybrid_score.score(False, tmp)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    jev, wiki, yago = Jev(), Wikipedia(), Yago()
    print("# Jev linking: what each part adds\n\nWritten by `python jev/ablation.py`: linking on the gold mentions, "
          "parts switched on one after another.\n")
    print("| parts | gold id among the candidates | link P / R / F1 at 0.7 | at 0.5 | no-id mentions left unlinked |")
    print("|---|---|---|---|---|")
    for name, parts in STEPS:
        r, at05 = run(jev, wiki, yago, parts, 0.7), run(jev, wiki, yago, parts, 0.5)["link"]
        p, rec, f = r["link"]
        print(f"| {name} | {r['shortlist'][0]} / {r['shortlist'][1]} | {p:.2f} / {rec:.2f} / **{f:.2f}** | "
              f"{at05[0]:.2f} / {at05[1]:.2f} / {at05[2]:.2f} | {r['no_id_unlinked'][0]} / {r['no_id_unlinked'][1]} |",
              flush=True)
