"""Scores runs in hybrid/pred/ against dataset/gold: mentions and links, over all 14 paragraphs.

    python hybrid/score.py hybrid_120b-20b_r1 names_120b_r1 [--overlap]
mention  a predicted span matches a gold span of the same paragraph if it is equal (strict) or overlaps it (--overlap),
         each gold span at most once
link     the matched mention also carries the gold YAGO id; a gold mention without an id never counts, and a link
         predicted for it is a false positive
cand     of the matched mentions with a gold id: share whose id is in the run's shortlist (only runs that have one)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "dataset", "gold")
DOCS = ("alford2026", "nechaeva2017")


def prf(tp, n_pred, n_gold):
    p, r = tp / n_pred if n_pred else 0.0, tp / n_gold if n_gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def score(run, overlap):
    gold = [{**m, "doc": d} for d in DOCS for m in json.load(open(os.path.join(DATA, d + ".json"), encoding="utf-8"))["mentions"]]
    pred = [{**e, "doc": d} for d in DOCS
            for e in json.loads(open(os.path.join(HERE, "pred", run, d + ".jsonl"), encoding="utf-8").readline())["entities"]]
    res = {}
    for name, keep in {"all 14": lambda k: True}.items():
        G = [m for m in gold if keep((m["doc"], m["para"]))]
        P = [e for e in pred if keep((e["doc"], e["para"]))]
        used, tp_m, tp_l, c_hit, c_n, nil_n, nil_ok = set(), 0, 0, 0, 0, 0, 0
        for e in P:
            best, best_ov = None, 0
            for i, g in enumerate(G):
                if i in used or (g["doc"], g["para"]) != (e["doc"], e["para"]):
                    continue
                ov = min(e["end"], g["end"]) - max(e["start"], g["start"])
                if (e["start"], e["end"]) == (g["start"], g["end"]) or (overlap and ov > best_ov):
                    best, best_ov = i, max(ov, 1)
                    if (e["start"], e["end"]) == (g["start"], g["end"]):
                        break
            if best is None:
                continue
            used.add(best)
            g, tp_m = G[best], tp_m + 1
            link = e.get("entity_id")
            if g["yago"]:
                tp_l += link == g["yago"]
                if "candidates" in e:
                    c_n += 1
                    c_hit += g["yago"] in [c["entity_id"] for c in e["candidates"]]
            else:
                nil_n += 1
                nil_ok += link is None
        res[name] = {"gold": len(G), "pred": len(P), "mention": prf(tp_m, len(P), len(G)),
                     "link": prf(tp_l, sum(1 for e in P if e.get("entity_id")), sum(1 for g in G if g["yago"])),
                     "cand": (c_hit, c_n) if c_n else None, "no_id_left_unlinked": (nil_ok, nil_n)}
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    overlap = "--overlap" in sys.argv
    print(f"spans: {'overlap' if overlap else 'strict'}\n")
    print("| run | group | gold | mention P / R / F1 | link P / R / F1 | right id in shortlist | no-id left unlinked |")
    print("|---|---|---|---|---|---|---|")
    for run in [a for a in sys.argv[1:] if not a.startswith("--")]:
        for name, r in score(run, overlap).items():
            m, l = r["mention"], r["link"]
            c = f"{r['cand'][0]}/{r['cand'][1]}" if r["cand"] else "n/a"
            print(f"| {run} | {name} | {r['gold']} | {m[0]:.2f} / {m[1]:.2f} / {m[2]:.2f} | "
                  f"{l[0]:.2f} / {l[1]:.2f} / {l[2]:.2f} | {c} | {r['no_id_left_unlinked'][0]}/{r['no_id_left_unlinked'][1]} |")
