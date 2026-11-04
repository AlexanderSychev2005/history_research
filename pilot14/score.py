"""Scores runs in pilot14/pred/ against pilot14/gold.json, in three groups: all 14 paragraphs, the two p004 paragraphs
(the pilot's, on which the pipeline was developed) and the other 12 (not used for any design choice).

    python pilot14/score.py hyb_120b_clean_r1 hyb_120b_clean_r2 ...  [--overlap]
mention  a predicted span matches a gold span of the same paragraph if it is equal (strict) or overlaps it (--overlap),
         each gold span at most once
link     the matched mention also carries the gold entity; gold mentions without an entity never count, a link
         predicted for them is a false positive
cand     of the matched mentions with a gold entity: share whose entity is in the run's shortlist (n/a without one)
Writes pilot14/results/<run>.json and prints a table.
"""
import json
import os
import sys


def prf(tp, n_pred, n_gold):
    p, r = tp / n_pred if n_pred else 0.0, tp / n_gold if n_gold else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


def load(run):
    out = []
    for doc in ("alford2026", "nechaeva2017"):
        row = json.loads(open(f"pilot14/pred/{run}/{doc}.jsonl", encoding="utf-8").readline())
        out += [{**e, "doc": doc} for e in row["entities"]]
    return out


def score(run, gold, tuned, overlap):
    pred = load(run)
    groups = {"all": lambda k: True, "p004 (tuned)": lambda k: k in tuned, "12 new": lambda k: k not in tuned}
    res = {}
    for name, keep in groups.items():
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
            if g["entity"]:
                tp_l += link == g["entity"]
                if "candidates" in e:
                    c_n += 1
                    c_hit += g["entity"] in [c["entity_id"] for c in e["candidates"]]
            else:
                nil_n += 1
                nil_ok += link is None
        n_link_pred = sum(1 for e in P if e.get("entity_id"))
        n_link_gold = sum(1 for g in G if g["entity"])
        res[name] = {"gold": len(G), "pred": len(P), "mention": prf(tp_m, len(P), len(G)),
                     "link": prf(tp_l, n_link_pred, n_link_gold), "cand": [c_hit, c_n] if c_n else None,
                     "no_entity_left_unlinked": [nil_ok, nil_n]}
    return res


def main(args):
    overlap = "--overlap" in args
    runs = [a for a in args if not a.startswith("--")]
    g = json.load(open("pilot14/gold.json", encoding="utf-8"))
    tuned = {tuple(t) for t in g["tuned"]}
    os.makedirs("pilot14/results", exist_ok=True)
    print(f"spans: {'overlap' if overlap else 'strict'}\n")
    print("| run | group | gold | mention P / R / F1 | link P / R / F1 | right entity in shortlist | no-entity left unlinked |")
    print("|---|---|---|---|---|---|---|")
    for run in runs:
        res = score(run, g["mentions"], tuned, overlap)
        json.dump(res, open(f"pilot14/results/{run}{'_overlap' if overlap else ''}.json", "w", encoding="utf-8"), indent=1)
        for name, r in res.items():
            m, l = r["mention"], r["link"]
            c = f"{r['cand'][0]}/{r['cand'][1]}" if r["cand"] else "n/a"
            print(f"| {run} | {name} | {r['gold']} | {m[0]:.2f} / {m[1]:.2f} / {m[2]:.2f} | {l[0]:.2f} / {l[1]:.2f} / {l[2]:.2f} | "
                  f"{c} | {r['no_entity_left_unlinked'][0]}/{r['no_entity_left_unlinked'][1]} |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
