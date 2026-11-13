"""Scores predictions against dataset/gold, over all 14 paragraphs:  python hybrid/score.py [pred_dir]   (default hybrid/pred)

mention  a predicted span matches a gold span of the same paragraph if it is equal (strict) or overlaps it (overlap),
         each gold span at most once
link     the matched mention also carries the gold YAGO id; a gold mention without an id never counts, and a link
         predicted for it is a false positive
shortlist  of the matched mentions with a gold id, how many have the gold id among the candidates
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = ("alford2026", "nechaeva2017")


def prf(tp, n_pred, n_gold):
    p, r = tp / n_pred if n_pred else 0.0, tp / n_gold if n_gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def score(overlap, pred_dir=os.path.join(HERE, "pred")):
    gold = [{**m, "doc": d} for d in DOCS
            for m in json.load(open(os.path.join(HERE, "..", "dataset", "gold", d + ".json"), encoding="utf-8"))["mentions"]]
    pred = [{**e, "doc": d} for d in DOCS
            for e in json.loads(open(os.path.join(pred_dir, d + ".jsonl"), encoding="utf-8").readline())["entities"]]
    used, tp_m, tp_l, hit, n_hit, no_id, no_id_ok = set(), 0, 0, 0, 0, 0, 0
    for e in pred:
        best, best_ov = None, 0
        for i, g in enumerate(gold):
            if i in used or (g["doc"], g["para"]) != (e["doc"], e["para"]):
                continue
            ov = min(e["end"], g["end"]) - max(e["start"], g["start"])
            if (e["start"], e["end"]) == (g["start"], g["end"]):
                best = i
                break
            if overlap and ov > best_ov:
                best, best_ov = i, ov
        if best is None:
            continue
        used.add(best)
        g, tp_m = gold[best], tp_m + 1
        if g["yago"]:
            tp_l += e["entity_id"] == g["yago"]
            n_hit += 1
            hit += g["yago"] in e["candidates"]
        else:
            no_id += 1
            no_id_ok += e["entity_id"] is None
    return {"mention": prf(tp_m, len(pred), len(gold)),
            "link": prf(tp_l, sum(1 for e in pred if e["entity_id"]), sum(1 for g in gold if g["yago"])),
            "shortlist": (hit, n_hit), "no_id_unlinked": (no_id_ok, no_id), "gold": len(gold), "pred": len(pred)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pred")
    print("| spans | gold / predicted mentions | mention P / R / F1 | link P / R / F1 | gold id in shortlist | no-id mentions left unlinked |")
    print("|---|---|---|---|---|---|")
    for name, ov in (("strict", False), ("overlap", True)):
        r = score(ov, pred_dir)
        m, l = r["mention"], r["link"]
        print(f"| {name} | {r['gold']} / {r['pred']} | {m[0]:.2f} / {m[1]:.2f} / {m[2]:.2f} | "
              f"{l[0]:.2f} / {l[1]:.2f} / {l[2]:.2f} | {r['shortlist'][0]} / {r['shortlist'][1]} | "
              f"{r['no_id_unlinked'][0]} / {r['no_id_unlinked'][1]} |")
