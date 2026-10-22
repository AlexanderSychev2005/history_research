"""Scores LELA runs against pilot/gold.json.

    python pilot/score.py quickstart gliner_bm25 ...      # configs = folders in pilot/pred/
Writes pilot/results/<config>.json (numbers + every error) and pilot/results/summary.md (one table, all configs).

mention  a predicted span counts only if it equals a gold span exactly (same paragraph, same start and end)
link     a predicted (span, entity) counts if the span matches AND the entity is the gold entity;
         gold mentions without an entity (NIL) never count, a link predicted for them is a false positive
cand@K   of the correctly found mentions that have a gold entity: share whose entity is among LELA's top-K candidates
         (what a better reranker or LLM could still reach)
nil      of the correctly found NIL mentions: share for which LELA returned no link
"""
import json
import os
import sys


def prf(tp, n_pred, n_gold):
    p, r = tp / n_pred if n_pred else 0.0, tp / n_gold if n_gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def load_pred(config):
    out = []
    for doc in ("alford2026", "nechaeva2017"):
        row = json.loads(open(f"pilot/pred/{config}/{doc}.jsonl", encoding="utf-8").readline())
        out += [{**e, "doc": doc} for e in row["entities"]]
    return out


def score(config, gold):
    G = {(m["doc"], m["start"], m["end"]): m for m in gold}
    pred = load_pred(config)
    found, wrong_link, fake, nil_seen, nil_ok, cand_n, cand_ok, link_tp = [], [], [], 0, 0, 0, 0, 0
    for e in pred:
        g = G.get((e["doc"], e["start"], e["end"]))
        linked = e.get("entity_id")
        if g is None:
            fake.append({"doc": e["doc"], "text": e["text"], "linked": linked})
            continue
        found.append(g)
        if g["entity"] is None:
            nil_seen += 1
            nil_ok += linked is None
            if linked:
                wrong_link.append({"text": g["text"], "gold": None, "pred": linked})
            continue
        cand_n += 1
        cand_ok += g["entity"] in [c["entity_id"] for c in e.get("candidates", [])]
        if linked == g["entity"]:
            link_tp += 1
        else:
            wrong_link.append({"text": g["text"], "gold": g["entity"], "pred": linked})
    n_link_pred = sum(1 for e in pred if e.get("entity_id"))
    n_link_gold = sum(1 for m in gold if m["entity"])
    missed = [m["text"] for k, m in G.items() if not any((e["doc"], e["start"], e["end"]) == k for e in pred)]
    res = {"config": config, "n_gold": len(gold), "n_pred": len(pred)}
    res["mention"] = dict(zip("PRF", prf(len(found), len(pred), len(gold))), tp=len(found))
    res["link"] = dict(zip("PRF", prf(link_tp, n_link_pred, n_link_gold)), tp=link_tp)
    res["cand_recall"] = {"hit": cand_ok, "of": cand_n, "K": max((len(e.get("candidates", [])) for e in pred), default=0)}
    res["nil"] = {"no_link": nil_ok, "of": nil_seen}
    res["errors"] = {"missed_mentions": missed, "false_mentions": fake, "wrong_links": wrong_link}
    return res


def main(configs):
    gold = json.load(open("pilot/gold.json", encoding="utf-8"))["mentions"]
    os.makedirs("pilot/results", exist_ok=True)
    rows = ["| config | mention P | R | F1 | link P | R | F1 | cand@K (K) | NIL ok |", "|---|---|---|---|---|---|---|---|---|"]
    for c in configs:
        r = score(c, gold)
        json.dump(r, open(f"pilot/results/{c}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        m, l = r["mention"], r["link"]
        rows.append(f"| {c} | {m['P']:.2f} | {m['R']:.2f} | {m['F']:.2f} | {l['P']:.2f} | {l['R']:.2f} | {l['F']:.2f} | "
                    f"{r['cand_recall']['hit']}/{r['cand_recall']['of']} (K={r['cand_recall']['K']}) | {r['nil']['no_link']}/{r['nil']['of']} |")
    table = "\n".join(rows)
    open("pilot/results/summary.md", "w", encoding="utf-8").write(
        f"2 paragraphs, {len(gold)} gold mentions ({sum(m['entity'] is None for m in gold)} NIL); strict spans.\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
