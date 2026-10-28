"""Scores every NER variant and relation run against eval/gold (no model calls) and writes
experiments/RESULTS.md and experiments/results/{ner,facts,agreement}.csv.

  uv run python experiments/compare.py
"""
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = ["alford2026", "nechaeva2017"]
NER = ["gliner", "gliner-clean", "jev"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
AGREEMENT = [("gliner-clean__openjev", "gliner-clean__jev-api"), ("gold__openjev", "gold__jev-api"),
             ("gold__openjev", "gold__jev-api-guided")]
OUT = Path("experiments/results")


def score(pred_dir, span="overlap"):
    """{level: (P, R, F1, tp, pred, gold)} from eval/evaluate.py"""
    out = subprocess.run([sys.executable, "eval/evaluate.py", "score", "--texts", "texts", "--gold", "eval/gold",
                          "--pred", str(pred_dir), "--span", span], capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    levels = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in ("mention", "link", "fact", "fact_pair"):
            p, r, f1 = (float(v) for v in re.findall(r"([\d.]+)%", line)[:3])
            levels[parts[0]] = (p, r, f1, *map(int, parts[-1].split("/")))
    return levels


def write_pred(directory, doc, mentions_by_para, facts):
    names = sorted({m["text"] for ms in mentions_by_para.values() for m in ms})
    eid = {n: f"m{i}" for i, n in enumerate(names)}
    pred = {"doc_id": doc, "entities": [{"eid": eid[n], "label": n} for n in names],
            "mentions": [{"para": pid, "start": m["start"], "end": m["end"], "text": m["text"], "entity": eid[m["text"]]}
                         for pid, ms in mentions_by_para.items() for m in ms],
            "facts": [{**f, "subj": eid[f["subj"]], "obj": eid[f["obj"]]} for f in facts]}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{doc}.json").write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def kept(decisions, t):
    for x in decisions:
        best = max(x["options"], key=lambda o: o["p"])
        if best["p"] >= t and best["p"] > x["p_none"]:
            yield {"para": x["paragraph_id"], "subj": best["subj"], "rel": best["rel"], "obj": best["obj"]}


def runs():
    found = {p.name.split("_relations_")[1][:-5] for p in Path("outputs").glob("*_relations_*.json")}
    order = {"gliner": 0, "gliner-clean": 1, "jev": 2, "gold": 3}
    return sorted(found, key=lambda r: (order.get(r.split("__")[0], 9), r))


def decision_table(run, doc):
    out = {}
    for x in load(f"outputs/{doc}_relations_{run}.json")["decisions"]:
        best = max(x["options"], key=lambda o: o["p"])
        pick = "none" if x["p_none"] >= best["p"] else (best["rel"], best["subj"], best["obj"])
        out[(doc, x["paragraph_id"], tuple(x["sentence"]), x["a"], x["b"])] = pick
    return out


def table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ner_rows, fact_rows, agree_rows = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for variant in NER:
            d = Path(tmp) / f"ner_{variant}"
            for doc in DOCS:
                ner = load(f"outputs/{doc}_ner_{variant}.json")
                write_pred(d, doc, {p["paragraph_id"]: p["mentions"] for p in ner["paragraphs"]}, [])
            for span in ("strict", "overlap"):
                p, r, f1, tp, n_pred, n_gold = score(d, span)["mention"]
                ner_rows.append(dict(ner=variant, span=span, P=p, R=r, F1=f1, tp=tp, pred=n_pred, gold=n_gold))

        for run in runs():
            ner_variant = run.split("__")[0]
            for t in THRESHOLDS:
                d = Path(tmp) / f"{run}_{t}"
                for doc in DOCS:
                    ner = load(f"outputs/{doc}_ner_{ner_variant}.json")
                    write_pred(d, doc, {p["paragraph_id"]: p["mentions"] for p in ner["paragraphs"]},
                               list(kept(load(f"outputs/{doc}_relations_{run}.json")["decisions"], t)))
                levels = score(d)
                for level in ("fact", "fact_pair"):
                    p, r, f1, tp, n_pred, n_gold = levels[level]
                    fact_rows.append(dict(run=run, threshold=t, level=level, P=p, R=r, F1=f1, tp=tp, pred=n_pred,
                                          gold=n_gold))
        legacy = score(Path("eval/pred/gliner__qwen-openie"))  # Alford only; Nechaeva counts as empty
        for level in ("fact", "fact_pair"):
            p, r, f1, tp, n_pred, n_gold = legacy[level]
            fact_rows.append(dict(run="gliner__qwen-openie", threshold="", level=level, P=p, R=r, F1=f1, tp=tp,
                                  pred=n_pred, gold=n_gold))

    for a, b in AGREEMENT:
        x, y = {}, {}
        for doc in DOCS:
            x.update(decision_table(a, doc))
            y.update(decision_table(b, doc))
        common = set(x) & set(y)
        agree_rows.append(dict(first=a, second=b, pairs=len(common),
                               same=f"{sum(x[k] == y[k] for k in common) / len(common):.1%}",
                               none_first=sum(x[k] == "none" for k in common),
                               none_second=sum(y[k] == "none" for k in common)))

    for name, rows in (("ner", ner_rows), ("facts", fact_rows), ("agreement", agree_rows)):
        with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)

    facts = [r for r in fact_rows if r["level"] == "fact"]
    best = []
    for run in dict.fromkeys(r["run"] for r in facts):
        rs = [r for r in facts if r["run"] == run]
        b = max(rs, key=lambda r: r["F1"])
        at08 = next((r for r in rs if r["threshold"] == 0.8), b)
        ner_variant, model = run.split("__")
        best.append({"mentions": ner_variant, "relations": model, "best F1": b["F1"], "at threshold": b["threshold"],
                     "P / R there": f"{b['P']} / {b['R']}", "F1 at 0.8": at08["F1"]})
    page = [
        "# Results",
        "",
        "Generated by `experiments/compare.py`; do not edit by hand. Gold: `eval/gold` (14 paragraphs, 159 mentions,"
        " 54 facts; a draft that still needs a historian's review). Spans are matched by overlap unless stated."
        " Fewer than 30 paragraphs, so the 95% intervals are wide (about ±10–15 points).",
        "",
        "## Mentions",
        "",
        table([r for r in ner_rows], ["ner", "span", "P", "R", "F1", "tp", "pred", "gold"]),
        "",
        "## Facts (YAGO relations), best threshold per run",
        "",
        "`gold` mentions = the relation stage alone. `gliner__qwen-openie` is the earlier free-text extractor, scored on"
        " Alford only and with free-text predicates, so its fact score is 0 by construction; see facts.csv for its"
        " fact-pair score.",
        "",
        table(best, ["mentions", "relations", "best F1", "at threshold", "P / R there", "F1 at 0.8"]),
        "",
        "## Agreement between relation models on the same pairs",
        "",
        table(agree_rows, ["first", "second", "pairs", "same", "none_first", "none_second"]),
        "",
        "Every threshold and the fact-pair level: `experiments/results/facts.csv`.",
        "",
    ]
    Path("experiments/RESULTS.md").write_text("\n".join(page), encoding="utf-8")
    print(Path("experiments/RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
