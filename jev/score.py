"""Scores jev/pred against dataset/gold on the three levels, as Markdown.

    python jev/score.py             the scores, with fact and link scores at other thresholds
    python jev/score.py --errors    what is wrong at each level

mentions, links   hybrid/score.py, unchanged, so the numbers compare with hybrid/README.md
facts             (paragraph, subject, relation, object) after mapping the names of a predicted fact to gold entities
                  by label and alias, as dataset/README.md says and as the archived evaluation did
pred/gold-mentions/ holds the same pipeline run on the gold mentions: the later stages alone.
jev/results/ holds the output of both commands.
"""
import collections
import importlib.util
import json
import os
import re
import sys
import tempfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from relations import facts                                        # noqa: E402

DOCS = ("alford2026", "nechaeva2017")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RUNS = (("End to end (Jev finds the mentions)", os.path.join(HERE, "pred")),
        ("On the gold mentions (stages 2 and 3 alone)", os.path.join(HERE, "pred", "gold-mentions")))
_spec = importlib.util.spec_from_file_location("hybrid_score", os.path.join(HERE, "..", "hybrid", "score.py"))
hybrid_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hybrid_score)

_TR = str.maketrans({"ł": "l", "Ł": "L", "đ": "d", "ø": "o", "ß": "ss", "æ": "ae", "œ": "oe", "ʿ": "", "ʾ": "", "ʻ": "",
                     "ʼ": "", "‘": "", "’": "", "'": ""})


def norm(s):
    """case-, diacritic- and punctuation-insensitive form of a name"""
    s = unicodedata.normalize("NFKC", str(s)).translate(_TR)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", s.casefold()).strip()


def prf(tp, n_pred, n_gold):
    p, r = tp / n_pred if n_pred else 0.0, tp / n_gold if n_gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def gold(doc):
    return json.load(open(os.path.join(HERE, "..", "dataset", "gold", doc + ".json"), encoding="utf-8"))


def load(pred_dir, suffix=".json"):
    return {d: json.load(open(os.path.join(pred_dir, d + suffix), encoding="utf-8")) for d in DOCS}


def gold_facts():
    return {(d, f["para"], f["subj"], f["rel"], f["obj"]) for d in DOCS for f in gold(d)["facts"]}


def fact_keys(doc, found):
    """the facts as (doc, paragraph, gold entity, relation, gold entity); a name that is not exactly one gold entity
    stays a name and matches nothing"""
    names = collections.defaultdict(set)
    for e in gold(doc)["entities"]:
        for n in [e["label"]] + e["aliases"]:
            names[norm(n)].add(e["eid"])
    eid = lambda name: next(iter(names[norm(name)])) if len(names.get(norm(name), ())) == 1 else "?" + name
    return {(doc, f["para"], eid(f["subj_text"]), f["rel"], eid(f["obj_text"])): f for f in found}


def score_facts(found):
    """found: {doc: [fact]} -> P, R, F1, right, predicted, gold"""
    g, p = gold_facts(), set().union(*[set(fact_keys(d, fs)) for d, fs in found.items()])
    return (*prf(len(g & p), len(p), len(g)), len(g & p), len(p), len(g))


def score_links(preds, threshold):
    """link P, R, F1 (strict spans) if the best candidate of an entity is taken at another threshold"""
    with tempfile.TemporaryDirectory() as tmp:
        for d, p in preds.items():
            top = {e["eid"]: e["candidates"][0] if e["candidates"] else {"p": 0} for e in p["entities"]}
            rows = [{**m, "entity_id": top[m["entity"]]["yago"] if top[m["entity"]]["p"] >= threshold else None,
                     "candidates": []} for m in p["mentions"]]
            open(os.path.join(tmp, d + ".jsonl"), "w", encoding="utf-8").write(json.dumps({"id": d, "entities": rows}) + "\n")
        return hybrid_score.score(False, tmp)["link"]


def cells(prf_by_threshold):
    return " | ".join(f"{p * 100:.0f} / {r * 100:.0f} / **{f * 100:.1f}**" for p, r, f in prf_by_threshold)


def scores(title, pred_dir):
    preds, decisions = load(pred_dir), load(pred_dir, ".decisions.json")
    print(f"\n## {title}\n")
    print("| spans | mention P / R / F1 | link P / R / F1 | gold id among the candidates | no-id mentions left unlinked |")
    print("|---|---|---|---|---|")
    for name, overlap in (("strict", False), ("overlap", True)):
        r = hybrid_score.score(overlap, pred_dir)
        m, l = r["mention"], r["link"]
        print(f"| {name} | {m[0]:.2f} / {m[1]:.2f} / {m[2]:.2f} | {l[0]:.2f} / {l[1]:.2f} / {l[2]:.2f} | "
              f"{r['shortlist'][0]} / {r['shortlist'][1]} | {r['no_id_unlinked'][0]} / {r['no_id_unlinked'][1]} |")
    f = score_facts({d: p["facts"] for d, p in preds.items()})
    print(f"\nFacts: P {f[0]:.1%} / R {f[1]:.1%} / F1 {f[2]:.1%} ({f[3]} right of {f[4]} predicted, {f[5]} gold).")

    at = lambda how: [score_facts({d: facts(decisions[d], t, how) for d in DOCS})[:3] for t in THRESHOLDS]
    print("\nP / R / **F1** at other thresholds (the pipeline uses 0.7 for both):\n")
    print("| threshold | " + " | ".join(str(t) for t in THRESHOLDS) + " |\n|---|" + "---|" * len(THRESHOLDS))
    print(f"| links, strict spans | {cells(score_links(preds, t) for t in THRESHOLDS)} |")
    print(f"| facts: related × which × states (the pipeline) | {cells(at('all'))} |")
    print(f"| facts: related × which only | {cells(at('choice'))} |")
    print(f"| facts: states only | {cells(at('states'))} |")

    # the threshold that is best on one paper, applied to the other: what tuning the threshold is really worth
    g = gold_facts()

    def f1(doc, t):
        p = set(fact_keys(doc, facts(decisions[doc], t)))
        return prf(len(p & g), len(p), sum(1 for k in g if k[0] == doc))[2]

    for dev, test in (DOCS, DOCS[::-1]):
        best = max(THRESHOLDS, key=lambda t: f1(dev, t))
        print(f"\nThe fact threshold that is best on {dev} ({best}) gives F1 {f1(test, best):.1%} on {test}.")


def errors(title, pred_dir):
    print(f"\n## {title}")
    g_facts = gold_facts()
    for d, p in load(pred_dir).items():
        g = gold(d)
        gm = {(m["para"], m["start"], m["end"]): m for m in g["mentions"]}
        pm = {(m["para"], m["start"], m["end"]): m for m in p["mentions"]}
        print(f"\n### {d}\n")
        print("Mentions missed: " + (", ".join(f"{m['text']} ({k[0]})" for k, m in gm.items() if k not in pm) or "none"))
        print("\nMentions that are not in the gold: "
              + (", ".join(f"{m['text']} ({k[0]})" for k, m in pm.items() if k not in gm) or "none"))
        wrong = collections.Counter((pm[k]["text"], pm[k]["yago"], gm[k]["yago"]) for k in pm
                                    if k in gm and pm[k]["yago"] != gm[k]["yago"])
        print("\n| mention | times | linked to | gold |\n|---|---|---|---|")
        for (text, mine, theirs), n in sorted(wrong.items(), key=lambda x: (-x[1], x[0][0])):
            print(f"| {text} | {n} | {mine or 'unlinked'} | {theirs or 'no entry in YAGO'} |")
        found = fact_keys(d, p["facts"])
        print("\n| fact that is not in the gold | paragraph | p |\n|---|---|---|")
        for k, f in found.items():
            if k not in g_facts:
                print(f"| {f['subj_text']} — {f['rel']} → {f['obj_text']} | {f['para']} | {f['p']:.2f} |")
        label = {e["eid"]: e["label"] for e in g["entities"]}
        print("\nGold facts missed: " + ("; ".join(f"{label[k[2]]} — {k[3]} → {label[k[4]]} ({k[1]})" for k in sorted(g_facts)
                                                   if k[0] == d and k not in found) or "none"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    report = errors if "--errors" in sys.argv else scores
    print("# Jev pipeline: " + ("errors" if report is errors else "scores") + " against dataset/gold\n\n"
          f"Written by `python jev/score.py{' --errors' if report is errors else ''}` from `jev/pred/`.")
    for title, pred_dir in RUNS:
        report(title, pred_dir)
