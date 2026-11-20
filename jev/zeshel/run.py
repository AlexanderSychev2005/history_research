"""Jev on ZESHEL in the format of dataset/zeshel_realms/, in the protocol of LELA (arXiv 2601.05192, Sec. 4.1, Table 3):
the 64 BM25 candidates that ship with ZESHEL, no "none" option, normalized accuracy.

Rounds (each switch can be turned off, for the ablation on a validation world):
  which    a Choice over the candidates (title -> first three sentences), asked in --orders option orders
           (2: given and reversed; 4: two shuffles more), because Jev leans to the first option
  gate     --gate: a Noul on the mention (a general kind or one individual?) and the same Noul on each of the top 10
           candidates; a candidate that clearly disagrees with the mention is dropped
  final    --final K: a Choice over the K best that are left, with the long description (the start of the page), asked
           in all K! option orders, for the mentions whose best `which` is below --unsure
  --drop-page  the page the mention stands on is not a candidate (a wiki link never points to its own page; 5 of the
               10,000 validation mentions do)
  --page-title Jev is told the title of that page

    python jev/zeshel/run.py --data <folder> --world elder_scrolls --split val --n 1000 [switches]
    python dataset/zeshel_realms/score.py jev/zeshel/pred/<name>.jsonl [--gold <gold file>]

Every answer is cached in jev/cache/answers.jsonl like the rest of jev/, so a rerun asks nothing twice.
"""
import argparse
import itertools
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from client import Jev                                             # noqa: E402

WINDOW = 80          # words of the page on each side of the mention
DESCRIPTION = 700    # characters of the short description, at most
CLEAR = 0.8          # the gate drops a candidate only when both readings are this clear, in opposite directions


def context(text, start, end):
    """the words around the mention, the mention between ⟦ ⟧"""
    left, right = text[:start].split(), text[end:].split()
    return " ".join(left[-WINDOW:] + ["⟦" + text[start:end] + "⟧"] + right[:WINDOW])


# ---------------------------------------------------------------- questions (general wording, no examples from a world)

def where(page):
    return f" `context` is a passage of the wiki page “{page}”." if page else ""


def which(text, options, page=None):
    return {"type": "choice",
            "instructions": f"Which entry of the knowledge base is the entity that the mention ⟦{text}⟧ in `context` "
                            f"refers to?{where(page)}",
            "criteria": options}


KIND = ("a general kind, class or concept: a species or race, a type of creature, object, spell or profession, a "
        "common noun or a group of things in general")
ONE = ("one particular individual: a named character, one particular place, organization, event, work or episode, or "
       "one specific object")


def mention_generic(text, page=None):
    return {"type": "noul",
            "instructions": f"In `context`, does the mention ⟦{text}⟧ stand for a general kind of thing rather than for "
                            f"one particular individual?{where(page)}",
            "criteria": {"true": f"⟦{text}⟧ stands for {KIND}.", "false": f"⟦{text}⟧ stands for {ONE}."}}


def entry_generic():
    return {"type": "noul",
            "instructions": "Is the knowledge-base entry in `entry` about a general kind of thing rather than about one "
                            "particular individual?",
            "criteria": {"true": f"The entry describes {KIND}.", "false": f"The entry describes {ONE}."}}


def orders(ids, n, key):
    """n option orders of ids: as given, reversed, then shuffles seeded by key"""
    out = [list(ids), list(reversed(ids))]
    rnd = random.Random(key)
    while len(out) < n:
        out.append(rnd.sample(list(ids), len(ids)))
    return out[:n]


def ask_choice(jev, jobs, n_orders=None):
    """jobs: [(state, make(option_ids) -> question, ids, key)] -> [{id: mean probability over the orders}].
    n_orders None: all permutations."""
    flat, meta = [], []
    for state, make, ids, key in jobs:
        perms = [list(p) for p in itertools.permutations(ids)] if n_orders is None else orders(ids, n_orders, key)
        flat.append((state, {f"o{k}": make(p) for k, p in enumerate(perms)}))
        meta.append(len(perms))
    out = []
    for (state, make, ids, key), n, ans in zip(jobs, meta, jev.ask_many(flat)):
        probs = {}
        for k in range(n):
            names = ans[f"o{k}"]["probabilities"]
            for i in ids:
                probs[i] = probs.get(i, 0.0) + names[make.name_of[i]] / n
        out.append(probs)
    return out


class Make:
    """builds a Choice from an option order; name_of maps an id to its option name"""

    def __init__(self, text, names, descriptions, page):
        self.text, self.name_of, self.desc, self.page = text, names, descriptions, page

    def __call__(self, ids):
        return which(self.text, {self.name_of[i]: self.desc[i] for i in ids}, self.page)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "dataset", "zeshel_realms"))
    ap.add_argument("--world", default="forgotten_realms")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0, help="a random sample of mentions with the gold among the candidates "
                                                     "(default 0: all mentions)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orders", type=int, default=2)
    ap.add_argument("--drop-page", action="store_true")
    ap.add_argument("--page-title", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--final", type=int, default=0, help="K of the final round (0: no final round)")
    ap.add_argument("--unsure", type=float, default=1.01, help="final round only below this `which` probability")
    ap.add_argument("--name", help="output name (default: from the switches)")
    a = ap.parse_args()

    doc = f"zeshel_{a.world}_{a.split}"
    load = lambda *p: json.load(open(os.path.join(a.data, *p), encoding="utf-8"))
    paras = {p["id"]: p for p in load("paragraphs", doc + ".json")["paragraphs"]}
    mentions = load("gold", doc + ".json")["mentions"]
    cands = load("candidates", doc + ".json")
    kb = {}
    for line in open(os.path.join(a.data, "kb", a.world + ".jsonl"), encoding="utf-8"):
        e = json.loads(line)
        kb[e["id"]] = e
    short = {i: e["description"][:DESCRIPTION] for i, e in kb.items()}
    if a.n:
        mentions = random.Random(a.seed).sample(sorted((m for m in mentions if m["gold_in_candidates"]),
                                                       key=lambda m: m["mention_id"]), a.n)

    jev, t0 = Jev(), time.time()
    rows = []
    for m in mentions:
        ids = [i for i in cands[m["mention_id"]] if not (a.drop_page and i == m["para"])]
        names, seen = {}, {}
        for i in ids:                                              # option names must be unique
            t = kb[i]["title"]
            seen[t] = seen.get(t, 0) + 1
            names[i] = t if seen[t] == 1 else f"{t} ({seen[t]})"
        page = paras[m["para"]]["title"] if a.page_title else None
        rows.append({"m": m, "ids": ids, "names": names, "page": page,
                     "state": {"context": context(paras[m["para"]]["text"], m["start"], m["end"])}})

    # which: one Choice over all candidates (a mention left without candidates stays unlinked)
    for r in rows:
        r["which"], r["top"] = {}, []
    asked = [r for r in rows if r["ids"]]
    probs = ask_choice(jev, [(r["state"], Make(r["m"]["text"], r["names"], short, r["page"]), r["ids"],
                              r["m"]["mention_id"]) for r in asked], a.orders)
    for r, p in zip(asked, probs):
        r["which"] = p
        r["top"] = sorted(r["ids"], key=lambda i: -p[i])[:10]
    rows_all, rows = rows, asked
    t1 = time.time()

    # gate: a general kind or one individual, for the mention and for each of the top 10
    if a.gate:
        ans_m = jev.ask_many([(r["state"], {"g": mention_generic(r["m"]["text"], r["page"])}) for r in rows])
        entries = sorted({i for r in rows for i in r["top"]})
        ans_e = jev.ask_many([({"entry": {"title": kb[i]["title"], "description": short[i]}}, {"g": entry_generic()})
                              for i in entries])
        generic = dict(zip(entries, (x["g"]["noul"] for x in ans_e)))
        for r, x in zip(rows, ans_m):
            g = r["mention_generic"] = x["g"]["noul"]
            clash = lambda i: (g >= CLEAR and generic[i] <= 1 - CLEAR) or (g <= 1 - CLEAR and generic[i] >= CLEAR)
            kept = [i for i in r["top"] if not clash(i)]
            r["dropped"] = [i for i in r["top"] if clash(i)]
            r["top"] = kept or r["top"]

    # final: the K best that are left, long descriptions, every option order
    if a.final:
        long_ = {i: e["description_long"] for i, e in kb.items()}
        todo = [r for r in rows if r["which"][r["top"][0]] < a.unsure and len(r["top"]) > 1]
        probs = ask_choice(jev, [(r["state"], Make(r["m"]["text"], r["names"], long_, r["page"]), r["top"][:a.final],
                                  r["m"]["mention_id"]) for r in todo])
        for r, p in zip(todo, probs):
            r["final"] = p
            r["top"] = sorted(r["top"][:a.final], key=lambda i: -p[i]) + r["top"][a.final:]

    name = a.name or "_".join([f"{a.split}.{a.world}" + (f".n{a.n}" if a.n else ""), f"o{a.orders}"]
                              + ["drop"] * a.drop_page + ["page"] * a.page_title + ["gate"] * a.gate
                              + ([f"final{a.final}u{a.unsure:g}"] if a.final else []))
    os.makedirs(os.path.join(HERE, "pred"), exist_ok=True)
    path = os.path.join(HERE, "pred", name + ".jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows_all:
            pred = r["top"][0] if r["top"] else None
            f.write(json.dumps({"mention_id": r["m"]["mention_id"], "kb_id": pred, "text": r["m"]["text"],
                                "pred_title": kb[pred]["title"] if pred else None,
                                "gold_title": kb[r["m"]["kb_id"]]["title"],
                                "which": [{"kb_id": i, "p": round(r["which"][i], 4)}
                                          for i in sorted(r["ids"], key=lambda i: -r["which"][i])[:10]],
                                "mention_generic": r.get("mention_generic"), "dropped": r.get("dropped", []),
                                "final": {i: round(p, 4) for i, p in r["final"].items()} if "final" in r else None},
                               ensure_ascii=False) + "\n")
    print(f"{len(rows_all)} mentions. Jev: {jev.report()}; {time.time() - t0:.0f} s (which {t1 - t0:.0f} s); "
          f"-> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
