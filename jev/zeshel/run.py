"""Jev on dataset/zeshel_realms/ (ZESHEL test, Forgotten Realms), in the protocol of LELA (arXiv 2601.05192, Sec. 4.1,
Table 3): the 64 BM25 candidates that ship with ZESHEL, an entity shown as its title and the first three sentences of
its page, no "none" option, normalized accuracy.

Jev cannot rank 64 entries in one judgment well, so two rounds, as LELA has a reranker before its selection LLM:
  which   a Choice over all candidates (title -> description), asked in both option orders
  is      for the top K of `which` (K = 10, LELA's k), a Noul per candidate: is the mention this entry?
  score   sqrt(which x is); the best of the top K is the prediction. `--pick choice` takes the Choice alone.

    python jev/zeshel/run.py                    all 1,200 mentions -> jev/zeshel/pred/choice.jsonl and noul.jsonl
    python jev/zeshel/run.py --n 100            100 mentions drawn at random among those with the gold among the candidates
    python dataset/zeshel_realms/score.py jev/zeshel/pred/choice.jsonl

Every answer is cached in jev/cache/answers.jsonl like the rest of jev/, so a rerun asks nothing twice.
"""
import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "dataset", "zeshel_realms")
DOC = "zeshel_forgotten_realms_test"
sys.path.insert(0, os.path.dirname(HERE))
from client import Jev                                             # noqa: E402
from questions import both_orders, choice_probs                    # noqa: E402

WINDOW = 80          # words of the page on each side of the mention
DESCRIPTION = 700    # characters of the entity description, at most


def context(text, start, end):
    """the words around the mention, the mention between ⟦ ⟧"""
    left, right = text[:start].split(), text[end:].split()
    return " ".join(left[-WINDOW:] + ["⟦" + text[start:end] + "⟧"] + right[:WINDOW])


# ---------------------------------------------------------------- questions (general wording, no history examples)

def which(text, options):
    return {"type": "choice",
            "instructions": f"Which entry of the knowledge base is the entity that the mention ⟦{text}⟧ in `context` "
                            "refers to?",
            "criteria": options}


def is_entry(text, title, desc):
    return {"type": "noul",
            "instructions": f"In `context`, does the mention ⟦{text}⟧ refer to the entity of the knowledge-base entry "
                            f"“{title}”: {desc}",
            "criteria": {"true": f"⟦{text}⟧ in the context and the entry “{title}” are one and the same character, "
                                 "thing, place, work or concept.",
                         "false": f"The entry “{title}” is something else than ⟦{text}⟧ in the context: another "
                                  "character or thing with a similar name, a different version or episode, or a topic "
                                  "that is only related to it."}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="a random sample of mentions with the gold among the candidates "
                                                     "(default 0: all 1,200 mentions)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=10, help="candidates of the first round that get a Noul")
    a = ap.parse_args()

    load = lambda *p: json.load(open(os.path.join(DATA, *p), encoding="utf-8"))
    paras = {p["id"]: p["text"] for p in load("paragraphs", DOC + ".json")["paragraphs"]}
    mentions = load("gold", DOC + ".json")["mentions"]
    cands = load("candidates", DOC + ".json")
    kb = {}
    for line in open(os.path.join(DATA, "kb", "forgotten_realms.jsonl"), encoding="utf-8"):
        e = json.loads(line)
        kb[e["id"]] = (e["title"], e["description"][:DESCRIPTION])
    if a.n:
        mentions = random.Random(a.seed).sample(sorted((m for m in mentions if m["gold_in_candidates"]),
                                                       key=lambda m: m["mention_id"]), a.n)

    jev, t0 = Jev(), time.time()
    rows = []
    for m in mentions:
        titles, seen = {}, {}
        for i in cands[m["mention_id"]]:                           # option names must be unique
            t = kb[i][0]
            seen[t] = seen.get(t, 0) + 1
            titles[i] = t if seen[t] == 1 else f"{t} ({seen[t]})"
        rows.append({"m": m, "ids": cands[m["mention_id"]], "titles": titles,
                     "state": {"context": context(paras[m["para"]], m["start"], m["end"])}})

    # round 1: one Choice over all candidates, both orders
    jobs = [(r["state"], both_orders("which", which(r["m"]["text"], {r["titles"][i]: kb[i][1] for i in r["ids"]})))
            for r in rows]
    for r, ans in zip(rows, jev.ask_many(jobs)):
        p = choice_probs(ans, "which")
        r["which"] = {i: p[r["titles"][i]] for i in r["ids"]}
        r["top"] = sorted(r["ids"], key=lambda i: -r["which"][i])[:a.top]
    t1 = time.time()

    # round 2: a Noul per candidate of the top K
    jobs = [(r["state"], {f"is|{i}": is_entry(r["m"]["text"], r["titles"][i], kb[i][1]) for i in r["top"]})
            for r in rows]
    for r, ans in zip(rows, jev.ask_many(jobs)):
        r["is"] = {i: ans[f"is|{i}"]["noul"] for i in r["top"]}
        r["score"] = {i: math.sqrt(r["which"][i] * r["is"][i]) for i in r["top"]}

    suffix = f".n{a.n}.seed{a.seed}" if a.n else ""
    os.makedirs(os.path.join(HERE, "pred"), exist_ok=True)
    for name, pick in (("choice", lambda r: r["top"][0]), ("noul", lambda r: max(r["top"], key=lambda i: r["score"][i]))):
        with open(os.path.join(HERE, "pred", name + suffix + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                pred = pick(r)
                f.write(json.dumps({"mention_id": r["m"]["mention_id"], "kb_id": pred, "text": r["m"]["text"],
                                    "pred_title": kb[pred][0], "gold_title": kb[r["m"]["kb_id"]][0],
                                    "top": [{"kb_id": i, "title": r["titles"][i], "which": round(r["which"][i], 4),
                                             "is": round(r["is"][i], 4)} for i in r["top"]]},
                                   ensure_ascii=False) + "\n")
    print(f"{len(rows)} mentions. Jev: {jev.report()}; round 1 {t1 - t0:.0f} s, round 2 {time.time() - t1:.0f} s; "
          f"written to jev/zeshel/pred/{{choice,noul}}{suffix}.jsonl")


if __name__ == "__main__":
    main()
