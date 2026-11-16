"""Stage 3, relations: for two entities named in one sentence, which relation of dataset/relations.json is stated.

Code decides which relations are possible at all (domain and range of the two kinds, one direction for symmetric
relations, partOf / follows only between two things of one kind, and never partOf between two places: the dataset
calls that schema:location). Jev then answers, per pair, in one batch per sentence:
  related    Noul    does the sentence connect the two at all?            (the companion of the Choice)
  which      Choice  which of the possible relations, in both option orders
  states     Noul    one per possible relation: "does the sentence say that A was held captive in B?"
The score of a relation is  related x which x states  to the power 1/3 (a geometric mean): the Choice makes the
relations compete, the Nouls check each claim on its own, and a claim has to pass all three. The best relation of a
pair is a fact if it reaches the threshold and YAGO's constraints allow it (`consistent`).
"""
import json
import os
from itertools import combinations

from mentions import sentences
from questions import (KINDS, RELATIONS, SAME_CLASS, SYMMETRIC, T_FACT, both_orders, choice_probs, option_name, related,
                       states, which_relation)

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = {k: v for k, v in json.load(open(os.path.join(HERE, "..", "dataset", "relations.json"), encoding="utf-8")).items()
          if k in RELATIONS}                                   # the 23 relations whose object is an entity
MIN_KIND = 0.2                                                 # a kind this probable counts when listing relations


def classes(m):
    return {c for k, p in m["kinds"].items() if p >= MIN_KIND for c in KINDS[k][2]}


def fits(cls, allowed):
    return bool(cls & set(allowed)) or "schema:Thing" in allowed


def options(a, b):
    """[(relation, subject, object)] for mentions a (first in the text) and b"""
    out = []
    for s, o in ((a, b), (b, a)):
        cs, co = classes(s), classes(o)
        for rel, r in SCHEMA.items():
            if (s is b and rel in SYMMETRIC) or not (fits(cs, r["domain"]) and fits(co, r["range"])):
                continue
            if rel in SAME_CLASS and (not cs & co or cs & co == {"schema:Person"}):
                continue
            if rel == "yago:partOf" and "schema:Place" in cs & co:
                continue
            out.append((rel, s, o))
    return out


def score(d, o, how="all"):
    """how: 'all' (default), or one of the ablations reported in the README"""
    if how == "choice":                      # the earlier pipeline: related x which
        return d["related"] * o["which"]
    if how == "states":                      # the per-relation Noul alone
        return o["states"]
    return (d["related"] * o["which"] * o["states"]) ** (1 / 3)


def decide(jev, paragraphs, mentions):
    """mentions carry `entity` and `kinds`. -> one decision per pair of entities per sentence, with every probability"""
    jobs, meta = [], []
    for pid, text in paragraphs:
        for s0, s1 in sentences(text):
            first = {}
            for m in mentions:
                if m["para"] == pid and s0 <= m["start"] and m["end"] <= s1:
                    first.setdefault(m["entity"], m)
            qs, pairs = {}, []
            for a, b in combinations(first.values(), 2):
                opts = options(a, b)
                if not opts:
                    continue
                i = len(pairs)
                pairs.append((a, b, opts))
                qs[f"{i}|related"] = related(a["text"], b["text"])
                qs.update(both_orders(f"{i}|which", which_relation(a["text"], b["text"],
                                                                 [(r, s["text"], o["text"]) for r, s, o in opts])))
                for j, (r, s, o) in enumerate(opts):
                    qs[f"{i}|states|{j}"] = states(r, s["text"], o["text"])
            if pairs:
                jobs.append(({"sentence": text[s0:s1]}, qs))
                meta.append((pid, s0, s1, pairs))
    out = []
    for (pid, s0, s1, pairs), ans in zip(meta, jev.ask_many(jobs)):
        for i, (a, b, opts) in enumerate(pairs):
            which = choice_probs(ans, f"{i}|which")
            out.append({"para": pid, "sentence": [s0, s1], "a": a["text"], "b": b["text"],
                        "related": ans[f"{i}|related"]["noul"],
                        "options": [{"rel": r, "subj": s["entity"], "obj": o["entity"], "subj_text": s["text"],
                                     "obj_text": o["text"], "which": round(which[option_name(r, s["text"], o["text"])], 4),
                                     "states": ans[f"{i}|states|{j}"]["noul"]} for j, (r, s, o) in enumerate(opts)]})
    return out


def consistent(found):
    """YAGO's own constraints on the facts of one document, the more probable fact winning:
    a relation with a maximal count in the schema (one birth place, one death place) keeps that many objects per
    subject, and a symmetric relation is one fact, not one in each direction."""
    keep, count, seen = [], {}, set()
    for f in sorted(found, key=lambda f: -f["p"]):
        limit = SCHEMA[f["rel"]]["max_count"]
        objects = count.setdefault((f["subj"], f["rel"]), set())
        if limit and f["obj"] not in objects and len(objects) >= limit:
            continue
        if f["rel"] in SYMMETRIC and (f["para"], f["obj"], f["rel"], f["subj"]) in seen:
            continue
        objects.add(f["obj"])
        seen.add((f["para"], f["subj"], f["rel"], f["obj"]))
        keep.append(f)
    return sorted(keep, key=lambda f: (f["para"], f["sentence"][0]))


def facts(decisions, threshold=T_FACT, how="all"):
    """The best relation of each pair if it reaches the threshold; one fact per (paragraph, subject, relation, object)."""
    best = {}
    for d in decisions:
        o = max(d["options"], key=lambda o: score(d, o, how))
        p = score(d, o, how)
        key = (d["para"], o["subj"], o["rel"], o["obj"])
        if p >= threshold and p > best.get(key, {"p": 0})["p"]:
            best[key] = {"para": d["para"], "subj": o["subj"], "rel": o["rel"], "obj": o["obj"],
                         "subj_text": o["subj_text"], "obj_text": o["obj_text"], "p": round(p, 4),
                         "sentence": d["sentence"]}
    return consistent(best.values())
