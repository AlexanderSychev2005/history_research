#!/usr/bin/env python3
"""Evaluation of history-paper IE pipelines (stdlib only). Spec and rationale: README.md.

python evaluate.py validate --texts texts/ --pred pred/A/
python evaluate.py score    --texts texts/ --gold gold/ --pred pred/A/ [pred/B/ ...] [--span strict|overlap]
python evaluate.py sample   --texts texts/ --pred pred/A/ --n 100 --out audit_A.csv     # then humans fill `verdict`
python evaluate.py audit    --csv audit_A.csv
"""

import argparse
import collections
import csv
import json
import math
import random
import re
import sys
import unicodedata
from pathlib import Path

_TR = str.maketrans(
    {
        "ł": "l",
        "Ł": "L",
        "đ": "d",
        "ø": "o",
        "ß": "ss",
        "æ": "ae",
        "œ": "oe",
        "ʿ": "",
        "ʾ": "",
        "ʻ": "",
        "ʼ": "",
        "‘": "",
        "’": "",
        "'": "",
    }
)


def norm(s):
    """case-, diacritic-, punctuation-insensitive form for labels, relations and literals"""
    s = unicodedata.normalize("NFKC", str(s)).translate(_TR)
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )
    return re.sub(r"[\W_]+", " ", s.casefold()).strip()


def prf(tp, n_pred, n_gold):
    """empty-vs-empty is perfect; nothing predicted while gold exists is P=0"""
    p = 1.0 if not n_pred and not n_gold else (tp / n_pred if n_pred else 0.0)
    r = 1.0 if not n_gold else tp / n_gold
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (used by YAGO and SOFIE for sampled precision)"""
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    c, h = (
        (p + z * z / (2 * n)) / d,
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d,
    )
    return (max(0.0, c - h), min(1.0, c + h))


def load_dir(path):
    out = {}
    for f in sorted(Path(path).glob("*.json")):
        if not f.name.startswith("_"):
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d.get("doc_id", f.stem)] = d
    return out


def load_texts(path):
    return {
        k: {p["id"]: p["text"] for p in d["paragraphs"]}
        for k, d in load_dir(path).items()
    }


def align(gold_entities, pred_entities):
    """predicted eid -> gold eid or None: equal QID, else a UNIQUE normalised label/alias match"""
    qid, lab = {}, collections.defaultdict(set)
    for g in gold_entities:
        if g.get("qid"):
            qid[g["qid"]] = g["eid"]
        for n in [g.get("label")] + list(g.get("aliases", [])):
            if n:
                lab[norm(n)].add(g["eid"])
    out = {}
    for p in pred_entities:
        e = qid.get(p.get("qid")) if p.get("qid") else None
        if e is None:
            hits = set().union(
                *[
                    lab.get(norm(n), set())
                    for n in [p.get("label")] + list(p.get("aliases", []))
                    if n
                ]
            )
            e = next(iter(hits)) if len(hits) == 1 else None
        out[p["eid"]] = e
    return out


def anchor(mentions, text):
    """mentions without offsets get the k-th occurrence of their text in the paragraph, in the order given"""
    nxt, out, n = {}, [], 0
    for m in mentions:
        m = dict(m)
        if m.get("start") is None:
            i = text.find(m["text"], nxt.get(m["text"], 0))
            if i < 0:
                i = text.find(m["text"])
            m["start"], m["end"] = (i, i + len(m["text"])) if i >= 0 else (-1, -1)
            if i >= 0:
                nxt[m["text"]] = m["end"]
                n += 1
        out.append(m)
    return out, n


def match(gold, pred, strict):
    """one-to-one span matching, greedy by overlap: -> {gold index: pred index}"""
    c = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if p["start"] < 0:
                continue
            if strict:
                if (g["start"], g["end"]) == (p["start"], p["end"]):
                    c.append((1.0, gi, pi))
            else:
                ov = min(g["end"], p["end"]) - max(g["start"], p["start"])
                if ov > 0:
                    c.append(
                        (
                            ov
                            / (max(g["end"], p["end"]) - min(g["start"], p["start"])),
                            gi,
                            pi,
                        )
                    )
    c.sort(key=lambda x: (-x[0], x[1], x[2]))
    ug, up, out = set(), set(), {}
    for _, gi, pi in c:
        if gi not in ug and pi not in up:
            ug.add(gi)
            up.add(pi)
            out[gi] = pi
    return out


def fact_sets(facts, resolve, rel_map):
    key = lambda o: (
        ("lit", norm(o.get("literal", ""))) if isinstance(o, dict) else resolve(o)
    )
    strict, pair = set(), set()
    for f in facts:
        s, o = resolve(f["subj"]), key(f["obj"])
        strict.add((s, norm(rel_map.get(f["rel"], f["rel"])), o))
        pair.add(frozenset((s, o)))
    return strict, pair


def count(gold, pred, al, para, text, strict, rel_map):
    gm = [m for m in gold.get("mentions", []) if m["para"] == para]
    pm, anchored = anchor(
        [m for m in pred.get("mentions", []) if m["para"] == para], text
    )
    pairs = match(gm, pm, strict)
    c = collections.Counter(
        anchored=anchored,
        mention_gold=len(gm),
        mention_pred=len(pm),
        mention_tp=len(pairs),
        link_gold=sum(1 for m in gm if m.get("entity")),
        link_pred=sum(1 for m in pm if m.get("entity")),
    )
    for gi, pi in pairs.items():
        e = pm[pi].get("entity")
        c["link_tp"] += bool(
            gm[gi].get("entity") and e and al.get(e) == gm[gi]["entity"]
        )
    gs, gp = fact_sets(
        [f for f in gold.get("facts", []) if f["para"] == para], lambda e: ("e", e), {}
    )
    ps, pp = fact_sets(
        [f for f in pred.get("facts", []) if f.get("para") == para],
        lambda e: ("e", al[e]) if al.get(e) else ("?", e),
        rel_map,
    )
    c.update(
        fact_gold=len(gs),
        fact_pred=len(ps),
        fact_tp=len(gs & ps),
        fact_pair_gold=len(gp),
        fact_pair_pred=len(pp),
        fact_pair_tp=len(gp & pp),
    )
    return c


LEVELS = ("mention", "link", "fact", "fact_pair")


def summarize(units):
    t = collections.Counter()
    for u in units:
        t.update(u)
    return {
        k: dict(
            zip(("P", "R", "F1"), prf(t[k + "_tp"], t[k + "_pred"], t[k + "_gold"])),
            tp=t[k + "_tp"],
            pred=t[k + "_pred"],
            gold=t[k + "_gold"],
        )
        for k in LEVELS
    } | {"anchored": t["anchored"]}


def bootstrap(units, B, seed=13):
    """95% interval of F1, resampling paragraphs (the unit that was sampled for annotation)"""
    if len(units) < 2:
        return {}
    rng, runs = random.Random(seed), {k: [] for k in LEVELS}
    for _ in range(B):
        s = summarize([units[rng.randrange(len(units))] for _ in units])
        for k in LEVELS:
            runs[k].append(s[k]["F1"])
    return {
        k: (sorted(v)[int(0.025 * B)], sorted(v)[min(B - 1, int(0.975 * B))])
        for k, v in runs.items()
    }


def check(pred, texts, doc_id):
    """problems that need no gold: unknown paragraph, offsets that do not spell the text, dangling ids, invented quotes"""
    paras, issues, st = texts.get(doc_id), [], collections.Counter()
    if paras is None:
        return [f"no canonical text for doc_id {doc_id!r}"], st
    eids = {e["eid"] for e in pred.get("entities", [])}
    for i, m in enumerate(pred.get("mentions", [])):
        if m.get("para") not in paras:
            issues.append(f"mention {i}: unknown paragraph {m.get('para')!r}")
            continue
        if m.get("start") is not None and paras[m["para"]][
            m["start"] : m["end"]
        ] != m.get("text"):
            issues.append(
                f"mention {i}: text {m.get('text')!r} is not paragraph[{m['start']}:{m['end']}]"
            )
        if m.get("entity") and m["entity"] not in eids:
            issues.append(f"mention {i}: entity {m['entity']!r} not in entities")
    for i, f in enumerate(pred.get("facts", [])):
        for k in ("subj", "obj"):
            if isinstance(f.get(k), str) and f[k] not in eids:
                issues.append(f"fact {i}: {k} {f[k]!r} not in entities")
        if f.get("evidence_quote") is not None:
            st["quotes"] += 1
            st["verbatim"] += norm(f["evidence_quote"]) in norm(
                paras.get(f.get("para"), "")
            )
    return issues, st


def evaluate(texts, gold, preds, strict, info, B):
    units, problems, quotes = [], [], collections.Counter()
    for doc_id, g in gold.items():
        scope = g.get("scope", {}).get("paragraphs")
        if scope is None:
            raise SystemExit(f"{doc_id}: gold has no scope.paragraphs")
        p = preds.get(doc_id) or {}
        if not p:
            problems.append(f"{doc_id}: no prediction file, scored as empty")
        else:
            iss, st = check(p, texts, doc_id)
            problems += [f"{doc_id}: {x}" for x in iss]
            quotes.update(st)
        al = align(g.get("entities", []), p.get("entities", []))
        units += [
            count(g, p, al, para, texts[doc_id][para], strict, info.get("rel_map", {}))
            for para in scope
        ]
    r = {
        "overall": summarize(units),
        "ci": bootstrap(units, B),
        "paragraphs": len(units),
        "problems": problems,
    }
    if quotes["quotes"]:
        r["quote_verbatim_rate"] = quotes["verbatim"] / quotes["quotes"]
    return r


def show(name, r, strict):
    print(
        f"\n== {name}   {r['paragraphs']} paragraphs, span match: {'strict' if strict else 'overlap'}"
    )
    print(f"{'':<11}{'P':>7}{'R':>7}{'F1':>7}   {'95% CI':<12}tp/pred/gold")
    for k in LEVELS:
        m, ci = r["overall"][k], r["ci"].get(k)
        print(
            f"{k:<11}{m['P']:>7.1%}{m['R']:>7.1%}{m['F1']:>7.1%}   "
            f"{('%d-%d%%' % (ci[0] * 100, ci[1] * 100)) if ci else '':<12}{m['tp']}/{m['pred']}/{m['gold']}"
        )
    if "quote_verbatim_rate" in r:
        print(
            f"evidence quotes found verbatim in the paragraph: {r['quote_verbatim_rate']:.0%}"
        )
    if r["overall"]["anchored"]:
        print(
            f"note: {r['overall']['anchored']} mentions had no offsets and were anchored by text"
        )
    if r["paragraphs"] < 30:
        print(
            "note: fewer than 30 paragraphs; the interval is wide, do not over-read differences"
        )
    for p in r["problems"][:6]:
        print("  problem:", p)
    if len(r["problems"]) > 6:
        print(f"  ... {len(r['problems']) - 6} more (run `validate`)")


def cmd_sample(a):
    texts, rows = load_texts(a.texts), []
    for doc_id, p in load_dir(a.pred).items():
        lab = {e["eid"]: e.get("label", e["eid"]) for e in p.get("entities", [])}
        name = lambda o: o.get("literal", "") if isinstance(o, dict) else lab.get(o, o)
        for f in p.get("facts", []):
            rows.append(
                {
                    "doc_id": doc_id,
                    "para": f.get("para", ""),
                    "subject": lab.get(f["subj"], f["subj"]),
                    "relation": f["rel"],
                    "object": name(f["obj"]),
                    "paragraph": texts.get(doc_id, {}).get(f.get("para"), ""),
                    "verdict": "",
                }
            )
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    with open(a.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]) if rows else ["verdict"])
        w.writeheader()
        w.writerows(rows)
    print(
        f"{len(rows)} facts sampled -> {a.out}; fill `verdict` with 1 (stated or directly implied by the paragraph) or 0"
    )


def cmd_audit(a):
    with open(a.csv, encoding="utf-8-sig", newline="") as fh:
        v = [r["verdict"].strip().lower() for r in csv.DictReader(fh)]
    k, n = (
        sum(x in ("1", "y", "yes", "correct") for x in v),
        sum(x in ("0", "1", "y", "yes", "n", "no", "correct", "incorrect") for x in v),
    )
    lo, hi = wilson(k, n)
    print(
        f"judged {n} of {len(v)} sampled facts: precision {k}/{n} = {k / n if n else 0:.1%}, 95% Wilson interval {lo:.1%}-{hi:.1%}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--texts", required=True)
    v.add_argument("--pred", required=True)
    s = sub.add_parser("score")
    s.add_argument("--texts", required=True)
    s.add_argument("--gold", required=True)
    s.add_argument("--pred", required=True, nargs="+")
    s.add_argument("--span", choices=["strict", "overlap"], default="strict")
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--out")
    m = sub.add_parser("sample")
    m.add_argument("--texts", required=True)
    m.add_argument("--pred", required=True)
    m.add_argument("--n", type=int, default=100)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--out", required=True)
    u = sub.add_parser("audit")
    u.add_argument("--csv", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        return cmd_sample(a) or 0
    if a.cmd == "audit":
        return cmd_audit(a) or 0
    texts = load_texts(a.texts)
    if a.cmd == "validate":
        bad = 0
        for doc_id, p in load_dir(a.pred).items():
            iss, st = check(p, texts, doc_id)
            bad += len(iss)
            print(
                f"{doc_id}: {len(p.get('mentions', []))} mentions, {len(p.get('facts', []))} facts, {len(iss)} problems"
            )
            for x in iss[:20]:
                print("   ", x)
        return 1 if bad else 0
    gold, rep = load_dir(a.gold), {}
    for d in a.pred:
        f = Path(d) / "_system.json"
        info = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        name = info.get("system", Path(d).name)
        rep[name] = evaluate(
            texts, gold, load_dir(d), a.span == "strict", info, a.bootstrap
        )
        show(name, rep[name], a.span == "strict")
    if a.out:
        Path(a.out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
