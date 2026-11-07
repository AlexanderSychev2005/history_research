"""Relation extraction by zero-shot prompting with constrained decoding.

An LLM (gpt-oss on Groq) is asked, with no examples, which YAGO relation a text states. The answer is forced into a
strict JSON schema whose legal values are enums: the relations of schema/history_relations.json, the names of the
paragraph or sentence, or (per pair) the relations that the types of the two names allow plus "none". The model cannot
invent a relation name, which is what made free-text Open IE unusable. Four ways of asking:

  paragraph  one call per paragraph: a list of (subject, relation, object) triples, evidence quoted
  sentence   the same, one call per sentence (shorter text, shorter list of names)
  pair       one question per pair of names in a sentence: one of [relation(subj, obj) ... | none]
  gated      first "does the sentence state how X and Y are connected?" (yes/no), then the relation without "none"

Triples that break the type constraints of the schema (domain/range) are dropped afterwards. Same inputs and output
format as pipeline/relations_jev.py: names from outputs/<doc>_ner_<ner>.json, relations from schema/history_relations.json
(entity-to-entity ones), predictions in pred/<run>/<doc>.json, run = <ner>__llm-<model>-<variant>-r<repeat>.

  GROQ_API_KEY=... python experiments/08_constrained_relations/relations_constrained.py --doc alford2026 \
      --ner gold --model 20b --variant paragraph
Run from the repository root; temperature 0. Pair runs keep a cache (cache/) and resume after a daily-limit stop.
Groq's "strict" mode is not a guarantee: an answer that breaks the schema is rejected with HTTP 400; such answers are
counted (invalid) and treated as "no fact".
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from pipeline.relations_jev import CLASS, SYMMETRIC, fits, options, sentences  # noqa: E402

MODELS = {"20b": "openai/gpt-oss-20b", "120b": "openai/gpt-oss-120b"}
URL = "https://api.groq.com/openai/v1/chat/completions"
USAGE = {"calls": 0, "tokens": 0, "invalid": 0}


def ask(model, prompt, schema, key):
    """One constrained call -> the parsed JSON answer, or None if the model broke the schema."""
    body = {"model": MODELS[model], "temperature": 0, "max_completion_tokens": 1500, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "history-research/0.1"})
    for attempt in range(8):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=180))
            USAGE["calls"] += 1
            USAGE["tokens"] += d["usage"]["total_tokens"]
            return json.loads(d["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            msg = e.read()
            if e.code == 400 and (b"json_validate_failed" in msg or b"does not match" in msg):
                USAGE["invalid"] += 1
                return None
            if e.code != 429:
                raise SystemExit(f"HTTP {e.code}: {msg[:400]!r}")
            wait = float(e.headers.get("retry-after") or 0) or 5 * (attempt + 1)
            if wait > 300:      # a daily token limit, not worth waiting for
                raise SystemExit(f"daily token limit of {MODELS[model]} reached (retry after {wait:.0f}s)")
            time.sleep(min(wait + 1, 90))
    raise SystemExit("rate limit did not clear")


def name(rel, s, o):
    return f"{rel.split(':')[1]}({s['text']}, {o['text']})"


def gloss_lines(rels, relations):
    return "\n".join(f"- {r.split(':')[1]}: {relations[r]['gloss']}" for r in sorted(rels))


def mentions_in(p, s0, s1):
    seen, out = set(), []
    for m in p["mentions"]:
        if s0 <= m["start"] and m["end"] <= s1 and m["label"] in CLASS and m["text"] not in seen:
            seen.add(m["text"])
            out.append(m)
    return out


def same_name(x, y):
    a, b = set(x["text"].split()), set(y["text"].split())
    return a <= b or b <= a          # 'Vahan' and 'Vahan Mamikonean' are one person


def pair_decision(model, key, sentence, x, y, opts, relations, gated):
    names = {name(r, s, o): (r, s, o) for r, s, o, _ in opts}
    head = ("You extract relations from a sentence of a history paper, using only the relations listed. "
            "In relation(a, b), a is the subject and b the object.\n"
            f"Sentence: {sentence}\nEntities: \"{x['text']}\" ({x['label']}), \"{y['text']}\" ({y['label']})\n"
            f"Relations:\n{gloss_lines({r for r, *_ in opts}, relations)}\n")
    if gated:
        gate = {"type": "object", "properties": {"related": {"type": "string", "enum": ["yes", "no"]}},
                "required": ["related"], "additionalProperties": False}
        yes = ask(model, head + f"\nDoes the sentence state how \"{x['text']}\" and \"{y['text']}\" are connected "
                  "(kinship, service, membership, captivity, a place, conflict, authorship or succession)? Being named in "
                  "the same sentence is not enough.", gate, key)
        if yes is None or yes["related"] == "no":
            return None, {"related": "no" if yes else "invalid"}
        choices = list(names)
        tail = f"\nOptions: {' | '.join(choices)}\nWhich one does the sentence state?"
    else:
        choices = list(names) + ["none"]
        tail = (f"\nOptions: {' | '.join(choices)}\nWhich one does the sentence state between the two entities? Answer none "
                "if it states none of these relations (being named in the same sentence is not enough).")
    ans = ask(model, head + tail, {"type": "object", "properties": {"relation": {"type": "string", "enum": choices}},
                                   "required": ["relation"], "additionalProperties": False}, key)
    if ans is None:
        return None, {"answer": "invalid"}
    return names.get(ans["relation"]), {"answer": ans["relation"]}


def run_pairs(model, key, ner, relations, gated, cache_file):
    """one question per pair; every answer is appended to cache_file, so a run stopped by a daily limit can resume"""
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            cache[c["k"]] = c
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    decisions = []
    for p in ner["paragraphs"]:
        text = p["text"]
        for s0, s1 in sentences(text):
            for x, y in combinations(mentions_in(p, s0, s1), 2):
                opts = options(x, y, relations) if not same_name(x, y) else []
                if not opts:
                    continue
                k = f"{p['paragraph_id']}|{x['text']}|{y['text']}"
                if k not in cache:
                    best, raw = pair_decision(model, key, text[s0:s1], x, y, opts, relations, gated)
                    cache[k] = {"k": k, "raw": raw, "fact": {"subj": best[1]["text"], "rel": best[0], "obj": best[2]["text"]} if best else None}
                    with cache_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(cache[k], ensure_ascii=False) + "\n")
                d = {"paragraph_id": p["paragraph_id"], "a": x["text"], "b": y["text"], "raw": cache[k]["raw"]}
                if cache[k]["fact"]:
                    d["fact"] = cache[k]["fact"]
                decisions.append(d)
    return decisions


def run_triples(model, key, ner, relations, unit):
    """one call per paragraph or per sentence: the model lists (subject, relation, object) triples"""
    decisions = []
    rel_names = {r.split(":")[1]: r for r in relations}
    for p in ner["paragraphs"]:
        spans = [(0, len(p["text"]))] if unit == "paragraph" else sentences(p["text"])
        for s0, s1 in spans:
            ms = {}
            for m in mentions_in(p, s0, s1):
                ms.setdefault(m["text"], m)
            if len(ms) < 2:
                continue
            schema = {"type": "object", "properties": {"facts": {"type": "array", "items": {
                "type": "object", "properties": {
                    "subject": {"type": "string", "enum": list(ms)}, "relation": {"type": "string", "enum": list(rel_names)},
                    "object": {"type": "string", "enum": list(ms)}, "evidence": {"type": "string"}},
                "required": ["subject", "relation", "object", "evidence"], "additionalProperties": False}}},
                "required": ["facts"], "additionalProperties": False}
            prompt = (f"You extract relations from a {unit} of a history paper, using only the listed relations. "
                      "In a triple, the subject and the object are names from the list of names; the relation says what "
                      "the text states about them (the subject is 'the subject' in the relation descriptions). "
                      "Give a triple only when ONE sentence names both participants and states the relation or directly "
                      "implies it; pronouns do not count; give the quoted evidence (a short phrase from that sentence). "
                      "Give an empty list if there are none.\n\n"
                      f"{unit.capitalize()}:\n{p['text'][s0:s1]}\n\nNames: "
                      + "; ".join(f"{t} ({m['label']})" for t, m in ms.items())
                      + f"\n\nRelations:\n{gloss_lines(set(relations), relations)}")
            out = ask(model, prompt, schema, key)
            if out is None:
                decisions.append({"paragraph_id": p["paragraph_id"], "a": None, "b": None, "raw": "invalid", "type_ok": False})
                continue
            for f in out["facts"]:
                rel = rel_names[f["relation"]]
                s, o = ms[f["subject"]], ms[f["object"]]
                if rel in SYMMETRIC and s["start"] > o["start"]:    # symmetric: the name that comes first is the subject
                    s, o = o, s
                r = relations[rel]
                ok = s["text"] != o["text"] and fits(CLASS[s["label"]], r["domain"]) and fits(CLASS[o["label"]], r["range"])
                d = {"paragraph_id": p["paragraph_id"], "a": s["text"], "b": o["text"], "raw": f, "type_ok": ok}
                if ok:
                    d["fact"] = {"subj": s["text"], "rel": rel, "obj": o["text"]}
                decisions.append(d)
    return decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ner", default="gold")
    ap.add_argument("--model", default="20b", choices=list(MODELS))
    ap.add_argument("--variant", default="paragraph", choices=["paragraph", "sentence", "pair", "gated"])
    ap.add_argument("--rep", type=int, default=1, help="repeat number: gpt-oss answers differ between runs even at temperature 0")
    a = ap.parse_args()
    key = os.environ["GROQ_API_KEY"]
    relations = {k: v for k, v in json.loads(Path("schema/history_relations.json").read_text(encoding="utf-8")).items()
                 if any(c.startswith("schema:") for c in v["range"])}
    ner = json.loads(Path(f"outputs/{a.doc}_ner_{a.ner}.json").read_text(encoding="utf-8"))
    run = f"{a.ner}__llm-{a.model}-{a.variant}-r{a.rep}"
    if a.variant in ("paragraph", "sentence"):
        decisions = run_triples(a.model, key, ner, relations, a.variant)
    else:
        decisions = run_pairs(a.model, key, ner, relations, a.variant == "gated", HERE / "cache" / f"{run}__{a.doc}.jsonl")
    facts, seen = [], set()
    for d in decisions:
        if "fact" in d and (d["paragraph_id"], d["fact"]["subj"], d["fact"]["rel"], d["fact"]["obj"]) not in seen:
            seen.add((d["paragraph_id"], d["fact"]["subj"], d["fact"]["rel"], d["fact"]["obj"]))
            facts.append(d)
    names = sorted({m["text"] for p in ner["paragraphs"] for m in p["mentions"]})
    eid = {n: f"m{i}" for i, n in enumerate(names)}
    pred = {"doc_id": a.doc, "entities": [{"eid": eid[n], "label": n} for n in names],
            "mentions": [{"para": p["paragraph_id"], "start": m["start"], "end": m["end"], "text": m["text"],
                          "entity": eid[m["text"]]} for p in ner["paragraphs"] for m in p["mentions"]],
            "facts": [{"para": d["paragraph_id"], "subj": eid[d["fact"]["subj"]], "rel": d["fact"]["rel"],
                       "obj": eid[d["fact"]["obj"]]} for d in facts]}
    (HERE / "pred" / run).mkdir(parents=True, exist_ok=True)
    (HERE / "decisions").mkdir(exist_ok=True)
    (HERE / "pred" / run / f"{a.doc}.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "decisions" / f"{run}__{a.doc}.json").write_text(json.dumps(
        {"run": run, "model": MODELS[a.model], "usage": USAGE, "decisions": decisions}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{run} {a.doc}: {len(decisions)} decisions, {len(facts)} facts, {USAGE['calls']} calls, "
          f"{USAGE['tokens']} tokens, {USAGE['invalid']} answers broke the schema")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
