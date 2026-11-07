"""Relation extraction with Jev: one decision per pair of names in a sentence over the YAGO relations their
types allow. Writes outputs/<doc>_relations_<run>.json and eval/pred/<run>/, run = <ner>__<model>[-guided].

  uv run python pipeline/relations_jev.py --doc alford2026 --ner jev --model jev-api --guided
"""
import argparse
import json
import math
import os
import re
import string
import time
from itertools import combinations
from pathlib import Path

import requests
from dotenv import load_dotenv

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
API = "https://api.typesafe.ai/v1/systemone"  # models: GET https://api.typesafe.ai/v1/models
API_MODEL = "jev-latest"
TEMPERATURE = 1.07
LETTERS = string.ascii_uppercase + string.ascii_lowercase

CLASS = {
    "person": "schema:Person",
    "location": "schema:Place",
    "organization": "schema:Organization",
    "event": "schema:Event",
    "written work": "schema:CreativeWork",
}
SYMMETRIC = {"schema:spouse", "yago:neighbor", "ex:relativeOf"}  # asked once, subject = the name that comes first
NONE = "none of these relations is stated in the sentence"

api_versions = set()  # what jev-latest resolved to, as reported by the API


def sentences(text):
    """(start, end) of each sentence; 'c. 470' and 'St. John' do not end one."""
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?”])\s+(?=[A-Z“‘(])", text):
        if not re.search(r"\b(?:c|St|cf|ed|eds|trans|vol|no|pp?)\.$", text[start:m.start()]):
            out.append((start, m.start()))
            start = m.end()
    out.append((start, len(text)))
    return out


def fits(cls, allowed):
    return cls in allowed or "schema:Thing" in allowed


def options(a, b, relations):
    """(relation, subject, object, description) for every relation the two types allow, in both directions."""
    out = []
    for s, o in ((a, b), (b, a)):
        for name, r in relations.items():
            if (s, o) == (b, a) and name in SYMMETRIC:
                continue
            if fits(CLASS[s["label"]], r["domain"]) and fits(CLASS[o["label"]], r["range"]):
                gloss = r["gloss"].replace("the subject", s["text"]).replace("the object", o["text"])
                gloss = gloss.replace("subject", s["text"]).replace("object", o["text"])
                out.append((name, s, o, gloss))
    return out


def post_api(state, questions, key):
    """One request to the hosted Jev API -> its answers."""
    for attempt in range(6):
        r = requests.post(API, json={"model": API_MODEL, "state": state, "questions": questions},
                          headers={"Authorization": f"Bearer {key}"}, timeout=120)
        if r.status_code in (429, 529):
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        r.raise_for_status()
        answer = r.json()
        api_versions.add(answer.get("model"))
        return answer["answers"]
    raise RuntimeError("Jev API: still rate limited after 6 attempts")


def ask_openjev(sentence, x, y, opts, key=None):
    """OpenJev on llama-server: lettered options + 'none' -> probabilities, 'none' last."""
    state = json.dumps({"sentence": sentence, "entities": {x["text"]: x["label"], y["text"]: y["label"]}},
                       ensure_ascii=False)
    descs = [d for *_, d in opts] + [NONE]
    lines = "\n".join(f"[{LETTERS[i]}] {d}" for i, d in enumerate(descs))
    prompt = (f"State:\n{state}\n\nQuestion: Which relation between {x['text']} and {y['text']} does the sentence "
              f"state?\nOptions:\n{lines}\n\nAnswer with the letter of the best option only.")
    r = requests.post(SERVER, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": min(len(descs) + 10, 100),
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120)
    r.raise_for_status()
    lp = {}
    for t in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        k = t["token"].strip()
        if k not in lp or t["token"] == k:
            lp[k] = t["logprob"]
    z = [lp.get(LETTERS[i], -30.0) / TEMPERATURE for i in range(len(descs))]
    e = [math.exp(v - max(z)) for v in z]
    return [v / sum(e) for v in e]


def ask_api(sentence, x, y, opts, key):
    """The OpenJev question sent to the API unchanged: one choice, options named by letter, 'none' as an option."""
    state = json.dumps({"sentence": sentence, "entities": {x["text"]: x["label"], y["text"]: y["label"]}},
                       ensure_ascii=False)
    descs = [d for *_, d in opts] + [NONE]
    names = [LETTERS[i] for i in range(len(descs))]
    question = {"type": "choice", "criteria": dict(zip(names, descs)),
                "instructions": f"Which relation between {x['text']} and {y['text']} does the sentence state?"}
    probs = post_api(state, {"relation": question}, key)["relation"]["probabilities"]
    return [probs[n] for n in names]


def ask_api_guided(sentence, x, y, opts, key):
    """p(option) = noul * choice, p(none) = 1 - noul."""
    names = [f"{rel.split(':')[1]}({s['text']}, {o['text']})" for rel, s, o, _ in opts]
    questions = {
        "related": {"type": "noul",
                    "instructions": f"Does `sentence` state how {x['text']} and {y['text']} are connected?",
                    "criteria": {"true": f"The sentence says how {x['text']} and {y['text']} relate: kinship, service, "
                                         "membership, captivity, a place, conflict, authorship or succession.",
                                 "false": f"{x['text']} and {y['text']} are only named in the same sentence."}},
        "relation": {"type": "choice",
                     "instructions": f"Which relation between {x['text']} and {y['text']} does `sentence` state?",
                     "criteria": {n: d for n, (*_, d) in zip(names, opts)}},
    }
    state = {"sentence": sentence, "entities": {x["text"]: x["label"], y["text"]: y["label"]}}
    answers = post_api(state, questions, key)
    noul, probs = answers["related"]["noul"], answers["relation"]["probabilities"]
    return [noul * probs[n] for n in names] + [1 - noul]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ner", default="gliner-clean", choices=["gliner", "gliner-clean", "jev", "gold"],
                    help="mentions: outputs/<doc>_ner_<ner>.json")
    ap.add_argument("--model", default="openjev", choices=["openjev", "jev-api"],
                    help="openjev: local llama-server (OpenJev-Flash-9B-Q4_K_M, port 8080); jev-api: JEV_API_KEY in .env")
    ap.add_argument("--guided", action="store_true",
                    help="jev-api only: descriptive option names and a yes/no question instead of a 'none' option")
    ap.add_argument("--paragraphs", nargs="*", help="only these paragraph ids (default: all)")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    if a.guided and a.model != "jev-api":
        ap.error("--guided needs --model jev-api")
    key = None
    if a.model == "jev-api":
        load_dotenv(dotenv_path=".env")
        key = os.getenv("JEV_API_KEY") or ap.error("JEV_API_KEY is missing from .env")
    ask = ask_openjev if a.model == "openjev" else ask_api_guided if a.guided else ask_api
    run = f"{a.ner}__{a.model}{'-guided' if a.guided else ''}"

    relations = {k: v for k, v in json.loads(Path("schema/history_relations.json").read_text(encoding="utf-8")).items()
                 if any(c.startswith("schema:") for c in v["range"])}  # entity-to-entity only; dates are not mentions
    ner = json.loads(Path(f"outputs/{a.doc}_ner_{a.ner}.json").read_text(encoding="utf-8"))
    decisions = []
    for p in ner["paragraphs"]:
        if a.paragraphs and p["paragraph_id"] not in a.paragraphs:
            continue
        text = p["text"]
        for s0, s1 in sentences(text):
            seen, ms = set(), []
            for m in p["mentions"]:  # one mention per name in a sentence
                if s0 <= m["start"] and m["end"] <= s1 and m["label"] in CLASS and m["text"] not in seen:
                    seen.add(m["text"])
                    ms.append(m)
            for x, y in combinations(ms, 2):
                if set(x["text"].split()) <= set(y["text"].split()) or set(y["text"].split()) <= set(x["text"].split()):
                    continue  # 'Vahan' and 'Vahan Mamikonean' are one person, not a relation
                opts = options(x, y, relations)
                if not opts:
                    continue
                probs = ask(text[s0:s1], x, y, opts, key)
                best = max(range(len(probs)), key=probs.__getitem__)
                d = {"paragraph_id": p["paragraph_id"], "sentence": [s0, s1], "a": x["text"], "b": y["text"],
                     "options": [{"rel": r, "subj": s["text"], "obj": o["text"], "p": round(q, 4)}
                                 for (r, s, o, _), q in zip(opts, probs)],
                     "p_none": round(probs[-1], 4)}
                if best < len(opts) and probs[best] >= a.threshold:
                    r, s, o, _ = opts[best]
                    d["fact"] = {"subj": s["text"], "rel": r, "obj": o["text"], "p": round(probs[best], 4)}
                decisions.append(d)

    model = (f"{API_MODEL} ({', '.join(sorted(v for v in api_versions if v))})" if a.model == "jev-api"
             else "OpenJev-Flash-9B-Q4_K_M (llama.cpp)")
    out = Path(f"outputs/{a.doc}_relations_{run}.json")
    out.write_text(json.dumps({"doc_id": a.doc, "run": run, "model": model, "threshold": a.threshold,
                               "decisions": decisions}, ensure_ascii=False, indent=1), encoding="utf-8")

    # eval format: an entity per distinct name (eval/evaluate.py aligns names with the gold labels and aliases)
    names = sorted({m["text"] for p in ner["paragraphs"] for m in p["mentions"]})
    eid = {n: f"m{i}" for i, n in enumerate(names)}
    pred = {
        "doc_id": a.doc,
        "entities": [{"eid": eid[n], "label": n} for n in names],
        "mentions": [{"para": p["paragraph_id"], "start": m["start"], "end": m["end"], "text": m["text"],
                      "entity": eid[m["text"]]} for p in ner["paragraphs"] for m in p["mentions"]],
        "facts": [{"para": d["paragraph_id"], "subj": eid[d["fact"]["subj"]], "rel": d["fact"]["rel"],
                   "obj": eid[d["fact"]["obj"]]} for d in decisions if "fact" in d],
    }
    dest = Path(f"eval/pred/{run}/{a.doc}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{run}: {len(decisions)} decisions, {len(pred['facts'])} facts -> {out}, {dest}")


if __name__ == "__main__":
    main()
