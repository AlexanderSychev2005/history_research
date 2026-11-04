"""Hybrid: the LLM names the entity, YAGO supplies the shortlist, the LLM chooses from it.

For every mention of a full-LLM run (llm_<model>_<prompt>_r<k>, made by llm_only.py):
  1. shortlist = YAGO entries whose title is most similar (WRatio, as in LELA's fuzzy search) to the mention text AND to
     the entity name the LLM gave, plus the exact title match of that name;
  2. the LLM sees the paragraph with the mention marked, its own first guess and the shortlist, and picks one entry or
     "none of them" (the same kind of question as LELA's LLM disambiguator).
Writes pilot/pred/hyb_<model>_<prompt>_r<k>/<doc>.jsonl; score it with score.py.

  GROQ_API_KEY=... YAGO_JSONL=<LELA>/data/yago/yago-entities.jsonl python pilot/hybrid.py 120b clean 1 2 3
Needs numpy and rapidfuzz (LELA's environment has them).
"""
import json
import os
import re
import sys

import numpy as np
from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.dirname(__file__))
from llm_only import MODELS, post  # noqa: E402

DOCS = ("alford2026", "nechaeva2017")
PER_QUERY = 10      # titles kept per query string
MAX_SHORTLIST = 20


def load_yago(path):
    ids, titles = [], []
    for line in open(path, encoding="utf-8"):
        i = line.find('"id": "')
        eid = line[i + 7:line.find('"', i + 7)]
        ids.append(eid)
        titles.append(eid[5:])
    return ids, titles


def descriptions(path, wanted):
    out, n = {}, 0
    for line in open(path, encoding="utf-8"):
        if n in wanted:
            out[n] = json.loads(line).get("description") or ""
        n += 1
    return out


def main(model, prompt, runs):
    key, yago = os.environ["GROQ_API_KEY"], os.environ["YAGO_JSONL"]
    ids, titles = load_yago(yago)
    index = {eid.lower(): k for k, eid in enumerate(ids)}
    texts = {d: json.load(open(f"pilot/texts/{d}.json", encoding="utf-8"))["paragraphs"][0]["text"] for d in DOCS}
    data = {}
    for r in runs:
        name = f"llm_{model}_{prompt}_r{r}"
        data[r] = {d: json.loads(open(f"pilot/pred/{name}/{d}.jsonl", encoding="utf-8").readline())["entities"]
                   for d in DOCS}
    queries = sorted({q for r in data.values() for es in r.values() for e in es
                      for q in (e["text"], e.get("llm_entity")) if q})
    print(len(queries), "query strings against", len(titles), "titles", flush=True)
    scores = process.cdist([q.replace(" ", "_") for q in queries], titles, scorer=fuzz.WRatio, dtype=np.uint8, workers=-1)
    top = {q: [int(k) for k in np.argsort(-scores[i].astype(np.int16), kind="stable")[:PER_QUERY]]  # uint8 would wrap
           for i, q in enumerate(queries)}

    def shortlist(e):
        out = []
        guess = e.get("llm_entity")
        exact = index.get("yago:" + guess.replace(" ", "_").lower()) if guess else None
        for k in ([exact] if exact is not None else []) + (top[guess] if guess else []) + top[e["text"]]:
            if k not in out:
                out.append(k)
        return out[:MAX_SHORTLIST]

    shortlists = {(r, d, e["start"]): shortlist(e) for r, rd in data.items() for d, es in rd.items() for e in es}
    desc = descriptions(yago, {k for s in shortlists.values() for k in s})

    for r, rd in data.items():
        out_dir = f"pilot/pred/hyb_{model}_{prompt}_r{r}"
        os.makedirs(out_dir, exist_ok=True)
        for d, es in rd.items():
            text, result = texts[d], []
            for e in es:
                sl = shortlists[(r, d, e["start"])]
                marked = text[:e["start"]] + "[[" + text[e["start"]:e["end"]] + "]]" + text[e["end"]:]
                lines = "\n".join(f"{n}. {titles[k]} ({desc[k]})" for n, k in enumerate(sl, 1))
                guess = (f'A first reader suggested the entity "{e["llm_entity"]}" (it may be wrong or spelled differently '
                         "from the entries below).\n") if e.get("llm_entity") else ""
                question = (f"Paragraph from a history paper, with one mention marked by [[ ]]:\n{marked}\n\n"
                            f'Mention: "{e["text"]}"\n{guess}'
                            "Which entry of the knowledge base is the historical entity this mention refers to in the "
                            f"paragraph?\n0. None of the listed candidates\n{lines}\n\nAnswer with the number only.")
                resp = post({"model": MODELS[model], "temperature": 0.3, "max_completion_tokens": 3000,
                             "messages": [{"role": "user", "content": question}]}, key)
                answer = resp["choices"][0]["message"].get("content") or ""
                m = re.search(r"\d+", answer)
                n = int(m.group()) if m else 0
                chosen = ids[sl[n - 1]] if 1 <= n <= len(sl) else None
                result.append({**e, "entity_id": chosen, "candidates": [{"entity_id": ids[k]} for k in sl]})
            open(f"{out_dir}/{d}.jsonl", "w", encoding="utf-8").write(
                json.dumps({"id": d, "entities": result}, ensure_ascii=False) + "\n")
            print("saved", out_dir, d, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
