"""Stage 2 of the hybrid on the 14 paragraphs: YAGO supplies a shortlist for each name and the LLM chooses.

For every name found by names.py (llm_<model>_clean_r<k>):
  1. shortlist = YAGO entries whose title is most similar (WRatio, as in LELA's fuzzy search) to the text AND to the
     entity name the LLM gave, plus the exact title match of that name (at most 20);
  2. the LLM sees the paragraph with the name marked, its own first guess and the shortlist, and picks one entry or
     "none of the listed candidates".
One decision per distinct (paragraph, text, LLM entity): a name repeated in a paragraph is decided once.
Writes pilot14/pred/hyb_<names model>-<chooser>_clean_r<k>/<doc>.jsonl.

  GROQ_API_KEY=... YAGO_JSONL=<LELA>/data/yago/yago-entities.jsonl python pilot14/hybrid.py 120b 20b 1 2 3
(names from the first model, the choice by the second; chooser calls use reasoning_effort=low to fit Groq's token limits)
Needs numpy and rapidfuzz (LELA's environment has them).
"""
import json
import os
import re
import sys

import numpy as np
from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pilot"))
from llm_only import MODELS, post  # noqa: E402

DOCS = ("alford2026", "nechaeva2017")
PER_QUERY, MAX_SHORTLIST = 10, 20


def main(model, chooser, runs):
    key, yago = os.environ["GROQ_API_KEY"], os.environ["YAGO_JSONL"]
    ids, titles, index = [], [], {}
    for n, line in enumerate(open(yago, encoding="utf-8")):
        i = line.find('"id": "')
        eid = line[i + 7:line.find('"', i + 7)]
        ids.append(eid)
        titles.append(eid[5:])
        index[eid.lower()] = n
    texts = {(d, p["id"]): p["text"] for d in DOCS
             for p in json.load(open(f"pilot14/texts/{d}.json", encoding="utf-8"))["paragraphs"]}
    data = {r: {d: json.loads(open(f"pilot14/pred/llm_{model}_clean_r{r}/{d}.jsonl", encoding="utf-8").readline())["entities"]
                for d in DOCS} for r in runs}
    queries = sorted({q for rd in data.values() for es in rd.values() for e in es
                      for q in (e["text"], e.get("llm_entity")) if q})
    print(len(queries), "query strings", flush=True)
    scores = process.cdist([q.replace(" ", "_") for q in queries], titles, scorer=fuzz.WRatio, dtype=np.uint8, workers=-1)
    top = {q: [int(k) for k in np.argsort(-scores[i].astype(np.int16), kind="stable")[:PER_QUERY]]  # uint8 would wrap
           for i, q in enumerate(queries)}

    def shortlist(e):
        guess = e.get("llm_entity")
        exact = index.get("yago:" + guess.replace(" ", "_").lower()) if guess else None
        out = []
        for k in ([exact] if exact is not None else []) + (top[guess] if guess else []) + top[e["text"]]:
            if k not in out:
                out.append(k)
        return out[:MAX_SHORTLIST]

    decisions = {}      # (run, doc, para, text, llm_entity) -> shortlist indices, chosen index
    need = set()
    for r, rd in data.items():
        for d, es in rd.items():
            for e in es:
                k = (r, d, e["para"], e["text"], e.get("llm_entity"))
                if k not in decisions:
                    decisions[k] = (shortlist(e), e["start"], e["end"])
                    need.update(decisions[k][0])
    desc, n = {}, 0
    for line in open(yago, encoding="utf-8"):
        if n in need:
            desc[n] = json.loads(line).get("description") or ""
        n += 1
    print(len(decisions), "decisions to ask", flush=True)

    chosen = {}
    for k, (sl, start, end) in decisions.items():
        r, d, para, mtext, guess = k
        text = texts[d, para]
        marked = text[:start] + "[[" + text[start:end] + "]]" + text[end:]
        lines = "\n".join(f"{i}. {titles[j]} ({desc[j]})" for i, j in enumerate(sl, 1))
        hint = (f'A first reader suggested the entity "{guess}" (it may be wrong or spelled differently from the entries '
                "below).\n") if guess else ""
        question = (f"Paragraph from a history paper, with one mention marked by [[ ]]:\n{marked}\n\n"
                    f'Mention: "{mtext}"\n{hint}'
                    "Which entry of the knowledge base is the historical entity this mention refers to in the "
                    f"paragraph?\n0. None of the listed candidates\n{lines}\n\nAnswer with the number only.")
        resp = post({"model": MODELS[chooser], "temperature": 0.3, "max_completion_tokens": 1500,
                     "reasoning_effort": "low", "messages": [{"role": "user", "content": question}]}, key)
        m = re.search(r"\d+", resp["choices"][0]["message"].get("content") or "")
        c = int(m.group()) if m else 0
        chosen[k] = ids[sl[c - 1]] if 1 <= c <= len(sl) else None
        print(len(chosen), "/", len(decisions), flush=True) if len(chosen) % 25 == 0 else None

    for r, rd in data.items():
        out = f"pilot14/pred/hyb_{model}-{chooser}_clean_r{r}"
        os.makedirs(out, exist_ok=True)
        for d, es in rd.items():
            rows = []
            for e in es:
                k = (r, d, e["para"], e["text"], e.get("llm_entity"))
                rows.append({**e, "entity_id": chosen[k], "candidates": [{"entity_id": ids[j]} for j in decisions[k][0]]})
            open(f"{out}/{d}.jsonl", "w", encoding="utf-8").write(json.dumps({"id": d, "entities": rows}, ensure_ascii=False) + "\n")
        print("saved", out, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
