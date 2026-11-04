"""Stage 1 of the hybrid on the 14 paragraphs: the LLM lists the names and the entity it thinks each one is (the clean
prompt of pilot/llm_only.py: our annotation rules, with examples that are not in the test paragraphs).

  GROQ_API_KEY=... python pilot14/names.py ask 120b 1 2 3                       # one request per paragraph
  YAGO_JSONL=<LELA>/data/yago/yago-entities.jsonl python pilot14/names.py build 120b 1 2 3
`build` marks every whole-word occurrence of each listed text (longest first) and looks the entity name up in YAGO by
exact title; that is the "full LLM" prediction llm_<model>_clean_r<k>. hybrid.py continues from it.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pilot"))
from llm_only import MODELS, PROMPTS, norm_id, occurrences, parse, post  # noqa: E402

DOCS = ("alford2026", "nechaeva2017")


def paragraphs():
    return [(d, p["id"], p["text"]) for d in DOCS
            for p in json.load(open(f"pilot14/texts/{d}.json", encoding="utf-8"))["paragraphs"]]


def ask(model, runs):
    key = os.environ["GROQ_API_KEY"]
    for r in runs:
        name = f"llm_{model}_clean_r{r}"
        for doc, para, text in paragraphs():
            out = f"pilot14/llm_raw/{name}/{doc}_{para}.json"
            if os.path.exists(out):
                continue
            for attempt in range(1, 5):          # a model sometimes returns nothing
                resp = post({"model": MODELS[model], "temperature": 0.3, "max_completion_tokens": 3000,
                             "messages": [{"role": "user", "content": PROMPTS["clean"] + text}]}, key)
                choice = resp["choices"][0]
                if choice["message"].get("content"):
                    break
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump({"model": MODELS[model], "content": choice["message"].get("content") or "",
                       "finish_reason": choice.get("finish_reason"), "attempts": attempt},
                      open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("saved", out, "attempts", attempt, flush=True)


def build(model, runs):
    answers = {}
    for r in runs:
        for doc, para, _ in paragraphs():
            raw = json.load(open(f"pilot14/llm_raw/llm_{model}_clean_r{r}/{doc}_{para}.json", encoding="utf-8"))
            answers[r, doc, para] = parse(raw["content"])
    wanted = {norm_id(x["entity"]).lower() for ms in answers.values() for x in ms or [] if x.get("entity")}
    yago = {}
    for line in open(os.environ["YAGO_JSONL"], encoding="utf-8"):
        i = line.find('"id": "')
        eid = line[i + 7:line.find('"', i + 7)]
        if eid.lower() in wanted:
            yago[eid.lower()] = eid
    rows = {}
    for doc, para, text in paragraphs():
        for r in runs:
            ms = answers[r, doc, para]
            taken, ents = [], []
            for x in sorted(ms or [], key=lambda x: -len(x["text"])):
                spans = [s for s in occurrences(text, x["text"]) if not any(s[0] < b and a < s[1] for a, b in taken)]
                link = yago.get(norm_id(x["entity"]).lower()) if x.get("entity") else None
                for a, b in spans:
                    taken.append((a, b))
                    ents.append({"para": para, "text": text[a:b], "start": a, "end": b, "entity_id": link,
                                 "llm_entity": x.get("entity")})
            ents.sort(key=lambda e: e["start"])
            rows.setdefault((r, doc), []).extend(ents)
    for (r, doc), ents in rows.items():
        out = f"pilot14/pred/llm_{model}_clean_r{r}"
        os.makedirs(out, exist_ok=True)
        open(f"{out}/{doc}.jsonl", "w", encoding="utf-8").write(json.dumps({"id": doc, "entities": ents}, ensure_ascii=False) + "\n")
    print("built", len(runs), "runs")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"ask": ask, "build": build}[sys.argv[1]](sys.argv[2], [int(x) for x in sys.argv[3:]])
