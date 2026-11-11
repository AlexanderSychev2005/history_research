"""Hybrid entity linker for the dataset: the LLM finds the names, YAGO supplies a shortlist, the LLM chooses.

    uv run --no-project --env-file .env --with numpy --with rapidfuzz python hybrid/hybrid.py [names_model] [chooser_model]

Reads dataset/paragraphs/, writes hybrid/pred/<doc>.jsonl; the LLM answers are cached in hybrid/cache/, so a rerun (for
example after a rate limit) continues where it stopped. Environment: GROQ_API_KEY, YAGO_JSONL (see .env.example).
"""
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "dataset")
CACHE = os.path.join(HERE, "cache")
DOCS = ("alford2026", "nechaeva2017")
MODELS = {"120b": "openai/gpt-oss-120b", "20b": "openai/gpt-oss-20b"}
URL = "https://api.groq.com/openai/v1/chat/completions"
PER_QUERY, MAX_SHORTLIST = 10, 20

NAMES_PROMPT = """Find the named entities in the paragraph of a history paper and link each to its Wikipedia article.
Rules:
- A mention is the proper name of a person, a place, a people or state, or a work.
- The text of a mention is the name only, spelled exactly as in the paragraph. Do not include titles or "of X"
  epithets: "Procopius of Caesarea" gives two mentions, "Procopius" and "Caesarea".
- Do not list adjectives (such as "Byzantine" or "Gothic"), descriptions without a name (such as "the emperor"), or dates.
- List every different written form separately (for example both "Justinian I" and "Justinian").
- For each mention give "entity": the English Wikipedia article title of the entity meant in this context, or null
  if the thing has no Wikipedia article. Use the historical entity, not a modern namesake.
Answer with JSON only, in this form:
{"mentions": [{"text": "<exact text from the paragraph>", "entity": "<English Wikipedia article title or null>"}]}

Paragraph:
"""


def post(payload):
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {
        "Authorization": "Bearer " + os.environ["GROQ_API_KEY"], "Content-Type": "application/json",
        "User-Agent": "history-research/0.1"})
    for attempt in range(10):
        try:
            return json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = float(e.headers.get("retry-after") or 0) or 5 * (attempt + 1)
            print(f"429, waiting {wait:.0f}s", file=sys.stderr, flush=True)
            time.sleep(min(wait + 1, 90))
    raise RuntimeError("rate limit did not clear; run again later, the cache keeps what is done")


def paragraphs():
    return [(d, p["id"], p["text"]) for d in DOCS
            for p in json.load(open(os.path.join(DATA, "paragraphs", d + ".json"), encoding="utf-8"))["paragraphs"]]


def occurrences(text, surface):
    """whole-word occurrences of a surface form"""
    i = 0
    while (i := text.find(surface, i)) >= 0:
        j = i + len(surface)
        if not (i > 0 and text[i - 1].isalnum()) and not (j < len(text) and text[j].isalnum()):
            yield i, j
        i = j


def find_names(model):
    """Stage 1: {doc: [mention]}, a mention is {para, text, start, end, llm_entity}."""
    rows = {d: [] for d in DOCS}
    for doc, para, text in paragraphs():
        path = os.path.join(CACHE, "names", f"{doc}_{para}.json")
        if not os.path.exists(path):
            for _ in range(4):                                       # a model sometimes returns nothing
                content = post({"model": MODELS[model], "temperature": 0.3, "max_completion_tokens": 3000,
                                "messages": [{"role": "user", "content": NAMES_PROMPT + text}]})
                if content:
                    break
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump({"content": content}, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print("names", doc, para, flush=True)
        content = re.sub(r"<think>.*?</think>", "", json.load(open(path, encoding="utf-8"))["content"], flags=re.S)
        try:
            listed = [x for x in json.loads(re.search(r"\{.*\}", content, flags=re.S).group(0)).get("mentions", [])
                      if isinstance(x, dict) and x.get("text")]
        except (AttributeError, ValueError):
            listed = []
        taken = []
        for x in sorted(listed, key=lambda x: -len(x["text"])):      # longest first: "Łazar P‘arpec‘i" before "Łazar"
            for a, b in occurrences(text, x["text"]):
                if not any(a < e and s < b for s, e in taken):
                    taken.append((a, b))
                    rows[doc].append({"para": para, "text": text[a:b], "start": a, "end": b, "llm_entity": x.get("entity")})
        rows[doc].sort(key=lambda m: (m["para"], m["start"]))
    return rows


class Yago:
    def __init__(self, path):
        self.path, self.ids = path, []
        for line in open(path, encoding="utf-8"):
            i = line.find('"id": "')
            self.ids.append(line[i + 7:line.find('"', i + 7)])
        self.titles = [i[5:] for i in self.ids]                      # drop "yago:"
        self.top = {}

    def search(self, queries):
        """the PER_QUERY titles most similar (rapidfuzz WRatio) to each query, over all YAGO titles"""
        queries = sorted(q for q in set(queries) if q not in self.top)
        for s in range(0, len(queries), 40):                        # chunks keep the score matrix small
            chunk = queries[s:s + 40]
            scores = process.cdist([q.replace(" ", "_") for q in chunk], self.titles, scorer=fuzz.WRatio,
                                   dtype=np.uint8, workers=-1)
            for i, q in enumerate(chunk):                           # int16: negating uint8 would wrap around
                self.top[q] = [int(k) for k in np.argsort(-scores[i].astype(np.int16), kind="stable")[:PER_QUERY]]
            print("search", min(s + 40, len(queries)), "/", len(queries), flush=True)

    def shortlist(self, text, guess):
        """the most similar to the LLM's guess, then to the written name; no repeats, at most MAX_SHORTLIST"""
        out = []
        for k in (self.top[guess] if guess else []) + self.top[text]:
            if k not in out:
                out.append(k)
        return out[:MAX_SHORTLIST]

    def descriptions(self, wanted):
        out = {}
        for n, line in enumerate(open(self.path, encoding="utf-8")):
            if n in wanted:
                out[n] = json.loads(line).get("description") or ""
        return out


def choose(yago, texts, rows, model):
    """Stage 2: the LLM picks one entry of each shortlist or none. One question per distinct (paragraph, text, guess)."""
    yago.search([q for ms in rows.values() for m in ms for q in (m["text"], m["llm_entity"]) if q])
    decisions = {(d, m["para"], m["text"], m["llm_entity"]): (yago.shortlist(m["text"], m["llm_entity"]), m["start"], m["end"])
                 for d, ms in rows.items() for m in ms}
    desc = yago.descriptions({j for sl, _, _ in decisions.values() for j in sl})
    path = os.path.join(CACHE, "choices.json")
    done = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for n, (k, (sl, start, end)) in enumerate(decisions.items(), 1):
        key = json.dumps(k, ensure_ascii=False)
        if key in done:
            continue
        doc, para, mtext, guess = k
        text = texts[doc, para]
        lines = "\n".join(f"{i}. {yago.titles[j]} ({desc[j]})" for i, j in enumerate(sl, 1))
        hint = (f'A first reader suggested the entity "{guess}" (it may be wrong or spelled differently from the entries '
                "below).\n") if guess else ""
        question = ("Paragraph from a history paper, with one mention marked by [[ ]]:\n"
                    f"{text[:start]}[[{text[start:end]}]]{text[end:]}\n\nMention: \"{mtext}\"\n{hint}"
                    "Which entry of the knowledge base is the historical entity this mention refers to in the "
                    f"paragraph?\n0. None of the listed candidates\n{lines}\n\nAnswer with the number only.")
        for _ in range(3):                                           # low reasoning effort sometimes returns nothing
            m = re.search(r"\d+", post({"model": MODELS[model], "temperature": 0.3, "max_completion_tokens": 1500,
                                        "reasoning_effort": "low", "messages": [{"role": "user", "content": question}]}))
            if m:
                break
        c = int(m.group()) if m else 0
        done[key] = yago.ids[sl[c - 1]] if 1 <= c <= len(sl) else None
        if n % 10 == 0:
            os.makedirs(CACHE, exist_ok=True)
            json.dump(done, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print("choice", n, "/", len(decisions), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    json.dump(done, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return {k: (done[json.dumps(k, ensure_ascii=False)], sl) for k, (sl, _, _) in decisions.items()}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    names_model, chooser = (sys.argv[1:] + ["120b", "20b"])[:2]
    texts = {(d, p): t for d, p, t in paragraphs()}
    rows = find_names(names_model)
    yago = Yago(os.environ["YAGO_JSONL"])
    result = choose(yago, texts, rows, chooser)
    out = os.path.join(HERE, "pred")
    os.makedirs(out, exist_ok=True)
    for d, ms in rows.items():
        ents = []
        for m in ms:
            link, sl = result[d, m["para"], m["text"], m["llm_entity"]]
            ents.append({**m, "entity_id": link, "candidates": [yago.ids[j] for j in sl]})
        open(os.path.join(out, d + ".jsonl"), "w", encoding="utf-8").write(json.dumps({"id": d, "entities": ents}, ensure_ascii=False) + "\n")
    print("saved", out)
