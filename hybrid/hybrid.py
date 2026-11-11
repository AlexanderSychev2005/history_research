"""Hybrid entity pipeline: the LLM finds the names, YAGO supplies a shortlist, the LLM chooses.

  names      the LLM reads each paragraph and lists the names, each with the entity it thinks the name is
  choose     shortlist from YAGO for every name (text + the LLM's guess), the LLM picks one entry or "none"
  shortlists the same shortlists without any LLM call, scored against the gold: how often the right entity is in them

  GROQ_API_KEY=... python hybrid/hybrid.py names 120b 1
  GROQ_API_KEY=... YAGO_JSONL=<yago-entities.jsonl> python hybrid/hybrid.py choose names_120b_r1 20b
  YAGO_JSONL=<yago-entities.jsonl> python hybrid/hybrid.py shortlists names_120b_r1
Needs numpy and rapidfuzz. Predictions go to hybrid/pred/<run>/<doc>.jsonl, the LLM's raw answers to hybrid/raw/.
"""
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "dataset")
DOCS = ("alford2026", "nechaeva2017")
MODELS = {"120b": "openai/gpt-oss-120b", "20b": "openai/gpt-oss-20b"}
URL = "https://api.groq.com/openai/v1/chat/completions"
PER_QUERY, MAX_SHORTLIST = 10, 20

PROMPT = """Find the named entities in the paragraph of a history paper and link each to its Wikipedia article.
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


def paragraphs():
    return [(d, p["id"], p["text"]) for d in DOCS
            for p in json.load(open(os.path.join(DATA, "paragraphs", d + ".json"), encoding="utf-8"))["paragraphs"]]


def post(payload, key):
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "history-research/0.1"})
    for attempt in range(10):
        try:
            return json.load(urllib.request.urlopen(req, timeout=180))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = float(e.headers.get("retry-after") or 0) or 5 * (attempt + 1)
            print(f"429, waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(min(wait + 1, 90))
    raise RuntimeError("rate limit did not clear")


def parse(content):
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    m = re.search(r"\{.*\}", content, flags=re.S)
    try:
        return [x for x in json.loads(m.group(0)).get("mentions", []) if isinstance(x, dict) and x.get("text")]
    except (AttributeError, ValueError):
        return []


def occurrences(text, surface):
    """whole-word occurrences of a surface form"""
    i = 0
    while (i := text.find(surface, i)) >= 0:
        j = i + len(surface)
        if not (i > 0 and text[i - 1].isalnum()) and not (j < len(text) and text[j].isalnum()):
            yield i, j
        i = j


def names(model, runs):
    key = os.environ["GROQ_API_KEY"]
    for r in runs:
        run = f"names_{model}_r{r}"
        rows = {d: [] for d in DOCS}
        for doc, para, text in paragraphs():
            raw = os.path.join(HERE, "raw", run, f"{doc}_{para}.json")
            if not os.path.exists(raw):
                for attempt in range(1, 5):          # a model sometimes returns nothing
                    resp = post({"model": MODELS[model], "temperature": 0.3, "max_completion_tokens": 3000,
                                 "messages": [{"role": "user", "content": PROMPT + text}]}, key)
                    choice = resp["choices"][0]
                    if choice["message"].get("content"):
                        break
                os.makedirs(os.path.dirname(raw), exist_ok=True)
                json.dump({"model": MODELS[model], "content": choice["message"].get("content") or "", "attempts": attempt},
                          open(raw, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            taken = []
            for x in sorted(parse(json.load(open(raw, encoding="utf-8"))["content"]), key=lambda x: -len(x["text"])):
                for a, b in occurrences(text, x["text"]):          # longest text first: "Łazar P‘arpec‘i" before "Łazar"
                    if not any(a < e and s < b for s, e in taken):
                        taken.append((a, b))
                        rows[doc].append({"para": para, "text": text[a:b], "start": a, "end": b,
                                          "entity_id": None, "llm_entity": x.get("entity")})
        out = os.path.join(HERE, "pred", run)
        os.makedirs(out, exist_ok=True)
        for doc, ents in rows.items():
            ents.sort(key=lambda e: (e["para"], e["start"]))
            open(os.path.join(out, doc + ".jsonl"), "w", encoding="utf-8").write(json.dumps({"id": doc, "entities": ents}, ensure_ascii=False) + "\n")
        print("saved", run, flush=True)


# --------------------------------------------------------------------------- shortlists
class Yago:
    def __init__(self, path):
        import numpy as np
        from rapidfuzz import fuzz, process
        self.np, self.fuzz, self.process = np, fuzz, process
        self.ids, self.titles, self.index = [], [], {}
        for n, line in enumerate(open(path, encoding="utf-8")):
            i = line.find('"id": "')
            eid = line[i + 7:line.find('"', i + 7)]
            self.ids.append(eid)
            self.titles.append(eid[5:])
            self.index[eid.lower()] = n
        self.top = {}

    def prepare(self, queries):
        """the PER_QUERY most similar titles (rapidfuzz WRatio) for every query string"""
        queries = sorted(q for q in set(queries) if q not in self.top)
        for s in range(0, len(queries), 40):             # chunks keep the score matrix small
            chunk = queries[s:s + 40]
            scores = self.process.cdist([q.replace(" ", "_") for q in chunk], self.titles, scorer=self.fuzz.WRatio,
                                        dtype=self.np.uint8, workers=-1)
            for i, q in enumerate(chunk):
                # int16: negating uint8 would wrap around
                self.top[q] = [int(k) for k in self.np.argsort(-scores[i].astype(self.np.int16), kind="stable")[:PER_QUERY]]
            print("fuzzy search", min(s + 40, len(queries)), "/", len(queries), flush=True)

    def shortlist(self, text, guess, parts=("exact", "guess", "text")):
        exact = self.index.get("yago:" + guess.replace(" ", "_").lower()) if guess else None
        out = []
        for k in (([exact] if "exact" in parts and exact is not None else [])
                  + (self.top[guess] if "guess" in parts and guess else [])
                  + (self.top[text] if "text" in parts else [])):
            if k not in out:
                out.append(k)
        return out[:MAX_SHORTLIST]


def load_names(run):
    return {d: json.loads(open(os.path.join(HERE, "pred", run, d + ".jsonl"), encoding="utf-8").readline())["entities"]
            for d in DOCS}


def choose(run, chooser):
    key = os.environ["GROQ_API_KEY"]
    yago = Yago(os.environ["YAGO_JSONL"])
    data = load_names(run)
    yago.prepare([q for es in data.values() for e in es for q in (e["text"], e.get("llm_entity")) if q])
    texts = {(d, p): t for d, p, t in paragraphs()}
    decisions, chosen = {}, {}
    for d, es in data.items():                      # one decision per distinct (paragraph, text, guess)
        for e in es:
            decisions.setdefault((d, e["para"], e["text"], e.get("llm_entity")),
                                 (yago.shortlist(e["text"], e.get("llm_entity")), e["start"], e["end"]))
    need = {j for sl, _, _ in decisions.values() for j in sl}
    desc = {}
    for n, line in enumerate(open(os.environ["YAGO_JSONL"], encoding="utf-8")):
        if n in need:
            desc[n] = json.loads(line).get("description") or ""
    for k, (sl, start, end) in decisions.items():
        d, para, mtext, guess = k
        text = texts[d, para]
        marked = text[:start] + "[[" + text[start:end] + "]]" + text[end:]
        lines = "\n".join(f"{i}. {yago.titles[j]} ({desc[j]})" for i, j in enumerate(sl, 1))
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
        chosen[k] = yago.ids[sl[c - 1]] if 1 <= c <= len(sl) else None
        if len(chosen) % 25 == 0:
            print(len(chosen), "/", len(decisions), flush=True)
    out = os.path.join(HERE, "pred", f"hybrid_{run.split('_', 1)[1]}-{chooser}")
    os.makedirs(out, exist_ok=True)
    for d, es in data.items():
        rows = []
        for e in es:
            k = (d, e["para"], e["text"], e.get("llm_entity"))
            rows.append({**e, "entity_id": chosen[k], "candidates": [{"entity_id": yago.ids[j]} for j in decisions[k][0]]})
        open(os.path.join(out, d + ".jsonl"), "w", encoding="utf-8").write(json.dumps({"id": d, "entities": rows}, ensure_ascii=False) + "\n")
    print("saved", out)


VARIANTS = [("exact title only (the LLM's guess looked up as it is)", ("exact",)),
            ("search by the text only", ("text",)),
            ("search by the guess only", ("guess",)),
            ("text + guess, no exact", ("guess", "text")),
            ("exact + guess + text (the hybrid)", ("exact", "guess", "text"))]


def shortlists(run):
    """Shortlist ceiling of each variant: no LLM call, only YAGO and the gold."""
    yago = Yago(os.environ["YAGO_JSONL"])
    data = load_names(run)
    yago.prepare([q for es in data.values() for e in es for q in (e["text"], e.get("llm_entity")) if q])
    gold = []
    for d in DOCS:
        g = json.load(open(os.path.join(DATA, "gold", d + ".json"), encoding="utf-8"))
        gold += [{**m, "doc": d} for m in g["mentions"] if m["yago"]]
    span = {(d, e["para"], e["start"], e["end"]): e for d, es in data.items() for e in es}
    print(f"{len(gold)} gold mentions with a YAGO id; {sum((g['doc'], g['para'], g['start'], g['end']) in span for g in gold)} "
          f"of them found as an exact span by the names stage\n")
    print("| shortlist | right entity in the list: of the found names | of all gold mentions with an id | mean list length |")
    print("|---|---|---|---|")
    for label, parts in VARIANTS:
        hit = found = size = 0
        for g in gold:
            e = span.get((g["doc"], g["para"], g["start"], g["end"]))
            if not e:
                continue
            found += 1
            sl = yago.shortlist(e["text"], e.get("llm_entity"), parts)
            size += len(sl)
            hit += g["yago"].lower() in {yago.ids[j].lower() for j in sl}
        print(f"| {label} | {hit}/{found} ({hit / found:.0%}) | {hit}/{len(gold)} ({hit / len(gold):.0%}) | {size / found:.1f} |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "names":
        names(args[0], [int(x) for x in args[1:]])
    elif cmd == "choose":
        choose(args[0], args[1])
    elif cmd == "shortlists":
        shortlists(args[0])
