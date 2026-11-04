"""LLM-only baseline: one LLM reads the paragraph and does every step itself (find the names, name the entity),
with no GLiNER, no shortlist and no reranker. Compared with the LELA configurations by the same score.py.

Run from the project root, in two stages:
  GROQ_API_KEY=... python pilot/llm_only.py ask     # calls Groq, saves raw answers to pilot/llm_raw/
  YAGO_JSONL=<path to yago-entities.jsonl> python pilot/llm_only.py build   # looks the names up in YAGO, writes pilot/pred/

How an answer becomes a prediction
* The LLM lists each distinct mention (exact text from the paragraph) with the English Wikipedia title of its entity,
  or null. Every whole-word occurrence of the text is marked, longest texts first, so "Łazar P‘arpec‘i" wins over
  "Łazar". A text that is not in the paragraph is dropped (and counted).
* The title is looked up in YAGO by exact id, `yago:<title with underscores>`, ignoring case. Not found = no link.
  So a name the LLM gives that YAGO writes differently (Vahan_II_ vs Vahan_I_Mamikonian) counts as no link.
"""
import json
import os
import re
import sys
import time
import urllib.request

DOCS = ("alford2026", "nechaeva2017")
MODELS = {"120b": "openai/gpt-oss-120b", "20b": "openai/gpt-oss-20b", "qwen27b": "qwen/qwen3.8-27b"}
RUNS = 3
URL = "https://api.groq.com/openai/v1/chat/completions"

# NOTE: the examples in these rules ("Damascius of Syria", "Łazar P‘arpec‘i", "Persian", "Armenian") come from the two
# test paragraphs, so the guided mention scores are an upper bound. Use other examples for a clean run.
RULES = """Rules:
- A mention is the proper name of a person, a place, a people or state, or a work.
- The text of a mention is the name only, spelled exactly as in the paragraph. Do not include titles or "of X"
  epithets: "Damascius of Syria" gives two mentions, "Damascius" and "Syria".
- Do not list adjectives (such as "Persian" or "Armenian"), descriptions without a name (such as "the king"), or dates.
- List every different written form separately (for example both "Łazar P‘arpec‘i" and "Łazar").
- For each mention give "entity": the English Wikipedia article title of the entity meant in this context, or null
  if the thing has no Wikipedia article. Use the historical entity, not a modern namesake."""
# The same rules with examples that do NOT occur in the test paragraphs: the clean run (no leak).
RULES_CLEAN = """Rules:
- A mention is the proper name of a person, a place, a people or state, or a work.
- The text of a mention is the name only, spelled exactly as in the paragraph. Do not include titles or "of X"
  epithets: "Procopius of Caesarea" gives two mentions, "Procopius" and "Caesarea".
- Do not list adjectives (such as "Byzantine" or "Gothic"), descriptions without a name (such as "the emperor"), or dates.
- List every different written form separately (for example both "Justinian I" and "Justinian").
- For each mention give "entity": the English Wikipedia article title of the entity meant in this context, or null
  if the thing has no Wikipedia article. Use the historical entity, not a modern namesake."""
FORMAT = """Answer with JSON only, in this form:
{"mentions": [{"text": "<exact text from the paragraph>", "entity": "<English Wikipedia article title or null>"}]}"""
PROMPTS = {
    "guided": "Find the named entities in the paragraph of a history paper and link each to its Wikipedia article.\n"
              + RULES + "\n" + FORMAT + "\n\nParagraph:\n",
    "clean": "Find the named entities in the paragraph of a history paper and link each to its Wikipedia article.\n"
             + RULES_CLEAN + "\n" + FORMAT + "\n\nParagraph:\n",
    "plain": "Find all named entities in this paragraph from a history paper and link each one to its English "
             "Wikipedia article (null if there is none).\n" + FORMAT + "\n\nParagraph:\n",
}
CONFIGS = [(f"llm_{m}_{p}_r{r}", m, p, r) for p, ms in (("guided", MODELS), ("plain", ("120b",)), ("clean", ("120b", "qwen27b")))
           for m in ms for r in range(1, RUNS + 1)]


def post(payload, key):
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "history-research-pilot/0.1"})
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


def ask():
    key = os.environ["GROQ_API_KEY"]
    for name, m, p, r in CONFIGS:
        for doc in DOCS:
            out = f"pilot/llm_raw/{name}/{doc}.json"
            if os.path.exists(out):
                continue
            text = json.load(open(f"pilot/texts/{doc}.json", encoding="utf-8"))["paragraphs"][0]["text"]
            for attempt in range(1, 5):      # a model sometimes spends its whole budget thinking and returns nothing
                resp = post({"model": MODELS[m], "temperature": 0.3, "max_completion_tokens": 6000,
                             "messages": [{"role": "user", "content": PROMPTS[p] + text}]}, key)
                choice = resp["choices"][0]
                if choice["message"].get("content"):
                    break
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump({"model": MODELS[m], "content": choice["message"].get("content") or "",
                       "finish_reason": choice.get("finish_reason"), "attempts": attempt},
                      open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("saved", out, "attempts", attempt)


def parse(content):
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    m = re.search(r"\{.*\}", content, flags=re.S)
    try:
        return [x for x in json.loads(m.group(0)).get("mentions", []) if isinstance(x, dict) and x.get("text")]
    except (AttributeError, ValueError):
        return None


def norm_id(title):
    return "yago:" + re.sub(r"^yago:", "", str(title).strip(), flags=re.I).replace(" ", "_")


def occurrences(text, surface):
    i = 0
    while (i := text.find(surface, i)) >= 0:
        j = i + len(surface)
        if not (i > 0 and text[i - 1].isalnum()) and not (j < len(text) and text[j].isalnum()):
            yield i, j
        i = j


def build():
    answers = {}
    for name, *_ in CONFIGS:
        for doc in DOCS:
            raw = json.load(open(f"pilot/llm_raw/{name}/{doc}.json", encoding="utf-8"))
            answers[name, doc] = parse(raw["content"])
    wanted = {norm_id(x["entity"]).lower() for ms in answers.values() for x in ms or [] if x.get("entity")}
    yago = {}
    for line in open(os.environ["YAGO_JSONL"], encoding="utf-8"):
        i = line.find('"id": "')
        eid = line[i + 7:line.find('"', i + 7)]
        if eid.lower() in wanted:
            yago[eid.lower()] = eid
    for (name, doc), ms in answers.items():
        text = json.load(open(f"pilot/texts/{doc}.json", encoding="utf-8"))["paragraphs"][0]["text"]
        taken, ents, dropped = [], [], 0
        for x in sorted(ms or [], key=lambda x: -len(x["text"])):
            spans = [s for s in occurrences(text, x["text"]) if not any(s[0] < b and a < s[1] for a, b in taken)]
            dropped += not spans
            link = yago.get(norm_id(x["entity"]).lower()) if x.get("entity") else None
            for a, b in spans:
                taken.append((a, b))
                ents.append({"text": text[a:b], "start": a, "end": b, "entity_id": link, "llm_entity": x.get("entity")})
        os.makedirs(f"pilot/pred/{name}", exist_ok=True)
        ents.sort(key=lambda e: e["start"])
        row = {"id": doc, "entities": ents, "meta": {"unparsed": ms is None, "texts_not_found": dropped}}
        open(f"pilot/pred/{name}/{doc}.jsonl", "w", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
    print("built", len(CONFIGS), "configurations")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"ask": ask, "build": build}[sys.argv[1]]()
