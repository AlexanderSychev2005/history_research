"""Runs LELA on the dataset paragraphs and writes lela/pred/<name>/<doc>.jsonl, the format hybrid/score.py reads.

    cd <LELA clone> && uv run python <this repository>/lela/run.py first      # or: llm
LELA must be started from its own folder (it looks for data/yago/yago-entities.jsonl there). `llm` needs GROQ_API_KEY in the
environment; the key is put into a temporary copy of lela/configs/llm.json, never into the repository.
LELA does not retry when the LLM API answers 429 (rate limit) and leaves the mention unlinked without a word; here a 429 is
retried after the pause the server asks for, and the run stops with an error if the limit does not clear.
"""
import json
import os
import runpy
import sys
import tempfile
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "dataset", "paragraphs")
DOCS = ("alford2026", "nechaeva2017")
name = sys.argv[1]

_post = requests.post


def post_with_retry(*args, **kwargs):
    for attempt in range(10):
        response = _post(*args, **kwargs)
        if response.status_code != 429:
            return response
        wait = float(response.headers.get("retry-after") or 0) or 5 * (attempt + 1)
        print(f"429, waiting {wait:.0f}s", file=sys.stderr, flush=True)
        time.sleep(min(wait + 1, 90))
    raise RuntimeError("rate limit did not clear; the run is incomplete, start it again later")


requests.post = post_with_retry

tmp = tempfile.mkdtemp()
config = open(os.path.join(HERE, "configs", name + ".json"), encoding="utf-8").read()
if "__KEY__" in config:
    config = config.replace("__KEY__", os.environ["GROQ_API_KEY"])
open(os.path.join(tmp, "config.json"), "w", encoding="utf-8").write(config)
with open(os.path.join(tmp, "in.jsonl"), "w", encoding="utf-8") as f:
    for doc in DOCS:
        for p in json.load(open(os.path.join(DATA, doc + ".json"), encoding="utf-8"))["paragraphs"]:
            f.write(json.dumps({"id": f"{doc}|{p['id']}", "text": p["text"]}, ensure_ascii=False) + "\n")

out = os.path.join(tmp, "out.jsonl")
sys.path.insert(0, os.getcwd())
sys.argv = ["lela.cli", "--config", os.path.join(tmp, "config.json"), "--input", os.path.join(tmp, "in.jsonl"), "--output", out]
runpy.run_module("lela.cli", run_name="__main__")

rows = {d: [] for d in DOCS}
for line in open(out, encoding="utf-8"):
    row = json.loads(line)
    doc, para = row["id"].split("|")
    for e in row["entities"]:
        rows[doc].append({"para": para, "text": e["text"], "start": e["start"], "end": e["end"], "label": e["label"],
                          "entity_id": e["entity_id"], "candidates": [c["entity_id"] for c in e["candidates"]]})
pred = os.path.join(HERE, "pred", name)
os.makedirs(pred, exist_ok=True)
for doc, ents in rows.items():
    ents.sort(key=lambda e: (e["para"], e["start"]))
    open(os.path.join(pred, doc + ".jsonl"), "w", encoding="utf-8").write(json.dumps({"id": doc, "entities": ents}, ensure_ascii=False) + "\n")
print("saved", pred, {d: len(e) for d, e in rows.items()}, "mentions")
