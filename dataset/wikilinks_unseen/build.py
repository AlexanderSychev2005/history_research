"""Builds dataset/wikilinks_unseen/ from the WikilinksNED Unseen-Mentions test file.

    uv run --no-project --env-file .env python dataset/wikilinks_unseen/build.py <test-00001-of-00001.jsonl>

The file is data/test-00001-of-00001.jsonl of https://huggingface.co/datasets/naist-nlp/unseen (10,000 one-sentence examples,
each with one mention and the id of its Wikipedia page). The page ids are turned into the current English Wikipedia titles with
the MediaWiki API (a redirect is followed to its target) and the titles into YAGO 4.5 ids: a YAGO id is the title with `_`;
`yago` is null when the YAGO dump (YAGO_JSONL) has no such entity. WIKIPEDIA_CONTACT (an e-mail or URL) goes into the
User-Agent, as Wikimedia asks.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://en.wikipedia.org/w/api.php"
AGENT = "history-research-linker/0.1" + (f" ({os.environ['WIKIPEDIA_CONTACT']})" if os.environ.get("WIKIPEDIA_CONTACT") else "")


def api(**params):
    time.sleep(0.5 if os.environ.get("WIKIPEDIA_CONTACT") else 6.5)
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode({"action": "query", "format": "json", **params}),
                                 headers={"User-Agent": AGENT})
    return json.load(urllib.request.urlopen(req, timeout=120))["query"]


def titles_of(page_ids):
    """{page id: current title or None}: gone pages are None, redirects are replaced by their target"""
    out, redirects = {}, {}
    for s in range(0, len(page_ids), 50):
        chunk = page_ids[s:s + 50]
        for p in api(pageids="|".join(chunk), prop="info")["pages"].values():
            out[str(p.get("pageid", ""))] = None if "missing" in p or "invalidreason" in p else p["title"]
            if "redirect" in p:
                redirects[str(p["pageid"])] = p["title"]
        print("titles", min(s + 50, len(page_ids)), "/", len(page_ids), flush=True)
    todo = sorted(set(redirects.values()))
    target = {}
    for s in range(0, len(todo), 50):
        q = api(titles="|".join(todo[s:s + 50]), redirects=1)
        step = {x["from"]: x["to"] for x in q.get("normalized", [])}
        step.update({x["from"]: x["to"] for x in q.get("redirects", [])})
        for t in todo[s:s + 50]:
            t0 = step.get(t, t)
            target[t] = step.get(t0, t0)
    return {i: (target.get(redirects[i], redirects[i]) if i in redirects else out.get(i)) for i in page_ids}


def yago_id(title):
    """YAGO 4.5 names an entity by its English Wikipedia title: Robert W. Thomson -> yago:Robert_W_Thomson"""
    return "yago:" + re.sub(r"_+", "_", re.sub(r"[^\w-]", "_", title.replace("–", "-"))).strip("_")


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    page_ids = sorted({e["label"][0] for r in rows for e in r["entities"]})
    cache = os.path.join(HERE, "titles.json")
    titles = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else titles_of(page_ids)
    json.dump(titles, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    exact = {}                                                       # lower-case YAGO id -> the id as the dump writes it
    for line in open(os.environ["YAGO_JSONL"], encoding="utf-8"):
        i = line.find('"id": "')
        exact[line[i + 7:line.find('"', i + 7)].lower()] = line[i + 7:line.find('"', i + 7)]
    real = {i: (exact[yago_id(t).lower()] if t and yago_id(t).lower() in exact else None) for i, t in titles.items()}

    paragraphs, mentions, entities = [], [], {}
    for r in rows:
        e = r["entities"][0]
        pid = e["label"][0]
        paragraphs.append({"id": r["id"], "text": r["text"]})
        mentions.append({"para": r["id"], "start": e["start"], "end": e["end"], "text": r["text"][e["start"]:e["end"]],
                         "entity": pid, "yago": real[pid], "mention_id": "w" + r["id"]})
        entities[pid] = {"eid": pid, "label": titles[pid], "wikipedia_page_id": pid, "yago": real[pid]}
    doc = "wikilinks_unseen_test"
    for sub, obj in (("paragraphs", {"doc_id": doc, "title": "WikilinksNED Unseen-Mentions, test split", "source": "naist-nlp/unseen on Hugging Face",
                                     "paragraphs": paragraphs}),
                     ("gold", {"doc_id": doc, "annotator": "Onoe and Durrett 2020 (automatic, from Wikipedia links); page ids resolved to titles and YAGO ids by build.py",
                               "scope": {"paragraphs": [p["id"] for p in paragraphs]}, "entities": list(entities.values()),
                               "mentions": mentions, "facts": []})):
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
        json.dump(obj, open(os.path.join(HERE, sub, doc + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    n = len(mentions)
    print(n, "mentions;", len(entities), "entities;", sum(t is None for t in titles.values()), "page ids without a page;",
          sum(m["yago"] is not None for m in mentions), "mentions with a YAGO id")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
