"""The hybrid's choice step on a random sample of ZESHEL, Forgotten Realms (dataset/zeshel_realms/), mentions given.

    uv run --no-project --env-file .env --with numpy --with rapidfuzz python hybrid/zeshel.py [n]      (default n = 100)

Only the part of the hybrid that does not depend on YAGO and Wikipedia is used, because the knowledge base here is a fan wiki:
  1. the LLM (NAMES_MODEL) guesses the title of the article the marked mention refers to
  2. shortlist = the 10 KB titles most similar to the guess, the 10 most similar to the mention, and the first 10 of the 64
     BM25 candidates that ship with ZESHEL (no repeats, at most 30)
  3. the LLM (CHOOSER_MODEL) picks one entry; there is no "none", as in LELA's ZESHEL setup
Mentions are taken in a fixed random order from those whose gold is among the 64 candidates (LELA's normalized accuracy) and
the answers are cached, so a rerun after a rate limit continues; the score covers the mentions done so far.
"""
import json
import os
import random
import re
import sys

import numpy as np
from rapidfuzz import fuzz, process

import hybrid as H

D = os.path.join(H.HERE, "..", "dataset", "zeshel_realms")
CACHE = os.path.join(H.CACHE, "zeshel")
WINDOW, BM25_TOP, MAX_SHORTLIST = 80, 10, 30

GUESS_PROMPT = """Paragraph from an encyclopedia page, with one mention marked by [[ ]]:
{context}

Give the title of the encyclopedia article that the marked mention refers to, as that article would be titled.
Answer with the title only, or with the word unknown."""


def load(name):
    return json.load(open(os.path.join(D, name), encoding="utf-8"))


def context(text, start, end):
    """WINDOW words on each side of the mention, which is marked"""
    before, after = text[:start].split()[-WINDOW:], text[end:].split()[:WINDOW]
    return f"{' '.join(before)} [[{text[start:end]}]] {' '.join(after)}"


def cached(name, key, ask):
    path = os.path.join(CACHE, name + ".json")
    done = H.load_json(path)
    if key not in done:
        done[key] = ask()
        H.save_json(done, path)
    return done[key]


def main(n):
    gold = load("gold/zeshel_forgotten_realms_test.json")["mentions"]
    candidates = load("candidates/zeshel_forgotten_realms_test.json")
    texts = {p["id"]: p["text"] for p in load("paragraphs/zeshel_forgotten_realms_test.json")["paragraphs"]}
    kb = [json.loads(line) for line in open(os.path.join(D, "kb", "forgotten_realms.jsonl"), encoding="utf-8")]
    ids, titles, desc = [r["id"] for r in kb], [r["title"] for r in kb], {r["id"]: r["description"] for r in kb}
    pool = [m for m in gold if m["gold_in_candidates"]]
    random.Random(0).shuffle(pool)
    sample = pool[:n]

    def similar(q):                                                  # the 10 most similar KB titles, ties in KB order
        s = process.cdist([q], titles, scorer=fuzz.WRatio, dtype=np.uint8, workers=-1)[0]
        near = np.nonzero(s >= np.partition(s, -10)[-10])[0]
        return [ids[j] for j in near[np.lexsort((near, -s[near].astype(np.int16)))][:10]]

    rows = []
    for i, m in enumerate(sample, 1):
        ctx = context(texts[m["para"]], m["start"], m["end"])
        guess = cached("guesses", m["mention_id"], lambda: (H.post({
            "model": H.NAMES_MODEL, "temperature": 0.3, "max_completion_tokens": 1500, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": GUESS_PROMPT.format(context=ctx)}]}).strip().splitlines() or [""])[0].strip(' ."\''))
        guess = None if guess.lower() in ("", "unknown", "null", "none") else guess
        shortlist = []
        for k in (similar(guess) if guess else []) + similar(m["text"]) + candidates[m["mention_id"]][:BM25_TOP]:
            if k not in shortlist:
                shortlist.append(k)
        shortlist = shortlist[:MAX_SHORTLIST]
        lines = "\n".join(f"{j}. {titles[ids.index(k)]} ({desc[k][:700]})" for j, k in enumerate(shortlist, 1))
        hint = f'A first reader suggested the entity "{guess}" (it may be wrong or spelled differently).\n' if guess else ""
        question = (f"Paragraph from an encyclopedia page, with one mention marked by [[ ]]:\n{ctx}\n\n"
                    f"Mention: \"{m['text']}\"\n{hint}Which entry of the knowledge base is the entity this mention refers to "
                    f"in the paragraph?\n{lines}\n\nAnswer with the number only.")

        def ask():
            for _ in range(3):
                found = re.search(r"\d+", H.post({"model": H.CHOOSER_MODEL, "temperature": 0.3, "max_completion_tokens": 1500,
                                                  "reasoning_effort": "low", "messages": [{"role": "user", "content": question}]}))
                if found:
                    return int(found.group())
            return 1                                                 # nothing came back: the first entry
        pick = cached("choices", json.dumps([m["mention_id"], shortlist]), ask)
        pick = shortlist[pick - 1] if 1 <= pick <= len(shortlist) else shortlist[0]
        rows.append({"mention_id": m["mention_id"], "kb_id": pick, "gold": m["kb_id"], "in_list": m["kb_id"] in shortlist,
                     "bm25_first": candidates[m["mention_id"]][0], "category": m["category"]})
        if i % 10 == 0:
            print("done", i, "/", len(sample), flush=True)

    with open(os.path.join(H.HERE, "zeshel_pred.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    k = len(rows)
    print(f"\n{k} mentions (random, gold among the 64 BM25 candidates)")
    for name, hit in (("hybrid choice step", sum(r["kb_id"] == r["gold"] for r in rows)),
                      ("gold in the shortlist (ceiling)", sum(r["in_list"] for r in rows)),
                      ("BM25 first candidate", sum(r["bm25_first"] == r["gold"] for r in rows))):
        p = hit / k
        print(f"{name:34} {100 * p:5.1f} ± {196 * (p * (1 - p) / k) ** 0.5:.1f}  ({hit} / {k})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
