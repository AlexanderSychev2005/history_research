"""The hybrid on a random sample of WikilinksNED Unseen-Mentions (dataset/wikilinks_unseen/), mentions given.

    uv run --no-project --env-file .env --with numpy --with rapidfuzz python hybrid/wikilinks.py [n]      (default n = 100)

The hybrid of hybrid.py without its first step (the mentions are given): the LLM (NAMES_MODEL) guesses the Wikipedia title of
the marked mention, the shortlist is built as in the hybrid (YAGO titles by similarity + Wikipedia search, for the guess and the
mention) and CHOOSER_MODEL picks one entry or none. The knowledge base is YAGO 4.5, so a mention whose Wikipedia page is not in
YAGO (25% of the benchmark: mostly common nouns such as "floor") can never be right. Writes hybrid/wikilinks_pred.jsonl.
"""
import json
import os
import random
import sys

import hybrid as H
from zeshel import GUESS_PROMPT

DOC = "wikilinks_unseen_test"
D = os.path.join(H.HERE, "..", "dataset", "wikilinks_unseen")


def main(n):
    H.CHOOSER_MODEL = os.environ.get("CHOOSER_MODEL", H.CHOOSER_MODEL)    # another Groq model when the daily limit of this one is spent
    paragraphs = {p["id"]: p["text"] for p in json.load(open(os.path.join(D, "paragraphs", DOC + ".json"), encoding="utf-8"))["paragraphs"]}
    sample = json.load(open(os.path.join(D, "gold", DOC + ".json"), encoding="utf-8"))["mentions"]
    random.Random(0).shuffle(sample)
    sample = sample[:n]
    guesses = H.load_json(os.path.join(H.CACHE, "wikilinks_guesses.json"))
    rows = {DOC: []}
    for i, m in enumerate(sample, 1):
        text = paragraphs[m["para"]]
        if m["mention_id"] not in guesses:
            ctx = f"{text[:m['start']]}[[{text[m['start']:m['end']]}]]{text[m['end']:]}"
            answer = H.post({"model": H.NAMES_MODEL, "temperature": 0.3, "max_completion_tokens": 1500, "reasoning_effort": "low",
                             "messages": [{"role": "user", "content": GUESS_PROMPT.format(context=ctx).replace("encyclopedia", "Wikipedia")}]})
            guesses[m["mention_id"]] = (answer.strip().splitlines() or [""])[0].strip(' ."\'')
            H.save_json(guesses, os.path.join(H.CACHE, "wikilinks_guesses.json"))
        guess = guesses[m["mention_id"]]
        guess = None if guess.lower() in ("", "unknown", "null", "none") else guess
        rows[DOC].append({"para": m["para"], "text": m["text"], "start": m["start"], "end": m["end"], "llm_entity": guess})
        if i % 20 == 0:
            print("guesses", i, "/", len(sample), flush=True)
    yago = H.Yago(os.environ["YAGO_JSONL"])
    decisions = H.candidates(yago, rows)
    chosen = H.choose(yago, {(DOC, p): t for p, t in paragraphs.items()}, decisions)
    out, hit = [], {"link": 0, "in_list": 0, "in_yago": 0, "link_in_yago": 0, "none": 0}
    for m, r in zip(sample, rows[DOC]):
        k = (DOC, m["para"], m["text"], r["llm_entity"], None)
        pick, cands = chosen[k], [yago.ids[j] for j in decisions[k][0]]
        out.append({"mention_id": m["mention_id"], "yago": pick, "gold": m["yago"], "candidates": cands})
        hit["link"] += pick == m["yago"] and pick is not None
        hit["in_yago"] += m["yago"] is not None
        hit["link_in_yago"] += pick == m["yago"] and pick is not None
        hit["in_list"] += m["yago"] in cands
        hit["none"] += pick is None
    with open(os.path.join(H.HERE, "wikilinks_pred.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in out)
    k, y = len(out), hit["in_yago"]
    ci = lambda h, t: f"{100 * h / t:5.1f} ± {196 * ((h / t) * (1 - h / t) / t) ** 0.5:.1f}  ({h} / {t})"
    print(f"\n{k} random mentions; {y} of them have a YAGO id")
    print("accuracy, all mentions             ", ci(hit["link"], k))
    print("accuracy, mentions with a YAGO id  ", ci(hit["link_in_yago"], y))
    print("gold in the shortlist (of those)   ", ci(hit["in_list"], y))
    print("left unlinked (none)               ", hit["none"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
