"""NER with the Jev API: candidate spans by rule (capitalised runs, "A of B"), Jev decides whether each is a
name, its kind, and whether "A of B" is one name. Writes outputs/<doc>_ner_jev.json.

  uv run python pipeline/ner_jev.py --doc nechaeva2017 [--paragraphs p004 p013]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from relations_jev import API, sentences  # noqa: E402

MAX_QUESTIONS = 8  # per request (TypeSafe limit)
WORD = r"[^\W\d_](?:[\w-]|[‘’ʻ](?=\w))*‘?"
RUN = re.compile(rf"{WORD}(?:\s+(?:of\s+(?:the\s+)?)?{WORD})*")
KINDS = {
    "person": "a person",
    "location": "a place, region, city, country or state",
    "organization": "a people, family, dynasty, army or institution such as a church",
    "written work": "a book, history, chronicle or other written work",
    "event": "a battle, rebellion, treaty or other named event",
}


def candidates(sentence):
    """(start, end, text, parts) within the sentence; parts = (A, B) offsets for an 'A of B' run."""
    out = []
    for m in RUN.finditer(sentence):
        words = [w for w in re.finditer(WORD, m.group()) if w.group() not in ("of", "the")]
        if not any(w.group()[0].isupper() for w in words):
            continue
        # a run is capitalised words only; split it where a lowercase word sits (re.finditer allowed any letter)
        cur = []
        for w in words + [None]:
            if w is not None and w.group()[0].isupper() and w.group() not in ("AD", "BC", "CE"):
                cur.append(w)
                continue
            if cur:
                s, e = m.start() + cur[0].start(), m.start() + cur[-1].end()
                text = re.sub(r"’s$", "", sentence[s:e])
                e = s + len(text)
                parts = None
                of = re.search(r"\s+of\s+(?:the\s+)?", text)
                if of:
                    parts = ((s, s + of.start()), (s + of.end(), e))
                out.append((s, e, text, parts))
            cur = []
    return out


def ask(state, questions, key, model):
    for attempt in range(6):
        r = requests.post(API, json={"model": model, "state": state, "questions": questions},
                          headers={"Authorization": f"Bearer {key}"}, timeout=120)
        if r.status_code in (429, 529):
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        r.raise_for_status()
        return r.json()["answers"]
    raise RuntimeError("Jev API: still rate limited after 6 attempts")


def name_questions(qid, name):
    # a noul "is it a name?" plus a choice of kind; the Jev guide prefers a noul to a "none" option
    return {
        f"{qid}_name": {"type": "noul",
                        "instructions": f"In `sentence`, is “{name}” the proper name of a specific person, place, "
                                        "group, work or event?",
                        "criteria": {"true": f"“{name}” names one particular person, place, group, work or event.",
                                     "false": f"“{name}” is an adjective (Sasanian forces), a title, a common word at "
                                              "the start of the sentence, or part of a citation."}},
        f"{qid}_kind": {"type": "choice", "instructions": f"What does “{name}” name in `sentence`?",
                        "criteria": {k.replace(" ", "_"): v for k, v in KINDS.items()}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--paragraphs", nargs="*")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default="jev-latest")
    a = ap.parse_args()
    load_dotenv(dotenv_path=".env")
    key = os.getenv("JEV_API_KEY") or ap.error("JEV_API_KEY is missing from .env")

    doc = json.loads(Path(f"texts/{a.doc}.json").read_text(encoding="utf-8"))
    results = []
    for p in doc["paragraphs"]:
        if a.paragraphs and p["id"] not in a.paragraphs:
            continue
        text, mentions = p["text"], []
        for s0, s1 in sentences(text):
            sent = text[s0:s1]
            # every span that may be asked about: the run itself and, for 'A of B', both parts
            spans, questions = {}, {}
            for s, e, name, parts in candidates(sent):
                for i, (x0, x1) in enumerate([(s, e)] + list(parts or [])):
                    spans[f"c{len(spans)}"] = (x0, x1, sent[x0:x1], parts is not None and i == 0, parts and i > 0)
            for qid, (x0, x1, name, has_of, _) in spans.items():
                questions.update(name_questions(qid, name))
                if has_of:
                    a_name, b_name = re.split(r"\s+of\s+(?:the\s+)?", name, maxsplit=1)
                    questions[f"{qid}_split"] = {
                        "type": "choice", "instructions": f"In `sentence`, how should “{name}” be read?",
                        "criteria": {"one_name": f"“{name}” is a single name, like John of Ephesus or Church of the East.",
                                     "name_and_origin": f"“{a_name}” is a name and “{b_name}” is a separate place "
                                                        f"or group {a_name} belongs to, like Damascius of Syria."}}
            answers = {}
            items = list(questions.items())
            for i in range(0, len(items), MAX_QUESTIONS):
                answers.update(ask({"sentence": sent}, dict(items[i:i + MAX_QUESTIONS]), key, a.model))

            ids = list(spans)
            skip = set()
            for qid in ids:  # an 'A of B' run is followed in `spans` by its two parts
                x0, x1, name, has_of, _ = spans[qid]
                if not has_of:
                    continue
                k = ids.index(qid)
                split = answers[f"{qid}_split"]["probabilities"]["name_and_origin"] >= 0.5
                skip.update([qid] if split else ids[k + 1:k + 3])
            for qid in ids:
                if qid in skip:
                    continue
                x0, x1, name, *_ = spans[qid]
                p_name = answers[f"{qid}_name"]["noul"]
                if p_name < a.threshold:
                    continue
                mentions.append({"text": name, "start": s0 + x0, "end": s0 + x1,
                                 "label": answers[f"{qid}_kind"]["choice"].replace("_", " "), "score": p_name})
        mentions.sort(key=lambda m: m["start"])
        results.append({"paragraph_id": p["id"], "page": p["page"], "section": p["section"], "text": text,
                        "mentions": mentions})
        print(f"{p['id']}: {len(mentions)} mentions", flush=True)

    out = Path(f"outputs/{a.doc}_ner_jev.json")
    out.write_text(json.dumps({"doc_id": a.doc, "model": a.model, "paragraphs": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
