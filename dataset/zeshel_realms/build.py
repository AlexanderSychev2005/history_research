"""Builds dataset/zeshel_realms/ from the ZESHEL download: the Forgotten Realms test world, in the format of dataset/.

    python dataset/zeshel_realms/build.py <zeshel folder>                         this folder (Forgotten Realms, test)
    python dataset/zeshel_realms/build.py <zeshel folder> --split val --world elder_scrolls --out <folder>

<zeshel folder> holds documents/, mentions/ (github.com/lajanugen/zeshel, the data link) and tfidf_candidates/ (the
"top-64 candidates" link of the same README, unpacked into it). Writes paragraphs/, gold/, candidates/ and kb/ here.
Standard library only.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = 1200         # characters of the long description: the start of the page, cut at a sentence end


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def description(text):
    """LELA's entity description: the first three sentences of the page (ZESHEL pages start with the title)"""
    return " ".join(re.split(r"(?<=[.!?])\s+", " ".join(text.split()))[:3])


def long_description(text):
    """the start of the page up to LONG characters, cut after the last full sentence"""
    text = " ".join(text.split())
    if len(text) <= LONG:
        return text
    cut = text[:LONG]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > 0 else cut


def main(src, WORLD, SPLIT, out):
    DOC_ID = f"zeshel_{WORLD}_{SPLIT}"
    docs = {d["document_id"]: d for d in load_jsonl(os.path.join(src, "documents", WORLD + ".json"))}
    mentions = [m for m in load_jsonl(os.path.join(src, "mentions", SPLIT + ".json")) if m["corpus"] == WORLD]
    cands = {c["mention_id"]: c["tfidf_candidates"]
             for c in load_jsonl(os.path.join(src, "tfidf_candidates", SPLIT + ".json"))}

    # paragraphs: every page that holds a mention, its whitespace tokens joined by one space (ZESHEL's tokenization)
    paras, starts = {}, {}
    for pid in sorted({m["context_document_id"] for m in mentions}):
        words = docs[pid]["text"].split()
        offsets, pos = [], 0
        for w in words:
            offsets.append(pos)
            pos += len(w) + 1
        paras[pid], starts[pid] = " ".join(words), offsets

    gold_mentions, entities = [], {}
    for m in sorted(mentions, key=lambda m: (m["context_document_id"], m["start_index"], m["mention_id"])):
        pid, text = m["context_document_id"], paras[m["context_document_id"]]
        start = starts[pid][m["start_index"]]
        end = starts[pid][m["end_index"]] + len(text[starts[pid][m["end_index"]]:].split(" ", 1)[0])
        assert text[start:end] == m["text"], (m["mention_id"], text[start:end], m["text"])
        gid = m["label_document_id"]
        e = entities.setdefault(gid, {"eid": gid, "label": docs[gid]["title"], "type": None, "aliases": [],
                                      "kb_id": gid, "note": ""})
        if m["text"] != e["label"] and m["text"] not in e["aliases"]:
            e["aliases"].append(m["text"])
        gold_mentions.append({"para": pid, "start": start, "end": end, "text": m["text"], "entity": gid, "kb_id": gid,
                              "mention_id": m["mention_id"], "category": m["category"],
                              "gold_in_candidates": gid in cands[m["mention_id"]]})

    dump = lambda obj, *path: json.dump(obj, open(os.path.join(out, *path), "w", encoding="utf-8"),
                                        ensure_ascii=False, indent=1)
    for d in ("paragraphs", "gold", "candidates", "kb"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    dump({"doc_id": DOC_ID, "title": f"ZESHEL {SPLIT}, world {WORLD}",
          "source": f"ZESHEL (Logeswaran et al., ACL 2019), documents/{WORLD}.json, CC BY-SA",
          "paragraphs": [{"id": pid, "title": docs[pid]["title"], "text": paras[pid]} for pid in paras]},
         "paragraphs", DOC_ID + ".json")
    dump({"doc_id": DOC_ID, "annotator": f"ZESHEL gold (hyperlinks of the {WORLD.replace('_', ' ').title()} wiki), unchanged",
          "scope": {"paragraphs": list(paras), "kb": f"kb/{WORLD}.jsonl"},
          "entities": sorted(entities.values(), key=lambda e: e["label"]), "mentions": gold_mentions, "facts": []},
         "gold", DOC_ID + ".json")
    dump({m["mention_id"]: cands[m["mention_id"]] for m in gold_mentions}, "candidates", DOC_ID + ".json")
    with open(os.path.join(out, "kb", WORLD + ".jsonl"), "w", encoding="utf-8") as f:
        for i, d in sorted(docs.items(), key=lambda kv: kv[1]["title"]):
            f.write(json.dumps({"id": i, "title": d["title"], "description": description(d["text"]),
                                "description_long": long_description(d["text"])},
                               ensure_ascii=False) + "\n")
    inside = sum(m["gold_in_candidates"] for m in gold_mentions)
    print(f"{len(gold_mentions)} mentions in {len(paras)} pages, {len(entities)} gold entities, "
          f"gold among the 64 candidates for {inside}; kb: {len(docs)} entities")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="the unpacked zeshel folder")
    ap.add_argument("--world", default="forgotten_realms")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    main(a.src, a.world, a.split, a.out)
