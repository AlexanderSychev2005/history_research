"""Scores entity-linking predictions on dataset/wikilinks_unseen/:   python dataset/wikilinks_unseen/score.py <pred.jsonl>

<pred.jsonl>: one line per mention, {"mention_id": ..., "yago": <predicted YAGO id or null>}; extra fields are ignored.
Only the mentions that have a line are scored (a sample is fine); a null prediction counts as wrong.

accuracy, all           mentions with a prediction whose predicted id equals the gold id
accuracy, with YAGO id  the same over the mentions whose gold page is an entity of YAGO 4.5 (a YAGO-based system cannot get
                        the other 25%, mostly common nouns); 95% interval, normal approximation
"""
import json
import math
import os
import sys

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold", "wikilinks_unseen_test.json")


def acc(hits):
    n = len(hits)
    p = sum(hits) / n if n else 0.0
    return f"{100 * p:.1f} ± {196 * math.sqrt(p * (1 - p) / n):.1f} ({sum(hits)} / {n})" if n else "–"


if __name__ == "__main__":
    gold = {m["mention_id"]: m for m in json.load(open(GOLD, encoding="utf-8"))["mentions"]}
    pred = {}
    for line in open(sys.argv[1], encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            pred[r["mention_id"]] = r.get("yago")
    ok = {i: gold[i]["yago"] is not None and p == gold[i]["yago"] for i, p in pred.items()}
    print("| mentions | accuracy |\n|---|---|")
    print(f"| all | {acc(list(ok.values()))} |")
    print(f"| gold has a YAGO id | {acc([ok[i] for i in ok if gold[i]['yago'] is not None])} |")
