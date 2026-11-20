"""Scores entity-linking predictions on dataset/zeshel_realms/:   python dataset/zeshel_realms/score.py <pred.jsonl>

<pred.jsonl>: one line per mention, {"mention_id": ..., "kb_id": <predicted kb id or null>}; extra fields are ignored,
a mention without a line counts as unlinked.

normalized accuracy    on the mentions whose gold entity is among the 64 candidates (1,000 of 1,200): the metric of
                       ZESHEL and of LELA (arXiv 2601.05192, Table 3, column "Realms")
unnormalized accuracy  on all 1,200 mentions
Both are given per mention category of ZESHEL as well, with a 95% interval (normal approximation).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold", "zeshel_forgotten_realms_test.json")


def acc(rows):
    n, k = len(rows), sum(r for r in rows)
    if not n:
        return "–"
    p = k / n
    return f"{100 * p:.2f} ± {196 * math.sqrt(p * (1 - p) / n):.1f} ({k} / {n})"


def main(path):
    sys.stdout.reconfigure(encoding="utf-8")
    gold = json.load(open(GOLD, encoding="utf-8"))["mentions"]
    pred = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            pred[r["mention_id"]] = r.get("kb_id")
    missing = sum(m["mention_id"] not in pred for m in gold)
    right = lambda m: pred.get(m["mention_id"]) == m["kb_id"]
    print(f"# {os.path.basename(path)}\n")
    if missing:
        print(f"{missing} gold mentions have no prediction and count as wrong.\n")
    print("| mentions | normalized accuracy | unnormalized accuracy |\n|---|---|---|")
    groups = [("all", gold)] + [(c, [m for m in gold if m["category"] == c]) for c in sorted({m["category"] for m in gold})]
    for name, ms in groups:
        print(f"| {name} | {acc([right(m) for m in ms if m['gold_in_candidates']])} | {acc([right(m) for m in ms])} |")


if __name__ == "__main__":
    main(sys.argv[1])
