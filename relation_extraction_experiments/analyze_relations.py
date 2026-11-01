import json
from collections import defaultdict
from pathlib import Path

DOCS = ["nechaeva2017", "alford2026"]

RUNS = {
    "NLI typed 0.95": "gold__nli_typed_t095",
    "NLI captive": "gold__nli_captive",
    "NLI templates": "gold__nli_templates",
    "Jev API": "gold__jev-api",
}


def load_facts(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    entities = {
        entity["eid"]: entity["label"].strip().casefold()
        for entity in data["entities"]
    }

    facts = set()

    for fact in data["facts"]:
        facts.add((
            fact["para"],
            entities[fact["subj"]],
            fact["rel"],
            entities[fact["obj"]],
        ))

    return facts


gold = set()
for doc in DOCS:
    for fact in load_facts(f"eval/gold/{doc}.json"):
        gold.add((doc, *fact))

for name, run in RUNS.items():
    predicted = set()

    for doc in DOCS:
        path = f"eval/pred/{run}/{doc}.json"
        for fact in load_facts(path):
            predicted.add((doc, *fact))

    relations = sorted(
        {fact[3] for fact in gold | predicted}
    )

    print(f"\n=== {name} ===")
    print(f"{'Relation':25} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")

    for relation in relations:
        g = {f for f in gold if f[3] == relation}
        p = {f for f in predicted if f[3] == relation}

        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        print(
            f"{relation:25} {tp:4} {fp:4} {fn:4} "
            f"{precision:7.1%} {recall:7.1%} {f1:7.1%}"
        )
