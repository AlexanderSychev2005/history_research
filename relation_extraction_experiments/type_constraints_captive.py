import json
from pathlib import Path

DOCS = ["nechaeva2017", "alford2026"]
THRESHOLD = 0.99
RUN = "gold__nli_captive_typed_t099"

output_dir = Path("eval/pred") / RUN
output_dir.mkdir(parents=True, exist_ok=True)

for doc in DOCS:
    source = json.loads(
        Path(f"outputs/{doc}_relations_gold__nli_captive.json")
        .read_text(encoding="utf-8")
    )

    original = json.loads(
        Path(f"eval/pred/gold__nli_captive/{doc}.json")
        .read_text(encoding="utf-8")
    )

    ner = json.loads(
        Path(f"outputs/{doc}_ner_gold.json")
        .read_text(encoding="utf-8")
    )

    labels = {}
    for paragraph in ner["paragraphs"]:
        for mention in paragraph["mentions"]:
            labels.setdefault(mention["text"], set()).add(mention["label"])

    eid = {entity["label"]: entity["eid"] for entity in original["entities"]}

    facts = []
    seen = set()

    for decision in source["decisions"]:
        allowed = []

        for option in decision["options"]:
            if option["rel"] == "yago:partOf":
                subj_types = labels.get(option["subj"], set())
                obj_types = labels.get(option["obj"], set())

                valid = (
                    ("location" in subj_types and "location" in obj_types)
                    or
                    ("organization" in subj_types and "organization" in obj_types)
                )

                if not valid:
                    continue

            allowed.append(option)

        if not allowed:
            continue

        best = max(allowed, key=lambda option: option["p"])

        if best["p"] < THRESHOLD:
            continue

        key = (
            decision["paragraph_id"],
            best["subj"],
            best["rel"],
            best["obj"],
        )

        if key in seen:
            continue

        seen.add(key)

        facts.append({
            "para": decision["paragraph_id"],
            "subj": eid[best["subj"]],
            "rel": best["rel"],
            "obj": eid[best["obj"]],
        })

    prediction = {
        "doc_id": doc,
        "entities": original["entities"],
        "mentions": original["mentions"],
        "facts": facts,
    }

    (output_dir / f"{doc}.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"{doc}: {len(facts)} predicted facts")

print(f"Saved to {output_dir}")
