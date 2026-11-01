import json
from pathlib import Path

THRESHOLDS = [0.5, 0.7, 0.9, 0.95, 0.99]
DOCS = ["nechaeva2017", "alford2026"]

for threshold in THRESHOLDS:
    run = f"gold__nli_templates_t{str(threshold).replace('.', '')}"
    folder = Path("eval/pred") / run
    folder.mkdir(parents=True, exist_ok=True)

    for doc in DOCS:
        source = json.loads(
            Path(f"outputs/{doc}_relations_gold__nli_templates.json")
            .read_text(encoding="utf-8")
        )
        original = json.loads(
            Path(f"eval/pred/gold__nli_templates/{doc}.json")
            .read_text(encoding="utf-8")
        )

        eid = {e["label"]: e["eid"] for e in original["entities"]}
        facts = []
        seen = set()

        for decision in source["decisions"]:
            if not decision["options"]:
                continue

            best = max(decision["options"], key=lambda o: o["p"])

            if best["p"] < threshold:
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

        (folder / f"{doc}.json").write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"{run}: {doc}, {len(facts)} facts")
