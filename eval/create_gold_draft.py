import json
from pathlib import Path

data = json.loads(Path("texts/alford2026.json").read_text(encoding="utf-8"))

annotations = {
    "p001": [
        ("Armenian rebellion", "event"),
        ("Yazdgerd II", "person"),
        ("Sasanian", "organization"),
    ],
    "p002": [
        ("Armenia", "location"),
        ("Sasanian Empire", "organization"),
        ("Huns", "organization"),
    ],
    "p003": [
        ("Armenian rebellion", "event"),
        ("Sasanian forces", "organization"),
        ("Awarayr", "location"),
        ("Yazdgerd II", "person"),
        ("Zoroastrianism", None),
        ("Vardan Mamikonean", "person"),
        ("Armenian Church", "organization"),
    ],
    "p004": [
        ("Łazar P‘arpec‘i", "person"),
        ("Gorgān", "location"),
        ("Hyrcania", "location"),
        ("Alborz", "location"),
        ("Caspian Sea", "location"),
        ("Khuzestan", "location"),
        ("Łazar", "person"),
        ("Vahan Mamikonean", "person"),
        ("Sasanians", "organization"),
        ("Vahan", "person"),
        ("Vałaršapat", "location"),
        ("Eǰmiacin", "location"),
    ],
    "p005": [
        ("Ełišē", "person"),
        ("Gorgān", "location"),
        ("Łazar", "person"),
        ("Robert Thomson", "person"),
        ("Akinean", "person"),
        ("Kiwlēsērean", "person"),
        ("Philo", "person"),
        ("John of Ephesus", "person"),
    ],
}

entities = {}
mentions = []
entity_ids = {}

for p in data["paragraphs"][:5]:
    pid, text = p["id"], p["text"]
    for name, category in annotations.get(pid, []):
        if category is None:
            continue
        if name not in entity_ids:
            eid = f"E{len(entity_ids) + 1}"
            entity_ids[name] = eid
            entities[eid] = {"eid": eid, "label": name}
        start = 0
        while (pos := text.find(name, start)) != -1:
            mentions.append({
                "para": pid,
                "start": pos,
                "end": pos + len(name),
                "text": name,
                "entity": entity_ids[name]
            })
            start = pos + len(name)

gold = {
    "doc_id": "alford2026",
    "annotator": "draft_requires_review",
    "scope": {"paragraphs": [p["id"] for p in data["paragraphs"][:5]]},
    "entities": list(entities.values()),
    "mentions": mentions,
    "facts": []
}

path = Path("eval/gold/alford2026.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Draft: {len(entities)} entities, {len(mentions)} mentions")
print(f"Saved: {path}")