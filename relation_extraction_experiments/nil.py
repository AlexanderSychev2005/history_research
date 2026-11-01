
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pipeline.relations_jev import options

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("mps")
model = model.to(device)

print(f"Using device: {device}")
sentence = "Pīrān-Gušnasp belonged to the house of Mihrān."

person = {"text": "Pīrān-Gušnasp", "label": "person"}
family = {"text": "house of Mihrān", "label": "organization"}

# Завантажуємо схему відношень тіммейтів
relations = json.loads(
    Path("schema/history_relations.json").read_text(encoding="utf-8")
)

# Генеруємо всі допустимі відношення для цієї пари
candidates = options(person, family, relations)

entailment_id = next(
    i for i, label in model.config.id2label.items()
    if label.lower() == "entailment"
)

results = []

for relation, subject, obj, hypothesis in candidates:
    if relation == "schema:memberOf":
        hypothesis = f"{subject['text']} is a member of {obj['text']}."
    inputs = tokenizer(
    sentence,
    hypothesis,
    return_tensors="pt",
    truncation=True
).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    score = probabilities[entailment_id].item()
    results.append((relation, subject["text"], obj["text"], score, hypothesis))

# Сортуємо від найвищої ймовірності до найнижчої
results.sort(key=lambda x: x[3], reverse=True)

for relation, subject, obj, score, hypothesis in results:
    print(f"{relation}({subject}, {obj}): {score:.4f}")
    print(f"  Hypothesis: {hypothesis}")
