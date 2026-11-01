
import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pipeline.relations_jev import sentences, options, CLASS


MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="nechaeva2017")
    parser.add_argument("--ner", default="gold")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    entailment_id = next(
        i for i, label in model.config.id2label.items()
        if label.lower() == "entailment"
    )

    relations = json.loads(
        Path("schema/history_relations.json").read_text(encoding="utf-8")
    )
    relations = {
        name: info for name, info in relations.items()
        if any(t.startswith("schema:") for t in info["range"])
    }

    ner = json.loads(
        Path(f"outputs/{args.doc}_ner_{args.ner}.json").read_text(encoding="utf-8")
    )

    decisions = []
    tasks = []

    for paragraph in ner["paragraphs"]:
        text = paragraph["text"]

        for start, end in sentences(text):
            seen = set()
            mentions = []

            for mention in paragraph["mentions"]:
                if (
                    start <= mention["start"]
                    and mention["end"] <= end
                    and mention["label"] in CLASS
                    and mention["text"] not in seen
                ):
                    seen.add(mention["text"])
                    mentions.append(mention)

            for a, b in combinations(mentions, 2):
                if (
                    set(a["text"].split()) <= set(b["text"].split())
                    or set(b["text"].split()) <= set(a["text"].split())
                ):
                    continue

                candidates = options(a, b, relations)
                if not candidates:
                    continue

                task = {
                    "paragraph_id": paragraph["paragraph_id"],
                    "sentence": [start, end],
                    "text": text[start:end],
                    "a": a["text"],
                    "b": b["text"],
                    "candidates": candidates,
                    "scores": [],
                }
                tasks.append(task)

    print(f"Pairs: {len(tasks)}", flush=True)

       # NLI: entity-specific hypothesis templates
    templates = {
        "yago:partOf": "{s} is part of {o}.",
        "yago:follows": "{s} succeeded {o} in a position or role.",
        "schema:location": "{s} is located in {o}.",
        "ex:relativeOf": "{s} and {o} are relatives.",
        "ex:captiveIn": "{s} was held captive in {o}.",
        "schema:memberOf": "{s} is a member of {o}.",
    }

    requests = []

    for task_index, task in enumerate(tasks):
        for relation, subject, obj, hypothesis in task["candidates"]:
            if relation in templates:
                hypothesis = templates[relation].format(
                    s=subject["text"],
                    o=obj["text"]
                )

            requests.append((task_index, task["text"], hypothesis)) 

    print(f"Hypotheses: {len(requests)}", flush=True)

    for start in range(0, len(requests), args.batch_size):
        batch = requests[start:start + args.batch_size]

        inputs = tokenizer(
            [item[1] for item in batch],
            [item[2] for item in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            scores = probabilities[:, entailment_id].cpu().tolist()

        for (task_index, _, _), score in zip(batch, scores):
            tasks[task_index]["scores"].append(score)

        processed = min(start + args.batch_size, len(requests))
        if processed % 160 == 0 or processed == len(requests):
            print(f"Processed {processed}/{len(requests)}", flush=True)

    for task in tasks:
        candidates = task["candidates"]
        scores = task["scores"]

        best_index = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_index]

        decision = {
            "paragraph_id": task["paragraph_id"],
            "sentence": task["sentence"],
            "a": task["a"],
            "b": task["b"],
            "options": [
                {
                    "rel": relation,
                    "subj": subject["text"],
                    "obj": obj["text"],
                    "p": round(score, 4),
                }
                for (relation, subject, obj, _), score
                in zip(candidates, scores)
            ],
            "p_none": None,
        }

        if best_score >= args.threshold:
            relation, subject, obj, _ = candidates[best_index]
            decision["fact"] = {
                "subj": subject["text"],
                "rel": relation,
                "obj": obj["text"],
                "p": round(best_score, 4),
            }

        decisions.append(decision)

        run = f"{args.ner}__nli_captive"
    output_path = Path(f"outputs/{args.doc}_relations_{run}.json")

    output_path.write_text(
        json.dumps(
            {
                "doc_id": args.doc,
                "run": run,
                "model": MODEL_NAME,
                "threshold": args.threshold,
                "decisions": decisions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    names = sorted({
        m["text"]
        for p in ner["paragraphs"]
        for m in p["mentions"]
    })
    eid = {name: f"m{i}" for i, name in enumerate(names)}

    prediction = {
        "doc_id": args.doc,
        "entities": [
            {"eid": eid[name], "label": name}
            for name in names
        ],
        "mentions": [
            {
                "para": p["paragraph_id"],
                "start": m["start"],
                "end": m["end"],
                "text": m["text"],
                "entity": eid[m["text"]],
            }
            for p in ner["paragraphs"]
            for m in p["mentions"]
        ],
        "facts": [
            {
                "para": d["paragraph_id"],
                "subj": eid[d["fact"]["subj"]],
                "rel": d["fact"]["rel"],
                "obj": eid[d["fact"]["obj"]],
            }
            for d in decisions
            if "fact" in d
        ],
    }

    prediction_path = Path(f"eval/pred/{run}/{args.doc}.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Decisions: {len(decisions)}")
    print(f"Predicted facts: {len(prediction['facts'])}")
    print(f"Saved: {output_path}")
    print(f"Saved: {prediction_path}")


if __name__ == "__main__":
    main()
