
import json
import re
from pathlib import Path

from gliner import GLiNER

INPUT = Path("texts/alford2026.json")
OUTPUT = Path("outputs/alford2026_ner.json")

LABELS = ["person", "location", "organization", "event"]
MODEL_NAME = "urchade/gliner_medium-v2.1"

# Conservative character limit to avoid token truncation.
MAX_CHARS = 500
OVERLAP_CHARS = 100

def make_chunks(text):
    """Return (start_offset, chunk_text) pairs."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + MAX_CHARS, len(text))

        if end < len(text):
            # Prefer ending at a sentence or word boundary.
            segment = text[start:end]
            boundaries = [
                m.end()
                for m in re.finditer(r"(?<=[.!?])\s+|\s+", segment)
                if m.end() >= MAX_CHARS // 2
            ]
            if boundaries:
                end = start + boundaries[-1]

        chunks.append((start, text[start:end]))

        if end == len(text):
            break

        next_start = max(start + 1, end - OVERLAP_CHARS)

        # Move backward to a word boundary when possible.
        while next_start > start + 1 and not text[next_start - 1].isspace():
            next_start -= 1

        start = next_start

    return chunks


def extract_mentions(model, text):
    mentions_by_span = {}

    for chunk_start, chunk in make_chunks(text):
        predictions = model.predict_entities(
            chunk,
            LABELS,
            threshold=0.4,
        )

        for entity in predictions:
            start = chunk_start + entity["start"]
            end = chunk_start + entity["end"]

            # Verify offsets against the original paragraph.
            if text[start:end] != entity["text"]:
                continue

            mention = {
                "text": entity["text"],
                "start": start,
                "end": end,
                "label": entity["label"],
                "score": float(entity["score"]),
            }

            key = (start, end, entity["label"])

            # Keep the highest-confidence duplicate.
            if (
                key not in mentions_by_span
                or mention["score"] > mentions_by_span[key]["score"]
            ):
                mentions_by_span[key] = mention

    return sorted(
        mentions_by_span.values(),
        key=lambda m: (m["start"], m["end"]),
    )


def main():
    model = GLiNER.from_pretrained(MODEL_NAME)

    with INPUT.open(encoding="utf-8") as f:
        document = json.load(f)

    results = []

    for paragraph in document["paragraphs"]:
        mentions = extract_mentions(model, paragraph["text"])

        results.append({
            "paragraph_id": paragraph["id"],
            "page": paragraph["page"],
            "section": paragraph["section"],
            "text": paragraph["text"],
            "mentions": mentions,
        })

        print(
            f"{paragraph['id']}: "
            f"{len(mentions)} mentions"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "doc_id": document["doc_id"],
                "paragraphs": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nProcessed {len(results)} paragraphs")
    print(f"Total mentions: {sum(len(p['mentions']) for p in results)}")
    print(f"Saved NER results to {OUTPUT}")


if __name__ == "__main__":
    main()
