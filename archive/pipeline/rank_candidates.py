
import argparse
import json
import re
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi


TOP_K = 5


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("mamikonean", "mamikonian")
    text = text.replace("yazdgerd", "yazdegerd")

    text = re.sub(r"[‘’ʼ'`]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def tokenize(text):
    return normalize(text).split()


def rank_entity(entity):
    candidates = entity.get("candidates", [])
    queries = entity.get(
        "search_queries", [entity["mention"]]
    )

    if not candidates:
        return {
            **entity,
            "ranked_candidates": [],
        }

    documents = [
        f"{c.get('label', '')} {c.get('description', '')}"
        for c in candidates
    ]

    tokenized_documents = [
        tokenize(document) for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    scores_by_query = [
        bm25.get_scores(tokenize(query))
        for query in queries
    ]

    bm25_scores = [
        max(scores[index] for scores in scores_by_query)
        for index in range(len(candidates))
    ]

    max_bm25 = max(
        (max(0.0, float(score)) for score in bm25_scores),
        default=0.0,
    )

    ranked = []

    for candidate, bm25_score in zip(candidates, bm25_scores):
        fuzzy_score = max(
            fuzz.ratio(
                normalize(query),
                normalize(candidate.get("label", "")),
            ) / 100
            for query in queries
        )

        normalized_bm25 = (
            max(0.0, float(bm25_score)) / max_bm25
            if max_bm25 > 0
            else 0.0
        )

        combined_score = (
            0.6 * fuzzy_score
            + 0.4 * normalized_bm25
        )

        ranked.append({
            **candidate,
            "fuzzy_score": round(fuzzy_score, 4),
            "bm25_score": round(float(bm25_score), 4),
            "combined_score": round(combined_score, 4),
        })

    ranked.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    return {
        **entity,
        "ranked_candidates": ranked[:TOP_K],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", type=Path, required=True
    )
    parser.add_argument(
        "--output", type=Path, required=True
    )

    args = parser.parse_args()

    data = json.loads(
        args.input.read_text(encoding="utf-8")
    )

    results = []

    for entity in data["entities"]:
        result = rank_entity(entity)
        results.append(result)

        print(f"\nMENTION: {entity['mention']}")

        for index, candidate in enumerate(
            result["ranked_candidates"], start=1
        ):
            print(
                f"  {index}. {candidate['id']} | "
                f"{candidate['label']} | "
                f"score={candidate['combined_score']:.4f}"
            )

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    args.output.write_text(
        json.dumps(
            {
                "doc_id": data["doc_id"],
                "entities": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
