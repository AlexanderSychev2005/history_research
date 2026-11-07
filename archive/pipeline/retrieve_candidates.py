
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import requests


API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": "HistoryResearch/0.1 (academic entity linking research)",
    "Accept": "application/json",
}

LABELS = {"person", "location", "organization", "event"}
REQUEST_DELAY = 2.0
MAX_RETRIES = 5
RESULTS_PER_QUERY = 10


def load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_unicode(text):
    """Normalize Unicode punctuation and spacing."""
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "‘": "'",
        "’": "'",
        "ʼ": "'",
        "ʻ": "'",
        "`": "'",
        "–": "-",
        "—": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return " ".join(text.split())


def strip_diacritics(text):
    """Remove combining diacritics for fallback queries."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char)
    )


def remove_titles(text):
    """Remove common personal titles at the start of a mention."""
    return re.sub(
        r"^(?:(?:the|a)\s+)?"
        r"(?:(?:Sasanian|Sassanian|Armenian|Persian|Roman|Byzantine)\s+)?"
        r"(?:King|Emperor|Queen|Prince|Princess|Saint|St\.?|"
        r"General|Bishop|Patriarch)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def generate_queries(name, label):
    """
    Generate general-purpose search variants.

    These variants improve retrieval but do not guarantee
    that Wikidata contains the correct entity.
    """
    queries = []

    def add(query):
        query = " ".join(query.split()).strip()
        if query and query not in queries:
            queries.append(query)

    original = name.strip()
    normalized = normalize_unicode(original)

    add(original)
    add(normalized)

    if label == "person":
        add(remove_titles(normalized))

    for query in list(queries):
        add(strip_diacritics(query))

    # Fallback: remove possessive endings.
    for query in list(queries):
        add(re.sub(r"(?:'s|’s)$", "", query, flags=re.I))

    # Avoid overly broad first-name-only searches.
    # Shorter queries are used only for names ending in Roman numerals.
    for query in list(queries):
        if re.search(r"\s+[IVXLCDM]+$", query):
            add(re.sub(r"\s+[IVXLCDM]+$", "", query))

    return queries


def search_wikidata(query, session):
    """Call Wikidata API with rate-limit handling."""

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                API,
                params={
                    "action": "wbsearchentities",
                    "search": query,
                    "language": "en",
                    "format": "json",
                    "limit": RESULTS_PER_QUERY,
                },
                timeout=30,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After", "")

                delay = (
                    max(2, int(retry_after))
                    if retry_after.isdigit()
                    else min(90, 5 * (2 ** attempt))
                )

                print(
                    f"    HTTP {response.status_code}: "
                    f"waiting {delay}s"
                )
                time.sleep(delay)
                continue

            response.raise_for_status()

            return response.json().get("search", [])

        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise

            delay = min(90, 5 * (2 ** attempt))
            print(f"    Request error: {exc}; retry in {delay}s")
            time.sleep(delay)

    raise RuntimeError(
        f"Wikidata request failed after {MAX_RETRIES} attempts"
    )


def collect_mentions(data, label_filter=None):
    """
    Collect unique (mention, label) pairs.

    All occurrences are preserved so that later stages
    can perform context-specific disambiguation.
    """
    entities = {}

    for paragraph in data["paragraphs"]:
        paragraph_id = paragraph["paragraph_id"]
        context = paragraph["text"]

        for mention in paragraph["mentions"]:
            label = mention["label"]
            name = mention["text"].strip()

            if label not in LABELS:
                continue

            if label_filter and label != label_filter:
                continue

            if not name:
                continue

            key = (name.casefold(), label)

            occurrence = {
                "paragraph_id": paragraph_id,
                "page": paragraph.get("page"),
                "start": mention["start"],
                "end": mention["end"],
                "context": context,
                "ner_score": mention.get("score"),
            }

            if key not in entities:
                entities[key] = {
                    "mention": name,
                    "label": label,
                    "paragraph_id": paragraph_id,
                    "context": context,
                    "occurrences": [],
                }

            entities[key]["occurrences"].append(occurrence)

    return list(entities.values())


def retrieve_candidates(entity, cache, session):
    name = entity["mention"]
    label = entity["label"]

    queries = generate_queries(name, label)

    candidates = {}
    failed_queries = []

    for query in queries:
        cache_key = f"en::{query}"

        if cache_key not in cache:
            try:
                cache[cache_key] = search_wikidata(query, session)
                time.sleep(REQUEST_DELAY)

            except (requests.RequestException, RuntimeError) as exc:
                print(f"    Search failed: {query}: {exc}")
                failed_queries.append(query)
                continue

        for item in cache[cache_key]:
            qid = item.get("id")

            if not qid:
                continue

            if qid not in candidates:
                candidates[qid] = {
                    "id": qid,
                    "label": item.get("label", ""),
                    "description": item.get("description", ""),
                    "url": item.get(
                        "concepturi",
                        f"https://www.wikidata.org/wiki/{qid}",
                    ),
                    "matched_queries": [],
                }

            if query not in candidates[qid]["matched_queries"]:
                candidates[qid]["matched_queries"].append(query)

    if failed_queries:
        status = "partial" if candidates else "retrieval_error"
    elif not candidates:
        status = "no_candidates"
    else:
        status = "complete"

    return {
        **entity,
        "search_queries": queries,
        "candidates": list(candidates.values()),
        "retrieval_status": status,
        "failed_queries": failed_queries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve Wikidata candidates for GLiNER mentions"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input GLiNER JSON",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output candidate JSON",
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/wikidata_cache.json"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum unique mentions; 0 = all",
    )

    parser.add_argument(
        "--label",
        choices=sorted(LABELS),
        default=None,
        help="Filter by entity type",
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    data = load_json(args.input)

    if data is None:
        raise FileNotFoundError(args.input)

    doc_id = data["doc_id"]

    output = args.output or Path(
        f"outputs/{doc_id}_candidates.json"
    )

    cache = load_json(args.cache, default={})

    entities = collect_mentions(data, args.label)

    if args.limit > 0:
        entities = entities[:args.limit]

    print(f"Document: {doc_id}")
    print(f"Unique mentions selected: {len(entities)}")

    # Resume previously completed entities only when
    # the output belongs to the same document.
    existing = load_json(output, default={})
    completed = {}

    if existing.get("doc_id") == doc_id:
        for entity in existing.get("entities", []):
            key = (
                entity["mention"].casefold(),
                entity["label"],
            )
            if entity.get("retrieval_status") in {
                "complete",
                "no_candidates",
            }:
                completed[key] = entity

    results = []

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for index, entity in enumerate(entities, start=1):
            key = (
                entity["mention"].casefold(),
                entity["label"],
            )

            print(
                f"[{index}/{len(entities)}] "
                f"{entity['mention']} ({entity['label']})"
            )

            if key in completed:
                result = {
                    **entity,
                    **{
                        field: completed[key][field]
                        for field in (
                            "search_queries",
                            "candidates",
                            "retrieval_status",
                            "failed_queries",
                        )
                        if field in completed[key]
                    },
                }
                print("  Reused previous result")
            else:
                result = retrieve_candidates(
                    entity, cache, session
                )
                print(
                    f"  Candidates: {len(result['candidates'])}; "
                    f"status: {result['retrieval_status']}"
                )

            results.append(result)

            save_json(args.cache, cache)
            save_json(
                output,
                {
                    "doc_id": doc_id,
                    "entities": results,
                },
            )

    print(f"\nFinished: {len(results)} mentions")
    print(f"Saved candidates to: {output}")
    print(f"Saved Wikidata cache to: {args.cache}")


if __name__ == "__main__":
    main()
