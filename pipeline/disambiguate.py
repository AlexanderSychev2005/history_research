import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, APIStatusError, RateLimitError

MODEL = "qwen/qwen3.8-27b"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def get_occurrences(entity):
    occurrences = entity.get("occurrences")
    if occurrences:
        return occurrences
    return [{
        "paragraph_id": entity.get("paragraph_id"),
        "context": entity.get("context", ""),
        "start": None,
        "end": None,
    }]


def disambiguate_mention(client, entity, occurrence):
    candidates = entity.get("ranked_candidates", [])
    if not candidates:
        return {
            "entity_id": None,
            "status": "no_candidates",
            "reason": "No Wikidata candidates were retrieved; this is not a verified NIL.",
        }

    candidate_data = [
        {
            "id": candidate["id"],
            "label": candidate.get("label", ""),
            "description": candidate.get("description", ""),
        }
        for candidate in candidates
    ]

    # Keep prompts compact to reduce Qwen output-token consumption.
    context = occurrence.get("context", "")
    start = occurrence.get("start")
    end = occurrence.get("end")
    if isinstance(start, int) and isinstance(end, int):
        left = max(0, start - 300)
        right = min(len(context), end + 300)
        context = context[left:right]
    else:
        context = context[:650]

    prompt = (
        "Link the historical mention to one listed Wikidata candidate. "
        "Use the paragraph and historical context. "
        "If it is a generic word or no candidate fits, return null. "
        "Never invent an ID. Return only compact JSON with entity_id and reason "
        "(reason: maximum 12 words).\n"
        f"Mention: {entity['mention']}\n"
        f"Type: {entity.get('label', 'unknown')}\n"
        f"Context: {context}\n"
        f"Candidates: {json.dumps(candidate_data, ensure_ascii=False)}"
    )

    response = None
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Return a compact valid JSON object only. Do not explain outside JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=350 if attempt == 0 else 700,
            )
            break
        except RateLimitError as exc:
            if attempt == 5:
                raise
            match = re.search(r"try again in ([0-9.]+)s", str(exc), re.I)
            wait = max(5.0, float(match.group(1)) + 2) if match else 30.0
            print(f"  Rate limit: waiting {wait:.1f}s", flush=True)
            time.sleep(wait)
        except APIStatusError as exc:
            # Qwen may exhaust its completion budget before closing JSON.
            if exc.status_code == 400 and "json_validate_failed" in str(exc) and attempt < 5:
                wait = 10.0
                print(f"  Incomplete JSON: retrying with larger budget after {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            raise

    if response is None:
        raise RuntimeError("No response after retries")

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned an empty response")
    result = json.loads(content)
    entity_id = result.get("entity_id")
    if entity_id in ("NIL", "null", ""):
        entity_id = None
    allowed_ids = {candidate["id"] for candidate in candidates}
    if entity_id is not None and entity_id not in allowed_ids:
        raise ValueError(f"Model returned invalid candidate ID: {entity_id}")

    return {
        "entity_id": entity_id,
        "status": "linked" if entity_id else "nil",
        "reason": str(result.get("reason", "")),
    }


def decision_key(decision):
    return (decision.get("paragraph_id"), decision.get("start"), decision.get("end"))


def main():
    parser = argparse.ArgumentParser(description="Disambiguate ranked Wikidata candidates using Qwen on Groq")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    client = Groq(api_key=api_key)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    existing = {}
    if args.output.exists():
        previous_data = json.loads(args.output.read_text(encoding="utf-8"))
        if previous_data.get("doc_id") == data["doc_id"]:
            for item in previous_data.get("entities", []):
                key = (item["mention"].casefold(), item.get("label"))
                existing[key] = item

    results = []
    for index, entity in enumerate(data["entities"], start=1):
        print(f"[{index}/{len(data['entities'])}] {entity['mention']} ({entity.get('label', '')})", flush=True)
        key = (entity["mention"].casefold(), entity.get("label"))
        old_decisions = {
            decision_key(d): d
            for d in existing.get(key, {}).get("decisions", [])
            if d.get("status") in {"linked", "nil", "no_candidates"}
        }
        decisions = []
        for occurrence in get_occurrences(entity):
            occurrence_key = decision_key(occurrence)
            if occurrence_key in old_decisions:
                decision = old_decisions[occurrence_key]
                print("  Reused saved decision", flush=True)
            else:
                try:
                    prediction = disambiguate_mention(client, entity, occurrence)
                except Exception as exc:
                    prediction = {"entity_id": None, "status": "error", "reason": str(exc)}
                decision = {
                    "paragraph_id": occurrence.get("paragraph_id"),
                    "start": occurrence.get("start"),
                    "end": occurrence.get("end"),
                    **prediction,
                }
                print(f"  -> {decision['entity_id']} ({decision['status']})", flush=True)
                if decision["status"] in {"linked", "nil"}:
                    time.sleep(1)

            decisions.append(decision)
            save_json(args.output, {
                "doc_id": data["doc_id"],
                "entities": results + [{**entity, "decisions": decisions}],
            })
        results.append({**entity, "decisions": decisions})

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
