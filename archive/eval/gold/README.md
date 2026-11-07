# Gold

Built by `eval/make_gold.py` (the annotation tables are in that script, the JSON files are generated).

| doc | paragraphs | mentions | facts |
|---|---|---|---|
| alford2026 | p001 p003 p004 p005 p013 p018 p024 p025 p032 | 90 | 31 |
| nechaeva2017 | p004 p013 p014 p018 p019 | 69 | 26 |

**Status:** draft by one annotator (Claude), annotated from the text before any relation output was seen. It needs a
historian's review before the numbers are reported. It replaces the earlier 5-paragraph Alford draft (in git history),
whose paragraphs p001–p005 are re-annotated here under the conventions below.

## Conventions

* **Mentions** follow `pilot/make_pilot.py`: proper names of persons, places, peoples/states, families, works and named
  events, name only (no titles, no "of X" epithets, no adjectives such as "Sasanian forces" or "Persian court").
  Citations in parentheses ("Mart. Grig. 8") are not mentions. One named phrase is kept as an event:
  "Armenian rebellion".
* **Entities** have no QIDs yet. `evaluate.py` matches them to predictions by label/alias, and every surface form in
  the text is an alias.
* **Facts** use the relations in `schema/history_relations.json` (YAGO 4.6 names plus three `ex:` extensions). A fact
  is annotated only when one sentence names both participants and states the relation or directly implies it.
  Pronouns ("he", "they") do not count, so the gold matches what a sentence-level extractor can find.
  * "X of Place" (Damascius of Syria, T‘at‘ik of Basean) and "a place inside a larger place" -> `schema:location`.
  * Being moved to, imprisoned in, or taken prisoner by a place or state -> `ex:captiveIn`.
  * Brother, nephew and other kin that are not parent or spouse -> `ex:relativeOf`. This is symmetric, so the name
    that comes first in the text is the subject (the same holds for `yago:neighbor` and `schema:spouse`).
  * A work and its author -> `yago:creator(work, author)`.
  * Claims are annotated as the text states them, even when the author is sceptical (Nerseh as a Gorgān captive,
    p013). Attribution and certainty are not annotated yet.
