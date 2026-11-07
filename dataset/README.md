# Dataset: 14 paragraphs with one gold set for the whole pipeline

The shared test set for all parts of the project. Fourteen paragraphs from two late-antique history papers, annotated for
every stage: **mentions** (names), **entity linking to YAGO 4.5** and **relations**. Everything is in one place so that
every method is scored against the same gold.

| paper | paragraphs | mentions | entities (with a YAGO id) | facts |
|---|---|---|---|---|
| Alford 2026, *Borderland Lords and their Centrally-Held Sons* | p001 p003 p004 p005 p013 p018 p024 p025 p032 | 90 | 45 (30) | 28 |
| Nechaeva 2017, *Seven Hellenes and One Christian in the Endless Peace Treaty of 532* | p004 p013 p014 p018 p019 | 69 | 38 (25) | 26 |
| total | 14 | 159 (117 with a YAGO id, 42 without) | 83 (55) | 54 |

**Status: draft.** Annotated by one annotator (Claude) and not yet reviewed by a historian; the YAGO ids were chosen from the
YAGO 4.5 dump by the same. Treat the numbers computed on it accordingly. Paragraph ids are those of `archive/prepare/prepare_texts.py`.

## Files
| file | content |
|---|---|
| `articles/*.pdf` | the two source papers (as given to the project; the `source` field of `paragraphs/` names the file) |
| `paragraphs/<doc>.json` | `{doc_id, title, source, paragraphs: [{id, text, section, page}]}`, the canonical text (offsets refer to it) |
| `gold/<doc>.json` | `{doc_id, annotator, scope, entities, mentions, facts}`, see below |
| `relations.json` | the relation vocabulary: 29 relations (YAGO 4.6 plus three `ex:` extensions) with domain, range and a one-line description, of which the 23 that have an entity as object are the candidates for facts (the gold uses 11 of them); copied from `archive/schema/history_relations.json` |
| `build.py` | provenance: how the files were assembled from `archive/eval/gold`, `archive/pilot14/entity_map.json` and the cleaned text (run once) |

`gold/<doc>.json` follows the format of `eval/evaluate.py` and adds the YAGO fields:
```
entities: [{eid, label, type, aliases, yago, note}]      yago = "yago:Title" or null; note explains doubtful decisions
mentions: [{para, start, end, text, entity, yago}]       offsets are 0-based characters in the paragraph, end exclusive
facts:    [{para, subj, rel, obj}]                       subj and obj are eids; rel is a name of relations.json
```

## Conventions
* **Mentions:** proper names of persons, places, peoples/states, families, works and named events, name only (no titles, no
  "of X" epithets, no adjectives such as "Sasanian forces"). Citations in parentheses are not mentions. One entity has many
  mentions: every written form is an alias ("Gorgān" and "Hyrcania" are one entity).
* **Entity linking:** `yago` is the id of the entity in **YAGO 4.5** (`yago-entities.jsonl`). `null` means YAGO has no matching
  entity: a system should leave such a mention unlinked, and a predicted link counts as an error. Where only a close
  entry exists the entity's `note` says "approx" (Khuzestan → the modern province, Širak and Siwnik' likewise, Awarayr → the
  plain, Callinicum → Raqqa, Armenia for historical Armenia). Homonyms are separated (Priscian of Lydia is not the grammarian;
  YAGO's Zabergan is a Kutrigur chieftain, not the Persian envoy, so the envoy is `null`).
* **Relations:** the 23 relations of `relations.json` whose object is an entity (YAGO 4.6 plus `ex:relativeOf`,
  `ex:captiveIn`, `ex:opposed`). A fact is annotated only when **one sentence names both participants** and states the
  relation or directly implies it; pronouns do not count. "X of Place" and "a place inside a larger place" are
  `schema:location`; being moved to, imprisoned in or taken prisoner by a place or state is `ex:captiveIn`; kin other than parent
  or spouse is `ex:relativeOf` (symmetric: the name that comes first in the text is the subject, likewise for
  `yago:neighbor` and `schema:spouse`); a work and its author is `yago:creator(work, author)`. Claims are annotated as the text
  states them; attribution and certainty are not annotated yet.

## Scoring
```bash
# mentions and relations (and entity matching by label/alias), as for every other run of the project
python archive/eval/evaluate.py score --texts dataset/paragraphs --gold dataset/gold --pred <pred dir> ... --span overlap
```
Entity linking to YAGO: compare each predicted mention's YAGO id with the gold mention's `yago` (a mention with `yago: null`
is correct only if left unlinked). `archive/pilot14/score.py` does this for a different prediction format and is easy to adapt.

Report **paragraph p004 of both papers separately**: the earlier pilot (`archive/pilot/`) was developed on them, so numbers on
the other 12 paragraphs are the unbiased ones.

## Known limits
One annotator; no inter-annotator agreement; 54 facts give wide confidence intervals (about ±12 points); the 42 mentions without
a YAGO id and the "approx" links are the decisions most in need of a historian's check; gold facts are sentence-level only.

This set replaces `archive/eval/gold` as the shared gold (the mentions and facts are the same). **The JSON in this folder is the
source from now on**; `archive/` is frozen. `archive/eval/evaluate.py` is stdlib-only and can be used from here as shown above.
