# Hybrid: entity linking to YAGO 4.5 with an LLM and a YAGO shortlist

Tested on the 14 paragraphs of [`../dataset/`](../dataset/README.md), scored against its gold (mentions and YAGO ids).

## How it works
1. **Names (LLM).** `gpt-oss-120b` reads a paragraph and lists the names, each with the entity it thinks the name is
   (a Wikipedia-style title, or null). Every whole-word occurrence of a listed text becomes a mention.
2. **Shortlist (YAGO, no LLM).** For every name, at most 20 entries of YAGO 4.5: the 10 titles most similar to the name as
   written in the text, the 10 most similar to the LLM's guess (rapidfuzz `WRatio` over all 5.8 million titles), and the
   entry whose title equals the guess.
3. **Choice (LLM).** `gpt-oss-20b` sees the paragraph with the name marked, the first guess and the shortlist with the
   descriptions, and picks one entry or "none of the listed candidates". A "none" leaves the mention unlinked.

## Result (14 paragraphs, 159 gold mentions, 117 with a YAGO id; one run)
Overlapping spans count as a match (strict boundaries in brackets, `results/score_strict.md`).

| run | mention F1 | link P / R | link F1 |
|---|---|---|---|
| hybrid (`pred/hybrid_120b-20b_r1`) | 93 (90) | 65 / 60 | **62** (58) |
| LLM only: the guess looked up as a YAGO title (`pred/names_120b_r1`) | 93 (90) | 67 / 38 | 48 (44) |

In 70 of the 115 matched mentions with an id the right entry was in the shortlist, and in exactly those 70 the choice was right.
Of the 40 gold mentions without a YAGO id the hybrid left 24 unlinked and gave the other 16 a wrong link.

## Which parts of the shortlist are needed
`python hybrid/hybrid.py shortlists names_120b_r1` (no LLM call): how often the right entity is in the list, over the 109 gold
mentions with an id that the names stage found as exact spans (`results/shortlists.md`).

| shortlist | right entity in the list | mean length |
|---|---|---|
| exact title only (the guess as it is) | 40 / 109 (37%) | 0.4 |
| search by the text only | 54 / 109 (50%) | 10.0 |
| search by the guess only | 51 / 109 (47%) | 6.3 |
| text + guess | 65 / 109 (60%) | 12.1 |
| exact + guess + text (the hybrid) | 65 / 109 (60%) | 12.1 |

* **Text and guess are both needed:** each alone finds the entity for about half of the names, together for 60%, so they
  find partly different entities.
* **The exact match adds nothing:** the lists with and without it contain the right entity in exactly the same cases, an
  exact title is already among the 10 most similar to the guess. It can be dropped (`parts` in `Yago.shortlist`); the choice
  step has not been rerun without it, so the link F1 of that variant is not measured.
* **The rest (40%)** is the ceiling of this design: whatever the LLM chooses, an entity that is not in the list cannot be linked.

## Files
| file | content |
|---|---|
| `hybrid.py` | `names`, `choose` and `shortlists` (the three commands above) |
| `score.py` | mention and link scores of a run against `dataset/gold` |
| `pred/<run>/<doc>.jsonl` | predictions: `para, text, start, end, entity_id, llm_entity` (and `candidates` for the hybrid) |
| `results/` | the tables above |

## Run it
Needs a Groq API key for the LLM steps (kept in the environment, never in the repository), the YAGO 4.5 entity dump
(`https://yago-knowledge.org/data/yago4.5/yago-entities.jsonl.zip`, 150 MB, 680 MB unpacked), `numpy` and `rapidfuzz`.
```bash
GROQ_API_KEY=... python hybrid/hybrid.py names 120b 1
GROQ_API_KEY=... YAGO_JSONL=<path>/yago-entities.jsonl python hybrid/hybrid.py choose names_120b_r1 20b
python hybrid/score.py hybrid_120b-20b_r1 names_120b_r1 --overlap
```

## Caveats
One run: gpt-oss answers differ between runs even at low temperature, so a single run can move by several points. The choice
uses `gpt-oss-20b` with low reasoning effort to stay within Groq's token limit. The gold is a draft by one annotator and has
not been reviewed by a historian.
