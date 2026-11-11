# Hybrid: entity linking to YAGO 4.5 with an LLM and a YAGO shortlist

Run on the 14 paragraphs of [`../dataset/`](../dataset/README.md) and scored against its gold (mentions and YAGO ids).

## How it works
1. **Names (LLM).** `gpt-oss-120b` reads a paragraph and lists the names, each with the entity it thinks the name is
   (a Wikipedia-style title, or null). Every whole-word occurrence of a listed text becomes a mention.
2. **Shortlist (YAGO, no LLM).** For every name, at most 20 entries of YAGO 4.5: the 10 titles most similar to the LLM's
   guess and the 10 most similar to the name as written in the text (rapidfuzz `WRatio` over all 5.8 million titles), without
   repeats. Without a guess only the 10 by the text.
3. **Choice (LLM).** `gpt-oss-20b` sees the paragraph with the name marked, the first guess and the shortlist with the
   descriptions, and picks one entry or "none of the listed candidates". A "none" leaves the mention unlinked.

## Result (one run)
159 gold mentions (117 with a YAGO id, 42 without), 166 predicted.

| spans | mention P / R / F1 | link P / R / F1 | gold id in shortlist |
|---|---|---|---|
| strict | 0.91 / 0.95 / 0.93 | 0.71 / 0.56 / **0.63** | 67 / 111 |
| overlap | 0.93 / 0.97 / 0.95 | 0.75 / 0.58 / **0.65** | 70 / 115 |

* A link counts only if the span and the YAGO id are both right (strict: the span is the gold span; overlap: any overlap).
* The hybrid made 91 links, 65 of them right (strict). When the right entity was in the shortlist the choice was right in 65 of
  67 cases (strict), so the limit is the shortlist: for 44 of 111 mentions with an id the right entry was not in it.
* Of the 40 mentions without a YAGO id, 28 were left unlinked and 12 got a wrong link.

## Files
| file | content |
|---|---|
| `hybrid.py` | the whole pipeline: names, shortlist, choice; writes `pred/` |
| `score.py` | mention and link scores of `pred/` against `dataset/gold` |
| `pred/<doc>.jsonl` | predictions: `para, text, start, end, llm_entity, entity_id, candidates` |
| `cache/` | the LLM's raw answers (git-ignored); a rerun reuses them |

## Run it
Needs the YAGO 4.5 entity dump (<https://yago-knowledge.org/data/yago4.5/yago-entities.jsonl.zip>, 150 MB, 680 MB unpacked)
and a Groq API key. Copy `.env.example` to `.env` (git-ignored) and fill in `GROQ_API_KEY` and `YAGO_JSONL`, then from the
repository root:
```bash
uv run --no-project --env-file .env --with numpy --with rapidfuzz python hybrid/hybrid.py
python hybrid/score.py
```
Delete `hybrid/cache/` to ask the LLM again. Groq's free tier answers 429 now and then; the script waits and continues, and
if the daily limit is reached a rerun continues from the cache.

## Caveats
One run: gpt-oss answers differ between runs even at low temperature, so a single run can move by several points. The choice
uses `gpt-oss-20b` with low reasoning effort to stay within Groq's token limit. The gold is a draft by one annotator and has
not been reviewed by a historian.
