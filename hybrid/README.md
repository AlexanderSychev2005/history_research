# Hybrid: entity linking to YAGO 4.5 with an LLM and a YAGO shortlist

Run on the 14 paragraphs of [`../dataset/`](../dataset/README.md) and scored against its gold (mentions and YAGO ids).

## How it works
1. **Names (LLM).** `gpt-oss-120b` reads a paragraph and lists the names, each with the entity it thinks the name is
   (a Wikipedia-style title, or null). Every whole-word occurrence of a listed text becomes a mention.
2. **Shortlist (no LLM).** For every name, at most 30 entries of YAGO 4.5, from two sources, each searched with two strings
   (the LLM's guess and the name as written in the text):
   * the 10 YAGO titles most similar to the string (rapidfuzz `WRatio` over all 5.8 million titles);
   * the 6 best results of a full-text search of English Wikipedia for the string, kept if they are entities of YAGO (a YAGO 4.5
     id is the Wikipedia title with `_`).

   A name the LLM gave no entity for (mostly Armenian and Persian names) first gets, from the LLM, the English spelling it is
   usually written with ("Siwnik‘" gives "Siwnik"), and that spelling is searched in place of the guess. Repeats are removed.
3. **Choice (LLM).** `gpt-oss-20b` sees the paragraph with the name marked, the first guess and the shortlist with the
   descriptions, and picks one entry or "none of the listed candidates". A "none" leaves the mention unlinked.

## Result
159 gold mentions (117 with a YAGO id, 42 without). A link counts only if the span and the YAGO id are both right (strict: the
span is the gold span; overlap: any overlap). One run of each method.

| method | spans | mention P / R / F1 | link P / R / F1 |
|---|---|---|---|
| LELA: GLiNER, fuzzy search, first candidate | strict | 0.73 / 0.88 / 0.80 | 0.15 / 0.24 / 0.18 |
| LELA: GLiNER, fuzzy search, first candidate | overlap | 0.76 / 0.92 / 0.84 | 0.16 / 0.26 / 0.19 |
| LELA + LLM: GLiNER, fuzzy search, `gpt-oss-20b` chooses | strict | 0.73 / 0.88 / 0.80 | 0.49 / 0.41 / 0.45 |
| LELA + LLM: GLiNER, fuzzy search, `gpt-oss-20b` chooses | overlap | 0.76 / 0.92 / 0.84 | 0.53 / 0.44 / 0.48 |
| **Hybrid** | strict | 0.91 / 0.95 / 0.93 | 0.69 / 0.72 / **0.70** |
| **Hybrid** | overlap | 0.93 / 0.97 / 0.95 | 0.72 / 0.75 / **0.74** |

* **LELA** ([dig-team/LELA](https://github.com/dig-team/LELA), commit `0202b81`, run with `lela/run.py`): GLiNER finds the names,
  a fuzzy search over the YAGO titles gives 20 candidates, and either the first candidate is taken or an LLM picks one (the
  same `gpt-oss-20b` as the hybrid's choice step). Both variants share the same names, so their mention scores are equal.
  The first-candidate row was not rerun for this table.
* The shortlist is where the hybrid gains: it holds the right entry for 89 of the 111 matched mentions with an id (strict), LELA's
  for 49 of 104.
* The hybrid made 122 links, 84 of them right (strict). When the right entity was in the shortlist the choice was right in 84 of
  89 cases. Of the 40 mentions without a YAGO id, 24 were left unlinked and 16 got a wrong link.

## What goes wrong
* **Not in the shortlist** (22 of 111 mentions, 7 distinct names): mostly names that the gold links to a neighbouring entry
  (Persia, Persians → Sasanian Empire; Romans → Byzantine Empire; Gorgān → Hyrcania) and the names Grigor, Mar Grigor and Širak.
* **In the shortlist but not chosen** (5): Awarayr (the battle instead of the plain), Callinicum (the battle instead of Raqqa),
  Romans (Roman Empire instead of Byzantine Empire), Church of the East (the Assyrian Church of the East), Łazar (none).

## Files
| file | content |
|---|---|
| `hybrid.py` | the whole pipeline: names, shortlist, choice; writes `pred/` |
| `score.py` | mention and link scores of `pred/` against `dataset/gold` |
| `pred/<doc>.jsonl` | predictions: `para, text, start, end, llm_entity, entity_id, candidates` |
| `cache/` | the LLM's raw answers and the Wikipedia search results (git-ignored); a rerun reuses them |

## Run it
Needs the YAGO 4.5 entity dump (<https://yago-knowledge.org/data/yago4.5/yago-entities.jsonl.zip>, 150 MB, 680 MB unpacked)
and a Groq API key. Copy `.env.example` to `.env` (git-ignored) and fill in `GROQ_API_KEY` and `YAGO_JSONL`, and, to make the
Wikipedia search fast, `WIKIPEDIA_CONTACT` (an e-mail or URL that Wikimedia may use to reach you; without it the script waits
6.5 s between requests). Then from the repository root:
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
