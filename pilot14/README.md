# pilot14: the pilot's entity pipeline on the colleagues' 14 paragraphs

The pilot (`pilot/`) developed mention detection and linking to YAGO 4.5 on two paragraphs. Here the same approaches
are run on the 14-paragraph gold of `eval/gold` (Max's, 159 mentions), 12 of whose paragraphs were not used for any
design choice. YAGO ids for its 83 entities were added in `entity_map.json` before any system output was looked at.

| file | what |
|---|---|
| `entity_map.json` | YAGO id (or null = no entity in YAGO) for each entity of `eval/gold`, with a note on doubtful cases |
| `build.py` | `eval/gold` + `entity_map.json` -> `gold.json` (mentions with YAGO ids) and `texts/` (the 14 paragraphs) |
| `names.py` | stage 1 of the hybrid and the full-LLM baseline: the LLM lists the names and the entity it thinks each is (clean prompt, no test examples) |
| `hybrid.py` | stage 2: YAGO shortlist from the text and from the LLM's guess, then the LLM chooses (or "none") |
| `lela_to_pred.py` | LELA's output (`lela_in/paragraphs.jsonl`, `configs/`) -> `pred/` |
| `score.py` | mention and link scores for all 14 paragraphs, for p004 (the pilot's, tuned on) and for the 12 new ones |

`pred/<run>/` are the predictions: `lela_ext_t03_first` (GLiNER, fuzzy 20, first candidate), `llm_120b_clean_r1` (full LLM),
`hyb_120b-20b_clean_r1` (hybrid: names and guesses by gpt-oss-120b, the choice by gpt-oss-20b). `llm_raw/` holds the raw
LLM answers; `pred_raw/` LELA's raw output.

## Results on the 12 new paragraphs (124 gold mentions; F1 %, overlap spans, strict in brackets)
| run | mention F1 | link P / R | link F1 |
|---|---|---|---|
| hybrid | 92 (87) | 56 / 51 | **53** (47) |
| full LLM (exact title lookup in YAGO) | 92 (87) | 64 / 34 | 44 (38) |
| LELA, first candidate | 82 (77) | 11 / 20 | 14 (13) |

On all 14 paragraphs: hybrid link F1 62 (58), full LLM 48 (44), LELA 19 (18). On the two paragraphs the pilot was tuned on the
hybrid reaches 87; the 12 new ones are the honest estimate.

## Where the hybrid loses (first run, 12 new paragraphs)
Of 78 names with a YAGO entity, 39 are linked correctly and for 39 the right entity is not in the shortlist (mostly Armenian
names the LLM left without a guess); the choice itself never failed when the entity was in the list. Of 36 names without a
YAGO entity, 15 got a wrong link. 10 gold names are not found as exact spans.

## Caveats
One run per variant; the choice was made by gpt-oss-20b with low reasoning effort because gpt-oss-120b's daily token limit was
used up; gpt-oss answers differ between runs (the pilot's hybrid ranged 0.79 to 0.88). The gold is a draft by one annotator
(Claude) and the YAGO ids were chosen by the same; a historian has to review them, above all the 42 "no entity in YAGO" decisions.

## Run it
```bash
YAGO_JSONL=<LELA>/data/yago/yago-entities.jsonl python pilot14/build.py
GROQ_API_KEY=... python pilot14/names.py ask 120b 1          # then: YAGO_JSONL=... python pilot14/names.py build 120b 1
GROQ_API_KEY=... YAGO_JSONL=... python pilot14/hybrid.py 120b 20b 1     # needs numpy and rapidfuzz
python pilot14/score.py hyb_120b-20b_clean_r1 llm_120b_clean_r1 lela_ext_t03_first [--overlap]
```
LELA: `python pilot/run_lela.py --config pilot14/configs/lela_ext_t03_first.json --input pilot14/lela_in/paragraphs.jsonl --output pilot14/pred_raw/lela_ext_t03_first.jsonl`
from the LELA folder, then `lela_to_pred.py`. Never commit the API key.
