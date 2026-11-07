# Results by stage

Best approaches for each stage of the pipeline, side by side. F1 in percent; spans matched by overlap (strict in
brackets). Gold: `eval/gold`, 14 paragraphs from two papers (159 mentions, 54 facts). Max = avojarot, Lili = LiliAlania,
Alex = AlexanderSychev2005. Per-stage details live in the experiment folders; this file only collects them.

## 1. Mentions (names)

| # | Model | Who | Precision / Recall | F1 |
|---|---|---|---|---|
| 1 | Jev NER (candidate rule + Jev API) | Max | 96.9 / 97.5 | **97.2** (94.0) |
| 2 | Full LLM `gpt-oss-120b` (1 run) | Alex | 90 / 97 | **93** (90) |
| 3 | GLiNER-clean | Max | 79.8 / 86.8 | 83.1 (73.5) |
| 4 | LELA: GLiNER with extra types, threshold 0.3 | Alex | 76 / 93 | 83 (80) |
| 5 | GLiNER (original) | Max | 66.2 / 89.9 | 76.3 (67.7) |

Sources: rows 1, 3, 5 `experiments/RESULTS.md`; rows 2, 4 `pilot14/` (`python pilot14/score.py ... --overlap`).

## 2. Entity linking to YAGO (only done in `pilot/` and `pilot14/`)

| # | Model | Who | Precision / Recall (all 14) | F1, all 14 | F1, 12 new |
|---|---|---|---|---|---|
| 1 | Hybrid: names by `gpt-oss-120b`, YAGO shortlist, choice by `gpt-oss-20b` | Alex | 65 / 60 | **62** (58) | **53** (47) |
| 2 | Full LLM `gpt-oss-120b`, exact title lookup in YAGO | Alex | 67 / 38 | 48 (44) | 44 (38) |
| 3 | LELA: GLiNER + `fuzzy` + first candidate | Alex | 15 / 26 | 19 (18) | 14 (13) |

"12 new" excludes the two p004 paragraphs on which the pipeline was developed (the pilot's hybrid scores 87 there).
Source: `pilot14/` (README, `results/`).

## 3. Relations on gold mentions (only the relation stage is measured)

| # | Model | Who | Threshold | Precision / Recall | F1 |
|---|---|---|---|---|---|
| 1 | Jev API guided | Max | 0.8 (tuned on this gold) | 71.2 / 77.8 | **74.3** |
| 2 | Jev API | Max | 0.9 (tuned) | 71.7 / 61.1 | 66.0 |
| 3 | Jev API | Max | 0.5 | 52.5 / 77.8 | 62.7 |
| 4 | Constrained LLM, per sentence (`gpt-oss-20b`) | Alex | none | 66.7 / 59.3 | **62.7** |
| 5 | Jev API guided | Max | 0.5 | 47.5 / 87.0 | 61.4 |
| 6 | Constrained LLM, per paragraph (`gpt-oss-20b`, 2 runs) | Alex | none | 75–78 / 50–52 | 60.0 and 62.2 |
| 7 | OpenJev 9B (local) | Max | 0.7 (tuned) | 52.1 / 70.4 | 59.8 |
| 8 | NLI (DeBERTa) | Lili | 0.99 (tuned) | 54.8 / 31.5 | 40.0 |

Sources: Max's rows `experiments/RESULTS.md`, `experiments/04_jev_relations`, `experiments/05_jev_api_vs_open`; rows 4, 6
`experiments/08_constrained_relations/RESULTS.md`; row 8 `experiments/07_nli_relations` scored with
`python eval/evaluate.py score --texts texts --gold eval/gold --span overlap --pred eval/pred/gold__nli_captive_typed_t099`.

## 4. Relations, end-to-end (names are predicted too)

| # | Pipeline | Who | Threshold | Precision / Recall | F1 |
|---|---|---|---|---|---|
| 1 | Jev NER → Jev API guided | Max | 0.8 | 63.8 / 68.5 | **66.1** |
| 2 | GLiNER-clean → Jev API guided | Max | 0.9 | 50.0 / 38.9 | 43.8 |
| 3 | GLiNER-clean → Jev API | Max | 0.9 | 41.7 / 37.0 | 39.2 |
| 4 | GLiNER-clean → OpenJev 9B | Max | 0.8 | 36.7 / 33.3 | 35.0 |

Source: `experiments/RESULTS.md`.

## Caveats
- The gold was annotated by Claude and has not been reviewed by a historian. For relations there are only 54 facts, so
  intervals are about ±12 points.
- Jev and NLI thresholds were tuned on this same gold, so their best numbers are optimistic. Rows 3–6 of the relations
  table are the comparable ones.
- The hybrid was developed on two paragraphs (p004), where it scores 87; the 12 new paragraphs (53) are the honest estimate.
- Jev NER and our LLM names both use annotation rules close to the gold's, which inflates the mention scores.
- Rows 2 and 4 of the mention table come from `pilot14/score.py`, which has its own overlap matching; the other rows come
  from `eval/evaluate.py`. The two are close but not guaranteed identical.
- gpt-oss answers differ between runs even at temperature 0; most of our rows are single runs.
- Entity linking is not yet connected to relation extraction; relations are scored on names.
