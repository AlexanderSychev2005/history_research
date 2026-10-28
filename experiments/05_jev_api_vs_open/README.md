# Experiment 05: hosted Jev (TypeSafe API) vs OpenJev (local)

## Question
Does TypeSafe's hosted Jev make better relation decisions than the open 9B model, and does calling it the way the
TypeSafe skill recommends make a difference?

## Setup
Same pairs, relations and gold as experiment 04 (`eval/gold`: 14 paragraphs, 54 facts). Only the 14 gold
paragraphs were sent to the API: 280 pairs with NER mentions and 322 with gold mentions.

| system | model | how it is asked |
|---|---|---|
| `openjev` | OpenJev-Flash-9B Q4_K_M, llama.cpp, local | the exact prompt from experiment 04: lettered options plus "none", letter softmax at T = 1.07 |
| `jev-api` | `jev-latest` (= jev-1.14.0), `api.typesafe.ai/v1/systemone` | the same options as one `choice` question; option names A, B, C…; state sent as a JSON string |
| `jev-api-guided` | same | as the [TypeSafe skill](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md) recommends: state as an object, descriptive option names (`relativeOf(Nerseh, Hrahat)`), and a companion `noul` "does the sentence state how X and Y are connected?" instead of a "none" option. A relation's score is noul × choice probability; "none" is 1 − noul |

Each model was run on `gliner-clean` mentions and on `gold` mentions (the relation stage alone), e.g. run
`gold__jev-api-guided` = `pipeline/relations_jev.py --ner gold --model jev-api --guided`. All numbers below are from
`experiments/compare.py` (`experiments/RESULTS.md`, `experiments/results/`).

## Results: facts, fact-level P / R / F1 (%)

**Gold mentions (relation stage alone)**

| threshold | `openjev` | `jev-api` | `jev-api-guided` |
|---|---|---|---|
| 0.5 | 40.4 / 85.2 / 54.8 | 52.5 / 77.8 / 62.7 | 47.5 / 87.0 / 61.4 |
| 0.7 | 52.1 / 70.4 / 59.8 | 56.7 / 70.4 / 62.8 | 60.0 / 83.3 / 69.8 |
| 0.8 | 59.3 / 59.3 / 59.3 | 64.3 / 66.7 / 65.5 | **71.2 / 77.8 / 74.3** |
| 0.9 | 74.1 / 37.0 / 49.4 | 71.7 / 61.1 / 66.0 | 72.7 / 44.4 / 55.2 |

**NER mentions (full pipeline)**

| threshold | `openjev` | `jev-api` | `jev-api-guided` |
|---|---|---|---|
| 0.5 | 21.4 / 46.3 / 29.2 | 21.1 / 44.4 / 28.6 | 20.9 / 51.9 / 29.8 |
| 0.8 | 36.7 / 33.3 / 35.0 | 31.8 / 38.9 / 35.0 | 34.8 / 42.6 / 38.3 |
| 0.9 | 48.1 / 24.1 / 32.1 | 41.7 / 37.0 / 39.2 | **50.0 / 38.9 / 43.8** |

**Agreement between open and API on the same pairs** (`experiments/results/agreement.csv`)

| pairs | same decision | "none": open / API |
|---|---|---|
| NER mentions, same prompt (280) | 73.9% | 110 / 144 |
| gold mentions, same prompt (322) | 79.2% | 181 / 221 |
| gold mentions, open vs guided (322) | 71.1% | 181 / 169 |

## Reading
* **The hosted model is better at the relation decision.** With the same prompt its best F1 is 66 against 60 for the
  open model. It also stays useful at high thresholds (61% recall at 0.9, against 37%), so its probabilities are
  sharper. It says "none" more often (221 vs 181 of 322).
* **Following the TypeSafe guide adds the most: best F1 74 (P 71, R 78 at 0.8).** The companion Noul separates
  "related at all" from "which relation", and descriptive option names replace bare letters. This matches the
  skill's advice for Jev 1.14.
* **With NER mentions, every model is limited to about 30–44 F1.** NER errors cap all three systems (experiment 04:
  21 of 54 gold facts lose a participant to NER). A better relation model does not fix that.
* The intervals overlap; 14 paragraphs and one draft annotator. Read the ordering, not the decimals.

## Caveats
* `jev-latest` is an alias (it answered as jev-1.14.0 on 28 Oct 2026, recorded in each output file under `model`).
  The models listing only offers `jev-latest` and `jev-preview`, so a pinned version could not be chosen.
* The local model has only the letter-prompt variant. The guided request format is an API feature, so the 74 F1
  combines a better model with a better question design. The "API, same prompt" column isolates the model.
* About 1,200 API requests were made (two variants × two mention sets, gold paragraphs only). The sentences of the
  papers were sent to TypeSafe.
