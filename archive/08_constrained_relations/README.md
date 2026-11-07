# Experiment 08: relation extraction by zero-shot prompting with constrained decoding

## Question
Fabian suggested three methods for relation extraction: NLI (experiment 07), Jev (experiments 04–06) and **zero-shot
prompting with constrained decoding**. This is the third. Can a general LLM, given no examples, pick the right YAGO
relation if its answer is forced into a strict JSON schema, so that it cannot invent a relation name (the failure of
the free-text Open IE, which produced 116 relation names for 148 facts)?

## Setup
* Text, gold, relations: as in experiments 04–07. `eval/gold` (14 paragraphs, 159 mentions, 54 facts; a draft that still
  needs a historian's review) and `schema/history_relations.json` (22 entity-to-entity relations: YAGO 4.6 plus
  `ex:relativeOf`, `ex:captiveIn`, `ex:opposed`).
* Names: the **gold mentions** (`outputs/<doc>_ner_gold.json`), so only the relation stage is measured, like the `gold__*`
  runs of Jev and NLI. Other mentions work through `--ner <variant>`.
* Model: `openai/gpt-oss-20b` on Groq, `response_format` = strict JSON schema, `reasoning_effort` low, temperature 0.
* Scoring: `eval/evaluate.py`, spans by overlap, as in `experiments/RESULTS.md`; `summarize.py` writes `RESULTS.md` and
  `errors.md` here.

## Variants (`relations_constrained.py`)
| variant | what is asked | what the schema allows |
|---|---|---|
| `paragraph` | one call per paragraph: list the facts stated | a list of {subject: one of the paragraph's names, relation: one of the 22, object: one of the names, evidence: text} |
| `sentence` | the same, one call per sentence | the same, with the names of the sentence only |
| `pair` | one question per pair of names in a sentence | one of `relation(subj, obj)` for the relations the two types allow, or `none` |
| `gated` | "does the sentence state how X and Y are connected?", then the relation | `yes`/`no`, then the options without `none` |

The prompt gives the sentence or paragraph, the names with their types and a one-line description of each relation (the
`gloss` of the schema file); the instruction is to give a fact only when **one sentence names both participants and
states the relation or directly implies it** (the gold convention). Triples that break the schema's domain/range are
dropped afterwards. For symmetric relations the name that comes first is the subject, as in the gold.

## Results (gold mentions, 54 facts, overlap spans; F1 and P/R in %)
| run | P | R | F1 | 95% CI | tp / pred / gold |
|---|---|---|---|---|---|
| `paragraph`, repeat 1 | 75.0 | 50.0 | 60.0 | 28–82 | 27 / 36 / 54 |
| `paragraph`, repeat 2 | 77.8 | 51.9 | 62.2 | 27–84 | 28 / 36 / 54 |
| `sentence`, repeat 1 | 66.7 | 59.3 | **62.7** | 50–73 | 32 / 48 / 54 |
| *Jev API, default threshold 0.5 (exp. 05)* | 52.5 | 77.8 | 62.7 | 52–71 | 42 / 80 / 54 |
| *Jev API guided, threshold 0.5* | 47.5 | 87.0 | 61.4 | 48–75 | 47 / 99 / 54 |
| *OpenJev 9B local, threshold 0.5* | 40.4 | 85.2 | 54.8 | 44–63 | 46 / 114 / 54 |
| *NLI, best variant, threshold 0.99* | 54.8 | 31.5 | 40.0 | 25–51 | 17 / 31 / 54 |

Fact-pair F1 (relation ignored): `sentence` 68.6, `paragraph` 62.9 / 65.2, Jev API 67.2. Every number comes from
`RESULTS.md` (generated).

## What the numbers say
1. **Without any threshold, the constrained LLM matches Jev API at its default threshold** (F1 62.7 both) and beats NLI
   (40.0). The intervals are wide (about ±12 points on 54 facts), so the order among the three top rows is not settled.
2. **Different profile:** the LLM is precise and misses facts (P 67–78, R 50–59); Jev over-predicts (P 48–53, R 78–87).
   Jev has a threshold to trade the two (its best, tuned on this gold, is F1 74.3 at 0.8); the constrained LLM has no
   probability to threshold, only repeats or voting could give one.
3. **Answers differ between runs even at temperature 0** (`paragraph` 60.0 and 62.2; an earlier run with the same
   prompt predicted 25 facts for Alford where this one predicts 12). Report the mean of several repeats.
4. **"Strict" is not a guarantee on Groq.** In the runs above 0 answers broke the schema, but (a) Qwen3.8-27b did not
   respect the schema (HTTP 400 `json_validate_failed`, the model answered `none` outside the JSON) and returned empty fact
   lists when asked for arrays; (b) an early variant that put every candidate fact of a sentence into one enum
   (hundreds of long strings) made gpt-oss-20b produce values outside the enum and, for long sentences, a 30,000-token
   request. Small enums (relations, names) worked.
5. **Errors** (`errors.md`, `sentence` run): most misses are convention cases: "X of Place" read as `schema:location`
   (7 of the 22 misses are the seven philosophers of Nechaeva p004: "Damascius of Syria"), and "the captives of Khuzestan
   comprised A, B, C" (4 `ex:captiveIn`). Several false positives are defensible readings the gold does not have
   (`schema:location(Nerseh Kamsarakan, Armenia)`, `influencedBy(Robert Thomson, Akinean)`); `ex:opposed(Pērōz, Kidarite
   Huns)` and `ex:opposed(Yazdgerd II, Kidarite Huns)` are plausible but the gold has only the Sasanian Empire as subject.
6. **Cost:** `paragraph` about 20,000 tokens for both papers, `sentence` about 40,000, `pair` about 850 tokens per pair, so
   about 275,000 for the 322 pairs: more than the 200,000 daily tokens of one gpt-oss model on Groq's free tier.

## Not done, and why
* **`pair` and `gated`** (the per-pair framing Fabian suggested): the code is ready and **resumes** from `cache/` after a
  daily-limit stop, but both gpt-oss models had used their daily tokens, so no pair run finished.
* **`gpt-oss-120b`**: the same, its daily limit was reached before the first call.
* **Repeats:** `paragraph` has 2, `sentence` 1; the others stopped at the limit.
* **Mentions from our entity-linking hybrid** (Fabian: "pairs of entities found by LELA"): the script takes any
  `outputs/<doc>_ner_<variant>.json`; the file for our hybrid has not been generated.
* The gold facts were written by one annotator (Claude) and not reviewed by a historian; 54 facts give wide intervals;
  the Jev and NLI thresholds were chosen on this same gold, the constrained LLM has none to tune.

## Run it
```bash
export GROQ_API_KEY=...        # never commit it
# from the repository root; uv gives the two small packages relations_jev.py imports
uv run --no-project --with python-dotenv --with requests python experiments/08_constrained_relations/relations_constrained.py \
    --doc alford2026 --ner gold --model 20b --variant sentence --rep 1          # then nechaeva2017; 120b, pair, gated likewise
python experiments/08_constrained_relations/summarize.py                         # -> RESULTS.md, errors.md
```
`pred/<run>/<doc>.json` is the format of `eval/pred/`; copy a folder there to include it in `experiments/compare.py`.
`decisions/` holds every answer with its evidence and token usage; `cache/` the resumable pair answers.
