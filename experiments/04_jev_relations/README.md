# Experiment 04: YAGO-relation extraction with OpenJev

## Question
Can a small decision model (OpenJev Flash 9B, Q4_K_M, local) pick the right YAGO relation for a pair of names in a
sentence, and do the NER fixes help?

## Setup
* Text: `prepare/prepare_texts.py`, unchanged (its QA report was clean for both papers).
* NER: `pipeline/ner.py --clean` (GLiNER medium, threshold 0.4) with three changes, all optional: without `--clean`
  the script behaves as before. The ‘ inside a transliterated word is
  replaced by ʻ for the model only, so offsets stay the same. Mentions with no capital letter, and bare dates, are
  dropped. A `written work` label is added.
* Relations: `schema/history_relations.json` (`schema/extract_history_relations.py`). That is 26 YAGO 4.6 relations
  plus `ex:relativeOf`, `ex:captiveIn` and `ex:opposed`. The pipeline uses only the 22 relations between two entities
  (dates are not mentions).
* Extractor: `pipeline/relations_jev.py`. It considers every pair of distinct names in a sentence. One OpenJev
  decision chooses among the relations the two types allow, in both directions, or "none". The probabilities are a
  softmax over the option letters at T = 1.07. A fact is kept if p ≥ threshold.
* Model: `OpenJev-Flash-9B-Q4_K_M.gguf` (SHA-256 checked) on llama.cpp b11951 (CUDA 12.4), RTX 3080 Laptop 8 GB.
  Both papers (1,103 decisions) took about 3 minutes.
* Gold: `eval/gold` (14 paragraphs, 159 mentions, 54 facts). This is a draft, see its README. Scored with
  `eval/evaluate.py score --span overlap`.

## Results

**NER, Alford, 9 gold paragraphs (90 mentions)**

| run | strict P / R / F1 | overlap P / R / F1 |
|---|---|---|
| `gliner` (no preprocessing) | 58.9 / 81.1 / 68.2 | 66.1 / 91.1 / 76.6 |
| `gliner-clean` (`ner.py --clean`) | 76.1 / 74.4 / 75.3 | 86.4 / 84.4 / 85.4 |

Precision improves because generic nouns and dates are gone (621 → 458 mentions over the paper). Recall falls for
two reasons:
* Borderline mentions with scores of 0.46–0.49 drop below the threshold.
* With the ‘ fix, GLiNER now reads "T‘at‘ik of Basean" as one span. That is a mismatch with the "name only"
  convention, not a missed person.

**Facts, both papers (54 gold facts)**: runs `gliner-clean__openjev` and `gold__openjev`, from `experiments/RESULTS.md`

| mentions | threshold | P | R | F1 |
|---|---|---|---|---|
| predicted (NER) | 0.5 | 21.4 | 46.3 | 29.2 |
| predicted (NER) | 0.8 | 36.7 | 33.3 | 35.0 |
| gold | 0.5 | 40.4 | 85.2 | 54.8 |
| gold | 0.7 | 52.1 | 70.4 | 59.8 |
| gold | 0.8 | 59.3 | 59.3 | 59.3 |
| gold | 0.9 | 74.1 | 37.0 | 49.4 |

For reference, on Alford only at pair level (relation ignored), the earlier Qwen OpenIE output
(`eval/pred/gliner__qwen-openie`) gets 1 of 28 gold pairs. JEV gets 18 of 28 at P = 34%. The OpenIE comparison is weak: its
entities are names from the old NER and its predicates are free text.

## Reading
* **The probabilities can be used.** Precision rises steadily with the threshold, so the threshold is a real
  precision/recall dial. This is the main gain over an LLM that returns plain JSON.
* **With gold mentions, recall is high (85% at 0.5).** The relation stage rarely misses. Most of what is left is NER
  error and pairs that NER never produced: fact F1 is 29 with predicted mentions against 55 with gold mentions.
* **Errors (gold mentions, threshold 0.5):**
  * `ex:captiveIn` and `schema:location` are over-predicted (18 and 29 false positives), e.g. Łazar → Khuzestan when
    a captive group is mentioned next to Łazar.
  * `schema:location` and `yago:partOf` are confused for place-in-place (Arzanene → Persarmenia).
  * The relation is attached to the wrong person (Łazar worksFor Vahan, the patron relation).
  * Several "false positives" may be gaps in the gold: Martyrdom of Mar Grigor `schema:about` Grigor, and Dagaris
    `worksFor` Sittas, which the gold leaves out because the text says "Sittas or Dorotheus".
* **Gold fixes made during the experiment, both mechanical:**
  * Three gold facts broke the stated "one sentence names both" rule (the participants were in different
    sentences, or one was a pronoun). `make_gold.py` now checks this rule and the three are removed.
  * The Sasanian and Byzantine empires are typed as places, as YAGO does. They had been typed as organizations, so
    `schema:location` was never offered for them.

## Limits
* The gold is a 14-paragraph draft by one annotator. The intervals are wide; read the differences, not the decimals.
* Facts are matched by entity name, not QID, so linking is not part of this score.
* No attribution or certainty yet ("Łazar claims…" is scored like a plain fact).
* The prompt follows the OpenJev model card, but not its calibrated helper (`shim.py` on vLLM). The probabilities
  are therefore uncalibrated letter softmaxes.

## Repeat
```bash
llama-server -m OpenJev-Flash-9B-Q4_K_M.gguf -ngl 999 -c 8192 -np 1 --port 8080
uv run python schema/extract_history_relations.py
uv run python pipeline/ner.py --doc alford2026 --clean      # and nechaeva2017
uv run python eval/make_gold.py
uv run python pipeline/relations_jev.py --doc alford2026 --ner gliner-clean --model openjev   # and --ner gold; and nechaeva2017
uv run python experiments/compare.py
```
