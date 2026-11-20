# Kno-Know — NEW NLI experiments (not a threshold sweep)

This directory is **a new, self-contained experiment**, separate from `archive/experiments/07_nli_relations/`. It does not re-score the earlier saved NLI predictions. Its `run.py` performs **fresh model inference** using the same DeBERTa MNLI model as the archived experiments, the live `dataset/relations.json` schema, the current 14-paragraph gold benchmark, and `lela/pred/llm/*.jsonl`.

## Research questions

1. **Hypothesis verbalization**: for a fixed model, pairs, and text, compare each schema relation's `baseline` and `alternative` hypotheses, plus a max-entailment `ensemble` over both (note: max is not a calibrated probability).
2. **Context length vs pair coverage**: `sentence_sentence` vs `sentence_paragraph` uses **identical candidate pairs** but changes ONLY the NLI premise from the source sentence to the full paragraph; `sentence_paragraph` vs `paragraph_paragraph` then adds cross-sentence candidate pairs. These are distinct interventions, so report them separately.
3. **Automatic mentions**: `gold` vs `lela` compare gold mentions and automatic LELA mentions. Predictions from LELA are evaluated using unique gold-name/alias or unique YAGO matches, as explained below. This is **not** the exact original Jev end-to-end pipeline (which uses its own entity finder).

## Run from repository root

```bash
python -m pip install torch transformers sentencepiece protobuf
python experiments/09_nli_extended/run.py --prepare-only
python -m unittest discover -s experiments/09_nli_extended -p 'test_*.py'
python experiments/09_nli_extended/run.py --device auto --batch-size 16 --threshold 0.90
```

`--prepare-only` writes candidate job JSON and coverage statistics but does **not** run NLI or produce F1. Full run writes `results/*_scores.json`, `results/pred/<run>/<doc>.json`, `results/metrics.csv`, and `results/RESULTS.md`. Both source papers are included by default. For a quicker first run:

```bash
python experiments/09_nli_extended/run.py --sources gold --modes sentence_sentence --batch-size 16
```

For strict mention type compatibility, add `--strict-types`. Default is conservative for unfamiliar LELA labels: unknown label types are allowed; this may generate more candidates than Jev's own typed system. Use `--multi-label` to permit more than one relation per pair; default picks **one** best relation per pair as in the archived NLI script. The fixed threshold of `0.90` is a **predefined starting point, not calibrated or asserted optimal**. The run scores each hypothesis only once and derives baseline, alternative and ensemble from the two freshly computed scores.

## Inputs and important differences

- Existing repository: `dataset/gold/{alford2026,nechaeva2017}.json`, `dataset/paragraphs/`, `dataset/relations.json`, `lela/pred/llm/{doc}.jsonl`. No inputs included in this ZIP; paste the `experiments/` folder at repository root.
- The benchmark has **54 facts across 14 selected paragraphs**, with a draft, single-annotator gold not yet reviewed by a historian. Scores are exploratory. It excludes date-valued relations because the task is entity–entity extraction.
- Gold mode groups mentions by gold entity ID; LELA mode groups by `entity_id` (or surface when unlinked). Both are processed only in gold's selected evaluation paragraphs. Mention spans must match paragraph text; alignment failures are silently skipped, so inspect `coverage.json`.
- Simple deterministic regex sentence boundaries are used here rather than importing the archived `pipeline.relations_jev.sentences`. Sentence segmentation differences can affect the comparison; for exact Jev matching replace `sentences` with that function after auditing its dependencies.
- Candidate restrictions are based on `dataset/relations.json` with broad local type mappings. **This is not a byte-for-byte reproduction of Jev's candidate generation or the original NLI candidate generation**, so comparison with published baselines is descriptive unless pair/candidate policy is aligned.
- Evaluation on `gold` uses the actual gold entity ID. On `lela`, a predicted YAGO ID is resolved if it uniquely maps to a gold entity; otherwise an exact normalized label/alias is tried. Incorrect LELA linking therefore can lower performance, but an alias can rescue an incorrectly linked mention. Strict end-to-end matching would require a dedicated mention+link+fact evaluator; this metric is *entity-normalized* and should not be presented as strict entity-linking accuracy.
- Entity pair direction matters except for symmetric relations, for which only one orientation is generated. One highest scoring relation per ordered entity pair is selected. No gold facts are used to select templates or the threshold, but gold provides the annotation scope and optional gold mentions; this is an oracle-mention research setting.
- Since all 14 paragraphs are reused for investigation, there is **no held-out generalization claim**. For a proper comparison, establish a document-level development/test split *before* choosing thresholds and templates, or annotate additional papers; evaluate at least one untouched paper.
- Because the hypothesis ensemble takes a maximum, its high NLI probabilities **are not calibrated confidence scores**. Consider training/dev calibration later, but do not tune on the final test paper.

## Reporting checklist

Compare only `doc=ALL` for micro-F1, and retain paper-by-paper lines as evidence of variability. Report TP / predicted / gold, precision, recall and F1. Explain that only two real papers are included, that the base templates and NLI model are the same as in the archive, and that the new work is fresh **hypothesis/context/mention-source experiments**, not another threshold sweep. Match exact candidate policy, mention source and scorer before placing a Jev/constrained row in a truly controlled model comparison.

## What has been verified

The `--prepare-only` pass and tests were executed without downloading weights. **No claim is made that the full neural inference has run** and no new measured F1 is supplied in this ZIP. 
