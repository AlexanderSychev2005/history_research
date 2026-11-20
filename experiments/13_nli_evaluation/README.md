# NLI Evaluation

This folder contains **source code only**. Results, predictions, CSV tables, and `RESULTS.md` are created **when the evaluation runs**. Nothing is pre-populated.

The evaluation uses the same `archive/eval/evaluate.py`, gold annotations, text files, span matching and bootstrap procedure used for the existing Jev / constrained zero-shot comparisons.

## Prerequisites

Run from the `history_research` repository root. The code requires the existing folders:

- `archive/eval/evaluate.py`, `archive/eval/gold/`, and `texts/`
- `experiments/09_nli_extended/` with `run.py`, `compare_all.py`, and saved `{gold,lela}_sentence_sentence_scores.json`
- `experiments/11_nli_error_driven/` with `common.py`, `evaluate_reverse.py`, and saved `{gold,lela}_ensemble_t0p9_paired_scores.json`

It **does not run NLI inference**; it evaluates the already-computed NLI and reverse-probe scores. If these caches are missing, produce them using the previous experiment scripts first.

## Run

```bash
python experiments/13_nli_evaluation/evaluation.py
```

Optional bootstrap size / output folder:

```bash
python experiments/13_nli_evaluation/evaluation.py --bootstrap 500 --out experiments/13_nli_evaluation/results
```

After execution, the script generates:

```text
experiments/13_nli_evaluation/results/
    comparison.csv
    per_relation.csv
    errors.csv
    validation.csv
    RESULTS.md
    pred/
        nli_gold_original/
        nli_gold_reverse_d0p2/
        nli_lela_original/
        nli_lela_reverse_d0p2/
```

To run the same archive evaluator directly against generated NLI predictions:

```bash
python archive/eval/evaluate.py score \
  --texts texts --gold archive/eval/gold \
  --span overlap --bootstrap 500 \
  --pred experiments/13_nli_evaluation/results/pred/nli_gold_reverse_d0p2
```

## Tests

```bash
python -m unittest discover -s experiments/13_nli_evaluation -p 'test_*.py'
```

Unlike the earlier package, the tests **compute their own temporary results** rather than depending on CSVs included in the archive.

## Methodology

- The NLI configuration is frozen at threshold `0.90`, reverse score margin `0.20`.
- Runs include gold entities and LELA entities, each with original and reverse-filtered predictions.
- Metrics: fact-level precision, recall, F1; fact-pair F1; paragraph bootstrap confidence intervals; errors and per-relation diagnostics.
- Archived Jev and zero-shot reference predictions are included in the report only when present locally.
- This reproduces the **same evaluation procedure**, but does *not* constitute an independent test set: the two development articles were used while developing NLI.
- Mention extraction and linking are not NLI stages in this experiment, so their metrics should not be interpreted as NLI performance.
