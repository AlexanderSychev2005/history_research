# Experiment 11 — Error-driven NLI: false positives and directionality

This adds **two distinct new research checks** rather than repeating threshold/type/margin sweeps:

1. **Gold-linked error taxonomy** (offline). Explain which false positives involve the wrong relation on a known pair, reversed direction, unmatched entity (LELA), or no annotated fact for the pair. Produces a relation-wise error table and a manual inspection queue. The labels are *diagnostic only*, and may be affected by incomplete gold annotations.
2. **Counterfactual role reversal** (new inference). For each selected *asymmetric* prediction, keep the premise unchanged, swap `subject` and `object` in the NLI hypothesis, and score its entailment. Under a **pre-declared** filtering rule, retain a forward prediction only if forward entailment minus reversed entailment is at least delta. Symmetric relations are left unchanged. A forward/reverse difference measures sensitivity, not proof of causality or semantic correctness.

Use commands from the root of `history_research` after unzipping. Existing score caches in `experiments/09_nli_extended/results` are required. **No new inference** is needed for diagnostics.

```bash
python -m unittest discover -s experiments/11_nli_error_driven -p 'test_*.py'
python experiments/11_nli_error_driven/diagnose.py --source gold --mode sentence_sentence --strategy ensemble
python experiments/11_nli_error_driven/diagnose.py --source lela --mode sentence_sentence --strategy ensemble --out experiments/11_nli_error_driven/results/diagnostics_lela
```

Files include `predictions_with_errors.csv`, `relation_errors.csv`, `false_negatives.csv`, `summary.json`, and `RESULTS.md`. Use `predictions_with_errors.csv` to review top false positives and manually label `error_category` / whether the source text supports the statement.

## New reverse-role inference

```bash
python experiments/11_nli_error_driven/reverse_probe.py --source gold --strategy ensemble --prepare-only
python experiments/11_nli_error_driven/reverse_probe.py --source gold --strategy ensemble --device auto --batch-size 8
python experiments/11_nli_error_driven/evaluate_reverse.py --source gold --strategy ensemble
```

For `lela`, repeat with `--source lela`. If wanting to test baseline, use `--strategy baseline` for both commands; the reverse inference always records both baseline/alternative entailment values for each pair.

`evaluate_reverse.py` reports an **original** row and exploratory contrasts with predeclared deltas `0, 0.1, 0.2`. Unlike a conventional threshold sweep, this asks whether the model distinguishes the same relation in the correct vs incorrect direction. Deltas are illustrative and require development-set selection. Do **not** call the maximum F1 of these options a final test score.

**Important:** the reverse hypothesis can be grammatically valid yet false, or the relation may actually hold in both directions; hence the check is a diagnostic ablation. Even for asymmetrical predicates, the document might support facts in both directions. Also relation and candidate errors can be caused by incomplete or draft gold facts.

**Reproducibility:** 09's NLI model is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; `reverse_probe.py` uses the same tokenizer/model/inference implementation. The new files do not overwrite 09 or 10 results. Scores are evaluated using the archive common fact evaluator; both papers were inspected during development, so all results remain exploratory.
