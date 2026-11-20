# NLI precision improvements — reproducible experiments

Install these files under `experiments/10_nli_precision/` in the existing Kno-Know repo, alongside your unchanged `experiments/09_nli_extended/` and `archive/eval/evaluate.py`.

## 1. No new model inference: type constraints + margins

Run from the **repository root**:

```bash
python experiments/10_nli_precision/checks.py
```

It automatically reads all 6 saved `experiments/09_nli_extended/results/{gold,lela}_*_{scores}.json` files if present (missing ones are logged), and writes:
- `results/ablations.csv`: fixed threshold (0.90) with three post-filters (`as_scored`, `conservative`, `strict`), strategies (`baseline`, `alternative`, `ensemble`), and score margins (0, .05, .10, .20).
- `results/cross_document.csv`: choose threshold + margin on Alford, evaluate Nechaeva; then reverse. **Exploratory transfer check**, not independent holdout.
- `results/missing_scores.json`, `results/validation_issues.json`, `results/RESULTS.md`.

**Critical methodological distinction:** original 09 candidate generation has already used schema filtering. The post-filter here can only remove existing candidates. This is *not* a test of entirely unrestricted versus constrained candidate generation. The `strict` option rejects unknown types, and may lower recall. No finding is guaranteed to improve F1.

The evaluator is imported from `archive/eval/evaluate.py`. The same source/entity conversion as `compare_all.py` is used. The outputs are written to a new folder; no existing score caches or summary CSVs are overwritten.

## 2. New NLI inference: entity-focused sentence context

First check coverage **without** loading model:

```bash
python experiments/10_nli_precision/focused_inference.py --source gold --prepare-only
```

Then perform the controlled new inference, on the same markable candidate pairs with original and marked premises:

```bash
python experiments/10_nli_precision/focused_inference.py --source gold --batch-size 8 --device auto
python experiments/10_nli_precision/compare_focused.py --source gold
```

Run with `--source lela` for the same ablation using automatically extracted mentions. These inference commands require `torch`, `transformers`, and the model checkpoint accessible from Hugging Face or local cache.

The marking method highlights entities using `[SUBJ] ... [/SUBJ]` and `[OBJ] ... [/OBJ]` in the premise. **These tags may harm performance** because the model has not been trained on them. It is an empirical hypothesis, not an assumed improvement.

Unmarkable or overlapping pairs are excluded from **both arms**. Consequently this is a matched-subset experiment and should not be compared directly to full-set 09 scores. Results use the canonical fact evaluator and the same 0.9 threshold. Training-free marker selection is not optimized on gold.

## 3. Tests

```bash
python -m unittest discover -s experiments/10_nli_precision -p 'test_*.py'
```

## Reporting

Report every metric (P/R/F1/TP/FP/FN) and the **number of predictions**. Avoid selecting the maximum over all variants and presenting it as test performance. Both articles have already informed development; independent final evaluation needs new, untouched historian-annotated text. For LELA, distinguish entity linking failures from relation classification errors; automatic entity IDs are aligned to gold by YAGO or unique aliases using the common evaluator.
