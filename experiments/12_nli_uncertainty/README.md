# NLI — paired uncertainty and prediction audit

This patch adds a **new experiment 12** without modifying the original evaluator or earlier experiment folders. It requires the existing repository folders `archive/eval/`, `dataset/`, `experiments/09_nli_extended/`, and `experiments/11_nli_error_driven/` plus the existing cached `*_scores.json` and `*_paired_scores.json`.

## Usage (from `history_research` repo root)

```bash
python -m unittest discover -s experiments/12_nli_uncertainty -p 'test_*.py'
python experiments/12_nli_uncertainty/paired_bootstrap.py --bootstrap 5000 --seed 13
```

To reproduce a different number of samples:

```bash
python experiments/12_nli_uncertainty/paired_bootstrap.py --bootstrap 10000 --seed 13
```

Outputs: `results/bootstrap.json`, `results/RESULTS.md`, `results/prediction_audit.csv`, `results/by_relation.csv`, `results/per_paragraph.csv`.

## Design

- The archive evaluator handles paragraph-scoped fact identity and entity alignment; no change to the gold standard.
- Bootstrap resamples annotated paragraphs **within each source document**, and applies identical sampled paragraph indices to Original NLI and Reverse NLI. F1 is recomputed from summed TP / predicted / gold counts each replicate. The resulting **paired difference interval** is relevant to the within-benchmark change. This is not a confidence interval for performance on unseen documents.
- Fixed `threshold=0.9`, `delta=0.2`, `ensemble`, `sentence_sentence`. These are previous development choices; **the same two articles were used to choose them**.
- The previous `ci_low` and `ci_high` in `semantics_aware/comparison.csv` are invalid as confidence intervals: their computation called the archive evaluator with `B=1`. This new script uses `B=5000` by default with a separately controlled seed.
- Audits use exact evaluator entity-alignment keys after deduplication and explicitly identify `removed_FP` vs `lost_TP`. The labels are only relative to currently annotated gold facts, not independent historian judgments.
- When assessing generalization, split by **whole document**, not by paragraph, and evaluate a pre-registered frozen model and parameters on new annotated texts. Two development documents do not establish external validity.

## What this does not do

It does not perform inference, modify the source experiment 11 files, create additional annotations, correct ambiguous ontology definitions, or provide a new independent test set. It does not claim statistical significance for generalization or infer that all gold-absent facts are historically false.
