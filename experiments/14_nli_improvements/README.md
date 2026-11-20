# NLI Improvements — reproducible experiments (code only)

This directory contains code and tests, **no precomputed results**. Run from your repository root.
All experiments start from the **same frozen baseline**: `sentence_sentence`, ensemble MAX of two NLI hypotheses, forward threshold `0.90`, one relation per entity pair, and reverse contrast `Δ = 0.20`. No labels are used in the filtering rules (except training data for the explicitly supervised calibration experiment).

## Prerequisites

Run the pre-existing inference experiments 09 and 11 for **both** `gold` and `lela` before using these scripts. Expected input caches:

```
experiments/09_nli_extended/results/{gold,lela}_sentence_sentence_scores.json
experiments/11_nli_error_driven/results/reverse_probe/{gold,lela}_ensemble_t0p9_paired_scores.json
```

The scripts import `run.py` from experiment 09, `common.py`, `evaluate_reverse.py` and `reverse_probe.py` from experiment 11, and use the exact `archive/eval/evaluate.py` evaluator, `archive/eval/gold/`, and `texts/`.

## Run

```bash
python -m unittest discover -s experiments/14_nli_improvements -p 'test_*.py'
python experiments/14_nli_improvements/run_all.py

# 1. Type constraints only
python experiments/14_nli_improvements/type_constraints.py --source both

# 2. Hard negatives: first prepare and inspect, then score using your local model
python experiments/14_nli_improvements/hard_negatives.py --source gold --prepare-only
python experiments/14_nli_improvements/hard_negatives.py --source gold --infer --device auto --batch-size 8
python experiments/14_nli_improvements/hard_negatives.py --source lela --infer --device auto --batch-size 8
# Later re-evaluate without model inference:
python experiments/14_nli_improvements/hard_negatives.py --source gold

# 3. Heuristic evidence inspection and filters
python experiments/14_nli_improvements/evidence.py --source both

# 4. Supervised logistic calibration: train on one document, evaluate on the other
python experiments/14_nli_improvements/calibration.py --source both
```

To execute all including model inference use `python experiments/14_nli_improvements/run_all.py --with-inference`.

## Outputs (created only when executed)

All output files are under `experiments/14_nli_improvements/results/`, in subdirectories `types/`, `hard_negatives/`, `evidence/`, `calibration/`.

* `*_comparison.csv`: same archive evaluator for TP, FP, predictions, precision, recall, F1, fact-pair F1, and paragraph-bootstrap CI.
* `*_rejected.csv`: type filter rejections with metadata.
* `*_manifest.json`, `*_scores.json`: fresh hard-negative model requests and resulting entailment probabilities.
* `*_evidence_audit.csv`: the evidence snippet, lexical cue match and whether each proposed evidence filter retained it.
* `*_prediction_audit.csv`: document-held-out calibration predictions and their labels **for analysis only**, not for selecting thresholds.

## Experimental cautions

- The two existing papers are **development data**, not an independent held-out benchmark. Type rules, hard-negative alternatives, margins and evidence heuristics are exploratory. Any newly selected model must be frozen before evaluation on untouched papers.
- Semantic types are incomplete, especially for LELA. Unknown types pass through and cannot independently correct entity linking.
- `yago:creator` in this corpus uses the work as subject and its creator as object (the inverse wording in templates).
- The alternative statement in a hard-negative can coexist with the true relation: NLI scores are **not** calibrated probabilities of mutually exclusive claims. Compare with frozen baseline before trusting any drop.
- Evidence lexical cues are weak proxies, not entailment, and both-mention filtering can lose cross-sentence relations.
- Logistic regression intentionally trains separately for `gold` and `lela`, using **one document at a time**, and never fits on the held-out article. Only two documents means cross-document estimates are highly unstable. The 0.5 decision threshold and C=1 are fixed defaults, not tuned on test labels.
- Bootstrap CIs are paragraph-level uncertainty within the two studied articles; they are not cross-paper generalization intervals.
- `run_all.py` excludes expensive NLI inference by default. Passing `--with-inference` runs it for gold and LELA.
