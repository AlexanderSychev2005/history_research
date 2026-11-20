# NLI Relation Extraction — Experimental Results

**20 November 2026 · Development-set evaluation (not held-out)**

## 1. Evaluation setup

- **Corpus:** `alford2026` and `nechaeva2017`; **14 annotated paragraphs**, **54 gold facts**.
- **Evaluator:** `archive/eval/evaluate.py` (fact-level exact relation matching). All figures below use this evaluator unless explicitly marked otherwise.
- **Gold:** manually annotated entity inputs. **LELA:** automatically detected/linked entity inputs. These are distinct evaluation conditions.
- **Frozen NLI baseline:** sentence → sentence, two-template entailment ensemble, forward threshold **0.90**, reverse-probe margin **Δ = 0.20**, one selected relation per pair.
- **Metrics:** precision = TP/(TP+FP), recall = TP/54, F1 = harmonic mean; **FP = predictions − TP**, **FN = 54 − TP**. Percentages are fact-level unless marked “pair F1.”

## 2. NLI versus other methods (same fact-level evaluator)

| Entity input | Method | TP | FP | FN | Precision | Recall | F1 | Pair F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gold | NLI original | 44 | 99 | 10 | 30.8% | 81.5% | 44.7% | 54.1% |
| Gold | NLI + reverse probe | 41 | 60 | 13 | 40.6% | 75.9% | 52.9% | 55.8% |
| Gold | **NLI + reverse + hard-negative margin** | **37** | **40** | **17** | **48.1%** | **68.5%** | **56.5%** | **60.0%** |
| LELA | NLI original | 34 | 209 | 20 | 14.0% | 63.0% | 22.9% | 31.1% |
| LELA | NLI + reverse probe | 32 | 102 | 22 | 23.9% | 59.3% | 34.0% | 36.6% |
| LELA | **NLI + reverse + evidence nearby** | **30** | **80** | **24** | **27.3%** | **55.6%** | **36.6%** | **39.3%** |
| Gold-input comparison | Zero-shot, sentence R1 | 32 | 16 | 22 | 66.7% | 59.3% | **62.7%** | 68.6% |
| Gold-input comparison | Zero-shot, paragraph R1 | 27 | 9 | 27 | 75.0% | 50.0% | 60.0% | 62.9% |
| Gold-input comparison | Zero-shot, paragraph R2 | 28 | 8 | 26 | 77.8% | 51.9% | 62.2% | 65.2% |
| Gold | Jev API | 42 | 38 | 12 | 52.5% | 77.8% | **62.7%** | 67.2% |
| Gold | Jev API guided | 47 | 52 | 7 | 47.5% | 87.0% | 61.4% | 66.7% |
| Gold | OpenJev | 46 | 68 | 8 | 40.4% | 85.2% | 54.8% | 59.5% |

**Interpretation.** On this common evaluator, **zero-shot sentence R1** and **Jev API** reach **62.7% F1**, versus **56.5%** for the best experimental gold-entity NLI variant. NLI's hard-negative filter improves precision but reduces recall; Jev API guided achieves the highest recall in the table (**87.0%**). The best LELA-input NLI result (**36.6%**) is **not directly comparable** to gold-input methods because entity recognition/linking errors are included. Jev's separately reported *end-to-end mention/link* pipeline scores are excluded: they measure different tasks.

## 3. Complete NLI improvement results — Gold entities

All policies below are applied to the **same frozen NLI + reverse** predictions; each was tested separately, not in combination.

| Experiment / policy | TP | FP | FN | Precision | Recall | F1 | Pair F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen reverse baseline | 41 | 60 | 13 | 40.59% | 75.93% | 52.90% | 55.84% |
| **Type constraints — conservative** | 41 | 57 | 13 | 41.84% | 75.93% | **53.95%** | 56.95% |
| Hard negatives — `max_negative` | 31 | 39 | 23 | 44.29% | 57.41% | 50.00% | 53.66% |
| **Hard negatives — `forward_minus_negative`** | **37** | **40** | **17** | **48.05%** | **68.52%** | **56.49%** | **60.00%** |
| Hard negatives — `both_rules` | 31 | 39 | 23 | 44.29% | 57.41% | 50.00% | 53.66% |
| Evidence — `both_mentions` | 41 | 60 | 13 | 40.59% | 75.93% | 52.90% | 55.84% |
| Evidence — `nearby` | 38 | 52 | 16 | 42.22% | 70.37% | 52.78% | 55.94% |
| Evidence — `nearby_with_cue` | 17 | 19 | 37 | 47.22% | 31.48% | 37.78% | 42.22% |
| Calibration — pooled leave-one-document-out | 25 | 23 | 29 | 52.08% | 46.30% | 49.02% | 50.98% |

## 4. Complete NLI improvement results — LELA entities

| Experiment / policy | TP | FP | FN | Precision | Recall | F1 | Pair F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen reverse baseline | 32 | 102 | 22 | 23.88% | 59.26% | 34.04% | 36.56% |
| Type constraints — conservative | 30 | 90 | 24 | 25.00% | 55.56% | 34.48% | 37.21% |
| Hard negatives — `max_negative` | 24 | 81 | 30 | 22.86% | 44.44% | 30.19% | 33.12% |
| Hard negatives — `forward_minus_negative` | 28 | 82 | 26 | 25.45% | 51.85% | 34.15% | 37.04% |
| Hard negatives — `both_rules` | 24 | 81 | 30 | 22.86% | 44.44% | 30.19% | 33.12% |
| Evidence — `both_mentions` | 32 | 102 | 22 | 23.88% | 59.26% | 34.04% | 36.56% |
| **Evidence — `nearby`** | **30** | **80** | **24** | **27.27%** | **55.56%** | **36.59%** | **39.26%** |
| Evidence — `nearby_with_cue` | 13 | 36 | 41 | 26.53% | 24.07% | 25.24% | 29.13% |
| Calibration — pooled leave-one-document-out | 12 | 17 | 42 | 41.38% | 22.22% | 28.92% | 31.33% |

## 5. What each experiment tested

| Experiment | Mechanism / parameters | Gold result | LELA result | Conclusion |
|---|---|---|---|---|
| **Reverse probe** (`11_nli_error_driven`) | Compare forward and reversed entity-role entailment; keep directional predictions with score margin ≥ **0.20**. | F1 **44.67 → 52.90**; 39 FP removed, 3 TP lost. | F1 **22.90 → 34.04**; 107 FP removed, 2 TP lost. | Most substantial improvement over original NLI for both input types. |
| **Type constraints** (`14/type_constraints.py`) | Reject type-incompatible relations; preserve unknown types. | F1 **52.90 → 53.95**; −3 FP, **no additional TP loss**. | F1 **34.04 → 34.48**; −12 FP, −2 TP. | Safe on gold types; automatic LELA typing can reject true relations. |
| **Hard negatives** (`14/hard_negatives.py`) | Run NLI on alternative interpretations for five relations: `schema:about`, `yago:partOf`, `schema:worksFor`, `schema:location`, `yago:creator`. Test negative score ≤0.50, forward–negative margin ≥0.20, or both. | Best policy: F1 **56.49**; −20 FP, −4 TP. | Best policy: F1 **34.15**; −20 FP, −4 TP. | **Highest gold F1**, but recall decreases. |
| **Evidence filtering** (`14/evidence.py`) | Check both literal mentions, entity gap ≤ **120 characters**, optionally require relation-specific lexical cue. | Best is no added filter; `nearby` F1 **52.78** and cues F1 **37.78**. | `nearby` reaches **36.59 F1**; cues fall to **25.24**. | **Highest LELA F1** using proximity only; strict cue matching overfilters. |
| **Calibration** (`14/calibration.py`) | Class-balanced logistic regression on NLI, reverse, types, distance and lexical features; train on one article, evaluate on the other; pool out-of-fold predictions. | F1 **49.02**, down from 52.90. | F1 **28.92**, down from 34.04. | More precision, much lower recall; unstable across the two articles. |

### Calibration by held-out article

| Input | Test article | Baseline P / R / F1 | Calibrated P / R / F1 |
|---|---|---|---|
| Gold | `alford2026` | 46.51 / 71.43 / **56.34%** | 52.94 / 32.14 / **40.00%** |
| Gold | `nechaeva2017` | 36.21 / 80.77 / **50.00%** | 51.61 / 61.54 / **56.14%** |
| LELA | `alford2026` | 26.42 / 50.00 / **34.57%** | 33.33 / 14.29 / **20.00%** |
| LELA | `nechaeva2017` | 22.22 / 69.23 / **33.64%** | 47.06 / 30.77 / **37.21%** |

## 6. Earlier NLI ablations (before reverse filtering)

| Tested approach | Gold fact F1 | Result |
|---|---:|---|
| Sentence → sentence, ensemble hypothesis | **44.7%** | Best of the listed initial context/template settings. |
| Sentence → sentence, baseline template | 43.2% | Slightly weaker. |
| Sentence → paragraph, ensemble | 34.3% | Worse context configuration. |
| Sentence → paragraph, baseline | 33.2% | Worse context configuration. |
| Sentence → sentence, alternative template | 32.6% | Worse template. |
| Entity-marked sentence → sentence, baseline | 43.9% | Did not beat original baseline template. |
| Entity-marked sentence → sentence, ensemble | 41.3% | Worse than unmarked ensemble. |

The broader `09_nli_extended` script evaluated **44 variants**; the table above lists the directly reported representative outcomes. This is not a claim that all 44 variants are reproduced here.

## 7. Conclusions

1. **Best observed gold-entity NLI:** reverse + hard-negative margin, **P 48.05%, R 68.52%, F1 56.49%**. Relative to the frozen reverse baseline it removes 20 false positives but loses 4 true positives.
2. **Best observed LELA-input NLI:** reverse + evidence proximity, **P 27.27%, R 55.56%, F1 36.59%**. It removes 22 false positives but loses 2 true positives.
3. **No-recall-loss option for gold:** conservative type constraints, **F1 53.95%** at unchanged **75.93% recall**.
4. **Cross-method comparison:** Jev API and zero-shot sentence R1 have **62.7% F1** in the common evaluation, ahead of the best experimental gold NLI (**56.5%**). Different entity-input conditions and end-to-end evaluation scores must not be conflated.
5. **Limitation:** all NLI rule choices were explored using these same two articles. The differences are **development-set observations, not held-out generalization claims**. No combined filters have yet been evaluated, and the reported F1 gains should not be described as statistically significant without a targeted uncertainty test and external validation.

**Sources:** `results.zip` (types, hard negatives, evidence, calibration comparison files); `experiments/13_nli_evaluation/evaluation.py` evaluation output for original NLI, Jev and zero-shot; `experiments/09_nli_extended/compare_all.py` and `experiments/10_nli_precision/compare_focused.py` outputs for earlier NLI ablations.
