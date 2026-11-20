# NLI Relation Extraction — Experimental Report

**Date:** 20 November 2026  
**Project:** Historical relation extraction from research articles  
**Evaluation type:** Development-set experiments; **not** an independent held-out benchmark

## 1. Objective

This study investigates ways to improve a Natural Language Inference (NLI) approach for extracting relations between historical entities. The primary problem observed in the original system was **overprediction**: plausible or co-occurring entity pairs were frequently assigned relations not supported by the gold annotations. The new experiments were designed to reduce false positives while tracking the resulting loss of true positives.

Four additions to a frozen reverse-probe NLI baseline were tested:

1. **Entity-type constraints:** discard relations that violate conservative subject/object type requirements.
2. **Relation-specific hard negatives:** contrast a supported-relation hypothesis with a plausible alternative interpretation.
3. **Evidence-based filtering:** inspect local text proximity and relation-related lexical cues.
4. **Learned score calibration:** train a logistic classifier on existing NLI and auxiliary features, using document-held-out evaluation.

All values in this report come from the supplied `results.zip` comparison files, except the *original NLI* reference values, which come from the previously reproduced evaluation logs. No untested combinations are presented as measured results.

## 2. Dataset and evaluation protocol

| Setting | Specification |
|---|---|
| Annotated articles | `alford2026` and `nechaeva2017` |
| Annotated units | 14 paragraphs |
| Gold relations | 54 facts (28 in Alford; 26 in Nechaeva) |
| Entity inputs | **Gold**: manually annotated entities; **LELA**: automatically generated entity mentions/linking |
| NLI context | Sentence → sentence |
| NLI score aggregation | Maximum of two hypothesis entailment scores (ensemble) |
| Forward acceptance threshold | 0.90 |
| Frozen reverse-probe margin | Forward − reverse ≥ 0.20 for probed relations |
| Relation selection | One selected relation per entity pair in the existing NLI prediction pipeline |
| Evaluator | Repository's `archive/eval/evaluate.py` |
| Reporting | Micro-averaged fact-level precision, recall, F1; fact-pair F1; TP, FP, FN |
| Uncertainty | Paragraph-bootstrap intervals saved in experiment CSVs (default 100 replicates) |

Every experiment starts from the **same frozen NLI + reverse-probe baseline**. This isolates the effect of each additional rule. For LELA, `selected_jobs` in the output can be larger than the number of *deduplicated predicted facts* counted by the evaluator; all TP/FP and scores below use the evaluator's `pred` count.

**Metric definitions:** Precision = TP/(TP + FP); Recall = TP/(TP + FN); F1 = 2PR/(P + R). All experiments compare predictions against the same 54 annotated facts. *Fact-pair F1* evaluates predicted entity pairs while disregarding the relation label; it is not interchangeable with fact-level F1.

**Critical limitation:** these two articles were already used for exploratory choices, including the reverse-probe margin and subsequent heuristics. Consequently, this report describes development-set performance and ablations, not unbiased generalization to unseen articles. The bootstrap intervals do not remove that selection bias.

## 3. Reference baseline: original NLI and reverse probe

The original NLI system accepted high-entailment hypotheses. A reverse-probe stage then evaluated hypotheses with subject and object roles exchanged and filtered relations with insufficient forward–reverse contrast.

| Entity source | Configuration | TP | FP | FN | Precision | Recall | Fact F1 | Fact-pair F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gold | Original NLI | 44 | 99 | 10 | 30.77% | 81.48% | 44.67% | 54.1% |
| Gold | **Frozen NLI + reverse** | 41 | 60 | 13 | 40.59% | 75.93% | **52.90%** | 55.84% |
| LELA | Original NLI | 34 | 209 | 20 | 13.99% | 62.96% | 22.90% | 31.1% |
| LELA | **Frozen NLI + reverse** | 32 | 102 | 22 | 23.88% | 59.26% | **34.04%** | 36.56% |

The reverse probe removed **39 false positives and 3 true positives** for gold, and **107 false positives and 2 true positives** for LELA. These changes motivate the more selective follow-up experiments below.

## 4. Experiment 1 — Entity-type constraints

### 4.1 Hypothesis

A portion of NLI's false positives may be eliminated by enforcing basic semantic compatibility between the **subject type**, **relation**, and **object type**, without rerunning the language model.

### 4.2 Method

The script `type_constraints.py` categorizes available entity types into `person`, `location`, `organization`, `work`, or `unknown`. It applies conservative relation-specific restrictions, including:

| Relation | Required subject type | Required object type |
|---|---|---|
| `ex:captiveIn` | person | location |
| `schema:birthPlace`, `schema:deathPlace`, `yago:burialPlace` | person | location |
| `yago:hasMother`, `yago:hasFather`, `schema:spouse`, `yago:studentOf` | person | person |
| `schema:worksFor` | person | person or organization |
| `yago:creator` | work | person |
| `schema:about` | work | unrestricted |

The `yago:creator` subject/object requirements reflect the **specific direction used in this corpus**: the work is the subject and its author is the object. Unknown entity types are retained, not rejected. Predictions for relations without an explicit restriction remain unchanged. Gold **relation labels** are not used to decide which facts pass the rules.

### 4.3 Results

| Entity source | Policy | TP | FP | FN | Precision | Recall | F1 | Fact-pair F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gold | Frozen baseline | 41 | 60 | 13 | 40.59% | 75.93% | 52.90% | 55.84% |
| Gold | **Conservative types** | **41** | **57** | **13** | **41.84%** | **75.93%** | **53.95%** | **56.95%** |
| LELA | Frozen baseline | 32 | 102 | 22 | 23.88% | 59.26% | 34.04% | 36.56% |
| LELA | **Conservative types** | **30** | **90** | **24** | **25.00%** | **55.56%** | **34.48%** | **37.21%** |

### 4.4 Interpretation

For **gold entities**, the constraints removed **three false positives and no true positives**. Precision and F1 improved while recall remained unchanged. This is the cleanest observed improvement without an additional recall cost in the gold setting.

For **LELA**, the rules removed 12 false positives **but also two true positives**. One recorded rejection was a valid creator relation involving *Ecclesiastical Histories* and *John of Ephesus*: the automatic type metadata incorrectly described both entities as literary works. Thus, type constraints are only as reliable as their upstream entity typing. The F1 gain for LELA was small (+0.44 percentage points).

## 5. Experiment 2 — Relation-specific hard-negative hypotheses

### 5.1 Hypothesis

High entailment for a proposed relation does not prove that a more mundane interpretation is excluded. A secondary NLI hypothesis describing a **plausible competing interpretation** may help reject unsupported relations.

### 5.2 Method

The script `hard_negatives.py` creates one alternative hypothesis for each selected candidate belonging to five targeted relations:

| Relation | Alternative hypothesis template |
|---|---|
| `schema:about` | “{s} merely mentions {o} without being primarily about {o}.” |
| `yago:partOf` | “{s} is associated with {o} but is not a constituent part of {o}.” |
| `schema:worksFor` | “{s} met or opposed {o}, but did not serve {o}.” |
| `schema:location` | “{s} travelled through {o} without being based in {o}.” |
| `yago:creator` | “{o} was mentioned in {s} but did not author {s}.” |

For these candidates, a fresh NLI inference computes the alternative-hypothesis entailment score. Other relation types pass through unchanged. **66 alternative probes** were scored for gold and **78** for LELA.

Three acceptance policies were evaluated on top of the frozen baseline:

- **`max_negative`**: keep if alternative entailment ≤ 0.50.
- **`forward_minus_negative`**: keep if forward entailment − alternative entailment ≥ 0.20.
- **`both_rules`**: require both conditions.

These alternatives are not guaranteed to be logical negations: historical claims may coexist with the alternatives. Therefore, alternative entailment is treated as a heuristic discriminative signal, **not** as a calibrated probability that the original fact is false.

### 5.3 Results — Gold entities

| Policy | TP | FP | FN | Precision | Recall | F1 | Fact-pair F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 41 | 60 | 13 | 40.59% | 75.93% | 52.90% | 55.84% |
| `max_negative` | 31 | 39 | 23 | 44.29% | 57.41% | 50.00% | 53.66% |
| **`forward_minus_negative`** | **37** | **40** | **17** | **48.05%** | **68.52%** | **56.49%** | **60.00%** |
| `both_rules` | 31 | 39 | 23 | 44.29% | 57.41% | 50.00% | 53.66% |

### 5.4 Results — LELA entities

| Policy | TP | FP | FN | Precision | Recall | F1 | Fact-pair F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 32 | 102 | 22 | 23.88% | 59.26% | 34.04% | 36.56% |
| `max_negative` | 24 | 81 | 30 | 22.86% | 44.44% | 30.19% | 33.12% |
| **`forward_minus_negative`** | **28** | **82** | **26** | **25.45%** | **51.85%** | **34.15%** | **37.04%** |
| `both_rules` | 24 | 81 | 30 | 22.86% | 44.44% | 30.19% | 33.12% |

### 5.5 Interpretation

**Gold:** `forward_minus_negative` produced the highest F1 observed among the four follow-up experiments: **56.49%**, up **3.59 percentage points** from the frozen baseline. It removed **20 false positives** but lost **four true positives**. Precision increased by **7.46 points** while recall decreased by **7.41 points**. The harder `max_negative` and `both_rules` criteria were overly aggressive, reducing F1 to 50.00%.

**LELA:** The same margin produced only a marginal F1 improvement (34.04% → 34.15%), with a substantial decrease in recall. This suggests that filtering relation interpretations alone does not resolve noisy automatically generated entities or incorrect entity links.

## 6. Experiment 3 — Evidence-based filtering

### 6.1 Hypothesis

Unsupported predictions might correlate with weak local textual evidence: missing literal entity mentions, large separation between entity names, or absence of relation-associated words.

### 6.2 Method

The script `evidence.py` inspects each frozen-baseline prediction. It searches for **literal case-insensitive occurrences of both entity surface forms** in its NLI text and calculates the shortest character gap. It also extracts a local snippet and checks relation-specific lexical patterns (e.g., “served” for `schema:worksFor`, “captive” for `ex:captiveIn`, “authored” for `yago:creator`).

Three policies were tested:

1. **`both_mentions`** — require both entity strings to occur.
2. **`nearby`** — require both strings and a minimum character gap of no more than **120**.
3. **`nearby_with_cue`** — apply the distance condition and additionally require a lexical cue for relations with a defined cue pattern.

These are **heuristic proxies for evidence**, not a validated entailment or evidence-span extraction model. Literal matching may miss coreference, aliases, paraphrases, or cross-sentence facts.

### 6.3 Results — Gold entities

| Policy | TP | FP | FN | Precision | Recall | F1 | Fact-pair F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 41 | 60 | 13 | 40.59% | 75.93% | 52.90% | 55.84% |
| `both_mentions` | 41 | 60 | 13 | 40.59% | 75.93% | 52.90% | 55.84% |
| `nearby` | 38 | 52 | 16 | 42.22% | 70.37% | 52.78% | 55.94% |
| `nearby_with_cue` | 17 | 19 | 37 | 47.22% | 31.48% | 37.78% | 42.22% |

### 6.4 Results — LELA entities

| Policy | TP | FP | FN | Precision | Recall | F1 | Fact-pair F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 32 | 102 | 22 | 23.88% | 59.26% | 34.04% | 36.56% |
| `both_mentions` | 32 | 102 | 22 | 23.88% | 59.26% | 34.04% | 36.56% |
| **`nearby`** | **30** | **80** | **24** | **27.27%** | **55.56%** | **36.59%** | **39.26%** |
| `nearby_with_cue` | 13 | 36 | 41 | 26.53% | 24.07% | 25.24% | 29.13% |

### 6.5 Interpretation

Requiring both literal surface forms did **not** change the predicted fact set in either setting. The **120-character proximity rule** improved LELA F1 by **2.55 percentage points**, removing **22 false positives** and losing **two true positives**. On gold entities, however, it removed eight false positives and three true positives, causing a slight F1 decline (52.90% → 52.78%).

Adding mandatory lexical cues was detrimental: gold recall dropped to 31.48%, and LELA recall to 24.07%. Relation meaning is expressed in many ways that short cue lists do not capture. Therefore, the lexical-cue policy should **not** be adopted as a global filter.

## 7. Experiment 4 — Supervised score calibration

### 7.1 Hypothesis

A lightweight classifier may combine NLI scores, reverse-probe confidence, relation identity, semantic types, and local evidence more effectively than manually specified thresholds.

### 7.2 Method

The script `calibration.py` trains a `StandardScaler` + class-balanced `LogisticRegression` model (`C = 1`, `max_iter = 3000`, fixed probability threshold **0.50**). Features include forward and alternative-hypothesis scores, disagreement between hypotheses, reverse score and margin, missing-reverse indicator, mention presence, normalized entity distance, lexical cue, type-constraint compatibility, one-hot relation identity, and one-hot subject/object categories.

**Document-held-out procedure:** train on `alford2026`, test on `nechaeva2017`; then train on `nechaeva2017`, test on `alford2026`. The **pooled out-of-fold** results combine predictions generated by models that did not train on their respective test articles. Gold relation labels are used **for classifier training on the training document only**, never to train on the corresponding test document. Gold and LELA are modeled separately.

This provides a stronger cross-document check than fitting and evaluating on the same article, but there are only **two** available articles, and the features and broader research direction were developed on this corpus. It is not a fully independent benchmark.

### 7.3 Results — Gold entities

| Test document | Policy | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Alford (train: Nechaeva) | Frozen baseline | 20 | 23 | 8 | 46.51% | 71.43% | 56.34% |
| Alford (train: Nechaeva) | Calibrated | 9 | 8 | 19 | 52.94% | 32.14% | 40.00% |
| Nechaeva (train: Alford) | Frozen baseline | 21 | 37 | 5 | 36.21% | 80.77% | 50.00% |
| Nechaeva (train: Alford) | Calibrated | 16 | 15 | 10 | 51.61% | 61.54% | 56.14% |
| **Pooled out-of-fold** | **Frozen baseline** | **41** | **60** | **13** | **40.59%** | **75.93%** | **52.90%** |
| **Pooled out-of-fold** | **Calibrated** | **25** | **23** | **29** | **52.08%** | **46.30%** | **49.02%** |

### 7.4 Results — LELA entities

| Test document | Policy | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Alford (train: Nechaeva) | Frozen baseline | 14 | 39 | 14 | 26.42% | 50.00% | 34.57% |
| Alford (train: Nechaeva) | Calibrated | 4 | 8 | 24 | 33.33% | 14.29% | 20.00% |
| Nechaeva (train: Alford) | Frozen baseline | 18 | 63 | 8 | 22.22% | 69.23% | 33.64% |
| Nechaeva (train: Alford) | Calibrated | 8 | 9 | 18 | 47.06% | 30.77% | 37.21% |
| **Pooled out-of-fold** | **Frozen baseline** | **32** | **102** | **22** | **23.88%** | **59.26%** | **34.04%** |
| **Pooled out-of-fold** | **Calibrated** | **12** | **17** | **42** | **41.38%** | **22.22%** | **28.92%** |

### 7.5 Interpretation

Calibration improved precision in both entity settings but became **too conservative**, significantly lowering recall. On gold, pooled F1 fell from 52.90% to **49.02%**; on LELA, from 34.04% to **28.92%**. Performance also differed sharply by held-out article: calibration improved Nechaeva F1 but degraded Alford F1 in both settings.

The experiment therefore does **not** justify replacing the rule-based baseline with learned calibration. A larger set of independently annotated documents and better external validation would be needed before relying on a supervised meta-classifier.

## 8. Consolidated results and trade-offs

The table below compares each **best-performing policy within its experiment**, along with the common frozen baseline. “Best” means best *observed F1 in these development files*, not a statistically validated model choice.

| Entity input | Method | TP | FP | FN | Precision | Recall | Fact F1 | Δ F1 vs frozen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gold | Frozen reverse baseline | 41 | 60 | 13 | 40.59% | 75.93% | 52.90% | — |
| Gold | Type constraints | 41 | 57 | 13 | 41.84% | 75.93% | 53.95% | +1.04 pp |
| Gold | **Hard-negative margin** | **37** | **40** | **17** | **48.05%** | **68.52%** | **56.49%** | **+3.59 pp** |
| Gold | Evidence nearby | 38 | 52 | 16 | 42.22% | 70.37% | 52.78% | −0.13 pp |
| Gold | Cross-document calibration | 25 | 23 | 29 | 52.08% | 46.30% | 49.02% | −3.88 pp |
| LELA | Frozen reverse baseline | 32 | 102 | 22 | 23.88% | 59.26% | 34.04% | — |
| LELA | Type constraints | 30 | 90 | 24 | 25.00% | 55.56% | 34.48% | +0.44 pp |
| LELA | Hard-negative margin | 28 | 82 | 26 | 25.45% | 51.85% | 34.15% | +0.10 pp |
| LELA | **Evidence nearby** | **30** | **80** | **24** | **27.27%** | **55.56%** | **36.59%** | **+2.54 pp** |
| LELA | Cross-document calibration | 12 | 17 | 42 | 41.38% | 22.22% | 28.92% | −5.13 pp |

**Precision–recall trade-off:** All improvements to the best observed F1 except **gold type constraints** came with lower recall. The higher F1 values reflect the removal of more false positives than true positives, **not** an improvement on every metric. A research application prioritizing a complete historical knowledge graph may prefer a configuration with higher recall, even at lower F1.

### Uncertainty note

Each experiment's comparison CSV includes `ci_low` and `ci_high` for fact F1 from a paragraph bootstrap (100 repetitions in the default scripts). Since the number of annotated paragraphs is small and the models/rules were explored on these same articles, these intervals should be treated as **descriptive within-corpus uncertainty** rather than evidence of superiority on unseen documents. In particular, this report does **not** claim that the +3.59-point hard-negative improvement is statistically significant.

## 9. Key findings

1. **Reverse-role contrast is the primary demonstrated improvement over original NLI.** It reduces overprediction substantially for both gold and LELA entity inputs.
2. **Hard-negative margin is the best observed gold configuration** (F1 = **56.49%**), at the cost of recall declining from 75.93% to 68.52% relative to the frozen baseline.
3. **Conservative type constraints are useful with reliable entity annotations.** In gold, they remove three false positives with no additional true-positive loss. Their benefit is weaker for LELA because incorrectly assigned types can suppress valid relations.
4. **Simple evidence proximity is useful for LELA** (F1 = **36.59%**) but not for gold. Mandatory lexical cues overfilter both inputs.
5. **Learned calibration does not generalize consistently across the two articles.** It raises precision but loses too much recall and reduces pooled F1.
6. **Gold and LELA need to be reported separately.** Their entity inputs have different error profiles; using gold entities is not an end-to-end performance estimate.

## 10. Recommended next steps

- **Freeze an interpretable baseline** for independent evaluation. Keep `sentence_sentence + ensemble + threshold 0.90 + reverse Δ = 0.20` as the common point of comparison.
- **Evaluate on newly annotated articles not used for development.** Select any proposed additional policy *before* looking at their gold labels.
- **Audit removed true positives, particularly hard negatives and LELA type mismatches.** Determine whether the lost facts involve subtle relation meanings, mistaken type metadata, long-distance evidence, or gold annotation gaps.
- **Test combined filters as a new, explicitly labeled exploratory ablation:** reverse + hard negatives + types, reverse + hard negatives + proximity, and selected per-relation variants. **No combined-filter F1 has been measured in the supplied files.**
- **Compare against zero-shot/Jev under the same evaluator and entity-input conditions** when reporting cross-method rankings. Keep end-to-end mention/link evaluation distinct from gold-entity fact extraction.

## 11. Reproducibility and source files

The analysis is based on the uploaded `results.zip`, particularly:

```text
results/types/{gold,lela}_comparison.csv
results/types/{gold,lela}_rejected.csv
results/hard_negatives/{gold,lela}_comparison.csv
results/hard_negatives/{gold,lela}_scores.json
results/hard_negatives/{gold,lela}_manifest.json
results/evidence/{gold,lela}_comparison.csv
results/evidence/{gold,lela}_evidence_audit.csv
results/calibration/{gold,lela}_comparison.csv
results/calibration/{gold,lela}_prediction_audit.csv
```

Experiment implementations: `experiments/14_nli_improvements/{shared.py,type_constraints.py,hard_negatives.py,evidence.py,calibration.py}`. These scripts use the common repository evaluator `archive/eval/evaluate.py` and precomputed NLI/reverse score caches from experiments 09 and 11. The frozen baseline evaluation was reproduced separately in experiment 13.

**Bottom line:** The highest observed development-set fact F1 is **56.49% with gold entities (hard-negative margin)** and **36.59% with LELA entities (evidence proximity)**. These are **different configurations**. Neither has yet been validated on an independent held-out article set.
