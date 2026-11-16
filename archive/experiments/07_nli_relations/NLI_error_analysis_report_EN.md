# NLI Error Analysis Report

## Objective

Analyze errors in the NLI-based approach to relation extraction from historical texts and identify the reasons for its relatively low recall.

## Current Results

The best tested configuration, using a threshold of **0.99**, achieved:

| Metric | Value |
|---|---:|
| Precision | 54.8% |
| Recall | 31.5% |
| F1-score | 40.0% |

These results come from a small dataset of 14 paragraphs. Configurations were evaluated on the same data used for experimentation, so the estimates are preliminary.

## Error Analysis

I examined **54 gold relations**. Diagnostic analysis of the saved scores produced the following breakdown:

- **17** — correctly selected according to the saved scores;
- **15** — the correct relation ranked highest but fell below the 0.99 threshold;
- **19** — another relation received a higher score than the correct one;
- **3** — the correct relation was missing from the candidate options.

This is a diagnostic categorization, not a separate official model evaluation.

## Main Observations

The most frequent issues involved **`schema:location`**. In particular, **`yago:creator`** received a higher score in six diagnosed cases. Reviewing the original historical paragraphs suggested that an NLI hypothesis such as *“is located in”* may not adequately capture historical relationships involving a person's origin or residence.

## Identified Issue with Type Constraints

In `type_constraints_captive.py`, type filtering is implemented only for **`yago:partOf`**. For other relations, the code does not check whether the subject and object satisfy the schema's `domain` and `range` constraints.

For example, for **Damascius (Person) → Sasanian Empire (Place)**, the model assigns a higher score to `yago:creator` than to `schema:location`, even though the entity types do not satisfy the constraints for `yago:creator`. However, fixing the type filter alone may not recover this relation because the correct relation's score is still below the threshold.

## Next Steps

1. Implement `domain` and `range` checks for all relations, mapping NER labels to schema types appropriately.
2. Re-evaluate precision, recall, and F1 using the saved NLI scores, without rerunning the model.
3. Test alternative NLI hypothesis formulations tailored to historical texts.
4. Compare the results and evaluate them separately on independent data.

## Conclusion

Lowering the threshold alone will not resolve all errors: some correct relations are outranked by other candidates, and the current filtering does not enforce all schema constraints. The next priorities are to test comprehensive type constraints and improve hypothesis wording.
