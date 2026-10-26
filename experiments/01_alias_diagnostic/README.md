# Experiment 01: Historical Alias Diagnostic

## Research question
Do historical spelling variants contribute to incorrect entity linking in LELA?

## Configuration
- NER: NuNER Zero-span, threshold 0.3
- Candidate generation: fuzzy, top_k=20
- Reranker: none
- Disambiguator: first
- Knowledge base: YAGO 4.5
- Source: Alford 2026, paragraph p004

## Method
Compare LELA predictions on the original paragraph and a modified
copy in which five historical mentions are replaced with canonical
YAGO names. The same LELA configuration is used for both runs.

## Results
- Original historical mentions: 0/5 correctly linked
- Canonical replacements: 5/5 correctly linked
- All five replacement mentions were detected and linked to the
  expected YAGO entity.

## Interpretation
The results support the hypothesis that historical spelling variants
contribute to entity-linking failures.

## Limitations
This is an oracle-assisted diagnostic experiment: replacement names
were chosen using known gold entities. It does not demonstrate
generalizable improvement on unseen texts.

## Next step
Test automatic normalization without using gold entity names.
