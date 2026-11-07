# Experiment 01: historical name variants (alias diagnostic)

## Question
Do historical spellings cause LELA's linking errors?

## Setup
LELA with NuNER Zero-span (threshold 0.3), fuzzy candidates (top 20), no reranker, `first` disambiguator, YAGO 4.5.
Alford 2026, paragraph p004. The paragraph was run as it stands (`predictions/baseline_alford.jsonl`), and again
with five historical names replaced by their canonical YAGO names (`alias_input.txt`,
`predictions/alias_alford.jsonl`).

## Results (`results.csv`)

| mention in the paper | replaced by | original text links to | replaced text links to |
|---|---|---|---|
| Łazar P‘arpec‘i | Ghazar Parpetsi | Azar (wrong) | Ghazar_Parpetsi |
| Gorgān | Hyrcania | Go_GO_GO_Go (wrong) | Hyrcania |
| Sasanians | Sasanian Empire | Asan (wrong) | Sasanian_Empire |
| Vałaršapat | Vagharshapat | Apate (wrong) | Vagharshapat |
| Eǰmiacin | Vagharshapat | Emilcin (wrong) | Vagharshapat |

0/5 correct with the original names, 5/5 with the canonical ones.

## Reading and limits
Spelling variants account for these failures. The test uses an oracle: the replacements were chosen from the gold
entities, so it shows the size of the problem, not a method. Experiment 02 tries an automatic normalisation (removing
diacritics); it fixes none of the five.
