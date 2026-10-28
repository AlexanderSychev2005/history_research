# Experiment 02: automatic Unicode normalisation

## Question
Does removing diacritics and transliteration marks (Łazar P‘arpec‘i → Lazar P'arpec'i) fix the five linking errors of
experiment 01, without using the gold names?

## Setup
Same LELA configuration and paragraph as experiment 01. The input (`inputs/alford2026.txt`) is the paragraph with the
five names normalised; the predictions are in `predictions/alford2026.jsonl`.

## Results (`results.csv`)
0/5 correct, the same as without normalisation. Gorgān and Vałaršapat now link to other wrong entities (Gorgan,
Alarsa).

## Reading
The scholarly transliteration differs from the YAGO/Wikipedia spelling in more than diacritics: Łazar is "Ghazar",
Vałaršapat is "Vagharshapat", Gorgān is listed under Hyrcania. A mapping between transliteration schemes (ł → gh,
-ean → -ian, …) or an alias list is needed, not just stripping marks.
