# Experiment 03: where LELA's linking fails

## Question
When LELA links a pilot mention wrongly, was the right entity among its candidates, i.e. was it a retrieval or a
ranking failure?

## Data
The 35 gold mentions of the pilot (`pilot/gold.json`, one paragraph per paper), with LELA's prediction, whether the
gold entity was among the 20 fuzzy candidates, and its rank (`results.csv`). Columns: `correct`,
`gold_in_candidates`, `gold_rank`, `candidate_count`, `detected`.

## Reading
* Most Alford errors are retrieval failures: the gold entity is not among the candidates (Łazar P‘arpec‘i, Łazar,
  Gorgān, Khuzestan, Vahan, Eǰmiacin). This matches experiments 01–02.
* The others are ranking failures: the gold entity is among the candidates but not first (Alford: Vahan Mamikonean
  and Sasanians at rank 6, Vałaršapat at 20; Nechaeva: Simplicius 13, Priscian 3, Diogenes 11).
* Persia and Romans (Nechaeva) are retrieval failures too: neither the Sasanian nor the Byzantine Empire is found.
* Gold-NIL mentions (Syria the region, Hermes, Isidore, History of the Armenians) are always linked to something.
