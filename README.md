# History research: entities and relations from history papers

Extracting named entities, links to YAGO 4.5 and relations from late-antique history papers, and measuring how well
different methods do it.

## Where things are

| folder | what |
|---|---|
| [`dataset/`](dataset/README.md) | **the shared test set**: 14 paragraphs from two papers (with the PDFs) and one gold set for mentions, entity linking to YAGO 4.5 and relations |
| [`archive/`](archive/README.md) | everything done before, frozen: text preparation, the LELA pilot, the colleagues' pipelines (NER, Jev, NLI, OpenIE), the earlier gold, all experiments and their results |

New work uses `dataset/` only. The gold is a draft by one annotator (Claude) and has not been reviewed by a historian.
Report paragraph `p004` of both papers separately: earlier methods were developed on it.

## At a glance (best earlier results, full table in `archive/README_RESULTS.md`)

| stage | best approach | F1 % on the 14 paragraphs |
|---|---|---|
| mentions | Jev NER (Max) | 97.2 (94.0 strict) |
| entity linking to YAGO | LLM names + YAGO shortlist + LLM choice (Alex) | 62 (58); 53 (47) on the 12 paragraphs not used for development |
| relations, gold mentions | Jev API guided (Max), threshold tuned on this gold | 74.3 |
| relations, zero-shot constrained LLM (Alex), no threshold | per sentence | 62.7 |
| relations, NLI (Lili) | threshold tuned on this gold | 40.0 |
