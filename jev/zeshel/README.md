# Jev on ZESHEL, Forgotten Realms

[`run.py`](run.py) runs Jev (`jev-1.14.0`) on [`dataset/zeshel_realms/`](../../dataset/zeshel_realms/README.md): the 1,200 test
mentions of the Forgotten Realms world with ZESHEL's 64 BM25 candidates. It follows LELA's protocol (arXiv 2601.05192, Table 3).

## Result
Normalized accuracy: on the 1,000 mentions whose gold entity is among the candidates.

| | normalized | unnormalized (all 1,200) |
|---|---|---|
| **Jev, Choice over the 64 candidates** | **86.70 ± 2.1** | 72.25 |
| Jev, Choice then a Noul on its top 10 | 86.60 ± 2.1 | 72.17 |
| LELA, Magistral-Small-2509 | 88.69 | |
| BLINK | 86.80 | |
| LELA, Qwen3-30B-A3B | 85.69 | |
| BM25 first candidate | 52.40 | 43.67 |

By ZESHEL category (normalized, Choice): high overlap 94.8, multiple categories 89.6, ambiguous substring 90.9, low overlap 81.6.
For the full table and the other systems see [`dataset/zeshel_realms/README.md`](../../dataset/zeshel_realms/README.md).

* **Cost and time:** 13,107 questions, 11.0 million input tokens, **$0.46**. Jev took 60 s for the Choice round and 51 s for the Noul
  round. LELA's Table 8 gives 26 to 44 s per mention for its LLM variants on ZESHEL.
* **The top 10 of the Choice holds the gold for 990 of the 1,000 mentions.** The errors are in the final pick, not in the
  ranking. The per-candidate Noul (score = √(which × is)) does not help: it gains one low-overlap mention and loses two.
* Typical errors: a related entry instead of the entity ("Pool of Radiance" goes to the town Phlan, not the novel; "imprisoned" goes
  to a prisoner, not Imprisonment). Another set are names the gold resolves through the wiki's redirects ("Misty Lake" is Lake of
  Mists; "Ravenloft" is the Demiplane of Dread).

## How it works
* Context: 80 words on each side of the mention, with the mention marked ⟦ ⟧.
* Entity: title + the first three sentences of its page, cut at 700 characters (LELA's description).
* Round 1: one Choice whose options are the 64 titles, each with its description. It is asked in both option orders, because Jev
  leans to the first option.
* Round 2: a Noul for each of the top 10 ("does ⟦mention⟧ refer to the entry X: description?"). The questions have general
  wording, with no examples from Forgotten Realms or from the history papers.
* No threshold and no "none" option, as in LELA's ZESHEL setup: the best candidate is always the prediction.

## Run it
```bash
python jev/zeshel/run.py
```
```bash
python dataset/zeshel_realms/score.py jev/zeshel/pred/choice.jsonl
```
`--n 100` runs a random sample of 100 mentions with the gold among the candidates. `JEV_API_KEY` must be in `.env`. Answers are
cached in `jev/cache/`, so a rerun asks nothing.

## Caveats
* One run, one world: LELA's headline number is the macro average over four test worlds (Magistral 83.11).
* The questions were written before this run and not changed after it. A 100-mention sample of the same mentions was run first
  (89.0% Choice, 87.0% with the Noul) with the same questions.
* The mention contexts and entity descriptions of ZESHEL were sent to TypeSafe.
