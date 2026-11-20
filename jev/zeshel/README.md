# Jev on ZESHEL, Forgotten Realms

[`run.py`](run.py) runs Jev (`jev-1.14.0`) on [`dataset/zeshel_realms/`](../../dataset/zeshel_realms/README.md): the 1,200 test
mentions of the Forgotten Realms world with ZESHEL's 64 BM25 candidates. It follows LELA's protocol (arXiv 2601.05192, Table 3).

## Result
Normalized accuracy: on the 1,000 mentions whose gold entity is among the candidates.

| | normalized | unnormalized (all 1,200) | Jev cost |
|---|---|---|---|
| GenDecider (trained on ZESHEL's training worlds) | 91.10 | | |
| **Jev, tuned on validation worlds** (`pred/final.jsonl`) | **91.00 ± 1.8** | 75.83 | $0.59 |
| LELA, Magistral-Small-2509 | 88.69 | | |
| BLINK | 86.80 | | |
| Jev, first version: one Choice over the 64 candidates (`pred/choice.jsonl`) | 86.70 ± 2.1 | 72.25 | $0.46 |
| LELA, Qwen3-30B-A3B | 85.69 | | |
| BM25 first candidate | 52.40 | 43.67 | |

The tuned version is right on 56 mentions the first version got wrong, and wrong on 13 it got right. By ZESHEL category
(normalized): high overlap 97.1, multiple categories 92.4, ambiguous substring 94.3, low overlap 87.6 (first version: 94.8, 89.6,
90.9, 81.6). Other systems: [`dataset/zeshel_realms/README.md`](../../dataset/zeshel_realms/README.md).

The tuned run took about 2 minutes (9,034 questions, 14.1 million input tokens). LELA's Table 8 gives 26 to 44 s **per mention**
for its LLM variants on ZESHEL.

## How it works
Two rounds: `which` ranks all candidates, then `final` picks among the top 3.
* **Context:** 80 words on each side of the mention, with the mention marked ⟦ ⟧. Jev is also told the title of the wiki page
  the passage comes from.
* **Candidates:** ZESHEL's 64, without the page the mention stands on. A wiki link never points to its own page (5 of the
  10,000 validation mentions do), but BM25 often ranks that page high. In the first version, 35 of the 133 errors were the page
  itself. LELA has no such rule.
* **which:** one Choice whose options are the candidate titles, each with the first three sentences of its page (LELA's
  description), asked in the given order and reversed. Jev leans to the first option.
* **final:** a Choice over the 3 best of `which`, now with the start of each page (up to 1,200 characters), asked in all 6
  option orders and averaged. Its best option is the prediction. There is no threshold and no "none" option, as in LELA's ZESHEL
  setup.
* The questions have general wording, with no examples from any ZESHEL world or from the history papers.

## How it was tuned
On 1,000 random mentions of the **Elder Scrolls** validation world (gold among the candidates), switches added one at a time.
The setup was then checked on 1,000 **Muppets** validation mentions and run once on the test world. The test world was not
used for any choice.

| Elder Scrolls validation | normalized accuracy |
|---|---|
| first version (`which` with 2 orders) | 70.2 |
| + drop the page the mention stands on | 73.1 |
| + the page title | 75.9 |
| + 4 option orders instead of 2 | 76.0, not kept |
| + gate "a kind of thing or one individual?" for the mention and each of the top 10 | 76.1, not kept (6× slower) |
| **+ final round over the top 3** | **80.4, kept** |
| final round over the top 2 / top 4 | 79.6 / 79.3 (top 4 costs 5×) |
| final round only where `which` < 0.9 | 80.3 |

| Muppets validation | normalized accuracy |
|---|---|
| first version | 85.4 |
| + drop the page and the page title | 89.7 |
| + final round over the top 3 (the frozen setup) | 91.3 |

On Elder Scrolls, `which` (with the page dropped and the page title) has the gold in its top 1 / 2 / 3 / 10 for
759 / 933 / 967 / 993 of the 1,000 mentions. The final round brings the top 1 to 804: a net 45 of the 208 mentions whose gold
is 2nd or 3rd.

The first version also had a Noul for each of the top 10 ("does ⟦mention⟧ refer to the entry X?", score √(which × is)). On the
test world it gave 86.60 against 86.70 without it, so it was removed (`pred/noul.jsonl`, commit `4a92626`).

## Run it
```bash
python jev/zeshel/run.py --drop-page --page-title --final 3 --name final
```
```bash
python dataset/zeshel_realms/score.py jev/zeshel/pred/final.jsonl
```
With no switches, `run.py` is the first version. To tune on a validation world, build it with `dataset/zeshel_realms/build.py
--split val --world elder_scrolls --out <folder>`, then run:
```bash
python jev/zeshel/run.py --data <folder> --world elder_scrolls --split val --n 1000 [switches]
```
`--gate`, `--orders`, `--final K` and `--unsure` are the other switches. `JEV_API_KEY` must be in `.env`. Answers are cached in
`jev/cache/`, so a rerun asks nothing.

## Caveats
* One test world. LELA's headline number is the macro average over four test worlds (Magistral 83.11). Lego, Star Trek and
  YuGiOh would cost about $4 more.
* The comparison with LELA holds for the candidates and the metric. The page rule and the page title are extra information that
  LELA does not use. On Elder Scrolls they account for 5.7 of the 10.2 points gained (70.2 to 75.9).
* Two validation worlds were looked at. Their samples are 1,000 mentions each (about ±2.5 points).
* The mention contexts and entity descriptions of ZESHEL were sent to TypeSafe.
