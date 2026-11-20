# WikilinksNED Unseen-Mentions: a Wikipedia entity-linking benchmark in the format of this repository

The test split of WikilinksNED Unseen-Mentions (Onoe and Durrett, AAAI 2020, derived from WikilinksNED, Eshel et al. 2017, which
is derived from Wikilinks), the benchmark in the appendix of the LELA paper ([arXiv 2601.05192](https://arxiv.org/abs/2601.05192),
Table 11). The mentions are links found on web pages, so the gold is the Wikipedia page that the link points to, and no mention
text of the test set occurs with the same entity in the training data. Source: [`naist-nlp/unseen`](https://huggingface.co/datasets/naist-nlp/unseen)
on Hugging Face, `data/test-00001-of-00001.jsonl` (CC BY 3.0).

| | |
|---|---|
| mentions | 10,000, one per one-sentence paragraph (about 300 characters); 2,514 distinct entities |
| knowledge base | Wikipedia; here the gold page is also given as a YAGO 4.5 id |
| mentions with a YAGO id | 7,547 (75.5%); the rest are pages that YAGO 4.5 lacks, mostly common nouns ("floor", "irradiation", "budget") |
| task | mentions are given (disambiguation only); no candidates are shipped, every method brings its own retrieval |
| licence | CC BY 3.0, as the source |

## Files
Same layout as [`../zeshel_realms/`](../zeshel_realms/README.md) and `dataset/`.

| file | content |
|---|---|
| `paragraphs/wikilinks_unseen_test.json` | `{doc_id, title, source, paragraphs: [{id, text}]}`; `id` is the id of the source file |
| `gold/wikilinks_unseen_test.json` | `{doc_id, annotator, scope, entities, mentions, facts}`; `entities: [{eid, label, wikipedia_page_id, yago}]` with `eid` the Wikipedia page id of the source and `label` its current title; `mentions: [{para, start, end, text, entity, yago, mention_id}]` with 0-based character offsets, end exclusive; `facts` is empty |
| `titles.json` | `{page id: current English Wikipedia title or null}`, the cache of `build.py` (14 of the 2,514 pages are gone) |
| `build.py` | builds all of this from the downloaded test file |
| `score.py` | the scorer |

The source gives only the page id of each mention. `build.py` turns it into the **current** title (a page keeps its id when it is
moved; a redirect is followed to its target) and the title into a YAGO 4.5 id (`yago:` + the title with `_`) when the YAGO dump
has the entity, as for the other gold sets of this repository. The pages of 2017 and the pages of today differ a little, which
is a source of noise of a few mentions.

## Protocol
LELA reports **accuracy** over the test set with its own candidates (64 by dense retrieval, recall 83%, or 64 from the GENRE
prior list, recall 84%); none of its candidate lists is in the benchmark file. Here every method brings its own retrieval, and
a method whose knowledge base is YAGO cannot reach more than **75.5%**, whatever it does. `score.py` therefore prints accuracy
over all mentions and over the mentions that have a YAGO id.

```bash
python dataset/wikilinks_unseen/score.py <pred.jsonl>     # one line per mention: {"mention_id": ..., "yago": ...}
```

## Rebuild
```bash
curl -L -o test.jsonl https://huggingface.co/datasets/naist-nlp/unseen/resolve/main/data/test-00001-of-00001.jsonl
uv run --no-project --env-file .env python dataset/wikilinks_unseen/build.py test.jsonl
```
Needs `YAGO_JSONL` and, to be quick, `WIKIPEDIA_CONTACT` in `.env` (see `.env.example`).
