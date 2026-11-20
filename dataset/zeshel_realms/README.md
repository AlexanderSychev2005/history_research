# ZESHEL, Forgotten Realms: a small entity-linking benchmark that lines up with LELA

One test world of [ZESHEL](https://github.com/lajanugen/zeshel) (Logeswaran et al., ACL 2019), the zero-shot entity-linking
benchmark that the LELA paper ([arXiv 2601.05192](https://arxiv.org/abs/2601.05192), Table 3) reports. It uses the files of
[`../`](../README.md) so that any method of this repository can be run on it. LELA reports a number for this world, so a run here
compares directly with LELA and the systems in its table.

**Why this world:** it is the smallest part of ZESHEL that a published number covers. 1,200 mentions give a 95% interval of about
±2 points; a sample of 100 gives about ±6, which is too wide to tell the systems below apart.

| | |
|---|---|
| mentions | 1,200 (the whole Forgotten Realms test world) in 1,064 pages; 918 distinct gold entities |
| knowledge base | 15,603 entities of the Forgotten Realms wiki, each a title and a description |
| candidates | the 64 BM25 (TF-IDF) candidates that ship with ZESHEL; the gold is among them for 1,000 mentions (83.3%) |
| task | mentions are given (disambiguation only); every gold entity is in the KB, so there is no NIL |
| licence | CC BY-SA, as ZESHEL (FANDOM wikis) |

## Files
Same layout as `dataset/` (see [`../README.md`](../README.md)). The KB id replaces the YAGO id.

| file | content |
|---|---|
| `paragraphs/zeshel_forgotten_realms_test.json` | `{doc_id, title, source, paragraphs: [{id, title, text}]}`: each page that holds a mention, ZESHEL's whitespace tokens joined by one space. `id` is ZESHEL's `context_document_id` |
| `gold/zeshel_forgotten_realms_test.json` | `{doc_id, annotator, scope, entities, mentions, facts}`. `entities: [{eid, label, type, aliases, kb_id, note}]`; `mentions: [{para, start, end, text, entity, kb_id, mention_id, category, gold_in_candidates}]`, with 0-based character offsets, end exclusive; `facts` is empty |
| `candidates/zeshel_forgotten_realms_test.json` | `{mention_id: [64 kb ids]}`, BM25 order |
| `kb/forgotten_realms.jsonl` | `{id, title, description}`; the description is the first three sentences of the page, as LELA uses |
| `build.py` | builds all of this from the ZESHEL download |
| `score.py` | the scorer |

`category` is ZESHEL's mention type: `HIGH_OVERLAP` (mention = title), `AMBIGUOUS_SUBSTRING`, `MULTIPLE_CATEGORIES` and
`LOW_OVERLAP` (the hardest kind, 656 of the mentions).

## Protocol (LELA's, Sec. 4.1)
* Use the 64 given candidates. Your own retrieval over `kb/` is allowed too, but say so: it is another column in LELA's table.
* Entity = title + description from `kb/`. Context = the page in `paragraphs/`. How much of it a method reads is up to the method.
* **Metric: normalized accuracy**, which is accuracy on the 1,000 mentions with `gold_in_candidates: true`. This is LELA's column
  "Realms". The scorer gives unnormalized accuracy over all 1,200 as well.

```bash
python dataset/zeshel_realms/score.py <pred.jsonl>      # one line per mention: {"mention_id": ..., "kb_id": ...}
```

## Results: normalized accuracy, Forgotten Realms test

| system | accuracy | source |
|---|---|---|
| GenDecider | 91.10 | LELA Table 3 (as reported in GenDecider) |
| Bi-MPR | 89.60 | LELA Table 3 |
| **LELA, Magistral-Small-2509** | **88.69** | LELA Table 3 |
| ReS | 88.10 | LELA Table 3 |
| Uni-MPR | 87.25 | LELA Table 3 |
| BLINK (zero-shot, trained on the ZESHEL train worlds) | 86.80 | LELA Table 3 |
| **Jev (`jev-1.14.0`), Choice over the 64 candidates** | **86.70 ± 2.1** | [`jev/zeshel/`](../../jev/zeshel/README.md), this repository |
| LELA, Qwen3-30B-A3B | 85.69 | LELA Table 3 |
| LLM-SFT | 85.59 | LELA Table 3 |
| BLINK, true zero-shot | 84.11 | LELA Table 3 |
| LELA, Qwen3-4B | 80.08 | LELA Table 3 |
| ExtEnD | 79.62 | LELA Table 3 |
| OneNet | 55.80 | LELA Table 3 |
| GENRE | 55.20 | LELA Table 3 |
| BM25 first candidate | 52.40 ± 3.1 | the order of `candidates/`, this repository |
| LELA without LLM selection | 31.23 | LELA Table 3 |

The numbers from LELA were taken from the paper, not rerun. Its "true zero-shot" systems (LELA, OneNet, GENRE, ExtEnD, the
second BLINK row) use no ZESHEL training data. Jev also uses none and belongs with them.

## Rebuild
Download ZESHEL (the data link in [its README](https://github.com/lajanugen/zeshel)), unpack it, and unpack the "top-64
candidates" archive from the same README into the same folder as `tfidf_candidates/`. Then:
```bash
python dataset/zeshel_realms/build.py <zeshel folder>
```
