# Evaluation of history-paper IE pipelines

One script, `evaluate.py` (Python stdlib only), scores every pipeline the same way.

```
python evaluate.py validate --texts texts/ --pred pred/A/        # file sanity, needs no gold
python evaluate.py score    --texts texts/ --gold gold/ --pred pred/A/ pred/B/ [--span strict|overlap]
python evaluate.py sample   --texts texts/ --pred pred/A/ --n 100 --out audit_A.csv   # humans fill `verdict`
python evaluate.py audit    --csv audit_A.csv                    # precision + Wilson interval
```
`example/` is a one-paragraph illustration of the format, not a result.

## What is evaluated
A pipeline turns a paper into **facts tied to entities and to the text**. Three levels, scored separately because
errors compound and because a pipeline may implement only the first ones:

| level | unit | note |
|---|---|---|
| `mention` | (paragraph, span) | where the named things are |
| `link` | (span, entity) | InKB micro P/R/F1, as in GERBIL/ELEVANT. A predicted link on a gold-NIL mention is a false positive |
| `fact` | (subject, relation, object) per paragraph | `fact_pair` ignores relation and direction: usable while relation vocabularies differ |

P = tp/predicted, R = tp/gold, F1 = harmonic mean. Micro-averaged over the annotated paragraphs. 95% interval:
bootstrap over paragraphs.

## Where these choices come from
* **Gold standard on a sample, P/R/F1** — Suchanek's *Evaluation* lecture (slides 1-24). Only annotated paragraphs
  count: output elsewhere is ignored, as in the lecture's "evaluation on a sample".
* **(span, entity) InKB F1 with 95% intervals** — LELA (IJCAI 2026 demo) reports exactly this via ELEVANT/GERBIL.
* **Two protocols, because recall needs a gold set and a gold set is expensive** — YAGO (WWW 2007): no ground truth
  of suitable size, so random facts were judged by humans *against the source text*, with a Wilson interval. SOFIE
  (WWW 2009): full gold on a small corpus for P and R, precision only by sampling on the large one; the KB was
  completed with the gold entities so that errors belong to the extractor, not to KB coverage.
* **Ambiguity matters** — the YAGO 4.5 paper reports linking accuracy by number of candidates per mention.

## Files
```
texts/<doc>.json   {"doc_id", "paragraphs": [{"id", "text"}]}        canonical text, shared by everyone
gold/<doc>.json    {"doc_id", "scope": {"paragraphs": [ids]},        the annotated sample (required)
                    "entities": [{"eid","label","aliases":[],"qid"?}],
                    "mentions": [{"para","start","end","text","entity": eid|null}],     null = NIL
                    "facts":    [{"para","subj": eid,"rel","obj": eid|{"literal"}}]}
pred/<sys>/<doc>.json   same, without scope; the system uses its own eids; `evidence_quote` optional on facts
pred/<sys>/_system.json optional {"system": name, "rel_map": {"its_relation": "shared_relation"}}
```
Offsets are 0-based character indices inside one paragraph's `text`, end exclusive. `start`/`end` may be omitted in a
prediction (anchored by text, reported). Extra fields such as `type` are allowed and ignored.

**Entities of different systems are matched by `qid`, else by a unique match of normalised label/alias**
(case, diacritics and transliteration marks ignored). The gold must therefore list the aliases that occur in the text.

## Protocol
1. Every system starts from the same `texts/`. PDF-to-text is not compared (it would change paragraphs and offsets).
2. Annotate before looking at any system output; two annotators on the same paragraphs, score one against the other
   with `score` to get agreement, then adjudicate.
3. Keep some annotated paragraphs unseen until the end (dev vs test); sample sections, not the first N.
4. To separate stages, run a stage on gold input (gold mentions for linking, gold entities for facts) and say so.
5. For output beyond the gold sample use `sample` + `audit`: the question asked of a judge is "is this fact stated or
   directly implied by this paragraph", not "is it historically true".
6. Report: levels, span mode, number of paragraphs, interval, `validate` problems.

## Not evaluated (v0.1)
Entity types, certainty and attribution of facts, qualifiers (dates, places), PDF parsing, cross-paper entity
resolution, query answering. Add them when there is a gold standard for them; each is a separate decision.
