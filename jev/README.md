# Jev: mentions, YAGO links and relations with one decision model, end to end

[Jev](https://docs.typesafe.ai) (TypeSafe, `jev-1.14.0`) is not a text generator. It answers typed questions about a piece of
text with probabilities: a **Noul** (yes or no) or a **Choice** (one of the options). It cannot write a span, a name or a search
query. This folder runs all three stages of the project with it, on the 14 paragraphs of [`../dataset/`](../dataset/README.md):
code proposes, Jev judges, code decides.

## Result
One run, scored against `dataset/gold` (159 mentions, 117 with a YAGO id, 54 facts). F1 in %, precision / recall in brackets.

| stage | **Jev, end to end** | Jev on the gold mentions | best other method in the repository |
|---|---|---|---|
| mentions, strict spans | **96** (95 / 97) | | hybrid, `gpt-oss-120b`: 93 (91 / 95); earlier Jev NER: 94 |
| mentions, overlap | **98** (96 / 99) | | hybrid: 95; earlier Jev NER: 97 |
| links to YAGO, strict spans | **80** (78 / 82) | 83 (82 / 84) | hybrid, LLM guess + YAGO shortlist + LLM choice: 64 (70 / 59) |
| links to YAGO, overlap | **82** (80 / 84) | 83 | hybrid: 68 (74 / 62); LELA + LLM: 48 |
| relations, threshold 0.7 | **77.0** (69 / 87) | 79.3 (72 / 89) | earlier Jev run at 0.7: 63.1 end to end, 69.8 on gold mentions |
| relations, best threshold (0.8) | 80.0 (82 / 78) | 81.9 (84 / 80) | earlier Jev run, best threshold: 66.1 and 74.3; constrained LLM `gpt-oss-20b`: 62.7; NLI: 40.0 |

* Full tables: [`results/scores.md`](results/scores.md) (every threshold), [`results/linking_ablation.md`](results/linking_ablation.md)
  (what each part of linking adds), [`results/errors.md`](results/errors.md) (every wrong mention, link and fact).
* The earlier Jev runs are `archive/pipeline/ner_jev.py` and `relations_jev.py`; their numbers were recomputed from
  `archive/outputs/` with this folder's scorer and equal the archived ones. The hybrid numbers are `python hybrid/score.py`.
* The thresholds (0.5 for a name, 0.7 for a link and for a fact) were fixed before scoring. A fact threshold picked on one paper and
  applied to the other gives F1 79.2 and 80.7.
* A run is about 16,000 questions, 10 million input tokens and **40 cents**.

## How it works
```
paragraphs ─► 1 mentions ─► 2 entities and YAGO links ─► 3 relations ─► pred/<doc>.json
              mentions.py    linking.py, sources.py       relations.py
```

**1. Mentions.** Code finds runs of capitalised words in a sentence. Jev answers, in one batch per sentence: is the run a name
(Noul), of what kind (Choice of six), is "A of B" one name or a person and a place (Choice), which words of a person's run are the
name without the title (Choice over its suffixes: "King Henry II" gives "Henry II"), and is a lowercase head noun part of the name
(Noul: "Kamsarakan family"). Code alone drops function words, months and abbreviations in citations.

**2. Entities and links.** Jev cannot guess "Ghazar Parpetsi" for "Łazar P‘arpec‘i" as an LLM does, so code finds candidates and
Jev judges them.
* *Forms.* Code proposes pairs of written forms ("Łazar" and "Łazar P‘arpec‘i"; "Gorgān (Latin. Hyrcania"), a Noul says whether
  they are one entity. "Vasak" and "Vasak Siwni" stay two.
* *Search.* Full-text search of English Wikipedia for the forms, for "Isidore of Gaza" where the text has it, and for scholarly
  transliterations respelled by rule (Aršawir gives Arshavir). A YAGO 4.5 id is the Wikipedia title with `_` for punctuation.
* *YAGO.* [`sources.py`](sources.py) answers what a lookup in the YAGO dump would, without the dump. YAGO is built from Wikidata
  ([yago-naga/yago-4.6](https://github.com/yago-naga/yago-4.6)): an entity is kept if one of its Wikidata classes lies under a class
  of the hand-made schema, and its names and facts are Wikidata's. So for every candidate, Wikidata gives its YAGO classes (none
  means it is not in YAGO: an ethnic group such as the Huns is not), its years, its alternate names and its facts. The alternate
  names go into the description Jev reads ("Khosrow I … also called Chosroes I").
* *Gates.* Jev says of what kind the entity is and in which century it lived, from every sentence of the paper that names it, and
  the same about each candidate from its description. Code compares and drops a candidate of another kind (a battle for a town)
  or, for persons and events, of another period (Yazdegerd III for a 5th-century Yazdgerd). YAGO's years and its class Person
  replace Jev's reading where YAGO has them. Jev compares dates badly, so it only reads them and code compares.
* *Identity.* Over the candidates that are left: a companion Noul (is any of them the entity?), a Choice in both option orders, and
  a Noul per candidate. The score is the geometric mean of the three; below 0.7 the entity stays unlinked.
* *Link graph.* The entities of one paper are linked to each other in Wikipedia as well. The articles that the sure links (p ≥ 0.9)
  point to, about 5,000 titles per paper, become candidates for every entity that is not sure yet. Jev picks among the bare
  titles, 120 per question, and the winners meet again until five are left; these go through YAGO, gates and identity like any
  other candidate. This is affordable only because a Jev question costs a few thousandths of a cent.

**3. Relations.** For two entities named in one sentence, code lists the relations of `dataset/relations.json` that their kinds
allow, in both directions. Jev answers: do the two connect at all (Noul), which relation (Choice, both orders), and one Noul per
possible relation ("does the sentence say that A was held captive in B?"). The score is the geometric mean of the three; the best
relation of a pair is kept if it reaches 0.7. YAGO's constraints then apply (a relation with a maximal count in the schema keeps
that many objects per subject; a symmetric relation is one fact), and every fact whose two ends are linked is marked `in_yago`:
whether YAGO already states it. Of the 68 facts found end to end, 24 could be checked and none is in YAGO.

All question texts and thresholds are in [`questions.py`](questions.py), written to the TypeSafe guide: one judgment per question,
a companion Noul in place of a "none" option, option names that say what they mean, and every Choice asked in both option orders
because Jev leans to the first option (the top relation differs between the two orders in 16% of the pairs).

## What changed against the earlier Jev scripts
| found in the earlier runs (`archive/`) | change | effect |
|---|---|---|
| 3 of 54 gold facts could never be predicted: "Sasanians" or "Roman Empire" got one type and the relation needed the other | a kind *people or state* that counts as place and as organization; every kind with p ≥ 0.2 counts | all 54 gold facts are among the options |
| one Choice over up to 20 relations decided everything, read in one option order | both option orders, and a Noul per possible relation; score = related × which × states | on gold mentions at 0.7: 69.8 before, 78.1 with the Choice alone, 79.3 with all three |
| `partOf` offered between a person and a place, and confused with `location` for a place in a place | code offers `partOf` and `follows` only between two things of one kind, and never `partOf` between two places (the dataset calls that `schema:location`) | no question spent on impossible relations |
| the threshold (0.8) was found on the gold | 0.7, fixed before scoring | see the result table for both |
| titles kept in names ("Bishop T‘at‘ik"), "Kamsarakan family" lost, sentences starting with "Ł" not split off | a Choice over the suffixes of a person's run, a Noul for a lowercase head noun, a sentence splitter that knows capitals beyond A–Z and brackets | strict mention F1 94.0 to 96.3 |
| one request per pair (about 1,200), alias `jev-latest`, no cache | all questions of a sentence in one batch, model pinned, every answer cached on disk | a rerun costs nothing |
| no entity linking with Jev | stage 2 | link F1 80, against 64 for the LLM hybrid |

## What the linking ablation shows
[`results/linking_ablation.md`](results/linking_ablation.md), on the gold mentions:

| parts, each added to the ones above | gold id among the candidates | link F1 |
|---|---|---|
| search only | 88 / 117 | 0.69 |
| + dossier, kind gate, period gate | 88 / 117 | 0.69 |
| + link graph | 104 / 117 | 0.76 |
| + YAGO: membership, class Person and years | 106 / 117 | 0.78 |
| + YAGO's alternate names (the pipeline) | 112 / 117 | 0.83 |

* **The link graph** finds what no search for the name returns: Robert W. Thomson for "Robert Thomson", the Avarayr plain for
  "Awarayr", the Byzantine Empire for "Romans", Piran Gushnasp for "Grigor" (10 mentions).
* **YAGO's alternate names** settle cases where the text uses another name than the entry: "Persia" and "Persians" in Nechaeva
  (8 mentions) go to the Sasanian Empire ("also called Neo-Persian Empire, Second Persian Empire"), not to Iran.
* **YAGO's membership rule** takes candidates out before Jev sees them, among them the articles on the Huns and on the Persians
  as peoples.
* The gates add little on this gold; they are there because they are right (they remove Yazdegerd III for Yazdgerd II, Hormizd I
  and II, Gregory the Illuminator for Grigor).
* Tried and removed from the code: a stricter identity question that refused an entry under a modern name (F1 0.67 against 0.69
  for search only); search queries with a century and a role that Jev chose from lists (no change); dropping every candidate
  whose YAGO class differs from Jev's kind (it lost the right entry for "Endless Peace treaty" on text outside the gold, because
  YAGO classes a treaty as a creative work and a reader as an event).

## What is still wrong
From [`results/errors.md`](results/errors.md).

* **Links** (33 of 159 gold mentions wrong on the gold mentions):
  * 14 are linked to an article that YAGO's rule admits but the gold says YAGO 4.5 does not have (Tokharistan, Pseudo-Zacharias
    Rhetor, Sasanian Armenia for Persarmenia, Dorotheus the magister militum, Domnentiolus, Phasiane for Basean, Arshusha II for
    Ašuša), two true mistakes among them (Isidore of Alexandria for Isidore of Gaza, modern Syria for the region). Either
    these articles are younger than the YAGO 4.5 dump or the gold missed them; only the dump can tell.
  * 7 are linked to a neighbouring entry: Vahan Mamikonean to `Vahan_II_Mamikonian` (5; Wikipedia's title today for the entity
    Q2085338, whose Wikidata label is still "Vahan I Mamikonian", the gold's id), Siwnik‘ to the historical province instead of
    the modern one, "Roman Empire" to the Roman Empire.
  * 12 are left unlinked: Nerseh (8, below), Armenia (2), Callinicum, Gaza.
* **Relations.** A pronoun is read as the named subject: "Łazar claims they were captured by Vasak Siwni … and kept in Siwnik‘"
  gives Łazar `captiveIn` Siwnik‘. "Following Akinean" gives `influencedBy`, and a rebellion "against" a king gives `winner`.
* **Mentions.** "John of Ephesus" is split like "Isidore of Gaza"; "Armenian rebellion" is refused as a name; adjectives pass
  four times ("Amidan source", "Hun allies", "Mamikonean boys", "Sasanian authority").

For the gold (nothing in `dataset/gold` was changed):
* A mention is missing: "Grigor" in nechaeva2017 p018 at 645–651 ("the prominence of Grigor’s position").
* The gold links Nerseh Kamsarakan (8 mentions) to `Nerses_Kamsarakan`. YAGO's source dates that entry to 691–700 and the Nerseh
  of the paper lived in the 5th century; the period gate drops it and is scored as wrong.
* Predicted facts that the gold does not have and the sentences seem to state: Łazar `worksFor` Vahan (p004, "his patron"),
  Hormizd `location` Arzanene (p013, "the ruler of Arzanene"), Grigor `worksFor` Persia (p018, "position in Persia"; the gold has
  `location`), Zabergān `worksFor` Chosroes (p019), Martyrdom `about` Grigor (p019).

## Files
| file | content |
|---|---|
| `pipeline.py` | the three stages for the dataset or for any paragraph file; writes `pred/` |
| `mentions.py`, `linking.py`, `relations.py` | the stages |
| `questions.py` | every question and threshold |
| `client.py` | the Jev API client: cache, batches, retries, pinned model |
| `sources.py` | Wikipedia search and links; YAGO by its build rule through Wikidata |
| `yago-schema.ttl` | the schema of [yago-naga/yago-4.6](https://github.com/yago-naga/yago-4.6) (CC BY 4.0), unchanged |
| `score.py`, `ablation.py` | write the files of `results/` |
| `pred/<doc>.json` | `mentions`; `entities` (forms, kind, century, YAGO id, Wikipedia title, Wikidata id, every candidate with its probability or the reason it was dropped); `facts` (YAGO ids of both ends, `in_yago`) |
| `pred/<doc>.decisions.json` | every probability of the relation stage, from which `score.py` computes other thresholds |
| `pred/<doc>.jsonl` | the mentions in the format of `hybrid/pred` |
| `pred/gold-mentions/` | the same from the gold mentions: stages 2 and 3 alone |
| `cache/` | Jev's answers and the Wikipedia and Wikidata results (git-ignored) |

## Run it
Python 3.9 or later, standard library only. `JEV_API_KEY` in `.env` (see `.env.example`). From the repository root:
```bash
python jev/pipeline.py
```
```bash
python jev/pipeline.py --gold-mentions
```
```bash
python jev/score.py > jev/results/scores.md
```
```bash
python jev/score.py --errors > jev/results/errors.md
```
```bash
python jev/ablation.py > jev/results/linking_ablation.md
```
Any other text: `python jev/pipeline.py --input texts/alford2026.json --out out/` (a file `{doc_id, paragraphs: [{id, text}]}`).

The first run is slow because of Wikipedia and Wikidata, not Jev: Wikimedia allows a client that does not say who it is 10 requests
a minute, and the 14 paragraphs need about 230 (25 minutes). With a contact in `WIKIPEDIA_CONTACT` (an e-mail or URL, sent in the
User-Agent) the limit is 200 a minute. Everything is cached in `jev/cache/`; a rerun takes seconds and asks nothing.

## Caveats
* **YAGO through its rule, not its dump.** yago-knowledge.org did not answer on 16 Nov 2026 (LELA downloads its YAGO file from the
  same address), so membership, classes, years and names come from Wikidata as it is today, and the YAGO id is derived from the
  Wikipedia title. An article younger than YAGO 4.5 therefore gets a link the gold does not have, and an article renamed since
  gets another id. With `YAGO_JSONL` pointing to the dump, candidates whose id is not in it are dropped as well; that was tried
  with a stand-in file only, never with the real dump.
* **The numbers are optimistic.** The questions and rules were written while looking at these 14 paragraphs, like the prompts of
  the other methods, and the gold is a draft by one annotator. What speaks for generality: no threshold was fitted; no question
  holds an example from the two papers (replacing the ones that did added one wrong mention and moved no score by more than a
  point); what a candidate is comes from YAGO's schema, not from a list made for these papers. Two paragraphs outside the gold (Nechaeva p001
  and p002) were run as a look, not a measurement: 14 mentions, 8 links (among them "Endless Peace treaty" to
  `Perpetual_Peace_532`), 2 facts.
* 54 facts and 117 linkable mentions: intervals of about ±10 points. One run; the cache holds its answers, and a run without the
  cache was not compared with it.
* The sentences of the two papers were sent to TypeSafe, and the names in them to Wikipedia's search and to Wikidata.
