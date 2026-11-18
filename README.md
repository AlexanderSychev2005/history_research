# History research: entities and relations from history papers

Extracting named entities, links to YAGO 4.5 and relations from late-antique history papers, and measuring how well
different methods do it.

## The dataset

[`dataset/`](dataset/README.md) is the shared test set: 14 paragraphs from two papers (with the PDFs) and one gold set for
all three stages.

| stage | gold | size |
|---|---|---|
| mentions (names) | `mentions`: paragraph, start, end, text | 159 |
| entity linking to YAGO 4.5 | `yago` id of each mention (null = no entity in YAGO) | 117 with an id, 42 without |
| relations | `facts`: subject, relation, object, from `relations.json` | 54 |

The gold is a draft by one annotator (Claude) and has not been reviewed by a historian.

To look at the gold, see [`dataset/mentions/`](dataset/mentions/README.md): the paragraphs with the mentions coloured by type and by YAGO link.
The entities with their YAGO ids are in [`dataset/linking/`](dataset/linking/README.md), the relations in [`dataset/relations/`](dataset/relations/README.md).

## Methods

[`hybrid/`](hybrid/README.md): entity linking to YAGO with an LLM, a YAGO shortlist and Wikipedia search, tested on the dataset
(mentions F1 93, links F1 70).

[`lela/`](lela/README.md): the LELA baseline run on the same dataset (links F1 18 with the first candidate, 45 with an LLM choice).

[`jev/`](jev/README.md): all three stages end to end with the Jev decision model (TypeSafe): mentions F1 96, links to YAGO F1 80,
relations F1 77, for about 40 cents.
