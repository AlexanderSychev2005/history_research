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

To look at the gold, open [`dataset/mentions.html`](dataset/mentions.html) in a browser: the paragraphs with the mentions coloured by type or by YAGO link.
