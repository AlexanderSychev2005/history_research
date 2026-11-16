# Jev linking: what each part adds

Written by `python jev/ablation.py`: linking on the gold mentions, parts switched on one after another.

| parts | gold id among the candidates | link P / R / F1 at 0.7 | at 0.5 | no-id mentions left unlinked |
|---|---|---|---|---|
| search only | 88 / 117 | 0.74 / 0.65 / **0.69** | 0.67 / 0.73 / 0.70 | 29 / 42 |
| + dossier: Jev reads every sentence about the entity, not one paragraph | 88 / 117 | 0.72 / 0.65 / **0.68** | 0.71 / 0.72 / 0.71 | 29 / 42 |
| + kind gate | 88 / 117 | 0.72 / 0.66 / **0.69** | 0.72 / 0.73 / 0.72 | 28 / 42 |
| + period gate | 88 / 117 | 0.72 / 0.66 / **0.69** | 0.70 / 0.66 / 0.68 | 28 / 42 |
| + link graph: candidates from the articles the sure entities link to | 104 / 117 | 0.75 / 0.78 / **0.76** | 0.72 / 0.79 / 0.76 | 26 / 42 |
| + YAGO: membership, class Person and years | 106 / 117 | 0.77 / 0.79 / **0.78** | 0.74 / 0.80 / 0.77 | 28 / 42 |
| + YAGO's alternate names in the description (the pipeline) | 112 / 117 | 0.82 / 0.84 / **0.83** | 0.77 / 0.85 / 0.81 | 28 / 42 |
