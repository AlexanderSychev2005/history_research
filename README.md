# History research: entities and relations from history papers

Extracts named entities and YAGO-typed relations from two late-antique history papers (Alford 2026, Nechaeva 2017)
and scores each pipeline variant against one hand-annotated gold set.

**Current results: [experiments/RESULTS.md](experiments/RESULTS.md)** (generated; every variant side by side).

## Layout

| folder | what |
|---|---|
| `prepare/` | PDF → canonical text (`texts/<doc>.json`, not in git: it quotes the papers) |
| `pipeline/` | NER, entity linking (Wikidata) and relation extraction |
| `schema/` | the YAGO 4.6 schema and the relations used for history (`history_relations.json`) |
| `eval/` | the scorer (`evaluate.py`), the gold set (`gold/`, built by `make_gold.py`) and predictions (`pred/<run>/`) |
| `outputs/` | intermediate files: `<doc>_ner_<variant>.json`, `<doc>_relations_<run>.json`, linking outputs |
| `experiments/` | one folder per experiment with its README; `compare.py` builds `RESULTS.md` and `results/*.csv` |
| `pilot/` | the earlier LELA pilot (two paragraphs, YAGO 4.5 linking) |

## Variants and run names

A run is named `<mentions>__<relations>`, e.g. `jev__jev-api-guided`.

| mentions (`outputs/<doc>_ner_<variant>.json`) | made by |
|---|---|
| `gliner` | `pipeline/ner.py` (GLiNER medium, as originally) |
| `gliner-clean` | `pipeline/ner.py --clean` (optional preprocessing: transliteration ‘ kept in words, generic nouns and dates dropped, `written work` label) |
| `jev` | `pipeline/ner_jev.py` (candidate spans by rule, decisions by the TypeSafe Jev API) |
| `gold` | `eval/make_gold.py` (the gold mentions; isolates the relation stage) |

| relations (`pipeline/relations_jev.py --model …`) | model |
|---|---|
| `openjev` | OpenJev Flash 9B Q4_K_M on a local llama-server |
| `jev-api` | TypeSafe's hosted Jev (`jev-latest`), same prompt as `openjev` |
| `jev-api-guided` | hosted Jev, asked as the TypeSafe skill recommends (`--guided`) |
| `qwen-openie` | the earlier free-text extractor (`pipeline/build_kg_openie.py`), kept for reference |

## Reproduce

```bash
uv sync
uv run python prepare/prepare_texts.py                      # needs the PDFs in papers/
uv run python schema/extract_history_relations.py
uv run python eval/make_gold.py
uv run python pipeline/ner.py --doc alford2026 [--clean]    # and nechaeva2017
uv run python pipeline/ner_jev.py --doc alford2026 --paragraphs …     # JEV_API_KEY in .env
uv run python pipeline/relations_jev.py --doc alford2026 --ner jev --model jev-api --guided
uv run python experiments/compare.py                        # -> experiments/RESULTS.md, experiments/results/
```

The Wikidata linking chain (`retrieve_candidates.py` → `rank_candidates.py` → `disambiguate.py`) and the OpenIE graph
(`build_kg_openie.py`) take explicit `--input`/`--output` paths; their outputs are `outputs/alford2026_all_*.json` and
`outputs/alford2026_openie_*`.

## Caveats
* The gold set (14 paragraphs) is a draft by one annotator and still needs a historian's review.
* `jev`, `jev-api` and `jev-api-guided` send sentences of the papers to TypeSafe's API.
