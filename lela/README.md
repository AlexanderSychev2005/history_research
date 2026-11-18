# LELA on the dataset

[LELA](https://github.com/dig-team/LELA) (commit `0202b81`) run on the 14 paragraphs of [`../dataset/`](../dataset/README.md);
the scores are in [`../hybrid/README.md`](../hybrid/README.md).

| file | content |
|---|---|
| `configs/first.json` | GLiNER (`numind/NuNER_Zero-span`, threshold 0.3), fuzzy search with 20 candidates, first candidate |
| `configs/llm.json` | the same, but `gpt-oss-20b` on Groq picks among the candidates (or none) |
| `run.py` | runs LELA on the dataset paragraphs and writes `pred/<config>/<doc>.jsonl` |
| `pred/llm/` | predictions of the LLM variant |

## Run it
Clone LELA, put the YAGO 4.5 dump at `<LELA>/data/yago/yago-entities.jsonl` (see [`../hybrid/README.md`](../hybrid/README.md)),
run `uv sync` there, then from the LELA folder:
```bash
uv run --env-file <this repository>/.env python <this repository>/lela/run.py llm
python <this repository>/hybrid/score.py <this repository>/lela/pred/llm
```
The first run downloads the GLiNER model (about 1.8 GB). `run.py` retries Groq's 429 answers, which LELA itself does not (it would
leave the mention unlinked without a word), and stops with an error if the limit does not clear. The key is put into a
temporary copy of the config, never into the repository.
