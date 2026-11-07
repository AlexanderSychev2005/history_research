# Archive

Our earlier work (Alex and Claude), frozen when the project moved to one shared gold set on 14 paragraphs
(`dataset/`). Nothing here is maintained.

| folder | what | where to read the outcome |
|---|---|---|
| `pilot/` | the LELA pilot: two paragraphs (one per paper), 35 hand-annotated mentions linked to YAGO 4.5, 12 LELA configurations, a full-LLM baseline and the hybrid | `pilot/README.md`, `pilot/results/summary.md` |
| `pilot14/` | the same approaches on the colleagues' 14 paragraphs, with YAGO ids for their 83 entities (`entity_map.json`, now part of `dataset/gold`) | `pilot14/README.md` |
| `08_constrained_relations/` | relation extraction by zero-shot prompting with constrained decoding (experiment 08) | `08_constrained_relations/README.md`, `RESULTS.md` |
| `README_RESULTS.md` | best models and metrics for each stage, as of November 2026 | itself |

## Notes
* **Paths inside are the old ones.** The scripts and READMEs refer to `pilot/`, `pilot14/` and `experiments/08_constrained_relations/`
  as top-level folders; to rerun one, run it from a checkout where the folder sits at that place (for example copy it back), or
  adjust the paths. `08_constrained_relations/relations_constrained.py` imports `pipeline.relations_jev` relative to its old depth.
* **What the results mean.** The main findings: GLiNER or an LLM for mentions (an LLM was better), LELA alone links poorly
  (first candidate: link F1 0.14 on 12 new paragraphs), an LLM that proposes a name and chooses from a YAGO shortlist is best
  (0.53), and the constrained LLM matches Jev API on relations without a threshold. Numbers on `p004` are biased: the pilot was
  developed on those two paragraphs.
* Kept out of the repository: the pilot's raw LELA output (`pilot/pred/`, git-ignored) and a PDF with speaker notes.
