# Full LLM against the GLiNER pipeline

Same two paragraphs, same 35 gold names, same YAGO 4.5, same `score.py` (exact spans).
LLM rows are the mean of 3 runs (temperature 0.3); GLiNER rows are one run each.

| Approach | Names found by | Entity chosen by | Mention F1 | Link P | Link R | Link F1 |
| --- | --- | --- | --- | --- | --- | --- |
| GLiNER pipeline | GLiNER, extra types, threshold 0.3 | top fuzzy candidate | 0.91 | 0.33 | 0.42 | 0.37 |
| GLiNER pipeline + LLM | GLiNER, extra types, threshold 0.3 | gpt-oss-120b from 20 candidates | 0.91 | 0.82 | 0.58 | 0.68 |
| Full LLM, gpt-oss-120b | LLM, rules in prompt | LLM, name looked up in YAGO | 1.00 | 0.76 | 0.45 | 0.57 (0.50 to 0.65) |
| Full LLM, qwen3.8-27b | LLM, rules in prompt | LLM, name looked up in YAGO | 0.96 | 0.71 | 0.42 | 0.53 (0.52 to 0.53) |
| Full LLM, gpt-oss-20b | LLM, rules in prompt | LLM, name looked up in YAGO | 0.95 | 0.74 | 0.37 | 0.49 (0.48 to 0.50) |
| Full LLM, gpt-oss-120b, plain prompt | LLM, no rules | LLM, name looked up in YAGO | 0.69 | 0.50 | 0.20 | 0.29 (0.27 to 0.33) |

Where the full-LLM link score is lost (guided runs, 9 runs, 31 names with a YAGO entity each):
* 10 to 14 names per run were named by the LLM but not found in YAGO under that spelling, for example
  "Lazar of Parpi" (YAGO: Ghazar_Parpetsi), "Ejmiatsin" (Vagharshapat), "Vahan Mamikonian" (Vahan_I_Mamikonian).
* A few were linked to a wrong entity: Gorgan (a city, not Hyrcania), Priscian (the grammarian), Roman_Empire for
  "the Romans".
* gpt-oss-20b gave 7 mention texts that are not in the paragraph over 3 runs; the other models gave none.

Caveats: the LLM prompt carries our annotation rules, and their examples come from the two test paragraphs
("Damascius of Syria", "Łazar P‘arpec‘i", "Persian"), a leak: the guided mention scores are an upper bound and the
plain-prompt row is the lower one. GLiNER cannot be given rules at all; name lookup in YAGO is an exact, case-insensitive title match, so a forgiving lookup would raise the
full-LLM link score (not tested); one gpt-oss-20b answer in four of the 24 first came back empty and was re-requested.
