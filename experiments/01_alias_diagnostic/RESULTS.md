# Experiment 01: Historical Alias Diagnostic

## Entity Linking Comparison

| Original mention | Modified mention | Baseline prediction | Modified prediction |
|---|---|---|---|
| Łazar P‘arpec‘i | Ghazar Parpetsi | ❌ Azar | ✅ Ghazar_Parpetsi |
| Gorgān | Hyrcania | ❌ Go_GO_GO_Go | ✅ Hyrcania |
| Sasanians | Sasanian Empire | ❌ Asan | ✅ Sasanian_Empire |
| Vałaršapat | Vagharshapat | ❌ Apate | ✅ Vagharshapat |
| Eǰmiacin | Vagharshapat | ❌ Emilcin | ✅ Vagharshapat |

## Summary

- Correct baseline predictions: **0/5**
- Correct modified predictions: **5/5**

## Interpretation

Replacing five historical name variants with canonical YAGO names corrected all five entity-linking errors in this controlled test.

**Limitation:** This is an oracle-assisted diagnostic experiment. The replacements were chosen using known gold-standard entities, so these results do not establish a generalizable improvement.
