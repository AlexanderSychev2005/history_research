# Experiment 06: NER with Jev instead of GLiNER

## Question
Experiment 04 found that NER is the bottleneck: 21 of 54 gold facts lost a participant, mostly because GLiNER
reads "Damascius of Syria" as one person. Can Jev, which answers questions but returns no spans, do the NER?

## Method (`pipeline/ner_jev.py`)
* **Candidates by rule:** runs of capitalised words, with "of" / "of the" allowed between them. A word keeps a ‘ in
  the middle or at the end (P‘arpec‘i, Gugark‘); a possessive ’s is dropped.
* **Jev decides, one request per sentence** (questions bundled, at most 8 per request), as the TypeSafe skill
  recommends:
  * a `noul` per candidate: "is X the proper name of a specific person, place, group, work or event?" The false
    criterion names the confusing cases: adjectives ("Sasanian forces"), titles, sentence-initial words, citations;
  * a `choice` of kind (person / location / organization / written work / event);
  * for "A of B" runs, a `choice` between one name (John of Ephesus) and a name plus a separate place (Damascius of
    Syria). The questions about A and B are asked in the same request, so code takes whichever reading wins.
* Kept if noul ≥ 0.5. Model `jev-latest` (= jev-1.14.0). About 90 requests for the 14 gold paragraphs.
* Relations afterwards: `relations_jev.py --ner jev --model jev-api --guided` (run `jev__jev-api-guided`, the best
  variant of experiment 05).

## Results (`eval/gold`, 14 paragraphs; from `experiments/RESULTS.md`)

**Mentions (159 gold)**

| NER | strict P / R / F1 | overlap P / R / F1 |
|---|---|---|
| `gliner-clean` (exp. 04) | 70.5 / 76.7 / 73.5 | 79.8 / 86.8 / 83.1 |
| **Jev** | **93.8 / 94.3 / 94.0** | **96.9 / 97.5 / 97.2** |

**Facts (54 gold), guided API relations**

| mentions | best threshold | P / R / F1 |
|---|---|---|
| GLiNER | 0.9 | 50.0 / 38.9 / 43.8 |
| **Jev NER** | 0.8 | **63.8 / 68.5 / 66.1** |
| gold mentions (ceiling) | 0.8 | 71.2 / 77.8 / 74.3 |
| for reference: open 9B model, GLiNER (exp. 04) | 0.8 | 36.7 / 33.3 / 35.0 |

## Reading
* **Jev-based NER removes most of the NER bottleneck.** The full pipeline reaches 66 F1, against 44 with GLiNER, and
  is 8 points below the gold-mention ceiling.
* What made the difference:
  * the "A of B" split question;
  * keeping the transliteration ‘ inside the span;
  * a named-entity noul that rejects adjectives, titles and sentence-initial words, which GLiNER kept or merged.
* **Bias warning:** the candidate rule and the gold conventions were written by the same annotator (Claude). The rule
  splits at "of" because the gold convention does, and GLiNER was never told that convention. Part of the gap is
  therefore convention-matching, not better language understanding. A historian-reviewed gold, and a held-out
  paper, are needed before this number is reported.
* The rule cannot find lowercase or one-word-lowercase names, and it drops "Armenian rebellion": "Armenian" is an
  adjective there and "rebellion" is lowercase. Named events written in lowercase are lost.

## Cost
About 90 NER requests plus 322 relation requests for 14 paragraphs. The papers' sentences were sent to TypeSafe.
