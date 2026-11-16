# Jev pipeline: scores against dataset/gold

Written by `python jev/score.py` from `jev/pred/`.

## End to end (Jev finds the mentions)

| spans | mention P / R / F1 | link P / R / F1 | gold id among the candidates | no-id mentions left unlinked |
|---|---|---|---|---|
| strict | 0.95 / 0.97 / 0.96 | 0.78 / 0.82 / 0.80 | 110 / 115 | 26 / 40 |
| overlap | 0.96 / 0.99 / 0.98 | 0.80 / 0.84 / 0.82 | 112 / 117 | 26 / 40 |

Facts: P 69.1% / R 87.0% / F1 77.0% (47 right of 68 predicted, 54 gold).

P / R / **F1** at other thresholds (the pipeline uses 0.7 for both):

| threshold | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| links, strict spans | 71 / 84 / **76.9** | 71 / 84 / **76.9** | 74 / 84 / **78.4** | 78 / 83 / **80.2** | 78 / 82 / **80.0** | 78 / 76 / **77.1** | 78 / 69 / **73.3** |
| facts: related × which × states (the pipeline) | 47 / 93 / **62.1** | 56 / 93 / **69.4** | 61 / 93 / **73.5** | 64 / 93 / **75.8** | 69 / 87 / **77.0** | 82 / 78 / **80.0** | 85 / 63 / **72.3** |
| facts: related × which only | 38 / 93 / **54.3** | 50 / 93 / **64.5** | 56 / 93 / **69.4** | 62 / 93 / **74.1** | 67 / 89 / **76.2** | 75 / 78 / **76.4** | 78 / 46 / **58.1** |
| facts: states only | 60 / 91 / **72.1** | 64 / 91 / **74.8** | 64 / 85 / **73.0** | 73 / 80 / **76.1** | 75 / 70 / **72.4** | 76 / 65 / **70.0** | 93 / 48 / **63.4** |

The fact threshold that is best on alford2026 (0.8) gives F1 79.2% on nechaeva2017.

The fact threshold that is best on nechaeva2017 (0.8) gives F1 80.7% on alford2026.

## On the gold mentions (stages 2 and 3 alone)

| spans | mention P / R / F1 | link P / R / F1 | gold id among the candidates | no-id mentions left unlinked |
|---|---|---|---|---|
| strict | 1.00 / 1.00 / 1.00 | 0.82 / 0.84 / 0.83 | 112 / 117 | 28 / 42 |
| overlap | 1.00 / 1.00 / 1.00 | 0.82 / 0.84 / 0.83 | 112 / 117 | 28 / 42 |

Facts: P 71.6% / R 88.9% / F1 79.3% (48 right of 67 predicted, 54 gold).

P / R / **F1** at other thresholds (the pipeline uses 0.7 for both):

| threshold | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| links, strict spans | 74 / 85 / **79.4** | 75 / 85 / **79.7** | 77 / 85 / **81.0** | 80 / 85 / **82.5** | 82 / 84 / **83.1** | 83 / 78 / **80.2** | 83 / 71 / **76.5** |
| facts: related × which × states (the pipeline) | 50 / 96 / **66.2** | 59 / 96 / **73.2** | 65 / 96 / **77.6** | 68 / 96 / **79.4** | 72 / 89 / **79.3** | 84 / 80 / **81.9** | 87 / 63 / **73.1** |
| facts: related × which only | 41 / 96 / **57.8** | 53 / 96 / **68.0** | 58 / 96 / **72.2** | 63 / 96 / **75.9** | 68 / 93 / **78.1** | 77 / 81 / **79.3** | 79 / 48 / **59.8** |
| facts: states only | 65 / 94 / **76.7** | 68 / 93 / **78.1** | 67 / 87 / **75.8** | 76 / 81 / **78.6** | 78 / 72 / **75.0** | 78 / 67 / **72.0** | 93 / 48 / **63.4** |

The fact threshold that is best on alford2026 (0.8) gives F1 80.9% on nechaeva2017.

The fact threshold that is best on nechaeva2017 (0.8) gives F1 82.8% on alford2026.
