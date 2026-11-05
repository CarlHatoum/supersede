Uncertain band [0.8, 0.98] of the cheap scorer, `strict` setting, 20 balanced test pairs (0 failed). Agent model `ollama:edtorre/qwen3.5-hermes`, 3.0 steps and 122 s per pair on average. Estimated precision uses the band's real positive rate (0.14 strict, 0.25 lineage).

| ground truth | method | TPR | TNR | balanced acc. [95% CI] | est. precision |
|---|---|---|---|---|---|
| strict | cheap scorer (librarian v0) | 0.40 | 0.90 | 0.65 [0.50, 0.85] | 0.40 |
| strict | librarian agent | 1.00 | 0.70 | 0.85 [0.70, 0.95] | 0.35 |
| lineage | cheap scorer (librarian v0) | 0.31 | 0.86 | 0.58 [0.40, 0.74] | 0.41 |
| lineage | librarian agent | 0.92 | 0.86 | 0.89 [0.71, 1.00] | 0.68 |

*lineage* also counts later versions through a chain as correct (3 sampled pairs change label).

Agent labels: {'amends': 2, 'supersedes': 13, 'unrelated': 5}
