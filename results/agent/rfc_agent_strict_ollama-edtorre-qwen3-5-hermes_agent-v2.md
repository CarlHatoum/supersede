Uncertain band [0.8, 0.98] of the cheap scorer, `strict` setting, 20 balanced test pairs (0 failed). Agent model `ollama:edtorre/qwen3.5-hermes`, 3.0 steps and 118 s per pair on average. Estimated precision uses the band's real positive rate (0.14 strict, 0.25 lineage).

| ground truth | method | TPR | TNR | balanced acc. | est. precision |
|---|---|---|---|---|---|
| strict | cheap scorer (librarian v0) | 0.40 | 0.90 | 0.65 | 0.40 |
| strict | librarian agent | 0.90 | 0.90 | 0.90 | 0.60 |
| lineage | cheap scorer (librarian v0) | 0.31 | 0.86 | 0.58 | 0.41 |
| lineage | librarian agent | 0.77 | 1.00 | 0.88 | 1.00 |

*lineage* also counts later versions through a chain as correct (3 sampled pairs change label).

Agent labels: {'amends': 2, 'supersedes': 10, 'unrelated': 8}
