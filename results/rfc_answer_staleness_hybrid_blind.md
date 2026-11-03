69 change-aware questions, retriever `rrf(bm25(k1=1.2,b=0.75),dense(BAAI/bge-small-en-v1.5);k=60)`, context top-5 (**blind**: no years, RFC numbers masked, neutral labels). Answers by `claude-cli:claude-sonnet-5-5`, judged by `claude-cli:claude-haiku-5-5` (not yet validated against human labels). 95% bootstrap CI in brackets.

| policy | n | stale answer ↓ | current answer ↑ | conflict flagged | abstain |
|---|---|---|---|---|---|
| naive | 69 | 0.217 [0.130, 0.319] | 0.652 [0.536, 0.754] | 0.420 [0.304, 0.536] | 0.130 [0.058, 0.217] |
| latest_only | 69 | 0.014 [0.000, 0.043] | 0.812 [0.725, 0.899] | 0.043 [0.000, 0.101] | 0.145 [0.072, 0.232] |
| show_conflict | 69 | 0.000 [0.000, 0.000] | 0.884 [0.812, 0.957] | 0.797 [0.696, 0.884] | 0.101 [0.043, 0.174] |
