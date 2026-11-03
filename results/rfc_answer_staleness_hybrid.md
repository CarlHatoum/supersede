69 change-aware questions, retriever `rrf(bm25(k1=1.2,b=0.75),dense(BAAI/bge-small-en-v1.5);k=60)`, context top-5. Answers by `claude-cli:claude-sonnet-5-5`, judged by `claude-cli:claude-haiku-5-5` (not yet validated against human labels). 95% bootstrap CI in brackets.

| policy | n | stale answer ↓ | current answer ↑ | conflict flagged | abstain |
|---|---|---|---|---|---|
| naive | 69 | 0.087 [0.029, 0.159] | 0.841 [0.754, 0.913] | 0.638 [0.522, 0.739] | 0.058 [0.014, 0.116] |
| latest_only | 69 | 0.014 [0.000, 0.043] | 0.884 [0.812, 0.957] | 0.043 [0.000, 0.101] | 0.072 [0.014, 0.130] |
| show_conflict | 69 | 0.000 [0.000, 0.000] | 0.942 [0.884, 0.986] | 0.826 [0.739, 0.913] | 0.043 [0.000, 0.087] |
