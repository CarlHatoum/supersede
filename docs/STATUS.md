# Status and how to resume

_Last updated: 2026-11-03_

## Where things stand
- **Phases 0–1 done**: RFC benchmark, BM25/dense/hybrid retrieval, oracle reranking.
- **Phase 2**:
  - 69 change-aware questions done (prompt q-v3, 300 of 1,352 pairs tried).
  - Answer policies, judge and labelling export are coded.
  - The answer + judge run (Sonnet 5.5 answers, Haiku 5.5 judge) was **~60% done** when the session ended. All finished calls are in `cache/llm.sqlite`; rerunning resumes from the cache.
- **Phase 3 started**: librarian v0 (embedding candidates + features + logistic regression).
  - Closes 61% (declared) / 51% (masked) of the no-lineage → oracle current@1 gap on title queries.
  - See `results/rfc_lineage_inference.md`.

## Resume
LLM calls go through `claude -p` on the Claude plan. From a regular terminal:

```bash
make answers   # or the command below
```

Inside a Claude Code session, the user has allowed bypassing the nested-session guard with `env -u CLAUDECODE`:

```bash
.venv/bin/python -m supersede.experiments.answer_staleness --data data/rfc --retriever hybrid \
  --backend claude-cli --model claude-sonnet-5-5 --judge-model claude-haiku-5-5 --workers 2
```

Then:
1. Commit `cache/llm.sqlite` and `results/rfc_answer_staleness_hybrid*`.
2. Export labels: `.venv/bin/python -m supersede.experiments.answer_staleness --export-labeling 100`. The user labels them; then run `--kappa <file>`.
3. Decide whether to generate questions for the remaining 1,052 pairs (about $17 notional).
4. Librarian next: LLM pairwise judge, then the agentic librarian, both compared against the logreg baseline.

## Known constraints
- Claude Code CLI 2.1.68 can't use Opus 5.5 (needs 2.1.280 or newer, via `claude update`). Haiku and Sonnet 5.5 work.
- No public push until written IP clearance from employer and the thesis supervisor.
