# Status and how to resume

_Last updated: 2026-11-03_

## Where things stand
- **Phases 0–1 done**: RFC benchmark, BM25/dense/hybrid retrieval, oracle reranking.
- **Phase 2**:
  - 69 change-aware questions done (prompt q-v3, 300 of 1,352 pairs tried).
  - Answer policies, judge and labelling export are coded.
  - The answer + judge run is **done** (Sonnet 5.5 answers, Haiku 5.5 judge, 414 calls, $5.19 notional). See `results/rfc_answer_staleness_hybrid.md`.
  - Stale-answer rate: naive 8.7%, latest_only 1.4%, show_conflict 0%. Current answer: 84% / 88% / 94%.
  - **Surprise.** Retrieval puts a stale doc first 65% of the time, yet Sonnet rarely answers stale and flags the conflict unprompted in 64% of naive answers. The likely cause is that the context shows publication **years**, and about 1/3 of abstracts say "obsoletes RFC n". This is unproven until two ablations are run: (a) hide years, (b) mask RFC numbers. If it holds, the finding is "a strong LLM resolves versions itself when dates and cues are visible, but not when they're missing", which points the story at enterprise corpora where dates and cues are absent.
  - The judge is **not yet validated** against human labels.
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
1. ~~Commit the cache and answer results~~ (done).
2. Ablations of the answer experiment: no years in the context, and masked RFC numbers (add an `--ablation` flag to `answer.render_context`). Also run the weaker local model (Ollama) for contrast.
3. Export labels: `.venv/bin/python -m supersede.experiments.answer_staleness --export-labeling 100`. The user labels them; then run `--kappa <file>`.
4. Decide whether to generate questions for the remaining 1,052 pairs (about $17 notional).
5. Librarian next: LLM pairwise judge, then the agentic librarian, both compared against the logreg baseline.

## Known constraints
- Claude Code CLI 2.1.68 can't use Opus 5.5 (needs 2.1.280 or newer, via `claude update`). Haiku and Sonnet 5.5 work.
- No public push until written IP clearance from employer and the thesis supervisor.
