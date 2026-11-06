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
  - **Blind ablation done** (no years, RFC numbers masked, neutral labels; $4.89 notional): naive stale 8.7% → **21.7%**, current 84% → **65%**, mixed_unflagged 1 → 12. show_conflict stays at 0% stale and 88% current. The hypothesis is confirmed: the LLM uses dates and cues when they're visible, and without them the lineage signal is what protects the answer. That's the case for the librarian.
  - **Judge check by a second model:** Claude Opus 5.5 labelled 100 answers blind. Agreement with the Haiku judge is κ = 0.846 (91/100), and the 9 disagreements are borderline. **The human spot check is pending**: 29 rows in `results/rfc_answer_staleness_hybrid_human_check.csv`; see `docs/LABELING.md`.
  - **Benchmark flaw found while labelling:** in 12/69 questions the edge's new doc is itself superseded (e.g. RFC 8415 → RFC 9915), so `answer_current` may be outdated. Conclusions hold on the 57 clean questions (blind naive stale 19.3% vs show_conflict 0%). **Fix for q-v4:** generate only from edges whose new doc is a lineage head, or from (old, head) pairs, and report the clean subset.
- **Phase 3 started**: librarian v0 (embedding candidates + features + logistic regression).
  - Closes 61% (declared) / 51% (masked) of the no-lineage → oracle current@1 gap on title queries.
  - See `results/rfc_lineage_inference.md`.

## Agent stack (locked 2026-11-03)
- **LangGraph** librarian agent (triage → investigate → decide), with tools served by an **MCP server** (`search`, `read_doc`, `compare`).
- **Mistral** (hosted) as the main agent model. **Needs a Mistral API key from the user.**
- **Ollama** open-weight baseline (Mistral Small or Qwen).
- **A2A** interface added in deployment.
- **Claude** only for question generation and judging (already cached). Never as the agent.
- **Cascade:** logreg (librarian v0) handles the clear pairs; the agent investigates the uncertain ones.

**Talk:** AI Tinkerers Paris, 7 December 2026. The agent and a minimal Cloud Run deployment must be demo-ready. Schedule: `~/.claude/plans/give-me-open-source-bright-eagle.md`.

## Librarian agent (started 2026-11-03)
- `serve/mcp_server.py`: MCP server (stdio) with `search`, `read_doc` and `compare`. No lineage metadata is exposed.
- `librarian/agent.py`: LangGraph graph triage → investigate ⇄ tools → decide, with structured verdicts.
- **First end-to-end run** (local Qwen 3.5 9.7B via Ollama, tools over MCP): RFC 2581 → RFC 5681 judged `supersedes` (0.95) after read_doc ×2 and compare. About 2 min per pair locally.
- **Leak found:** the masked setting hides numbers but keeps sentences like "This document obsoletes [REF]", and the agent used that cue. Add a `strict` setting that removes supersession sentences before evaluating.
- **Strict setting added** (also drops supersession sentences). It's the agent default.
- **First agent evaluation** (local Qwen 3.5 9.7B, strict, 20 balanced pairs from the cheap scorer's uncertain band [0.8, 0.98), `results/agent/`):
  - Agent TPR 1.00 / TNR 0.70, against cheap scorer 0.40 / 0.90 (strict truth).
  - With lineage truth (later versions in the same chain count as correct): balanced accuracy 0.89 vs 0.58, estimated precision 0.68 vs 0.41.
  - Weaknesses: the agent always does read, read, compare (never `search`), and its confidence is almost always 0.95.
  - This is a smoke test only. The real run needs Mistral and n ≥ 100, with CIs.
- **Old next step** (strict setting and harness now done): response caching for agent calls, then an evaluation harness (cascade on uncertain test pairs vs logreg vs oracle). Mistral once the key is in. The machine has 16 GB of RAM, so Mistral Small can't run locally; Qwen 3.5 9.7B is the local baseline.

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
2. ~~Blind ablation~~ (done). Optional: separate years-only vs cues-only ablations, and a weaker local model for contrast.
3. Export labels: `.venv/bin/python -m supersede.experiments.answer_staleness --export-labeling 100`. The user labels them; then run `--kappa <file>`.
4. Decide whether to generate questions for the remaining 1,052 pairs (about $17 notional).
5. Librarian next: MCP server, then the LangGraph agent (Ollama first, Mistral once the key is in), evaluated against the logreg baseline and the oracle.

## Clearance
- **Publication approved (employer 2026-11-03, thesis supervisor 2026-11-05).** Code is public at https://github.com/CarlHatoum/supersede; never commit employer or client names or data.

## Known constraints
- Claude Code CLI 2.1.68 can't use Opus 5.5 (needs 2.1.280 or newer, via `claude update`). Haiku and Sonnet 5.5 work.

## Paper (living draft, synced with results)

- `make paper` regenerates every number and table from `results/` (`python -m supersede.report.paper` → `paper/generated/`) and compiles `paper/main.pdf` with Tectonic (`brew install tectonic`).
- In `paper/main.tex`, numbers are written as `\res{key}` (keys are in `paper/generated/numbers.tex`). A missing result prints as a red **??**, so pending runs are visible. Each section carries a `\status{...}` tag (final / preliminary).
- After any experiment: `make paper`, then commit the results, `paper/generated/` and `paper/main.pdf` together.
- Agent runs are rescored from their logs with no model calls: `python -m supersede.experiments.agent_lineage --prompt-version agent-v1 --n 20`.
- Agent models are local only (Ollama): `ministral-3:8b` (main), `edtorre/qwen3.5-hermes` (comparison). No API key.

## Hugging Face dataset

- https://huggingface.co/datasets/hatoum/supersede-rfc, **public** since 2026-11-05.
- Rebuild: `make hf` (folder in `dist/hf/supersede-bench`); upload: `python -m supersede.bench.export_hf --push hatoum/supersede-rfc` (add `--private` only for a new repo).
- Licence: RFC abstracts ship verbatim (IETF TLP 5.0 §3.c allows unmodified, attributed excerpts; no right to modify), annotations CC BY 4.0. Masked/strict views are computed by the bundled `masking.py`, never uploaded.
- Still to add to the card: a citation (once the paper is on arXiv).

## Plan v2 in progress (from 2026-11-05)

The execution plan (workstreams A–F, dated to the 7 Dec talk) lives in the private plan file. Main claim: **inferring lineage when metadata is missing**, with a cheap scorer and then an agent.

- **A. Positioning (done):** related work in `paper/notes/related.md`. Note that HoH already reports "better retriever ⇒ more stale".
- **B1. Recency baselines (done):** `rerank.newest_first` and `rerank.time_decay`, tuned on dev and added to `experiments/staleness.py`.
- **B2. Year-noise sweep:** `experiments/recency_noise.py` → `results/rfc_recency_noise.json`. It orders pairs by year only (`candidates.year_only_order`), never by document number.
- **C. Agent evaluation:**
  - Decisions are cached in `results/agent/decisions/<setting>_<model>_<variant>_<prompt>_t<temp>_r<run>.jsonl`.
  - Metrics come from `python -m supersede.experiments.agent_compare --n 100` → `results/agent/compare_strict_n100.json`.
  - Queues: `scripts/agent_queue.sh scripts/queues/nightN.txt` (sequential and resumable; put `/usr/local/bin` on PATH for `ollama`).
  - Nights: 1 = Qwen, 3 variants × 100 pairs; 2 = Ministral + variance; 3 = downstream questions (203 pairs); 4 = downstream titles with llm_only (1,798 pairs).
  - The old 20-pair runs (sample s1, not nested) are in `results/agent/history/`.

### 2026-11-06: agent variants on 100 nested band pairs (night 1)
- Qwen 3.5, prompt v2, strict band [0.8, 0.98), 50/50 sample. Balanced accuracy (strict):
  cheap scorer 0.72 [0.63, 0.80]; llm_only 0.76 (35 s/pair); no_find_later 0.77 (96 s);
  full 0.75 (151 s). No difference significant (McNemar p 0.5–1.0). Lineage truth:
  0.69 vs 0.78–0.82; no_find_later vs scorer p = 0.02.
- Tools add cost, not accuracy: llm_only and full agree on 93/100 pairs. All model
  variants over-say "supersedes" (est. precision at the band prior 0.29 vs 0.36).
- Paper: Q3b rebuilt from `results/agent/compare_strict_n100.json` (Figs. 7–8, Tables 5–6);
  the 20-pair runs moved to Appendix B as prompt-development history.
- Night 1 lost ~14 h to laptop sleep; `scripts/agent_queue.sh` now runs under `caffeinate`.
  Night 2 (Ministral full + temperature-0.3 repeats) started 14:22.
