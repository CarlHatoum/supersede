# Supersede

**Your RAG is citing an obsolete standard.** Knowledge bases rot. Documents get replaced by newer versions, contradict each other, or are made up by an LLM and dropped into a shared drive. Retrievers rank by similarity, so a superseded document that matches the query well wins over its current replacement.

Supersede is an open benchmark and toolkit for measuring and fixing this:

1. **Benchmark.** Public corpora where the replacement relation is already recorded, so no labelling is needed. IETF RFCs are available now; FAA Advisory Circulars and UNECE vehicle regulations are planned.
2. **Fixes**, at two points: an ingestion-time *librarian* that infers lineage and quarantines suspect documents, and retrieval/answer-time policies that surface the current version and cite the old one.
3. **Production path:** an incremental ingestion service, an MCP server, and published latency and cost numbers *(planned)*.

> Status: **v0.2**. RFC slice, with BM25, dense (`bge-small-en-v1.5`) and hybrid (RRF) retrievers, and lineage-aware reranking with known (oracle) lineage. No LLM is involved yet, so every number below is deterministic and free to reproduce.

## First result: every retriever prefers superseded documents, and hybrid is the worst

The RFC index lists 9,852 RFCs. 1,384 of them are obsoleted by a later RFC, through 1,543 `obsoletes` links that form 800 lineage chains, the longest with 35 documents. For each link we query the topic in two ways: by the old document's title (the name users know) and by the new one's. The gold answer is the *current* head of the lineage.

Test split, top-10, 95% bootstrap CI. "Promote" puts the current version of each stale hit just above it, using oracle lineage:

| retriever | query wording | n | stale@1 ↓ (no policy) | current@1 ↑ (no policy) | current@1 ↑ (promote) |
|---|---|---|---|---|---|
| BM25 | new title | 727 | 0.417 [0.382, 0.451] | 0.440 [0.406, 0.477] | 0.857 [0.832, 0.882] |
| dense | new title | 727 | 0.459 [0.421, 0.494] | 0.446 [0.411, 0.484] | 0.904 [0.882, 0.924] |
| hybrid RRF | new title | 727 | **0.528** [0.491, 0.561] | 0.395 [0.362, 0.432] | **0.923** [0.902, 0.942] |
| BM25 | old title | 784 | 0.722 [0.691, 0.755] | 0.152 [0.126, 0.177] | 0.874 [0.849, 0.895] |
| dense | old title | 784 | 0.749 [0.718, 0.779] | 0.170 [0.143, 0.196] | 0.916 [0.895, 0.934] |
| hybrid RRF | old title | 784 | **0.816** [0.790, 0.844] | 0.111 [0.089, 0.133] | **0.927** [0.909, 0.945] |

- **This isn't a keyword-matching artefact.** Even when the query uses the *new* document's own wording, every retriever ranks a superseded version first 42–53% of the time.
- **Hybrid fusion makes it worse.** An old version and its replacement are close under *both* lexical and semantic similarity, so rank fusion rewards whichever of them matches more consistently, and that is often the older, more widely cited one. Hybrid is still the best base once lineage is applied (current@1 0.92), because it retrieves the whole lineage chain.
- Dropping superseded hits isn't enough (see the `filter` rows in [results/](results/)): recall is lost when the replacement was never retrieved.
- **Natural-language questions make it worse.** On 69 LLM-written change-aware questions (see below), a superseded RFC ranks first for 56% (BM25), 55% (dense) and **65% (hybrid)** of questions [CIs ±12 pts], against 42–53% for title queries. Oracle promotion brings current@1 to 0.84–0.93. Table: [results/rfc_questions_all.md](results/rfc_questions_all.md).
- These "promote" numbers **assume lineage metadata is available**. Real corpora rarely have it, and inferring it is the next research step.

Full tables, with all metrics for both splits, are in [results/](results/).

## Reproduce

```bash
make install
make test
make reproduce RFC_INDEX_SHA256=e1669ddd5a87561fe548e737865f486e376ae58b5011a73a92a72f0aac32363a
```

`make reproduce` downloads `rfc-index.xml` if it's missing, builds `data/rfc/` (`corpus.jsonl`, `lineage.jsonl`, `queries.jsonl`, `manifest.json`) and writes `results/rfc_staleness_{dev,test}.{json,md}`. The first run downloads the embedding model and caches document embeddings under `data/rfc/emb/`; it takes about 2.5 minutes on an Apple Silicon laptop, with no API keys. Reruns produce byte-identical results. For the dependency-free BM25-only run, use `make reproduce RETRIEVERS=bm25`.

The RFC index is a live file. The published numbers use the snapshot with the SHA-256 above, and the build fails if the hash doesn't match. A frozen copy of the snapshot will be released alongside the dataset.

### Change-aware questions (LLM step)

Title queries favour lexical matching, so the next query set is LLM-written questions whose correct answer *differs* between a document and its replacement. Each item carries the current and stale answers with a verbatim evidence span from each abstract, and items that fail that grounding check are rejected.

```bash
make questions QUESTIONS_LIMIT=50     # run from a regular terminal, not inside Claude Code
```

The default backend is the Claude Code CLI in headless mode (`claude -p`, model `claude-sonnet-5-5`), which runs on a Claude subscription with no API key. `LLM_BACKEND=ollama LLM_MODEL=qwen2.5:7b` uses a local model instead. Every response goes into `cache/llm.sqlite`, which is committed, so `make reproduce` rebuilds the questions with `--offline` and makes no model calls. The acceptance rate and rejection reasons are written to `data/rfc/questions_manifest.json`.

Prompt iterations so far, on the same seeded sample of supersession pairs:

| prompt | change | accepted | what the inspection showed |
|---|---|---|---|
| q-v1 | baseline | 3 / 5 | 2 of 3 questions were about editorial wording ("official" status report), and they listed the answer options |
| q-v2 | substantive change types only; no answer leaks | 6 / 20 | all substantive, but 2 "before" answers argued from silence ("the abstract mentions no support") |
| q-v3 | answers must be positively stated | 4 / 20 | all grounded and substantive. Most pairs are skipped because the abstracts differ only editorially, so full texts are needed to scale past roughly 270 questions |

Every response for all three prompt versions stays in the cache, for transparency.

**First batch (q-v3, 300 sampled pairs):** 69 accepted (23%). The model skipped 224 pairs, and the deterministic checks rejected 7 that it produced (3 non-verbatim evidence, 2 RFC-number leaks, 1 answer leak, 1 argument from silence). By change type: scope 48, mechanism 9, reference 8, requirement 3, value 1. Scope questions dominate because abstracts describe coverage more than values. Notional API price: $4.72 for 280 new calls.

### Benchmark design notes

- **Supersession vs. amendment.** Only `obsoletes` counts as supersession. `updates` (partial amendment) is kept in the corpus for the conflict-aware answering work, but it doesn't make a document stale.
- **Splits are by lineage component** (a 30/70 dev/test split hashed on the component), so no topic appears in both dev and test.
- **Split replacements.** A document replaced by several others has several current heads. Retrieving any one of them counts as current.
- **Data quality is reported.** The builder counts links declared on only one side and links to documents outside the corpus. Both are 0 for the RFC snapshot.
- **v0 limitation:** queries are document titles, which favours lexical matching. Natural-language questions answerable from abstracts (and later from full text) come next.

## Roadmap

| | Milestone |
|---|---|
| ✅ | v0.1: RFC lineage benchmark, BM25 baseline, oracle reranking, bootstrap CIs, CI tests |
| ✅ | v0.2: dense and hybrid (RRF) retrievers, cached embeddings |
| 🚧 | v0.3: LLM layer (Claude CLI, Ollama) with a published response cache; change-aware questions *(code done, generation pending)*; answer-level staleness with a human-validated judge |
| | Librarian agent that **infers** lineage without metadata (dates, titles, NLI, semantic entropy across versions) |
| | Contamination injector: synthetic fabricated standards and contradictions at controlled rates (dose-response) |
| | FAA Advisory Circulars (`cancels AC xx-yyA`) and UNECE regulations (series of amendments) |
| | Answer policies: "latest only" vs. "show the conflict and cite both", with human-validated judging |
| | Ingestion service + MCP server (`search`, `get_lineage`, `conflicts`), Terraform on GCP, latency and cost table |
| | Dataset on the Hugging Face Hub, paper, leaderboard |

## Layout

```
src/supersede/
  corpus.py        Doc model, JSONL I/O
  lineage.py       supersession graph: current versions, components
  retrieval.py     BM25 (dependency-free), dense, hybrid RRF
  rerank.py        version-aware policies (none / filter / promote)
  metrics.py       stale@1, current@k, MRR, bootstrap CI
  llm.py           LLM backends (claude -p, Ollama, mock) + SQLite response cache
  bench/rfc.py     RFC corpus + query builder
  bench/questions.py  change-aware question generator with grounding checks
  experiments/     one module per experiment, one command each
results/           committed outputs of `make reproduce`
```

## License

Code: Apache-2.0. RFC metadata comes from the [RFC Editor](https://www.rfc-editor.org/), under the IETF Trust's legal provisions.
