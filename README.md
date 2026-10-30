# Supersede

**Your RAG is citing an obsolete standard.** Knowledge bases rot. Documents get replaced by newer versions, contradict each other, or are made up by an LLM and dropped into a shared drive. Retrievers rank by similarity, so a superseded document that matches the query well wins over its current replacement.

Supersede is an open benchmark and toolkit for measuring and fixing this:

1. **Benchmark.** Public corpora where the replacement relation is already recorded, so no labelling is needed. IETF RFCs are available now; FAA Advisory Circulars and UNECE vehicle regulations are planned.
2. **Fixes**, at two points: an ingestion-time *librarian* that infers lineage and quarantines suspect documents, and retrieval/answer-time policies that surface the current version and cite the old one.
3. **Production path:** an incremental ingestion service, an MCP server, and published latency and cost numbers *(planned)*.

> Status: **v0.1**. RFC slice, BM25 baseline and lineage-aware reranking with known (oracle) lineage. No LLM is involved yet, so every number below is deterministic and free to reproduce.

## First result: lexical retrieval prefers superseded documents

The RFC index lists 9,852 RFCs. 1,384 of them are obsoleted by a later RFC, through 1,543 `obsoletes` links that form 800 lineage chains, the longest with 35 documents. For each link we query the topic in two ways: by the old document's title (the name users know) and by the new one's. The gold answer is the *current* head of the lineage.

Test split, BM25 top-10, 95% bootstrap CI:

| query wording | policy | n | stale@1 ↓ | current@1 ↑ | current@10 ↑ |
|---|---|---|---|---|---|
| new title | none | 727 | **0.417** [0.382, 0.451] | 0.440 [0.406, 0.477] | 0.931 |
| new title | filter superseded | 727 | 0.000 | 0.750 [0.719, 0.781] | 0.931 |
| new title | promote current | 727 | 0.000 | **0.857** [0.832, 0.882] | 0.983 |
| old title | none | 784 | **0.722** [0.691, 0.755] | 0.152 [0.126, 0.177] | 0.790 |
| old title | filter superseded | 784 | 0.000 | 0.568 [0.532, 0.602] | 0.790 |
| old title | promote current | 784 | 0.000 | **0.874** [0.849, 0.895] | 0.986 |

- Even when the query uses the **new** document's own wording, a superseded version ranks first **42%** of the time. With the familiar old name it's **72%**.
- Dropping superseded hits isn't enough: recall is lost when the replacement was never retrieved.
- Replacing each stale hit with its current version (while keeping the stale one below it) fixes most of the gap. *This assumes lineage metadata is available.* Real corpora rarely have it, and inferring it is the next research step.

Full tables, with all metrics for both splits, are in [results/](results/).

## Reproduce

```bash
make install
make test
make reproduce RFC_INDEX_SHA256=e1669ddd5a87561fe548e737865f486e376ae58b5011a73a92a72f0aac32363a
```

`make reproduce` downloads `rfc-index.xml` if it's missing, builds `data/rfc/` (`corpus.jsonl`, `lineage.jsonl`, `queries.jsonl`, `manifest.json`) and writes `results/rfc_staleness_{dev,test}.{json,md}`. It takes about 30 seconds on a laptop, with no GPU and no API keys.

The RFC index is a live file. The published numbers use the snapshot with the SHA-256 above, and the build fails if the hash doesn't match. A frozen copy of the snapshot will be released alongside the dataset.

### Benchmark design notes

- **Supersession vs. amendment.** Only `obsoletes` counts as supersession. `updates` (partial amendment) is kept in the corpus for the conflict-aware answering work, but it doesn't make a document stale.
- **Splits are by lineage component** (a 30/70 dev/test split hashed on the component), so no topic appears in both dev and test.
- **Split replacements.** A document replaced by several others has several current heads. Retrieving any one of them counts as current.
- **Data quality is reported.** The builder counts links declared on only one side and links to documents outside the corpus. Both are 0 for the RFC snapshot.
- **v0 limitation:** queries are document titles, which favours lexical matching. Natural-language questions answerable from abstracts (and later from full text) come next.

## Roadmap

| | Milestone |
|---|---|
| ✅ | RFC lineage benchmark, BM25 baseline, oracle reranking, bootstrap CIs, CI tests |
| ⏭ | Dense and hybrid retrievers; natural-language questions; full-text chunks |
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
  retrieval.py     dependency-free BM25
  rerank.py        version-aware policies (none / filter / promote)
  metrics.py       stale@1, current@k, MRR, bootstrap CI
  bench/rfc.py     RFC corpus + query builder
  experiments/     one module per experiment, one command each
results/           committed outputs of `make reproduce`
```

## License

Code: Apache-2.0. RFC metadata comes from the [RFC Editor](https://www.rfc-editor.org/), under the IETF Trust's legal provisions.
