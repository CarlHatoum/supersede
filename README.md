# Supersede

**Document versions in retrieval-augmented generation.** Collections keep old documents
next to the ones that replace them, and retrievers rank them as equally valid. Supersede
is an open benchmark and toolkit for measuring that, and for fixing it when the
collection doesn't record which document replaces which.

- **Benchmark:** [`hatoum/supersede-rfc`](https://huggingface.co/datasets/hatoum/supersede-rfc)
  on Hugging Face. 9,852 IETF RFCs, 1,543 replacement links, title queries,
  change-aware questions and labelled candidate pairs, built from the RFC Editor's
  index.
- **Librarian:** infers the lineage (which document replaces which) without metadata. A
  cheap feature-based scorer settles clear pairs; ambiguous ones go to a model that
  reads both documents, either in one call or as a LangGraph agent with tools served
  over MCP. Runs on local models through Ollama.
- **Service:** a FastAPI app with lineage-aware search, the assessor, the MCP tools
  over HTTP and an A2A agent card, plus Dockerfiles for Cloud Run.
- **Experiments:** one module and one command per experiment. LLM calls are cached
  and committed, so every result rebuilds offline.

Results will be published in a blog post. Progress notes are in
[`docs/STATUS.md`](docs/STATUS.md).

## Quick start

```bash
make install            # uv venv + all extras (Python 3.12)
make data               # download the pinned RFC index, build data/rfc/
make test
```

**Use the dataset** without this repository:

```python
from datasets import load_dataset

queries = load_dataset("hatoum/supersede-rfc", "queries", split="test")
q = queries[0]
print(q["text"], "| current:", q["current"], "| stale:", q["stale"])
```

**Run the service** (first start builds the inferred lineage and embeddings, then
about 20 s):

```bash
.venv/bin/uvicorn supersede.serve.app:app --port 8080
curl "localhost:8080/search?q=TCP%20Congestion%20Control&k=3"      # current version first
curl "localhost:8080/search?q=TCP%20Congestion%20Control&lineage=none"
curl -X POST localhost:8080/assess -H 'content-type: application/json' \
     -d '{"old": "RFC2246", "new": "RFC4346"}'                      # needs Ollama
scripts/demo.sh                                                     # the whole walkthrough
```

| endpoint | what it does |
|---|---|
| `GET /search?q=&k=&lineage=inferred\|declared\|none` | hybrid retrieval (BM25 + dense), current versions promoted above what they replace |
| `GET /lineage/{doc_id}` | what replaces a document and what it replaces, inferred and declared |
| `POST /assess {"old", "new"}` | the cascade: cheap scorer outside its uncertain band, the model inside it |
| `/mcp` | MCP tools `search`, `read_doc`, `compare`, `find_later` over streamable HTTP |
| `/.well-known/agent-card.json`, `POST /a2a` | A2A agent card and synchronous `message/send` |

Settings: `SUPERSEDE_MODEL` (default `ollama:edtorre/qwen3.5-hermes`),
`SUPERSEDE_VARIANT` (`llm_only` or `full`), `OLLAMA_BASE_URL`, `SUPERSEDE_DATA`.

**Use the tools from Claude Code:**

```bash
claude mcp add --transport http supersede http://localhost:8080/mcp/       # over HTTP
claude mcp add supersede -- .venv/bin/python -m supersede.serve.mcp_server  # or over stdio
```

**Deploy to Cloud Run:** two services, defined in [`infra/terraform/`](infra/terraform/).
`supersede-api` (CPU; corpus, embeddings and lineage built into the image) is public;
`supersede-llm` (Ollama with the model built in, one NVIDIA L4, scales to zero)
accepts only internal traffic, which the API sends through Direct VPC egress. Images
are built on Cloud Build, so no local Docker is needed.

```bash
gcloud auth login && gcloud auth application-default login
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars  # set project_id
deploy/deploy.sh            # base infra -> image builds -> services -> smoke test
```

The project needs Cloud Run L4 GPU quota in the chosen region. Setting
`billing_account` in `terraform.tfvars` adds a monthly budget alert.

## Experiments

| command | question |
|---|---|
| `make reproduce` | how often retrieval ranks a superseded version first (title queries and questions; BM25, dense, hybrid; no policy, filter, promote, newest-first, time decay) |
| `make questions` | generate change-aware questions (LLM, cached) |
| `make answers` | how often a RAG answer gives the outdated answer, per answer policy |
| `python -m supersede.experiments.closed_book` | what the answering model says with no documents |
| `make lineage` | inferring the lineage from content: heuristic, version chains, logistic regression; edge P/R/F1 and the downstream gap closed |
| `python -m supersede.experiments.recency_noise` | recency and inferred lineage when years are missing or wrong |
| `python -m supersede.experiments.agent_lineage` | model decisions on ambiguous pairs (variants: `llm_only`, `no_find_later`, `full`); cached per pair, resumable |
| `python -m supersede.experiments.agent_compare` | balanced accuracy with bootstrap CIs, exact McNemar tests, cost, run-to-run stability |
| `python -m supersede.experiments.agent_downstream` | retrieval with the cascade's lineage |
| `scripts/agent_queue.sh <queue>` | long agent runs, one model at a time, keeping the Mac awake |

**LLM calls.** Question generation, answering and judging use the Claude Code CLI in
headless mode (`claude -p`, run from a regular terminal) or a local Ollama model.
Every response goes into `cache/llm.sqlite`, which is committed, so `--offline` reruns
make no calls. The agent runs only on local models.

**Reproducibility.** The RFC index is a live file; the build checks the SHA-256 of
the pinned snapshot (`make data` / `make reproduce` use it by default) and fails on a
mismatch. Retrieval results are byte-identical across reruns.

## Benchmark design notes

- **Supersession vs. amendment.** Only `obsoletes` counts as replacement. `updates`
  (partial amendment) stays in the corpus but doesn't make a document stale.
- **Splits are by lineage** (about 30% dev, 70% test, hashed on the connected
  component), so no lineage appears in both.
- **Split replacements.** A document replaced by several others has several current
  versions; retrieving any of them counts as current.
- **No leakage for lineage inference.** Many abstracts say "This document obsoletes
  RFC 2581". The `masked` setting hides RFC numbers; `strict` also drops sentences with
  a supersession verb. The dataset ships verbatim text (as the IETF licence requires)
  with the script that derives both views.
- **Data quality is reported:** links declared on one side only, and links to
  documents outside the corpus (both 0 for the pinned snapshot).

## Layout

```
src/supersede/
  corpus.py, lineage.py      documents, supersession graph, current versions
  retrieval.py, rerank.py    BM25, dense, hybrid RRF; version-aware and recency re-rankers
  metrics.py, judge.py       stale@1, current@k, bootstrap CIs; answer judge
  answer.py, llm.py          answer policies; LLM backends with a SQLite cache
  bench/                     RFC builder, question generator, Hugging Face export
  librarian/                 candidates, features, cheap scorers, masking, agent, MCP-facing tools
  serve/                     FastAPI app (search, lineage, assess, A2A) and MCP server
  experiments/               one module per experiment
deploy/                      Dockerfiles, Cloud Build config, deploy script
infra/terraform/             Cloud Run services, registry, VPC, budget
scripts/                     agent queue, demo
results/                     committed outputs
```

## License

Code: Apache-2.0. RFC metadata and abstracts come from the
[RFC Editor](https://www.rfc-editor.org/) index under the IETF Trust Legal Provisions;
the dataset's own annotations are CC BY 4.0.
