PY ?= .venv/bin/python
# Snapshot used for the published numbers. Leave empty to accept any snapshot.
RFC_INDEX_SHA256 ?= e1669ddd5a87561fe548e737865f486e376ae58b5011a73a92a72f0aac32363a
RETRIEVERS ?= bm25,dense,hybrid
# Change-aware questions: how many supersession edges to try (seeded sample).
QUESTIONS_LIMIT ?= 300
LLM_BACKEND ?= claude-cli
LLM_MODEL ?= claude-sonnet-5-5

.PHONY: install test lint data questions answers lineage reproduce paper hf

install:
	uv venv -p 3.12 .venv && uv pip install --python $(PY) -e '.[dev,dense,agent,hub,paper,serve]'

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests && $(PY) -m ruff format --check src tests

data:
	$(PY) -m supersede.bench.rfc --out data/rfc $(if $(RFC_INDEX_SHA256),--expect-sha256 $(RFC_INDEX_SHA256))

# Calls the LLM for cache misses. With the claude-cli backend, run it from a
# regular terminal (not inside Claude Code); it uses your Claude plan.
questions: data
	$(PY) -m supersede.bench.questions --data data/rfc --backend $(LLM_BACKEND) --model $(LLM_MODEL) --limit $(QUESTIONS_LIMIT)

# Rebuilds every committed result. LLM outputs come from cache/llm.sqlite only.
reproduce: data
	$(PY) -m supersede.experiments.staleness --data data/rfc --split dev --retrievers $(RETRIEVERS)
	$(PY) -m supersede.experiments.staleness --data data/rfc --split test --retrievers $(RETRIEVERS)
	@if [ -f cache/llm.sqlite ]; then \
	  $(PY) -m supersede.bench.questions --data data/rfc --backend $(LLM_BACKEND) --model $(LLM_MODEL) --limit $(QUESTIONS_LIMIT) --offline && \
	  $(PY) -m supersede.experiments.staleness --data data/rfc --split all --retrievers $(RETRIEVERS) --queries questions.jsonl; \
	fi

# Answers (Sonnet) + judge (Haiku) on the change-aware questions. Resumes from the cache.
answers:
	$(PY) -m supersede.experiments.answer_staleness --data data/rfc --retriever hybrid --backend $(LLM_BACKEND) --model $(LLM_MODEL) --judge-model claude-haiku-5-5

lineage:
	$(PY) -m supersede.experiments.lineage_inference --data data/rfc

# Regenerates every number and table from results/, then compiles paper/main.pdf.
TECTONIC ?= $(shell command -v tectonic || echo /opt/homebrew/bin/tectonic)
paper:
	$(PY) -m supersede.report.paper --results results --data data/rfc --out paper/generated
	$(PY) -m supersede.report.figures --results results --out paper/generated
	$(TECTONIC) --keep-logs -o paper paper/main.tex

# Builds the Hugging Face dataset folder in dist/hf/ (no upload).
# Upload: $(PY) -m supersede.bench.export_hf --push <user>/supersede-bench [--private]
hf:
	$(PY) -m supersede.bench.export_hf --data data/rfc --out dist/hf/supersede-bench
