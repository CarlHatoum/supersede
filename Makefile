PY ?= .venv/bin/python
# Snapshot used for the published numbers. Leave empty to accept any snapshot.
RFC_INDEX_SHA256 ?= e1669ddd5a87561fe548e737865f486e376ae58b5011a73a92a72f0aac32363a
RETRIEVERS ?= bm25,dense,hybrid

.PHONY: install test lint data reproduce

install:
	uv venv -p 3.12 .venv && uv pip install --python $(PY) -e '.[dev,dense]'

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests && $(PY) -m ruff format --check src tests

data:
	$(PY) -m supersede.bench.rfc --out data/rfc $(if $(RFC_INDEX_SHA256),--expect-sha256 $(RFC_INDEX_SHA256))

reproduce: data
	$(PY) -m supersede.experiments.staleness --data data/rfc --split dev --retrievers $(RETRIEVERS)
	$(PY) -m supersede.experiments.staleness --data data/rfc --split test --retrievers $(RETRIEVERS)
