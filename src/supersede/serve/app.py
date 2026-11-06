"""HTTP API: lineage-aware search, lineage lookup and pair assessment, with MCP at ``/mcp``.

Endpoints:

* ``GET /healthz``
* ``GET /search?q=...&k=10&lineage=inferred|declared|none``: hybrid retrieval
  (top 50), then current versions promoted above the documents they replace.
* ``GET /lineage/{doc_id}``: what replaces this document and what it replaces,
  in the inferred graph (and in the declared one, for comparison).
* ``POST /assess {"old": ..., "new": ...}``: the cascade. The cheap scorer decides
  pairs outside its uncertain band; the model decides the rest.
* ``/mcp``: the librarian's MCP tools over streamable HTTP.
* A2A: agent card at ``/.well-known/agent-card.json``; JSON-RPC ``message/send``
  at ``/a2a`` (synchronous; data part ``{"old": ..., "new": ...}``).

The inferred lineage is librarian v0 on the ``strict`` setting (every explicit
cue removed), computed once and cached next to the data.

Configuration (environment):
    SUPERSEDE_DATA     corpus directory (default data/rfc)
    SUPERSEDE_MODEL    model for /assess (default ollama:edtorre/qwen3.5-hermes)
    SUPERSEDE_VARIANT  llm_only (default; as accurate as the agent, 4x cheaper) or full
    OLLAMA_BASE_URL    remote Ollama (default: local)

Run:
    uvicorn supersede.serve.app:app --port 8080
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from supersede.corpus import Doc, load_docs
from supersede.lineage import LineageGraph
from supersede.rerank import promote_current

DEPTH = 50
BAND = (0.8, 0.98)
Assessor = Callable[[str, str], Awaitable[dict]]


@dataclass
class Service:
    docs: dict[str, Doc]
    retriever: object  # anything with search(query, k) -> [(doc_id, score)]
    inferred: LineageGraph
    declared: LineageGraph
    pair_proba: dict[tuple[str, str], float] = field(default_factory=dict)
    threshold: float = 0.5
    assessor: Assessor | None = None
    info: dict = field(default_factory=dict)

    @classmethod
    def load(cls, data: Path, setting: str = "strict") -> Service:
        from supersede.experiments.staleness import build_retrievers

        docs = load_docs(data / "corpus.jsonl")
        declared = LineageGraph(docs)
        state = inferred_lineage(data, docs, declared, setting)
        edges = [tuple(e) for e in state["edges"]]
        hybrid = build_retrievers(["hybrid"], [(d.doc_id, d.text) for d in docs], data)["hybrid"]
        return cls(
            docs={d.doc_id: d for d in docs},
            retriever=hybrid,
            inferred=LineageGraph.from_edges(docs, edges),
            declared=declared,
            pair_proba={(o, n): p for o, n, p in state["band"]},
            threshold=state["threshold"],
            info={"setting": setting, "inferred_edges": len(edges)},
        )


def inferred_lineage(data: Path, docs, declared: LineageGraph, setting: str) -> dict:
    """Librarian-v0 edges, threshold and band-pair scores, cached in ``data``."""
    cache = data / f"serve_lineage_{setting}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from supersede.experiments.lineage_inference import cheap_scores
    from supersede.librarian.classifiers import select_edges

    cs = cheap_scores(setting, data, docs, declared)
    state = {
        "setting": setting,
        "threshold": cs.threshold,
        "edges": select_edges(cs.pairs, cs.proba, cs.threshold),
        # Every candidate pair's score would be ~120k rows; /assess needs the band only,
        # plus enough to say "confidently no" for the rest (absent = not a candidate).
        "band": [
            (o, n, round(float(p), 4))
            for (o, n), p in zip(cs.pairs, cs.proba, strict=True)
            if p >= 0.5 * BAND[0]
        ],
    }
    cache.write_text(json.dumps(state))
    return state


def model_assessor(spec: str, variant: str, data: Path, base_url: str | None) -> Assessor:
    graph = None

    async def assess(old: str, new: str) -> dict:
        nonlocal graph
        if graph is None:
            from supersede.librarian.agent import build_variant, make_chat_model
            from supersede.librarian.tools import load_corpus_tools

            model = make_chat_model(spec, base_url=base_url)
            graph = build_variant(variant, model, load_corpus_tools(data, "strict"), max_steps=8)
        state = await graph.ainvoke({"old": old, "new": new, "score": None})
        return {**state["decision"], "model": spec, "variant": variant}

    return assess


class Pair(BaseModel):
    old: str
    new: str


def doc_view(d: Doc, graph: LineageGraph) -> dict:
    return {
        "doc_id": d.doc_id,
        "title": d.title,
        "year": d.year,
        "superseded": graph.is_superseded(d.doc_id),
        "current_versions": sorted(graph.current_versions(d.doc_id))
        if graph.is_superseded(d.doc_id)
        else [],
    }


def create_app(service: Service | Callable[[], Service]) -> FastAPI:
    from supersede.serve.mcp_server import build_mcp

    mcp = build_mcp()
    mcp.settings.streamable_http_path = "/"
    mcp_app = mcp.streamable_http_app()
    holder: dict[str, Service] = {}

    def svc() -> Service:
        if "s" not in holder:
            holder["s"] = service() if callable(service) else service
        return holder["s"]

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        svc().retriever.search("warm-up", 1)  # load index and encoder before serving
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="Supersede", version="0.1", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        s = svc()
        return {"status": "ok", "docs": len(s.docs), **s.info}

    @app.get("/search")
    def search(q: str, k: int = 10, lineage: str = "inferred") -> dict:
        s = svc()
        graphs = {"inferred": s.inferred, "declared": s.declared, "none": None}
        if lineage not in graphs:
            raise HTTPException(400, f"lineage must be one of {sorted(graphs)}")
        t0 = time.perf_counter()
        ranking = [d for d, _ in s.retriever.search(q, DEPTH)]
        graph = graphs[lineage]
        if graph is not None:
            ranking = promote_current(ranking, graph)
        shown = graph or s.inferred
        return {
            "query": q,
            "lineage": lineage,
            "results": [doc_view(s.docs[d], shown) for d in ranking[:k]],
            "ms": round(1000 * (time.perf_counter() - t0), 1),
        }

    @app.get("/lineage/{doc_id}")
    def lineage(doc_id: str) -> dict:
        s = svc()
        if doc_id not in s.docs:
            raise HTTPException(404, f"unknown document {doc_id}")

        def side(g: LineageGraph) -> dict:
            return {
                "replaced_by": sorted(g.successors.get(doc_id, ())),
                "replaces": sorted(g.predecessors.get(doc_id, ())),
                "current_versions": sorted(g.current_versions(doc_id)),
            }

        return {"doc_id": doc_id, "inferred": side(s.inferred), "declared": side(s.declared)}

    @app.post("/assess")
    async def assess(pair: Pair) -> dict:
        s = svc()
        for d in (pair.old, pair.new):
            if d not in s.docs:
                raise HTTPException(404, f"unknown document {d}")
        p = s.pair_proba.get((pair.old, pair.new), 0.0)
        if not BAND[0] <= p < BAND[1]:
            label = "supersedes" if p >= s.threshold else "unrelated"
            return {"label": label, "decided_by": "cheap_scorer", "score": p}
        if s.assessor is None:
            raise HTTPException(503, "no model configured for uncertain pairs")
        t0 = time.perf_counter()
        out = await s.assessor(pair.old, pair.new)
        return {**out, "decided_by": "model", "score": p, "seconds": time.perf_counter() - t0}

    @app.get("/.well-known/agent-card.json")
    @app.get("/.well-known/agent.json")
    def agent_card(request: Request) -> dict:
        base = str(request.base_url).rstrip("/")
        return {
            "protocolVersion": "0.3.0",
            "name": "Supersede librarian",
            "description": "Decides whether a newer document replaces an older one, "
            "without relying on declared lineage or dates.",
            "url": f"{base}/a2a",
            "preferredTransport": "JSONRPC",
            "version": "0.1.0",
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": ["application/json"],
            "defaultOutputModes": ["application/json"],
            "skills": [
                {
                    "id": "assess_supersession",
                    "name": "Assess supersession",
                    "description": 'Send a data part {"old": <doc id>, "new": <doc id>}; get a '
                    "label (supersedes, amends, conflicts, unrelated), a confidence "
                    "and a rationale.",
                    "tags": ["documents", "versioning", "rag"],
                    "examples": ['{"old": "RFC2581", "new": "RFC5681"}'],
                }
            ],
        }

    @app.post("/a2a")
    async def a2a(body: dict) -> dict:
        """A2A JSON-RPC, synchronous ``message/send`` only."""
        rid = body.get("id")

        def error(code: int, message: str) -> dict:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

        if body.get("method") != "message/send":
            return error(-32601, "only message/send is supported")
        parts = body.get("params", {}).get("message", {}).get("parts", [])
        data = next((p.get("data") for p in parts if p.get("kind") == "data"), None)
        if not isinstance(data, dict) or not {"old", "new"} <= data.keys():
            return error(-32602, 'expected a data part {"old": ..., "new": ...}')
        try:
            out = await assess(Pair(old=str(data["old"]), new=str(data["new"])))
        except HTTPException as e:
            return error(-32602, str(e.detail))
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "kind": "message",
                "role": "agent",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "data", "data": out}],
            },
        }

    app.mount("/mcp", mcp_app)
    return app


def default_service() -> Service:
    data = Path(os.environ.get("SUPERSEDE_DATA", "data/rfc"))
    s = Service.load(data)
    s.assessor = model_assessor(
        os.environ.get("SUPERSEDE_MODEL", "ollama:edtorre/qwen3.5-hermes"),
        os.environ.get("SUPERSEDE_VARIANT", "llm_only"),
        data,
        os.environ.get("OLLAMA_BASE_URL"),
    )
    return s


app = create_app(default_service)
