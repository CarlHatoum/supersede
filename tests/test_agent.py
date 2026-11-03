import asyncio

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage  # noqa: E402

from supersede.corpus import Doc  # noqa: E402
from supersede.librarian.agent import (  # noqa: E402
    Decision,
    build_librarian,
    corpus_langchain_tools,
    trace,
)
from supersede.librarian.tools import CorpusTools  # noqa: E402
from supersede.retrieval import BM25  # noqa: E402

DOCS = [
    Doc(
        "RFC2581",
        "TCP Congestion Control",
        "Defines four congestion control algorithms.",
        year=1999,
        obsoleted_by=("RFC5681",),
        status="OBSOLETE",
    ),
    Doc(
        "RFC5681",
        "TCP Congestion Control",
        "Specifies four TCP congestion control algorithms. This document obsoletes RFC 2581.",
        year=2009,
        obsoletes=("RFC2581",),
    ),
    Doc("RFC9000", "QUIC Transport", "A UDP-based transport.", year=2021),
]


def corpus(setting="masked"):
    return CorpusTools(DOCS, BM25([(d.doc_id, d.text) for d in DOCS]), setting)


def test_tools_hide_lineage_and_mask_references():
    c = corpus()
    doc = c.read_doc("RFC5681")
    assert set(doc) == {"doc_id", "title", "year", "keywords", "abstract"}
    assert "2581" not in doc["abstract"] and "[REF]" in doc["abstract"]
    assert "2581" in corpus("declared").read_doc("RFC5681")["abstract"]
    assert c.read_doc("nope") == {"error": "unknown document 'nope'"}
    hits = c.search("tcp congestion control", k=2)
    assert {h["doc_id"] for h in hits} == {"RFC2581", "RFC5681"}
    cmp = c.compare("RFC2581", "RFC5681")
    assert cmp["older"] == "RFC2581" and cmp["title_word_overlap"] == 1.0
    assert "error" in c.compare("RFC2581", "nope")


def test_mcp_server_serves_the_same_tools(monkeypatch):
    from supersede.serve import mcp_server

    monkeypatch.setattr(mcp_server, "corpus", lambda: corpus())
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert names == {"search", "read_doc", "compare"}
    result = asyncio.run(mcp_server.mcp.call_tool("read_doc", {"doc_id": "RFC2581"}))
    assert "TCP Congestion Control" in str(result)


class ScriptedModel:
    """Duck-typed chat model: plays back tool calls, then returns a fixed verdict."""

    def __init__(self, replies, verdict):
        self.replies, self.verdict, self.seen = list(replies), verdict, []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        outer = self

        class Judge:
            async def ainvoke(self, messages):
                outer.seen.append(messages)
                return outer.verdict

        return Judge()

    async def ainvoke(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


def run(graph, **state):
    return asyncio.run(graph.ainvoke({"old": "RFC2581", "new": "RFC5681", **state}))


def test_agent_investigates_with_tools_then_decides():
    model = ScriptedModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "compare", "id": "1", "args": {"doc_a": "RFC2581", "doc_b": "RFC5681"}}
                ],
            ),
            AIMessage(
                "",
                tool_calls=[
                    {"name": "search", "id": "2", "args": {"query": "tcp congestion control"}}
                ],
            ),
            AIMessage("Same topic, newer and broader. Done."),
        ],
        Decision(label="supersedes", confidence=0.8, rationale="same title, later year"),
    )
    graph = build_librarian(model, corpus_langchain_tools(corpus()), lambda d: ("t", 2000))
    state = run(graph, score=0.5)
    assert state["decision"]["label"] == "supersedes"
    assert state["decision"]["source"] == "agent" and state["decision"]["steps"] == 3
    assert trace(state)[0].startswith("→ compare(")
    tool_outputs = [m for m in state["messages"] if m.type == "tool"]
    assert len(tool_outputs) == 2 and '"older": "RFC2581"' in tool_outputs[0].content


def test_triage_skips_the_model_on_clear_cases():
    model = ScriptedModel([], Decision(label="unrelated", confidence=1, rationale="-"))
    graph = build_librarian(model, corpus_langchain_tools(corpus()), low=0.2, high=0.9)
    assert run(graph, score=0.95)["decision"]["label"] == "supersedes"
    assert run(graph, score=0.05)["decision"]["label"] == "unrelated"
    assert model.seen == []


def test_step_budget_forces_a_decision():
    loop = AIMessage("", tool_calls=[{"name": "search", "id": "x", "args": {"query": "tcp"}}])
    model = ScriptedModel([loop] * 5, Decision(label="conflicts", confidence=0.4, rationale="-"))
    graph = build_librarian(model, corpus_langchain_tools(corpus()), max_steps=2)
    state = run(graph, score=None)
    assert state["decision"]["label"] == "conflicts" and state["decision"]["steps"] == 2
