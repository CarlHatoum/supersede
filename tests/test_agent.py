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
    assert names == {"search", "find_later", "read_doc", "compare"}
    result = asyncio.run(mcp_server.mcp.call_tool("read_doc", {"doc_id": "RFC2581"}))
    assert "TCP Congestion Control" in str(result)


class ScriptedModel:
    """Duck-typed chat model: plays back tool calls, then returns a fixed verdict."""

    def __init__(self, replies, verdict):
        self.replies, self.verdict, self.seen = list(replies), verdict, []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, **kwargs):
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


def test_band_sampling_and_rates():
    from types import SimpleNamespace

    from supersede.experiments.agent_lineage import est_precision, rates, sample_band

    cs = SimpleNamespace(
        pairs=[("a", str(i)) for i in range(8)],
        y=[1, 0, 1, 0, 0, 0, 1, 0],
        split=["test"] * 7 + ["dev"],
        proba=[0.85, 0.85, 0.95, 0.95, 0.5, 0.99, 0.9, 0.9],
    )
    picked, prior, band = sample_band(cs, 0.8, 0.98, n=4, seed=0)
    assert len(band) == 5
    # Band = test pairs 0, 1, 2, 3, 6 (pair 7 is dev; 4 and 5 are outside).
    assert prior == 3 / 5
    assert sum(p["gold"] for p in picked) == 2 and len(picked) == 4
    rows = [{"gold": 1, "x": True}, {"gold": 1, "x": False}, {"gold": 0, "x": False}]
    m = rates(rows, lambda r: r["x"])
    assert m == {"tpr": 0.5, "tnr": 1.0, "balanced_accuracy": 0.75}
    assert est_precision(0.5, 1.0, 0.1) == 1.0


def test_find_later_lists_later_same_topic_documents():
    c = corpus()
    later = [h["doc_id"] for h in c.find_later("RFC2581")]
    assert later[0] == "RFC5681" and "RFC2581" not in later
    assert "RFC2581" not in [h["doc_id"] for h in c.find_later("RFC5681")]
    assert "error" in c.find_later("nope")[0]


def test_balanced_accuracy_ci_brackets_the_estimate():
    from supersede.experiments.agent_lineage import balanced_accuracy_ci

    rows = [{"gold": 1, "x": i < 8} for i in range(10)] + [
        {"gold": 0, "x": i < 3} for i in range(10)
    ]
    lo, hi = balanced_accuracy_ci(rows, lambda r: r["x"])
    assert lo <= 0.75 <= hi and hi - lo > 0.1


def test_band_sample_is_nested_and_balanced():
    from types import SimpleNamespace

    from supersede.experiments.agent_lineage import sample_band

    n_pairs = 60
    cs = SimpleNamespace(
        pairs=[("a", str(i)) for i in range(n_pairs)],
        y=[int(i % 3 == 0) for i in range(n_pairs)],
        split=["test"] * n_pairs,
        proba=[0.9] * n_pairs,
    )
    small, _, _ = sample_band(cs, 0.8, 0.98, n=6)
    large, _, _ = sample_band(cs, 0.8, 0.98, n=20)
    assert small == large[:6]
    assert sum(p["gold"] for p in small) == 3 and sum(p["gold"] for p in large) == 10


def test_variants_build_and_llm_only_makes_one_call():
    from supersede.librarian.agent import build_variant

    verdict = Decision(label="supersedes", confidence=0.7, rationale="same scope")
    model = ScriptedModel([], verdict)
    graph = build_variant("llm_only", model, corpus())
    state = asyncio.run(graph.ainvoke({"old": "RFC2581", "new": "RFC5681", "score": None}))
    assert state["decision"]["label"] == "supersedes" and state["decision"]["calls"] == 1
    assert len(model.seen) == 1 and "RFC5681" in model.seen[0][-1].content
    for variant in ("full", "no_find_later"):
        assert build_variant(variant, ScriptedModel([], verdict), corpus()) is not None
    tools = {t.name for t in corpus_langchain_tools(corpus(), exclude=("find_later",))}
    assert tools == {"search", "read_doc", "compare"}


def test_decision_log_names_and_cache(tmp_path):
    import json

    from supersede.experiments.agent_lineage import decision_log, load_decisions

    p = decision_log(tmp_path, "strict", "ollama:org/m", "llm_only", "agent-v2", 0.3, 1)
    assert p.name == "strict_ollama-org-m_llm_only_agent-v2_t0.3_r1.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps({"old": "a", "new": "b", "label": "amends"})
        + "\n"
        + json.dumps({"old": "a", "new": "c", "label": None})
        + "\n"
    )
    assert set(load_decisions(p)) == {("a", "b")}
