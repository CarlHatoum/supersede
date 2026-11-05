"""The librarian agent: decide whether one document supersedes another.

A LangGraph state machine with three stages:

    triage ──(clear case)──────────────────────────────► END
       │
       └─(uncertain)─► investigate ⇄ tools ─► decide ──► END

* ``triage`` settles pairs that the cheap scorer (librarian v0) is confident
  about, so the expensive agent only runs where it adds value (the cascade).
* ``investigate`` is a tool-calling loop over ``search``, ``find_later``,
  ``read_doc`` and ``compare``, served by the MCP server or called in-process.
* ``decide`` asks for a structured verdict: supersedes / amends / unrelated /
  conflicts, with confidence and rationale.

The model is any LangChain chat model with tool calling: Mistral (hosted) as
the main model, an Ollama open-weight model as the baseline.

Usage:
    python -m supersede.librarian.agent RFC2581 RFC5681 --model ollama:edtorre/qwen3.5-hermes
    python -m supersede.librarian.agent RFC2581 RFC5681 --model mistral:mistral-medium-latest --mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

PROMPT_VERSION = "agent-v2"
LABELS = ("supersedes", "amends", "unrelated", "conflicts")

SYSTEM = """You are a librarian for a collection of technical standards. Documents get replaced \
by newer versions over time, but the collection does not record which replaces which. Your job \
is to find out, for one pair of documents, whether the NEWER document supersedes the OLDER one.

Labels:
- supersedes: the newer document replaces the older one as the reference for the same topic \
(directly, or as a later version in the same series).
- amends: the newer document changes or extends the older one, which stays in force.
- conflicts: they cover the same topic and disagree, but neither clearly replaces the other.
- unrelated: different topics, companion documents, or merely related work.

Investigate in this order:
1. compare the two documents.
2. find_later on the OLDER document, to see which documents could replace it.
3. If the list shows a document that fits better as the older one's replacement, or that \
suggests the newer one is a companion rather than a successor (for example a different \
transport, profile, or extension of the same base), read_doc it.
Similar titles alone are not enough: a successor covers the same scope and normally restates \
or revises it; a companion adds something next to it. Dates matter: a document can only be \
superseded by a later one. Stop calling tools once you can decide."""

TASK = """Pair to assess:
- OLDER: {old} ({old_title}, {old_year})
- NEWER: {new} ({new_title}, {new_year})
Investigate, then stop calling tools when you can decide."""

DECIDE = """Give your final verdict on whether {new} supersedes {old}, based on your \
investigation. Confidence: 0.9 or more only if the evidence is explicit (same scope restated \
and revised); 0.6 to 0.8 if it rests on titles, dates and overlap; below 0.6 if you are guessing."""


class Decision(BaseModel):
    label: Literal["supersedes", "amends", "unrelated", "conflicts"]
    confidence: float = Field(ge=0.0, le=1.0, description="How sure you are, from 0 to 1")
    rationale: str = Field(description="One or two sentences citing the evidence")


class LibrarianState(TypedDict, total=False):
    old: str
    new: str
    score: float | None  # cheap-scorer probability, None to always investigate
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int
    decision: dict


def corpus_langchain_tools(corpus) -> list[BaseTool]:
    """In-process LangChain tools over a ``CorpusTools`` (same contract as the MCP server)."""
    return [
        StructuredTool.from_function(
            corpus.search, name="search", description=corpus.search.__doc__
        ),
        StructuredTool.from_function(
            corpus.read_doc, name="read_doc", description=corpus.read_doc.__doc__
        ),
        StructuredTool.from_function(
            corpus.find_later, name="find_later", description=corpus.find_later.__doc__
        ),
        StructuredTool.from_function(
            corpus.compare, name="compare", description=corpus.compare.__doc__
        ),
    ]


async def mcp_tools(data: str = "data/rfc", setting: str = "masked") -> list[BaseTool]:
    """The same tools, fetched from the MCP server launched over stdio."""
    import os
    import sys

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "supersede": {
                "command": sys.executable,
                "args": ["-m", "supersede.serve.mcp_server"],
                "transport": "stdio",
                "env": {
                    **os.environ,
                    "SUPERSEDE_DATA": data,
                    "SUPERSEDE_SETTING": setting,
                },
            }
        }
    )
    return await client.get_tools()


def build_librarian(
    model,
    tools: list[BaseTool],
    describe=None,
    low: float = 0.2,
    high: float = 0.9,
    max_steps: int = 8,
):
    """Compile the librarian graph.

    ``describe(doc_id) -> (title, year)`` fills in the task prompt.
    Pairs with ``score >= high`` are accepted and ``score <= low`` rejected
    without calling the model.
    """
    tool_model = model.bind_tools(tools)
    judge = model.with_structured_output(Decision)

    def triage(state: LibrarianState) -> dict:
        score = state.get("score")
        if score is not None and score >= high:
            return {"decision": _verdict("supersedes", score, "triage: cheap scorer is confident")}
        if score is not None and score <= low:
            return {"decision": _verdict("unrelated", 1 - score, "triage: cheap scorer rejects")}
        old, new = state["old"], state["new"]
        (ot, oy), (nt, ny) = (describe(old), describe(new)) if describe else (("?", "?"),) * 2
        task = TASK.format(old=old, new=new, old_title=ot, old_year=oy, new_title=nt, new_year=ny)
        return {"messages": [SystemMessage(SYSTEM), HumanMessage(task)], "steps": 0}

    async def investigate(state: LibrarianState) -> dict:
        reply = await tool_model.ainvoke(state["messages"])
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    async def decide(state: LibrarianState) -> dict:
        prompt = HumanMessage(DECIDE.format(old=state["old"], new=state["new"]))
        verdict: Decision = await judge.ainvoke([*state["messages"], prompt])
        return {"decision": {**verdict.model_dump(), "source": "agent", "steps": state["steps"]}}

    def after_triage(state: LibrarianState) -> str:
        return END if state.get("decision") else "investigate"

    def after_investigate(state: LibrarianState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        return "tools" if wants_tools and state["steps"] < max_steps else "decide"

    graph = StateGraph(LibrarianState)
    graph.add_node("triage", triage)
    graph.add_node("investigate", investigate)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("decide", decide)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", after_triage, ["investigate", END])
    graph.add_conditional_edges("investigate", after_investigate, ["tools", "decide"])
    graph.add_edge("tools", "investigate")
    graph.add_edge("decide", END)
    return graph.compile()


def _verdict(label: str, confidence: float, rationale: str) -> dict:
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "rationale": rationale,
        "source": "triage",
        "steps": 0,
    }


def make_chat_model(spec: str, temperature: float = 0.0):
    """``ollama:<model>`` or ``mistral:<model>``."""
    provider, _, name = spec.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=name, temperature=temperature, reasoning=False, seed=0)
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=name, temperature=temperature)
    raise ValueError(f"unknown model spec {spec!r}; use ollama:<name> or mistral:<name>")


def trace(state: dict[str, Any]) -> list[str]:
    """A readable log of the agent's tool calls."""
    lines = []
    for m in state.get("messages", []):
        for call in getattr(m, "tool_calls", None) or []:
            lines.append(f"→ {call['name']}({json.dumps(call['args'])})")
    return lines


async def _main(args) -> None:
    from supersede.librarian.tools import load_corpus_tools

    corpus = load_corpus_tools(args.data, args.setting)
    tools = await mcp_tools(args.data, args.setting) if args.mcp else corpus_langchain_tools(corpus)

    def describe(doc_id: str) -> tuple[str, Any]:
        d = corpus.read_doc(doc_id)
        return d.get("title", "?"), d.get("year", "?")

    graph = build_librarian(make_chat_model(args.model), tools, describe, max_steps=args.max_steps)
    state = await graph.ainvoke({"old": args.old, "new": args.new, "score": None})
    print("\n".join(trace(state)))
    print(json.dumps(state["decision"], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--model", default="ollama:edtorre/qwen3.5-hermes")
    ap.add_argument("--data", default="data/rfc")
    ap.add_argument("--setting", choices=("declared", "masked", "strict"), default="strict")
    ap.add_argument("--mcp", action="store_true", help="fetch tools from the MCP server")
    ap.add_argument("--max-steps", type=int, default=8)
    asyncio.run(_main(ap.parse_args()))


if __name__ == "__main__":
    main()
