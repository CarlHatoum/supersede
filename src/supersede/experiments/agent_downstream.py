"""Experiment 3 (downstream): how much of the lineage gap does the agent close?

The cascade keeps the cheap scorer's decision outside the uncertain band and
uses the agent's verdict inside it. Re-deciding every band pair of the corpus is
too slow on a laptop, so we re-decide only the band pairs that can change a test
ranking: those whose older document appears in the hybrid top 10 of a test query
(``agent_lineage --pairs questions|titles``). Every other pair keeps the cheap
scorer's decision. This is exact for the measured queries, provided every
relevant pair has been decided (coverage is reported).

Graphs compared with ``promote`` on the hybrid top 10: no lineage, cheap scorer
(v0), cascade (v0 + agent in the band), oracle. Main number: share of the
current@1 gap between no lineage and oracle lineage that is closed.

Usage:
    python -m supersede.experiments.agent_downstream --queries questions --variant full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from supersede.corpus import load_docs, read_jsonl
from supersede.experiments.agent_lineage import decision_log, downstream_pairs, load_decisions
from supersede.experiments.lineage_inference import K, cheap_scores, downstream
from supersede.experiments.staleness import build_retrievers
from supersede.librarian.agent import PROMPT_VERSION
from supersede.librarian.classifiers import select_edges
from supersede.lineage import LineageGraph


def cascade_scores(cs, decisions: dict, low: float, high: float) -> tuple[list[float], int]:
    """v0 probabilities, with the agent's verdict (1 or 0) replacing band pairs it decided."""
    out, used = [], 0
    for pair, split, p in zip(cs.pairs, cs.split, cs.proba, strict=True):
        d = decisions.get(pair) if split == "test" and low <= p < high else None
        if d is None:
            out.append(p)
        else:
            out.append(1.0 if d["label"] == "supersedes" else 0.0)
            used += 1
    return out, used


def run(args) -> dict:
    data = Path(args.data)
    docs = load_docs(data / "corpus.jsonl")
    gold = LineageGraph(docs)
    cs = cheap_scores(args.setting, data, docs, gold)
    relevant = downstream_pairs(cs, data, args.queries, args.low, args.high)
    log = decision_log(args.out, args.setting, args.model, args.variant, args.prompt, 0.0, 0)
    decisions = load_decisions(log)
    covered = sum((p["old"], p["new"]) in decisions for p in relevant)

    scores, used = cascade_scores(cs, decisions, args.low, args.high)
    v0_edges = select_edges(cs.pairs, cs.proba, cs.threshold)
    cascade_edges = select_edges(cs.pairs, scores, cs.threshold)

    file = {"titles": "queries.jsonl", "questions": "questions.jsonl"}[args.queries]
    queries = [q for q in read_jsonl(data / file) if q["split"] == "test"]
    hybrid = build_retrievers(["hybrid"], [(d.doc_id, d.text) for d in docs], data)["hybrid"]
    rankings = [[d for d, _ in h] for h in hybrid.search_many([q["text"] for q in queries], K)]
    graphs = {
        "no_lineage": None,
        "cheap_scorer": LineageGraph.from_edges(docs, v0_edges),
        "cascade": LineageGraph.from_edges(docs, cascade_edges),
        "oracle": gold,
    }
    return {
        "experiment": "agent_downstream",
        "queries": args.queries,
        "n_queries": len(queries),
        "setting": args.setting,
        "model": args.model,
        "variant": args.variant,
        "prompt": args.prompt,
        "band": [args.low, args.high],
        "n_relevant_pairs": len(relevant),
        "n_relevant_decided": covered,
        "coverage": covered / len(relevant) if relevant else 0.0,
        "n_band_decisions_used": used,
        "n_edges": {"cheap_scorer": len(v0_edges), "cascade": len(cascade_edges)},
        "downstream": downstream(rankings, queries, graphs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--queries", choices=("questions", "titles"), default="questions")
    ap.add_argument("--model", default="ollama:edtorre/qwen3.5-hermes")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--prompt", default=PROMPT_VERSION)
    ap.add_argument("--data", default="data/rfc")
    ap.add_argument("--setting", default="strict")
    ap.add_argument("--low", type=float, default=0.8)
    ap.add_argument("--high", type=float, default=0.98)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    result = run(args)
    out = Path(args.out) / "agent" / f"downstream_{args.queries}_{args.variant}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"coverage {result['coverage']:.0%} of {result['n_relevant_pairs']} relevant pairs")
    for name, d in result["downstream"].items():
        print(f"  {name:13} current@1 {d['current@1']:.3f}  gap closed {d['gap_closed']:.0%}")


if __name__ == "__main__":
    main()
