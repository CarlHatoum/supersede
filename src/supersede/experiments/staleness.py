"""Experiment 0: how often does lexical retrieval surface a superseded version?

This needs no LLM, so it is deterministic and free to reproduce. It runs BM25
with each version-aware policy over the RFC queries and writes
``results/<name>.json`` and ``results/<name>.md``.

Usage:
    python -m supersede.experiments.staleness --data data/rfc --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from supersede.corpus import load_docs, read_jsonl
from supersede.lineage import LineageGraph
from supersede.metrics import (
    bootstrap_ci,
    current_at_k,
    reciprocal_rank,
    stale_at_1,
    stale_before_current,
)
from supersede.rerank import POLICIES
from supersede.retrieval import BM25

K = 10
METRICS = ("stale@1", "stale_before_current", "current@1", "current@10", "mrr_current")


def score(ranking: list[str], q: dict) -> dict[str, float]:
    current, stale = set(q["current"]), set(q["stale"])
    return {
        "stale@1": stale_at_1(ranking, current, stale),
        "stale_before_current": stale_before_current(ranking, current, stale),
        "current@1": current_at_k(ranking, current, 1),
        "current@10": current_at_k(ranking, current, 10),
        "mrr_current": reciprocal_rank(ranking, current),
    }


def run(data: Path, split: str, seed: int = 0) -> dict:
    docs = load_docs(data / "corpus.jsonl")
    graph = LineageGraph(docs)
    index = BM25([(d.doc_id, d.text) for d in docs])
    queries = [q for q in read_jsonl(data / "queries.jsonl") if split in ("all", q["split"])]

    rows = []
    for variant in sorted({q["variant"] for q in queries}):
        qs = [q for q in queries if q["variant"] == variant]
        rankings = [[d for d, _ in index.search(q["text"], K)] for q in qs]
        for policy, fn in POLICIES.items():
            per_query = [score(fn(r, graph), q) for r, q in zip(rankings, qs, strict=True)]
            row = {"variant": variant, "policy": policy, "n": len(qs)}
            for m in METRICS:
                vals = [s[m] for s in per_query]
                row[m] = sum(vals) / len(vals) if vals else float("nan")
                row[f"{m}_ci95"] = bootstrap_ci(vals, seed=seed)
            rows.append(row)

    manifest = json.loads((data / "manifest.json").read_text())
    return {
        "experiment": "staleness",
        "retriever": "bm25(k1=1.2,b=0.75)",
        "k": K,
        "split": split,
        "seed": seed,
        "index_sha256": manifest["index_sha256"],
        "rows": rows,
    }


def to_markdown(result: dict) -> str:
    lines = [
        f"BM25 top-{result['k']}, split `{result['split']}`, "
        f"index `{result['index_sha256'][:12]}`. 95% bootstrap CI in brackets.",
        "",
        "| query variant | policy | n | stale@1 ↓ | stale before current ↓ "
        "| current@1 ↑ | current@10 ↑ | MRR current ↑ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        cells = [f"{r[m]:.3f} [{r[m + '_ci95'][0]:.3f}, {r[m + '_ci95'][1]:.3f}]" for m in METRICS]
        lines.append(f"| {r['variant']} | {r['policy']} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/rfc"))
    ap.add_argument("--split", choices=("dev", "test", "all"), default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    result = run(args.data, args.split, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"rfc_staleness_{args.split}"
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    md = to_markdown(result)
    (args.out / f"{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
