"""Experiment 2b: what happens to each fix when publication years are missing or wrong?

Recency baselines (newest-first, time decay) need correct years. So does
lineage inference, which uses years to orient a pair (older -> newer) and as a
feature. Oracle lineage needs neither. We corrupt a share ``p`` of the years
(half removed, half shifted by 1-10 years either way), re-tune the recency
parameters and re-fit the cheap lineage scorer on the corrupted data, and
measure current@1 and stale@1 of the hybrid retriever's top 10 on test queries.

Document numbers are never used for ordering here (``year_only_order``), so
they cannot leak the true order.

Usage:
    python -m supersede.experiments.recency_noise --data data/rfc
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from supersede.corpus import load_docs, read_jsonl
from supersede.experiments.lineage_inference import cheap_scores
from supersede.experiments.staleness import (
    DEPTH,
    K,
    build_retrievers,
    recency_ranking,
    score,
    tune_recency,
)
from supersede.librarian.candidates import year_only_order
from supersede.librarian.classifiers import select_edges
from supersede.lineage import LineageGraph
from supersede.metrics import bootstrap_ci
from supersede.rerank import promote_current

SHARES = (0.0, 0.1, 0.3, 0.5)
SEEDS = (0, 1, 2)
METHODS = ("none", "newest_first", "time_decay", "lineage_v0", "oracle")


def corrupt_years(docs, p: float, seed: int) -> dict[str, int | None]:
    """Remove half of a random share ``p`` of years and shift the other half by 1-10 years."""
    rng = random.Random(seed)
    years = {d.doc_id: d.year for d in docs}
    dated = sorted(d.doc_id for d in docs if d.year)
    hit = rng.sample(dated, int(round(p * len(dated))))
    for i, doc_id in enumerate(hit):
        if i % 2 == 0:
            years[doc_id] = None
        else:
            years[doc_id] = years[doc_id] + rng.choice([-1, 1]) * rng.randint(1, 10)
    return years


def run(data: Path, setting: str = "strict", shares=SHARES, seeds=SEEDS) -> dict:
    docs = load_docs(data / "corpus.jsonl")
    gold = LineageGraph(docs)
    hybrid = build_retrievers(["hybrid"], [(d.doc_id, d.text) for d in docs], data)["hybrid"]
    sets = {
        "title queries": [q for q in read_jsonl(data / "queries.jsonl") if q["split"] == "test"],
        "questions": [q for q in read_jsonl(data / "questions.jsonl") if q["split"] == "test"],
    }
    deep = {name: hybrid.search_many([q["text"] for q in qs], DEPTH) for name, qs in sets.items()}

    rows = []
    for p in shares:
        for seed in seeds if p > 0 else seeds[:1]:
            years = corrupt_years(docs, p, seed)
            docs_c = [replace(d, year=years[d.doc_id]) for d in docs]
            tuned = tune_recency(hybrid, data, years)
            cs = cheap_scores(setting, data, docs_c, gold, order_key=year_only_order)
            inferred = LineageGraph.from_edges(docs, select_edges(cs.pairs, cs.proba, cs.threshold))
            for name, qs in sets.items():
                ranked = {
                    "none": [[d for d, _ in h[:K]] for h in deep[name]],
                    "newest_first": [
                        recency_ranking("newest_first", h, years, tuned["newest_first"])
                        for h in deep[name]
                    ],
                    "time_decay": [
                        recency_ranking("time_decay", h, years, tuned["time_decay"])
                        for h in deep[name]
                    ],
                }
                ranked["lineage_v0"] = [promote_current(r, inferred) for r in ranked["none"]]
                ranked["oracle"] = [promote_current(r, gold) for r in ranked["none"]]
                for method in METHODS:
                    per_query = [score(r, q) for r, q in zip(ranked[method], qs, strict=True)]
                    row = {"share": p, "seed": seed, "queries": name, "method": method}
                    for m in ("current@1", "stale@1"):
                        vals = [s[m] for s in per_query]
                        row[m] = sum(vals) / len(vals)
                        row[f"{m}_ci95"] = bootstrap_ci(vals, seed=seed)
                    rows.append(row)
            print(f"share {p} seed {seed}: done (recency params {tuned})", flush=True)
    return {
        "experiment": "recency_noise",
        "setting": setting,
        "retriever": hybrid.name,
        "k": K,
        "depth": DEPTH,
        "shares": list(shares),
        "seeds": list(seeds),
        "rows": rows,
    }


def summary(result: dict) -> list[dict]:
    """Mean over seeds per (share, queries, method)."""
    groups: dict[tuple, list[dict]] = {}
    for r in result["rows"]:
        groups.setdefault((r["share"], r["queries"], r["method"]), []).append(r)
    out = []
    for (share, queries, method), rs in sorted(groups.items()):
        out.append(
            {
                "share": share,
                "queries": queries,
                "method": method,
                "n_seeds": len(rs),
                **{m: sum(r[m] for r in rs) / len(rs) for m in ("current@1", "stale@1")},
                "current@1_min": min(r["current@1"] for r in rs),
                "current@1_max": max(r["current@1"] for r in rs),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/rfc"))
    ap.add_argument("--setting", default="strict")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    result = run(args.data, args.setting)
    result["summary"] = summary(result)
    (args.out / "rfc_recency_noise.json").write_text(json.dumps(result, indent=2) + "\n")
    for s in result["summary"]:
        print(
            f"{s['share']:.1f} {s['queries']:13} {s['method']:13} "
            f"current@1 {s['current@1']:.3f} stale@1 {s['stale@1']:.3f}"
        )


if __name__ == "__main__":
    main()
