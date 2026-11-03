"""Experiment 2: infer lineage without metadata, and measure what it buys downstream.

For each setting (``declared`` keeps text cues such as "obsoletes RFC 2833";
``masked`` hides every RFC number):

  1. Candidate pairs: the 10 nearest neighbours of each document by text
     embedding and by title embedding, ordered old -> new by publication date.
  2. Scorers: a hand-written heuristic, and a logistic regression trained on
     dev-split pairs with its threshold tuned on dev F1.
  3. Edge precision, recall and F1 on test-split pairs.
  4. Downstream: retrieval with the ``promote`` policy using the *inferred*
     graph, compared with no lineage and with the oracle graph. Main number:
     the share of the no-lineage -> oracle gap in current@1 that is closed.

Usage:
    python -m supersede.experiments.lineage_inference --data data/rfc
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from supersede.bench.rfc import split_of
from supersede.corpus import load_docs, read_jsonl
from supersede.experiments.staleness import DENSE_MODEL, build_retrievers, score
from supersede.librarian import masking
from supersede.librarian.candidates import neighbour_pairs
from supersede.librarian.classifiers import (
    LogisticRegression,
    edge_prf,
    heuristic_score,
    select_edges,
)
from supersede.librarian.features import pair_features
from supersede.lineage import LineageGraph
from supersede.rerank import promote_current
from supersede.retrieval import DenseRetriever

N_NEIGHBOURS = 10
K = 10
# Fine steps near 1: the class-weighted model puts most true edges above 0.9.
THRESHOLDS = [t / 20 for t in range(2, 19)] + [0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 0.995]


def candidates(docs, data: Path):
    """Candidate pairs with features, from text and title embeddings."""
    text = DenseRetriever([(d.doc_id, d.text) for d in docs], DENSE_MODEL, cache_dir=data / "emb")
    title = DenseRetriever([(d.doc_id, d.title) for d in docs], DENSE_MODEL, cache_dir=data / "emb")
    ranks = neighbour_pairs(docs, text.emb, N_NEIGHBOURS)
    for pair, r in neighbour_pairs(docs, title.emb, N_NEIGHBOURS).items():
        ranks[pair] = min(ranks.get(pair, r), r)
    idx = {d.doc_id: i for i, d in enumerate(docs)}
    by_id = {d.doc_id: d for d in docs}
    pairs = sorted(ranks)
    feats = []
    for old, new in pairs:
        i, j = idx[old], idx[new]
        feats.append(
            pair_features(
                by_id[old],
                by_id[new],
                title_cos=float(title.emb[i] @ title.emb[j]),
                text_cos=float(text.emb[i] @ text.emb[j]),
                rank=ranks[(old, new)],
            )
        )
    return pairs, feats


def downstream(rankings, queries, graphs: dict[str, LineageGraph | None]) -> dict[str, dict]:
    out = {}
    for name, graph in graphs.items():
        scores = [
            score(promote_current(r, graph) if graph is not None else r, q)
            for r, q in zip(rankings, queries, strict=True)
        ]
        out[name] = {m: sum(s[m] for s in scores) / len(scores) for m in ("current@1", "stale@1")}
    lo, hi = out["no_lineage"]["current@1"], out["oracle"]["current@1"]
    for name in out:
        out[name]["gap_closed"] = (out[name]["current@1"] - lo) / (hi - lo) if hi > lo else 0.0
    return out


@dataclass
class CheapScores:
    """Librarian v0 on one setting: candidate pairs, labels, splits and fitted scores."""

    setting: str
    pairs: list[tuple[str, str]]
    feats: list[dict[str, float]]
    y: list[int]
    split: list[str]
    proba: list[float]
    threshold: float
    model: LogisticRegression


def cheap_scores(setting: str, data: Path, docs_orig, gold: LineageGraph) -> CheapScores:
    docs = masking.apply(docs_orig, setting)
    pairs, feats = candidates(docs, data)
    gold_edges = set(gold.edges)
    y = [int(p in gold_edges) for p in pairs]
    split = [split_of(gold.component(old)) for old, _ in pairs]
    dev = [i for i, s in enumerate(split) if s == "dev"]
    model = LogisticRegression().fit([feats[i] for i in dev], [y[i] for i in dev])
    proba = model.predict_proba(feats)
    gold_dev = {e for e in gold_edges if split_of(gold.component(e[0])) == "dev"}

    def dev_f1(t: float) -> float:
        predicted = select_edges(pairs, proba, t)
        pred = {e for e in predicted if split_of(gold.component(e[0])) == "dev"}
        return edge_prf(pred, gold_dev)["f1"]

    best_t = max(THRESHOLDS, key=dev_f1)
    return CheapScores(setting, pairs, feats, y, split, proba, best_t, model)


def run_setting(setting: str, data: Path, docs_orig, gold: LineageGraph, rankings) -> dict:
    cs = cheap_scores(setting, data, docs_orig, gold)
    pairs, feats = cs.pairs, cs.feats
    gold_edges = set(gold.edges)
    gold_split = {e: split_of(gold.component(e[0])) for e in gold_edges}

    def evaluate(scores, threshold, which="test"):
        predicted = select_edges(pairs, scores, threshold)
        pred = {e for e in predicted if split_of(gold.component(e[0])) == which}
        gold_s = {e for e in gold_edges if gold_split[e] == which}
        return predicted, edge_prf(pred, gold_s)

    covered = set(pairs) & gold_edges
    result = {
        "setting": setting,
        "n_candidate_pairs": len(pairs),
        "candidate_recall": len(covered) / len(gold_edges),
        "methods": {},
    }

    heur = [heuristic_score(f) for f in feats]
    predicted_h, prf_h = evaluate(heur, 0.5)

    proba, best_t, model = cs.proba, cs.threshold, cs.model
    predicted_lr, prf_lr = evaluate(proba, best_t)

    result["methods"]["heuristic"] = {"edges": prf_h}
    result["methods"]["logreg"] = {
        "edges": prf_lr,
        "threshold": best_t,
        "weights": model.weights(),
    }

    graphs = {
        "no_lineage": None,
        "heuristic": LineageGraph.from_edges(docs_orig, predicted_h),
        "logreg": LineageGraph.from_edges(docs_orig, predicted_lr),
        "oracle": gold,
    }
    result["downstream"] = {name: downstream(r, qs, graphs) for name, (r, qs) in rankings.items()}
    return result


def to_markdown(results: list[dict]) -> str:
    lines = [
        "Lineage inferred without metadata. Edge scores on test-split pairs; downstream is "
        "hybrid top-10 with the `promote` policy on test-split queries. *Gap closed* is the "
        "share of the current@1 gap between no lineage and oracle lineage.",
        "",
    ]
    for r in results:
        lines += [
            f"**{r['setting']}**: {r['n_candidate_pairs']} candidate pairs, "
            f"candidate recall {r['candidate_recall']:.3f}",
            "",
            "| method | edge P | edge R | edge F1 | "
            + " | ".join(f"{q}: current@1 (gap closed)" for q in r["downstream"])
            + " |",
            "|---|---|---|---|" + "---|" * len(r["downstream"]),
        ]
        for m in ("no_lineage", "heuristic", "logreg", "oracle"):
            e = r["methods"].get(m, {}).get("edges")
            cells = (
                [f"{e['precision']:.3f}", f"{e['recall']:.3f}", f"{e['f1']:.3f}"]
                if e
                else ["–"] * 3
            )
            for q in r["downstream"].values():
                d = q[m]
                cells.append(f"{d['current@1']:.3f} ({d['gap_closed']:.0%})")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/rfc"))
    ap.add_argument("--settings", default="declared,masked,strict")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    docs_orig = load_docs(args.data / "corpus.jsonl")
    gold = LineageGraph(docs_orig)
    hybrid = build_retrievers(["hybrid"], [(d.doc_id, d.text) for d in docs_orig], args.data)[
        "hybrid"
    ]
    rankings = {}
    query_sets = {
        "title queries": [
            q for q in read_jsonl(args.data / "queries.jsonl") if q["split"] == "test"
        ],
    }
    if (args.data / "questions.jsonl").exists():
        query_sets["questions"] = [
            q for q in read_jsonl(args.data / "questions.jsonl") if q["split"] == "test"
        ]
    for name, qs in query_sets.items():
        hits = hybrid.search_many([q["text"] for q in qs], K)
        rankings[name] = ([[d for d, _ in h] for h in hits], qs)

    results = [
        run_setting(s, args.data, docs_orig, gold, rankings) for s in args.settings.split(",")
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rfc_lineage_inference.json").write_text(json.dumps(results, indent=2) + "\n")
    md = to_markdown(results)
    (args.out / "rfc_lineage_inference.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
