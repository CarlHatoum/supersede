"""Experiment 3: does the librarian agent beat the cheap scorer where that scorer is unsure?

The cascade sends only uncertain pairs to the agent: test-split candidate
pairs whose librarian-v0 probability falls in ``[--low, --high)``. From that
band we draw a seeded sample with equal numbers of true supersession edges and
non-edges, run the agent on each, and compare it with librarian v0's own
decision (probability >= its dev-tuned threshold) **on the same pairs**.

Reported: true-positive rate, true-negative rate, balanced accuracy, and an
estimated precision at the band's real class balance (the sample is balanced,
the band is not). Two ground truths:

* ``strict``: positive only if the index records "new obsoletes old" directly.
* ``lineage``: also positive if new is a later version of old through a chain
  (A -> B -> C makes "C supersedes A" correct). Saying a document is replaced
  by its replacement's replacement is not an error for a reader.

Decisions are appended to ``results/agent/<name>.jsonl`` as they finish, so an
interrupted run resumes where it stopped.

Usage:
    python -m supersede.experiments.agent_lineage --model ollama:edtorre/qwen3.5-hermes --n 20
    python -m supersede.experiments.agent_lineage --model mistral:mistral-medium-latest --n 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from supersede.corpus import load_docs, read_jsonl
from supersede.experiments.lineage_inference import cheap_scores
from supersede.librarian.agent import (
    PROMPT_VERSION,
    build_librarian,
    corpus_langchain_tools,
    make_chat_model,
    trace,
)
from supersede.librarian.tools import load_corpus_tools
from supersede.lineage import LineageGraph


def sample_band(cs, low: float, high: float, n: int, seed: int = 0):
    """Balanced sample of test pairs in the band, the band's positive rate, and the band."""
    band = [
        {"old": o, "new": nw, "gold": y, "cheap_proba": round(p, 4)}
        for (o, nw), y, s, p in zip(cs.pairs, cs.y, cs.split, cs.proba, strict=True)
        if s == "test" and low <= p < high
    ]
    pos = [b for b in band if b["gold"]]
    neg = [b for b in band if not b["gold"]]
    rng = random.Random(seed)
    picked = rng.sample(pos, min(n // 2, len(pos))) + rng.sample(neg, min(n - n // 2, len(neg)))
    rng.shuffle(picked)
    return picked, (len(pos) / len(band) if band else 0.0), band


def rates(rows: list[dict], predict) -> dict:
    pos = [r for r in rows if r["gold"]]
    neg = [r for r in rows if not r["gold"]]
    tpr = sum(predict(r) for r in pos) / len(pos) if pos else float("nan")
    tnr = sum(not predict(r) for r in neg) / len(neg) if neg else float("nan")
    return {"tpr": tpr, "tnr": tnr, "balanced_accuracy": (tpr + tnr) / 2}


def est_precision(tpr: float, tnr: float, prior: float) -> float:
    tp, fp = tpr * prior, (1 - tnr) * (1 - prior)
    return tp / (tp + fp) if tp + fp else 0.0


async def run(args) -> dict:
    data = Path(args.data)
    docs = load_docs(data / "corpus.jsonl")
    gold = LineageGraph(docs)
    cs = cheap_scores(args.setting, data, docs, gold)
    sample, prior, band = sample_band(cs, args.low, args.high, args.n, args.seed)
    prior_lineage = (
        sum(b["new"] in gold.descendants(b["old"]) for b in band) / len(band) if band else 0.0
    )

    slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
    name = f"rfc_agent_{args.setting}_{slug}_{PROMPT_VERSION}"
    out_dir = Path(args.out) / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{name}.jsonl"
    done = {(r["old"], r["new"]): r for r in read_jsonl(log)} if log.exists() else {}

    corpus = load_corpus_tools(data, args.setting)
    graph = build_librarian(
        make_chat_model(args.model),
        corpus_langchain_tools(corpus),
        describe=lambda d: (corpus.read_doc(d).get("title"), corpus.read_doc(d).get("year")),
        max_steps=args.max_steps,
    )
    gate = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def assess(pair: dict) -> dict:
        key = (pair["old"], pair["new"])
        if key in done:
            return done[key]
        async with gate:
            t0 = time.monotonic()
            try:
                state = await graph.ainvoke({"old": pair["old"], "new": pair["new"], "score": None})
                row = {**pair, **state["decision"], "trace": trace(state)}
            except Exception as e:  # one failed pair must not stop the run
                row = {**pair, "label": None, "error": f"{type(e).__name__}: {e}"[:300]}
            row["seconds"] = round(time.monotonic() - t0, 1)
            row.update(model=args.model, setting=args.setting, prompt_version=PROMPT_VERSION)
        async with lock:
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"{pair['old']}->{pair['new']} gold={pair['gold']} -> {row.get('label')}", flush=True)
        return row

    rows = await asyncio.gather(*(assess(p) for p in sample))
    ok = [r for r in rows if r.get("label")]
    for r in ok:
        r["gold_lineage"] = int(r["new"] in gold.descendants(r["old"]))
    lineage_rows = [{**r, "gold": r["gold_lineage"]} for r in ok]

    def compare_on(rs, base_rate):
        agent = rates(rs, lambda r: r["label"] == "supersedes")
        cheap = rates(rs, lambda r: r["cheap_proba"] >= cs.threshold)
        for m in (agent, cheap):
            m["est_precision"] = est_precision(m["tpr"], m["tnr"], base_rate)
        return agent, cheap

    agent, cheap = compare_on(ok, prior)
    agent_lin, cheap_lin = compare_on(lineage_rows, prior_lineage)
    return {
        "experiment": "agent_lineage",
        "name": name,
        "setting": args.setting,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "band": [args.low, args.high],
        "band_positive_rate": prior,
        "band_positive_rate_lineage": prior_lineage,
        "cheap_threshold": cs.threshold,
        "n_sampled": len(sample),
        "n_failed": len(rows) - len(ok),
        "labels": dict(sorted(Counter(r["label"] for r in ok).items())),
        "mean_steps": sum(r.get("steps", 0) for r in ok) / len(ok) if ok else 0,
        "mean_seconds": sum(r["seconds"] for r in ok) / len(ok) if ok else 0,
        "n_lineage_only_positives": sum(r["gold_lineage"] and not r["gold"] for r in ok),
        "agent": agent,
        "cheap_scorer": cheap,
        "lineage_truth": {"agent": agent_lin, "cheap_scorer": cheap_lin},
    }


def to_markdown(r: dict) -> str:
    def row(truth, name, m):
        return (
            f"| {truth} | {name} | {m['tpr']:.2f} | {m['tnr']:.2f} "
            f"| {m['balanced_accuracy']:.2f} | {m['est_precision']:.2f} |"
        )

    lin = r["lineage_truth"]

    return "\n".join(
        [
            f"Uncertain band {r['band']} of the cheap scorer, `{r['setting']}` setting, "
            f"{r['n_sampled']} balanced test pairs ({r['n_failed']} failed). "
            f"Agent model `{r['model']}`, {r['mean_steps']:.1f} steps and "
            f"{r['mean_seconds']:.0f} s per pair on average. Estimated precision uses the "
            f"band's real positive rate ({r['band_positive_rate']:.2f} strict, "
            f"{r['band_positive_rate_lineage']:.2f} lineage).",
            "",
            "| ground truth | method | TPR | TNR | balanced acc. | est. precision |",
            "|---|---|---|---|---|---|",
            row("strict", "cheap scorer (librarian v0)", r["cheap_scorer"]),
            row("strict", "librarian agent", r["agent"]),
            row("lineage", "cheap scorer (librarian v0)", lin["cheap_scorer"]),
            row("lineage", "librarian agent", lin["agent"]),
            "",
            f"*lineage* also counts later versions through a chain as correct "
            f"({r['n_lineage_only_positives']} sampled pairs change label).",
            "",
            f"Agent labels: {r['labels']}",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="ollama:edtorre/qwen3.5-hermes")
    ap.add_argument("--data", default="data/rfc")
    ap.add_argument("--setting", choices=("declared", "masked", "strict"), default="strict")
    ap.add_argument("--low", type=float, default=0.8)
    ap.add_argument("--high", type=float, default=0.98)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    result = asyncio.run(run(args))
    out = Path(args.out) / "agent"
    (out / f"{result['name']}.json").write_text(json.dumps(result, indent=2) + "\n")
    md = to_markdown(result)
    (out / f"{result['name']}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
