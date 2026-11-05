"""Experiment 3 (runs): librarian-agent decisions on pairs the cheap scorer is unsure about.

The cascade sends only uncertain pairs to the agent: test-split candidate pairs
whose librarian-v0 probability falls in ``[--low, --high)`` (the band). This
module *produces decisions*; ``agent_compare`` turns them into metrics.

Pair sets (``--pairs``):

* ``sample``: a balanced, seeded, **nested** sample of the band (the first 20
  pairs are inside the first 50, which are inside the first 100), for the
  head-to-head comparison of variants.
* ``questions`` / ``titles``: every band pair whose older document appears in
  the hybrid top 10 of some test query of that set. These are the only band
  pairs that can change a test ranking, so they suffice to measure the agent's
  downstream effect.

Variants (``--variant``): ``full`` (all tools), ``no_find_later`` (ablation),
``llm_only`` (one call on both documents, no tools). See ``librarian.agent``.

Decisions are cached per (setting, model, variant, prompt, temperature, run) in
``results/agent/decisions/``, keyed by pair, so every run reuses what earlier
runs decided and an interrupted run resumes where it stopped.

Usage:
    python -m supersede.experiments.agent_lineage --variant llm_only --n 100
    python -m supersede.experiments.agent_lineage --variant full --pairs questions
    python -m supersede.experiments.agent_lineage --variant full --n 50 --temperature 0.3 --run 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

from supersede.corpus import load_docs, read_jsonl
from supersede.experiments.lineage_inference import cheap_scores
from supersede.librarian.agent import (
    PROMPT_VERSION,
    VARIANTS,
    build_variant,
    make_chat_model,
    trace,
)
from supersede.librarian.tools import load_corpus_tools
from supersede.lineage import LineageGraph

SAMPLE_VERSION = "s2"  # nested sampling (s1 was the non-nested 20-pair sample)


def band_pairs(cs, low: float, high: float) -> list[dict]:
    return [
        {"old": o, "new": nw, "gold": y, "cheap_proba": round(p, 4)}
        for (o, nw), y, s, p in zip(cs.pairs, cs.y, cs.split, cs.proba, strict=True)
        if s == "test" and low <= p < high
    ]


def sample_band(cs, low: float, high: float, n: int, seed: int = 0):
    """Balanced sample of band pairs, the band's positive rate, and the band.

    Nested: positives and negatives are shuffled once (independently of ``n``)
    and the sample takes prefixes, so a smaller sample is inside a larger one.
    """
    band = band_pairs(cs, low, high)

    def key(b):
        return (b["old"], b["new"])

    pos = sorted((b for b in band if b["gold"]), key=key)
    neg = sorted((b for b in band if not b["gold"]), key=key)
    random.Random(seed).shuffle(pos)
    random.Random(seed + 1).shuffle(neg)
    k = n // 2
    picked = []
    for i in range(max(k, n - k)):  # interleave so any prefix stays balanced
        if i < min(k, len(pos)):
            picked.append(pos[i])
        if i < min(n - k, len(neg)):
            picked.append(neg[i])
    return picked, (len(pos) / len(band) if band else 0.0), band


def downstream_pairs(cs, data: Path, which: str, low: float, high: float, k: int = 10):
    """Band pairs whose older document is in the hybrid top ``k`` of a test query."""
    from supersede.experiments.staleness import build_retrievers

    docs = load_docs(data / "corpus.jsonl")
    hybrid = build_retrievers(["hybrid"], [(d.doc_id, d.text) for d in docs], data)["hybrid"]
    file = {"titles": "queries.jsonl", "questions": "questions.jsonl"}[which]
    queries = [q for q in read_jsonl(data / file) if q["split"] == "test"]
    top = set()
    for hits in hybrid.search_many([q["text"] for q in queries], k):
        top |= {d for d, _ in hits}
    return [b for b in band_pairs(cs, low, high) if b["old"] in top]


def rates(rows: list[dict], predict) -> dict:
    pos = [r for r in rows if r["gold"]]
    neg = [r for r in rows if not r["gold"]]
    tpr = sum(predict(r) for r in pos) / len(pos) if pos else float("nan")
    tnr = sum(not predict(r) for r in neg) / len(neg) if neg else float("nan")
    return {"tpr": tpr, "tnr": tnr, "balanced_accuracy": (tpr + tnr) / 2}


def balanced_accuracy_ci(
    rows: list[dict], predict, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Stratified bootstrap CI: positives and negatives are resampled separately."""
    pos = [bool(predict(r)) for r in rows if r["gold"]]
    neg = [not predict(r) for r in rows if not r["gold"]]
    if not pos or not neg:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = sorted(
        (
            sum(rng.choice(pos) for _ in pos) / len(pos)
            + sum(rng.choice(neg) for _ in neg) / len(neg)
        )
        / 2
        for _ in range(n_resamples)
    )
    return stats[int(alpha / 2 * n_resamples)], stats[int((1 - alpha / 2) * n_resamples) - 1]


def est_precision(tpr: float, tnr: float, prior: float) -> float:
    tp, fp = tpr * prior, (1 - tnr) * (1 - prior)
    return tp / (tp + fp) if tp + fp else 0.0


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def decision_log(out, setting, model, variant, prompt, temperature, run) -> Path:
    name = f"{setting}_{slug(model)}_{variant}_{prompt}_t{temperature:g}_r{run}.jsonl"
    return Path(out) / "agent" / "decisions" / name


def load_decisions(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    return {(r["old"], r["new"]): r for r in read_jsonl(path) if r.get("label")}


async def run(args) -> dict:
    data = Path(args.data)
    docs = load_docs(data / "corpus.jsonl")
    gold = LineageGraph(docs)
    cs = cheap_scores(args.setting, data, docs, gold)
    if args.pairs == "sample":
        pairs, _, _ = sample_band(cs, args.low, args.high, args.n, args.seed)
    else:
        pairs = downstream_pairs(cs, data, args.pairs, args.low, args.high)
        if args.n:
            pairs = pairs[: args.n]

    tag = args.setting + (f"-noise{round(100 * args.year_noise)}" if args.year_noise else "")
    log = decision_log(
        args.out,
        tag,
        args.model,
        args.variant,
        args.prompt_version,
        args.temperature,
        args.run,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    done = load_decisions(log)
    todo = [p for p in pairs if (p["old"], p["new"]) not in done]
    print(f"{len(pairs)} pairs, {len(pairs) - len(todo)} cached, {len(todo)} to run -> {log}")
    if todo and args.prompt_version != PROMPT_VERSION:
        raise SystemExit(f"only the current prompt ({PROMPT_VERSION}) can run new pairs")

    corpus = load_corpus_tools(data, args.setting, year_noise=args.year_noise)
    model = make_chat_model(args.model, temperature=args.temperature, seed=args.run)
    graph = build_variant(args.variant, model, corpus, max_steps=args.max_steps)
    gate = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def assess(pair: dict) -> dict:
        async with gate:
            t0 = time.monotonic()
            try:
                state = await graph.ainvoke({"old": pair["old"], "new": pair["new"], "score": None})
                tr = trace(state)
                row = {**pair, **state["decision"], "trace": tr, "n_tool_calls": len(tr)}
            except Exception as e:  # one failed pair must not stop the run
                row = {**pair, "label": None, "error": f"{type(e).__name__}: {e}"[:300]}
            row["seconds"] = round(time.monotonic() - t0, 1)
            row.update(
                model=args.model,
                variant=args.variant,
                setting=args.setting,
                prompt_version=args.prompt_version,
                temperature=args.temperature,
                run=args.run,
            )
        async with lock:
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(
            f"{pair['old']}->{pair['new']} gold={pair['gold']} -> {row.get('label')} "
            f"({row['seconds']}s)",
            flush=True,
        )
        return row

    new_rows = await asyncio.gather(*(assess(p) for p in todo))
    decided = load_decisions(log)
    rows = [
        {**p, **decided[(p["old"], p["new"])]} for p in pairs if (p["old"], p["new"]) in decided
    ]
    return {
        "pairs": args.pairs,
        "n_pairs": len(pairs),
        "n_decided": len(rows),
        "n_failed_now": sum(1 for r in new_rows if not r.get("label")),
        "log": str(log),
        "agent": rates(rows, lambda r: r["label"] == "supersedes"),
        "cheap_scorer": rates(rows, lambda r: r["cheap_proba"] >= cs.threshold),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="ollama:edtorre/qwen3.5-hermes")
    ap.add_argument("--variant", choices=VARIANTS, default="full")
    ap.add_argument("--pairs", choices=("sample", "questions", "titles"), default="sample")
    ap.add_argument("--data", default="data/rfc")
    ap.add_argument("--setting", choices=("declared", "masked", "strict"), default="strict")
    ap.add_argument("--low", type=float, default=0.8)
    ap.add_argument("--high", type=float, default=0.98)
    ap.add_argument("--n", type=int, default=100, help="sample size (cap for downstream sets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--run", type=int, default=0, help="repeat index; also the model seed")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prompt-version", default=PROMPT_VERSION)
    ap.add_argument(
        "--year-noise",
        type=float,
        default=0.0,
        help="share of documents whose year the agent sees removed or shifted (seed 0)",
    )
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
