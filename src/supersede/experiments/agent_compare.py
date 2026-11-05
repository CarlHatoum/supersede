"""Experiment 3 (analysis): compare agent variants and the cheap scorer on the same band pairs.

Reads the decision caches written by ``agent_lineage`` and the nested balanced
band sample, and reports for every (model, variant, prompt) at temperature 0:

* TPR, TNR, balanced accuracy with a stratified bootstrap 95% CI, and the
  precision expected at the band's real share of true links, under two ground
  truths (``strict``: the index records the link; ``lineage``: a later version
  through a chain also counts);
* paired comparisons on the same pairs: exact McNemar test on per-pair
  correctness, and a paired bootstrap CI of the difference in balanced
  accuracy;
* cost per pair: seconds, model calls, tokens, tool calls, and how often each
  tool is used;
* run-to-run stability for configurations repeated at temperature > 0.

Usage:
    python -m supersede.experiments.agent_compare --n 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from supersede.corpus import load_docs, read_jsonl
from supersede.experiments.agent_lineage import (
    SAMPLE_VERSION,
    balanced_accuracy_ci,
    est_precision,
    rates,
    sample_band,
)
from supersede.experiments.lineage_inference import cheap_scores
from supersede.lineage import LineageGraph

MIN_COVERAGE = 0.95  # a configuration is reported once it has decided this share of the sample


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_ba_diff_ci(rows, pred_a, pred_b, n_resamples=2000, seed=0) -> tuple[float, float]:
    """Stratified paired bootstrap CI of balanced accuracy(a) minus balanced accuracy(b)."""
    pos = [r for r in rows if r["gold"]]
    neg = [r for r in rows if not r["gold"]]
    if not pos or not neg:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)

    def ba(sp, sn, pred):
        return (sum(pred(r) for r in sp) / len(sp) + sum(not pred(r) for r in sn) / len(sn)) / 2

    diffs = []
    for _ in range(n_resamples):
        sp = [rng.choice(pos) for _ in pos]
        sn = [rng.choice(neg) for _ in neg]
        diffs.append(ba(sp, sn, pred_a) - ba(sp, sn, pred_b))
    diffs.sort()
    return diffs[int(0.025 * n_resamples)], diffs[int(0.975 * n_resamples) - 1]


def parse_log_name(path: Path) -> dict:
    """``<setting>_<model>_<variant>_<prompt>_t<temp>_r<run>.jsonl`` from the first row."""
    first = next(iter(read_jsonl(path)), None) or {}
    return {
        "model": first.get("model"),
        "variant": first.get("variant", "full"),
        "prompt": first.get("prompt_version"),
        "temperature": float(first.get("temperature", 0.0)),
        "run": int(first.get("run", 0)),
        "setting": first.get("setting"),
        "path": path,
    }


def model_alias(spec: str) -> str:
    return spec.split(":", 1)[-1].split("/")[-1].split(":")[0]


def summarise(rows, predict, prior: float) -> dict:
    m = rates(rows, predict)
    m["balanced_accuracy_ci95"] = balanced_accuracy_ci(rows, predict)
    m["est_precision"] = est_precision(m["tpr"], m["tnr"], prior)
    m["n_pos"] = sum(1 for r in rows if r["gold"])
    m["n_neg"] = sum(1 for r in rows if not r["gold"])
    return m


def cost(rows) -> dict:
    n = len(rows)
    tools = Counter(t.split("(")[0].lstrip("→ ").strip() for r in rows for t in r.get("trace", []))
    used = Counter()
    for r in rows:
        for name in {t.split("(")[0].lstrip("→ ").strip() for t in r.get("trace", [])}:
            used[name] += 1
    return {
        "seconds": sum(r.get("seconds", 0) for r in rows) / n,
        "calls": sum(r.get("calls", r.get("steps", 0) + 1) for r in rows) / n,
        "input_tokens": sum(r.get("input_tokens", 0) for r in rows) / n,
        "output_tokens": sum(r.get("output_tokens", 0) for r in rows) / n,
        "tool_calls": sum(len(r.get("trace", [])) for r in rows) / n,
        "tool_calls_by_name": {k: v / n for k, v in sorted(tools.items())},
        "share_of_pairs_using": {k: v / n for k, v in sorted(used.items())},
    }


def run(args) -> dict:
    data = Path(args.data)
    docs = load_docs(data / "corpus.jsonl")
    gold = LineageGraph(docs)
    cs = cheap_scores(args.setting, data, docs, gold)
    sample, prior, band = sample_band(cs, args.low, args.high, args.n, args.seed)
    prior_lineage = sum(b["new"] in gold.descendants(b["old"]) for b in band) / len(band)
    keys = [(p["old"], p["new"]) for p in sample]
    lineage_gold = {k: int(k[1] in gold.descendants(k[0])) for k in keys}

    def with_truth(rows, truth):
        return (
            rows
            if truth == "strict"
            else [{**r, "gold": lineage_gold[(r["old"], r["new"])]} for r in rows]
        )

    configs = []
    for path in sorted((Path(args.out) / "agent" / "decisions").glob(f"{args.setting}_*.jsonl")):
        meta = parse_log_name(path)
        decided = {(r["old"], r["new"]): r for r in read_jsonl(path) if r.get("label")}
        rows = [
            {**p, **decided[(p["old"], p["new"])], "gold": p["gold"]}
            for p in sample
            if (p["old"], p["new"]) in decided
        ]
        meta["coverage"] = len(rows) / len(sample)
        meta["rows"] = rows
        configs.append(meta)

    def agent_pred(r):
        return r["label"] == "supersedes"

    def cheap_pred(r):
        return r["cheap_proba"] >= cs.threshold

    result = {
        "experiment": "agent_compare",
        "setting": args.setting,
        "sample_version": SAMPLE_VERSION,
        "n": len(sample),
        "band": [args.low, args.high],
        "band_size": len(band),
        "band_positive_rate": prior,
        "band_positive_rate_lineage": prior_lineage,
        "cheap_threshold": cs.threshold,
        "cheap_scorer": {
            t: summarise(
                with_truth(sample, t), cheap_pred, prior if t == "strict" else prior_lineage
            )
            for t in ("strict", "lineage")
        },
        "methods": [],
        "paired": [],
        "stability": [],
    }

    main = [
        c
        for c in configs
        if c["temperature"] == 0 and c["run"] == 0 and c["coverage"] >= MIN_COVERAGE
    ]
    for c in main:
        entry = {
            "model": c["model"],
            "alias": model_alias(c["model"]),
            "variant": c["variant"],
            "prompt": c["prompt"],
            "n_decided": len(c["rows"]),
            "labels": dict(Counter(r["label"] for r in c["rows"])),
            "cost": cost(c["rows"]),
        }
        for t in ("strict", "lineage"):
            entry[t] = summarise(
                with_truth(c["rows"], t), agent_pred, prior if t == "strict" else prior_lineage
            )
        result["methods"].append(entry)

    # Paired comparisons on the pairs both sides decided (strict and lineage truth).
    def correct(pred, r):
        return pred(r) == bool(r["gold"])

    def compare(name_a, rows_a, pred_a, name_b, rows_b, pred_b):
        b_by = {(r["old"], r["new"]): r for r in rows_b}
        out = {"a": name_a, "b": name_b}
        for t in ("strict", "lineage"):
            ra = with_truth([r for r in rows_a if (r["old"], r["new"]) in b_by], t)
            rb = with_truth(
                [b_by[(r["old"], r["new"])] for r in rows_a if (r["old"], r["new"]) in b_by], t
            )
            pairs = list(zip(ra, rb, strict=True))
            b = sum(correct(pred_a, x) and not correct(pred_b, y) for x, y in pairs)
            c_ = sum(not correct(pred_a, x) and correct(pred_b, y) for x, y in pairs)
            merged = [{**x, "_a": pred_a(x), "_b": pred_b(y)} for x, y in pairs]
            lo, hi = paired_ba_diff_ci(merged, lambda r: r["_a"], lambda r: r["_b"])
            ba_a = rates(merged, lambda r: r["_a"])["balanced_accuracy"]
            ba_b = rates(merged, lambda r: r["_b"])["balanced_accuracy"]
            out[t] = {
                "n": len(pairs),
                "a_only_correct": b,
                "b_only_correct": c_,
                "mcnemar_p": mcnemar_exact(b, c_),
                "ba_diff": ba_a - ba_b,
                "ba_diff_ci95": [lo, hi],
            }
        return out

    names = {id(c): f"{model_alias(c['model'])}/{c['variant']}/{c['prompt']}" for c in main}
    for c in main:
        result["paired"].append(
            compare(names[id(c)], c["rows"], agent_pred, "cheap scorer", c["rows"], cheap_pred)
        )
    by_key = {(c["model"], c["variant"], c["prompt"]): c for c in main}
    for (model, variant, prompt), c in by_key.items():
        if variant == "full":
            for other in ("llm_only", "no_find_later"):
                o = by_key.get((model, other, prompt))
                if o:
                    result["paired"].append(
                        compare(
                            names[id(c)], c["rows"], agent_pred, names[id(o)], o["rows"], agent_pred
                        )
                    )

    # Stability across repeated runs (temperature > 0) on the pairs every run decided.
    groups: dict[tuple, list[dict]] = {}
    for c in configs:
        if c["temperature"] > 0:
            groups.setdefault((c["model"], c["variant"], c["prompt"], c["temperature"]), []).append(
                c
            )
    for (model, variant, prompt, temp), runs in groups.items():
        common = set.intersection(*({(r["old"], r["new"]) for r in c["rows"]} for c in runs))
        if len(runs) < 2 or not common:
            continue
        per_run = []
        labels = {}
        for c in runs:
            rows = [r for r in c["rows"] if (r["old"], r["new"]) in common]
            per_run.append(rates(rows, agent_pred)["balanced_accuracy"])
            for r in rows:
                labels.setdefault((r["old"], r["new"]), []).append(agent_pred(r))
        mean = sum(per_run) / len(per_run)
        sd = (sum((x - mean) ** 2 for x in per_run) / (len(per_run) - 1)) ** 0.5
        result["stability"].append(
            {
                "model": model,
                "alias": model_alias(model),
                "variant": variant,
                "prompt": prompt,
                "temperature": temp,
                "runs": len(runs),
                "n_pairs": len(common),
                "ba_per_run": per_run,
                "ba_mean": mean,
                "ba_sd": sd,
                "unanimous_share": sum(len(set(v)) == 1 for v in labels.values()) / len(labels),
            }
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="data/rfc")
    ap.add_argument("--setting", default="strict")
    ap.add_argument("--low", type=float, default=0.8)
    ap.add_argument("--high", type=float, default=0.98)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    result = run(args)
    out = Path(args.out) / "agent" / f"compare_{args.setting}_n{args.n}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['methods'])} configurations -> {out}")
    for m in result["methods"]:
        s = m["strict"]
        lo, hi = s["balanced_accuracy_ci95"]
        print(
            f"  {m['alias']:16} {m['variant']:14} {m['prompt']}  "
            f"BA {s['balanced_accuracy']:.2f} [{lo:.2f}, {hi:.2f}]  "
            f"{m['cost']['seconds']:.0f}s/pair"
        )
    for p in result["paired"]:
        s = p["strict"]
        lo, hi = s["ba_diff_ci95"]
        print(
            f"  {p['a']} vs {p['b']}: diff {s['ba_diff']:+.2f} [{lo:+.2f}, {hi:+.2f}], "
            f"McNemar p={s['mcnemar_p']:.3f} (n={s['n']})"
        )


if __name__ == "__main__":
    main()
