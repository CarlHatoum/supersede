"""Paper figures, drawn from ``results/`` (vector PDF, light print style).

Colours: the first three slots of a CVD-validated categorical palette (blue,
orange, aqua; all pairs pass the colour-vision checks) plus neutral greys.
Aqua is below 3:1 contrast on white, so every bar carries a direct value
label. Colour follows the entity: a retriever, method or model keeps its
colour in every figure.

Usage:
    python -m supersede.report.figures --results results --out paper/generated
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#86b6ef"  # same hue, lighter step (sequential ramp, step 250)
GREY, GREY_LIGHT, INK, INK_2 = "#8a8985", "#d6d5d0", "#1f1f1e", "#52514e"
RETRIEVER = {"bm25": BLUE, "dense": ORANGE, "hybrid": AQUA}
WIDTH = 6.3  # inches, the text width of the paper

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8,
        "axes.edgecolor": GREY_LIGHT,
        "axes.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "text.color": INK,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, out: Path, name: str) -> None:
    """The vector PDF for the paper and a PNG (dataset card, slides)."""
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200, facecolor="white")


def _row(rows, **match):
    return next(r for r in rows if all(r[k] == v for k, v in match.items()))


def _pct_axis(ax, lim=1.0):
    ax.set_xlim(0, lim)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)


def fig_retrieval(res: Path, out: Path) -> bool:
    titles, qs = load(res / "rfc_staleness_test.json"), load(res / "rfc_questions_all.json")
    if not titles or not qs:
        return False
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.35), gridspec_kw={"wspace": 0.55})

    groups = [
        ("old title", titles["rows"], {"variant": "old_title"}),
        ("new title", titles["rows"], {"variant": "new_title"}),
        ("change-aware\nquestion", qs["rows"], {}),
    ]
    h = 0.26
    for gi, (_name, rows, m) in enumerate(groups):
        for ri, ret in enumerate(RETRIEVER):
            r = _row(rows, retriever=ret, policy="none", **m)
            y = gi + (ri - 1) * h
            v, (lo, hi) = r["stale@1"], r["stale@1_ci95"]
            a.barh(y, v, height=h - 0.04, color=RETRIEVER[ret], zorder=2)
            a.plot([lo, hi], [y, y], color=INK, lw=0.7, zorder=3)
            a.text(hi + 0.015, y, f"{v:.0%}", va="center", fontsize=7, color=INK)
    a.set_yticks(range(len(groups)), [g[0] for g in groups])
    a.invert_yaxis()
    _pct_axis(a)
    a.set_xlabel("stale@1: top result is an obsolete version")
    a.set_title("(a) The obsolete version often ranks first")

    # (b) current@1 without lineage -> with oracle lineage (promote), per retriever.
    items = []
    for ret in RETRIEVER:
        items.append(
            (
                f"{ret}, new title",
                _row(titles["rows"], retriever=ret, variant="new_title", policy="none"),
                _row(titles["rows"], retriever=ret, variant="new_title", policy="promote"),
                ret,
            )
        )  # noqa: E501
    for ret in RETRIEVER:
        items.append(
            (
                f"{ret}, question",
                _row(qs["rows"], retriever=ret, policy="none"),
                _row(qs["rows"], retriever=ret, policy="promote"),
                ret,
            )
        )  # noqa: E501
    for i, (_name, before, after, ret) in enumerate(items):
        x0, x1 = before["current@1"], after["current@1"]
        b.plot([x0, x1], [i, i], color=GREY_LIGHT, lw=2.2, zorder=1, solid_capstyle="round")
        b.scatter([x0], [i], s=26, color="white", edgecolor=GREY, lw=1.2, zorder=3)
        b.scatter([x1], [i], s=30, color=RETRIEVER[ret], edgecolor="white", lw=1.2, zorder=3)
        rows = titles["rows"] if "title" in _name else qs["rows"]
        match = {"variant": "new_title"} if "title" in _name else {}
        newest = next(
            (
                r
                for r in rows
                if r["retriever"] == ret
                and r["policy"] == "newest_first"
                and all(r[k] == v for k, v in match.items())
            ),
            None,
        )
        if newest:
            b.scatter(
                [newest["current@1"]],
                [i],
                s=26,
                marker="D",
                color="white",
                edgecolor=RETRIEVER[ret],
                lw=1.2,
                zorder=3,
            )
        b.text(x0 - 0.025, i, f"{x0:.0%}", va="center", ha="right", fontsize=7, color=INK_2)
        b.text(x1 + 0.025, i, f"{x1:.0%}", va="center", fontsize=7)
    b.set_yticks(range(len(items)), [it[0] for it in items])
    b.invert_yaxis()
    _pct_axis(b, 1.12)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_xlabel("current@1: top result is in force")
    b.set_title("(b) Known lineage fixes it; recency only partly")
    fig.legend(
        handles=[
            *[Line2D([], [], color=c, lw=6, label=k) for k, c in RETRIEVER.items()],
            Line2D([], [], ls="", marker="o", mfc="white", mec=GREY, label="as retrieved"),
            Line2D([], [], ls="", marker="D", mfc="white", mec=INK_2, label="newest first"),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                mfc=INK_2,
                mec="white",
                label="promote (oracle lineage)",
            ),
        ],
        loc="lower center",
        ncol=6,
        handlelength=0.9,
        columnspacing=1.0,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, -0.2),
    )
    _save(fig, out, "fig_retrieval")
    plt.close(fig)
    return True


ANSWER_SEGMENTS = [
    ("current only", ("current",), BLUE),
    ("current, old one flagged", ("mixed_flagged",), BLUE_LIGHT),
    ("stale (or mixed, unflagged)", ("stale", "mixed_unflagged"), ORANGE),
    ("abstain / other", ("abstain", "other"), GREY_LIGHT),
]


def fig_answers(res: Path, out: Path) -> bool:
    sources = [
        ("metadata visible", res / "rfc_answer_staleness_hybrid_items.jsonl"),
        ("blind", res / "rfc_answer_staleness_hybrid_blind_items.jsonl"),
    ]
    if not all(p.exists() for _, p in sources):
        return False
    policies = ("naive", "latest_only", "show_conflict")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.55), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (title, path) in zip(axes, sources, strict=True):
        rows = [json.loads(line) for line in path.open()]
        for i, pol in enumerate(policies):
            labels = Counter(r["label"] for r in rows if r["policy"] == pol and r.get("label"))
            n = sum(labels.values())
            left = 0.0
            for _, keys, colour in ANSWER_SEGMENTS:
                v = sum(labels[k] for k in keys) / n
                if v > 0:
                    ax.barh(i, v - 0.004, left=left + 0.002, height=0.62, color=colour, zorder=2)
                    if v >= 0.06:
                        ink = "white" if colour in (BLUE, ORANGE) else INK
                        ax.text(
                            left + v / 2,
                            i,
                            f"{v:.0%}",
                            ha="center",
                            va="center",
                            fontsize=6.8,
                            color=ink,
                        )
                left += v
        ax.set_yticks(range(3), [p.replace("_", " ") for p in policies])
        ax.set_ylim(2.5, -0.5)
        _pct_axis(ax)
        ax.grid(False)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{title} context")
    fig.legend(
        handles=[Line2D([], [], color=c, lw=6, label=k) for k, _, c in ANSWER_SEGMENTS],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.13),
        handlelength=0.9,
    )
    _save(fig, out, "fig_answers")
    plt.close(fig)
    return True


def fig_lineage(res: Path, out: Path) -> bool:
    r = load(res / "rfc_lineage_inference.json")
    if not r:
        return False
    qsets = list(r[0]["downstream"])
    fig, axes = plt.subplots(
        1, len(qsets), figsize=(WIDTH, 1.6), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    axes = [axes] if len(qsets) == 1 else axes
    marks = [("heuristic", ORANGE, "s"), ("version_chains", AQUA, "^"), ("logreg", BLUE, "o")]
    for ax, q in zip(axes, qsets, strict=True):
        for i, s in enumerate(r):
            d = s["downstream"][q]
            lo, hi = d["no_lineage"]["current@1"], d["oracle"]["current@1"]
            ax.plot([lo, hi], [i, i], color=GREY_LIGHT, lw=2.2, zorder=1, solid_capstyle="round")
            ax.scatter([lo], [i], s=24, color="white", edgecolor=GREY, lw=1.2, zorder=3)
            ax.scatter([hi], [i], s=28, color=INK, marker="|", lw=1.6, zorder=3)
            for m, colour, shape in marks:
                if m not in d:
                    continue
                x = d[m]["current@1"]
                ax.scatter(
                    [x], [i], s=30, color=colour, marker=shape, edgecolor="white", lw=1, zorder=4
                )
            g = d["logreg"]["gap_closed"]
            ax.text(
                d["logreg"]["current@1"],
                i - 0.32,
                f"{g:.0%} of gap",
                ha="center",
                fontsize=6.8,
                color=BLUE,
            )
        ax.set_yticks(range(len(r)), [s["setting"] for s in r])
        ax.set_ylim(len(r) - 0.5, -0.75)
        _pct_axis(ax)
        ax.set_xlim(0.15, 1.02)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{q}: current@1 with promote")
    fig.legend(
        handles=[
            Line2D([], [], ls="", marker="o", mfc="white", mec=GREY, label="no lineage"),
            Line2D([], [], ls="", marker="s", mfc=ORANGE, mec="white", label="heuristic"),
            Line2D([], [], ls="", marker="^", mfc=AQUA, mec="white", label="version chains"),
            Line2D([], [], ls="", marker="o", mfc=BLUE, mec="white", label="logistic regression"),
            Line2D([], [], ls="", marker="|", color=INK, mew=1.6, label="oracle lineage"),
        ],
        loc="lower center",
        ncol=5,
        columnspacing=1.2,
        handletextpad=0.3,
        bbox_to_anchor=(0.5, -0.26),
    )
    _save(fig, out, "fig_lineage")
    plt.close(fig)
    return True


def _model_style(model: str, prompt: str):
    colour = BLUE if "qwen" in model else ORANGE if "ministral" in model else AQUA
    filled = prompt.endswith("v2")
    return colour, filled


VARIANT_STYLE = {  # colour follows the variant; hollow markers for the comparison model
    "llm_only": (AQUA, "LLM only"),
    "no_find_later": (ORANGE, "agent, no find_later"),
    "full": (BLUE, "agent, all tools"),
}
MAIN_MODEL = "qwen3.5-hermes"


def _compare(res: Path, n: int = 100):
    return load(res / "agent" / f"compare_strict_n{n}.json")


def _method_label(m) -> str:
    name = VARIANT_STYLE.get(m["variant"], (GREY, m["variant"]))[1]
    return name if m["alias"] == MAIN_MODEL else f"{name} ({m['alias']})"


def _method_style(m):
    colour = VARIANT_STYLE.get(m["variant"], (GREY, ""))[0]
    return colour, m["alias"] == MAIN_MODEL


def fig_agent(res: Path, out: Path) -> bool:
    r = _compare(res)
    if not r:
        return False
    order = {"llm_only": 0, "no_find_later": 1, "full": 2}
    methods = sorted(
        r["methods"], key=lambda m: (m["alias"] != MAIN_MODEL, order.get(m["variant"], 9))
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH, 0.5 + 0.3 * (len(methods) + 1)),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for ax, truth in zip(axes, ("strict", "lineage"), strict=True):
        rows = [("cheap scorer (v0)", r["cheap_scorer"][truth], GREY, True)]
        rows += [(_method_label(m), m[truth], *_method_style(m)) for m in methods]
        for i, (_name, m, colour, filled) in enumerate(rows):
            lo, hi = m["balanced_accuracy_ci95"]
            v = m["balanced_accuracy"]
            ax.plot(
                [lo, hi], [i, i], color=colour, lw=1.4, alpha=0.55, zorder=2, solid_capstyle="round"
            )
            ax.scatter(
                [v],
                [i],
                s=38,
                zorder=3,
                color=colour if filled else "white",
                edgecolor=colour,
                lw=1.4,
            )
            ax.text(hi + 0.012, i, f"{v:.2f}", va="center", fontsize=7)
        ax.axvline(0.5, color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(0.505, len(rows) - 0.45, "chance", fontsize=6.5, color=GREY)
        ax.set_yticks(range(len(rows)), [x[0] for x in rows])
        ax.set_ylim(len(rows) - 0.35, -0.6)
        ax.set_xlim(0.45, 1.0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("balanced accuracy, 95% bootstrap CI")
        ax.set_title(f"{truth} ground truth")
    _save(fig, out, "fig_agent")
    plt.close(fig)
    return True


def fig_agent_cost(res: Path, out: Path) -> bool:
    r = _compare(res)
    if not r:
        return False
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.1))
    cheap = r["cheap_scorer"]["strict"]["balanced_accuracy"]
    ax.axhline(cheap, color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(
        13,
        0.565,
        f"dashed: cheap scorer, {cheap:.2f} (< 1 ms per pair)",
        fontsize=6.5,
        color=INK_2,
        va="bottom",
    )
    for m in r["methods"]:
        colour, filled = _method_style(m)
        s = m["strict"]
        x, (lo, hi), v = m["cost"]["seconds"], s["balanced_accuracy_ci95"], s["balanced_accuracy"]
        ax.plot(
            [x, x], [lo, hi], color=colour, lw=1.4, alpha=0.55, zorder=2, solid_capstyle="round"
        )
        ax.scatter(
            [x], [v], s=40, zorder=3, color=colour if filled else "white", edgecolor=colour, lw=1.4
        )
        # Fixed spots keep neighbouring labels apart: above or below the interval,
        # or beside the marker.
        if m["alias"] != MAIN_MODEL:
            xy, off, ha, va = (x, lo), (0, -4), "center", "top"
        elif m["variant"] == "no_find_later":
            xy, off, ha, va = (x, hi), (0, 4), "center", "bottom"
        elif m["variant"] == "llm_only":
            xy, off, ha, va = (x, v), (-7, 0), "right", "center"
        else:
            xy, off, ha, va = (x, v), (7, 0), "left", "center"
        ax.annotate(
            f"{_method_label(m)}\n{v:.2f}, {x:.0f} s",
            xy,
            xytext=off,
            textcoords="offset points",
            fontsize=6.5,
            ha=ha,
            va=va,
        )
    ax.set_xscale("log")
    ax.set_xlim(12, 600)
    ax.set_xticks([15, 30, 60, 120, 240, 480], ["15", "30", "60", "120", "240", "480"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0.55, 0.95)
    ax.set_xlabel("seconds per pair (local model, Apple M4, log scale)")
    ax.set_ylabel("balanced accuracy (strict)")
    _save(fig, out, "fig_agent_cost")
    plt.close(fig)
    return True


NOISE_METHODS = [
    ("none", "as retrieved", GREY, (0, (2, 2)), "o"),
    ("newest_first", "newest first", ORANGE, "-", "D"),
    ("time_decay", "time decay", AQUA, "-", "s"),
    ("lineage_v0", "inferred lineage (v0)", BLUE, "-", "o"),
    ("oracle", "oracle lineage", INK, (0, (4, 2)), ""),
]


def fig_noise(res: Path, out: Path) -> bool:
    r = load(res / "rfc_recency_noise.json")
    if not r:
        return False
    summ = r["summary"]
    qsets = sorted(
        dict.fromkeys(s["queries"] for s in summ), key=lambda q: not q.startswith("title")
    )
    fig, axes = plt.subplots(
        1, len(qsets), figsize=(WIDTH, 2.0), sharey=True, gridspec_kw={"wspace": 0.1}
    )
    axes = [axes] if len(qsets) == 1 else axes
    for ax, q in zip(axes, qsets, strict=True):
        for method, _, colour, ls, marker in NOISE_METHODS:
            pts = sorted(
                (s for s in summ if s["queries"] == q and s["method"] == method),
                key=lambda s: s["share"],
            )
            xs = [s["share"] for s in pts]
            ax.fill_between(
                xs,
                [s["current@1_min"] for s in pts],
                [s["current@1_max"] for s in pts],
                color=colour,
                alpha=0.12,
                lw=0,
            )
            ax.plot(
                xs,
                [s["current@1"] for s in pts],
                color=colour,
                ls=ls,
                lw=1.6,
                marker=marker or None,
                ms=4,
                mfc="white" if marker else None,
                zorder=3,
            )
        ax.set_xticks(xs, [f"{x:.0%}" for x in xs])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("documents with a missing or wrong year")
        ax.set_title(f"{q}: current@1")
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=c,
                ls=ls,
                lw=1.6,
                marker=m or None,
                ms=4,
                mfc="white" if m else None,
                label=lab,
            )
            for _, lab, c, ls, m in NOISE_METHODS
        ],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.24),
        handlelength=2.2,
    )
    _save(fig, out, "fig_noise")
    plt.close(fig)
    return True


# --- Figures for the dataset card (PNG only) -------------------------------------
# A worked example of the data (no results): one lineage, one query row on it, and
# the explicit cue that the strict view removes.
EXAMPLE_CHAIN = ["RFC2001", "RFC2581", "RFC5681"]
EXAMPLE_QID = "rfc-new_title-RFC2001-RFC2581"


def _box(ax, x, y, w, h, ec, lw=0.9, fc="white"):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015", fc=fc, ec=ec, lw=lw
        )
    )


def _cue(abstract: str, old: str) -> str:
    """The sentence of ``abstract`` that names ``old`` with a supersession verb."""
    import re

    num = old.replace("RFC", "")
    for sent in re.split(r"(?<=[.])\s+", abstract):
        if re.search(rf"RFC\s*{num}\b", sent) and re.search(r"obsolet|supersed|replac", sent):
            return sent.strip()
    return ""


def fig_card_example(docs: dict, graph, query: dict | None, out: Path) -> bool:
    if not all(d in docs for d in EXAMPLE_CHAIN) or query is None:
        return False
    fig = plt.figure(figsize=(WIDTH, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 1. the lineage
    ax.text(
        0.01,
        0.95,
        "1  a lineage in the index (lineage + corpus configs)",
        fontsize=8,
        fontweight="bold",
        va="center",
    )
    xs, w = [0.02, 0.355, 0.69], 0.29
    for i, (x, d) in enumerate(zip(xs, EXAMPLE_CHAIN, strict=True)):
        doc = docs[d]
        current = not graph.is_superseded(d)
        colour = AQUA if current else ORANGE
        _box(ax, x, 0.66, w, 0.2, colour, lw=1.2)
        ax.text(
            x + 0.012,
            0.815,
            f"{d.replace('RFC', 'RFC ')} · {doc.year}",
            fontsize=7.5,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x + w - 0.012,
            0.815,
            "current" if current else "superseded",
            fontsize=6.5,
            fontweight="bold",
            color=colour,
            ha="right",
            va="center",
        )
        title = doc.title if len(doc.title) <= 40 else doc.title[:38] + "…"
        ax.text(x + 0.012, 0.715, title, fontsize=6.3, color=INK_2, va="center")
        if i:
            ax.annotate(
                "",
                (x - 0.004, 0.76),
                (xs[i - 1] + w + 0.004, 0.76),
                arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 0.9},
            )
    ax.text((xs[0] + w + xs[1]) / 2, 0.895, "obsoleted by", fontsize=6, color=INK_2, ha="center")
    ax.text((xs[1] + w + xs[2]) / 2, 0.895, "obsoleted by", fontsize=6, color=INK_2, ha="center")

    # 2. a query row on that lineage
    ax.text(
        0.01,
        0.56,
        "2  a row of the queries config on it",
        fontsize=8,
        fontweight="bold",
        va="center",
    )
    _box(ax, 0.02, 0.33, 0.96, 0.18, GREY_LIGHT, fc="#f7f6f3")
    ax.text(0.035, 0.455, f"text: “{query['text']}”", fontsize=7.5, va="center")
    ax.text(0.035, 0.455 - 0.0, "", fontsize=7)
    ax.text(
        0.97,
        0.455,
        f"variant: {query['variant']}  ·  split: {query['split']}",
        fontsize=6.5,
        color=INK_2,
        ha="right",
        va="center",
    )
    cur = ", ".join(x.replace("RFC", "RFC ") for x in query["current"])
    stale = ", ".join(x.replace("RFC", "RFC ") for x in query["stale"])
    ax.text(0.035, 0.375, "current:", fontsize=7, color=INK_2, va="center")
    ax.text(0.095, 0.375, cur, fontsize=7, fontweight="bold", color=AQUA, va="center")
    ax.text(0.23, 0.375, "stale:", fontsize=7, color=INK_2, va="center")
    ax.text(0.275, 0.375, stale, fontsize=7, fontweight="bold", color=ORANGE, va="center")
    ax.text(
        0.97,
        0.375,
        "a good system ranks a current ID first",
        fontsize=6.5,
        color=INK_2,
        ha="right",
        va="center",
        style="italic",
    )

    # 3. the cue that the strict view removes
    new, old = EXAMPLE_CHAIN[-1], EXAMPLE_CHAIN[-2]
    cue = _cue(docs[new].abstract, old)
    ax.text(
        0.01,
        0.23,
        "3  the explicit cue hidden for lineage inference",
        fontsize=8,
        fontweight="bold",
        va="center",
    )
    ax.text(
        0.035,
        0.14,
        f"{new.replace('RFC', 'RFC ')}'s abstract says: “{cue}”",
        fontsize=7,
        va="center",
    )
    ax.text(
        0.035,
        0.07,
        "masking.py masks RFC numbers (masked) and drops such "
        "sentences (strict); a model must infer the link from content.",
        fontsize=6.5,
        color=INK_2,
        va="center",
    )
    fig.savefig(out / "fig_card_example.png", dpi=200, facecolor="white")
    plt.close(fig)
    return True


def fig_card_corpus(docs: list, graph, out: Path) -> bool:
    by = Counter()
    for d in docs:
        if d.year:
            by[(d.year // 10 * 10, graph.is_superseded(d.doc_id))] += 1
    decades = sorted({k[0] for k in by})
    if not decades:
        return False
    cur = [by[(x, False)] for x in decades]
    sup = [by[(x, True)] for x in decades]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    xs = range(len(decades))
    ax.bar(xs, cur, color=BLUE_LIGHT, width=0.62, label="still in force", edgecolor="white", lw=1)
    ax.bar(
        xs,
        sup,
        bottom=cur,
        color=ORANGE,
        width=0.62,
        label="superseded since",
        edgecolor="white",
        lw=1,
    )
    for x, c, s_ in zip(xs, cur, sup, strict=True):
        ax.text(x, c + s_ + 40, f"{s_ / (c + s_):.0%}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(list(xs), [f"{x}s" for x in decades])
    ax.set_ylabel("RFCs published")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(c + s_ for c, s_ in zip(cur, sup, strict=True)) * 1.15)
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("RFCs per decade; above each bar, the share later superseded")
    fig.savefig(out / "fig_card_corpus.png", dpi=200, facecolor="white")
    plt.close(fig)
    return True


def card_figures(data: Path, out: Path) -> dict[str, bool]:
    from supersede.corpus import load_docs
    from supersede.lineage import LineageGraph

    if not (data / "corpus.jsonl").exists():
        return {}
    from supersede.corpus import read_jsonl

    docs = load_docs(data / "corpus.jsonl")
    graph = LineageGraph(docs)
    qpath = data / "queries.jsonl"
    query = (
        next((q for q in read_jsonl(qpath) if q["qid"] == EXAMPLE_QID), None)
        if qpath.exists()
        else None
    )
    out.mkdir(parents=True, exist_ok=True)
    return {
        "fig_card_example": fig_card_example({d.doc_id: d for d in docs}, graph, query, out),
        "fig_card_corpus": fig_card_corpus(docs, graph, out),
    }


def build(res: Path, out: Path) -> dict[str, bool]:
    out.mkdir(parents=True, exist_ok=True)
    return {
        "fig_retrieval": fig_retrieval(res, out),
        "fig_answers": fig_answers(res, out),
        "fig_lineage": fig_lineage(res, out),
        "fig_agent": fig_agent(res, out),
        "fig_agent_cost": fig_agent_cost(res, out),
        "fig_noise": fig_noise(res, out),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("paper/generated"))
    ap.add_argument("--data", type=Path, default=Path("data/rfc"))
    args = ap.parse_args()
    print(build(args.results, args.out) | card_figures(args.data, args.out))


if __name__ == "__main__":
    main()
