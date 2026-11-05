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
    a.legend(
        handles=[Line2D([], [], color=c, lw=6, label=k) for k, c in RETRIEVER.items()],
        loc="lower right",
        ncol=3,
        handlelength=0.9,
        columnspacing=0.9,
        bbox_to_anchor=(1.0, -0.42),
    )

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
        b.text(x0 - 0.025, i, f"{x0:.0%}", va="center", ha="right", fontsize=7, color=INK_2)
        b.text(x1 + 0.025, i, f"{x1:.0%}", va="center", fontsize=7)
    b.set_yticks(range(len(items)), [it[0] for it in items])
    b.invert_yaxis()
    _pct_axis(b, 1.12)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_xlabel("current@1: top result is in force")
    b.set_title("(b) Known lineage fixes it")
    b.legend(
        handles=[
            Line2D([], [], ls="", marker="o", mfc="white", mec=GREY, label="as retrieved"),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                mfc=INK_2,
                mec="white",
                label="with promote (oracle lineage)",
            ),
        ],
        loc="lower right",
        ncol=2,
        bbox_to_anchor=(1.0, -0.42),
    )
    fig.savefig(out / "fig_retrieval.pdf")
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
    fig.savefig(out / "fig_answers.pdf")
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
    marks = [("heuristic", ORANGE, "s"), ("logreg", BLUE, "o")]
    for ax, q in zip(axes, qsets, strict=True):
        for i, s in enumerate(r):
            d = s["downstream"][q]
            lo, hi = d["no_lineage"]["current@1"], d["oracle"]["current@1"]
            ax.plot([lo, hi], [i, i], color=GREY_LIGHT, lw=2.2, zorder=1, solid_capstyle="round")
            ax.scatter([lo], [i], s=24, color="white", edgecolor=GREY, lw=1.2, zorder=3)
            ax.scatter([hi], [i], s=28, color=INK, marker="|", lw=1.6, zorder=3)
            for m, colour, shape in marks:
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
            Line2D([], [], ls="", marker="o", mfc=BLUE, mec="white", label="logistic regression"),
            Line2D([], [], ls="", marker="|", color=INK, mew=1.6, label="oracle lineage"),
        ],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.savefig(out / "fig_lineage.pdf")
    plt.close(fig)
    return True


def _model_style(model: str, prompt: str):
    colour = BLUE if "qwen" in model else ORANGE if "ministral" in model else AQUA
    filled = prompt.endswith("v2")
    return colour, filled


def fig_agent(res: Path, out: Path) -> bool:
    runs = []
    for p in sorted((res / "agent" / "history").glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("experiment") == "agent_lineage" and r["agent"].get("balanced_accuracy_ci95"):
            runs.append(r)
    if not runs:
        return False
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH, 0.42 + 0.3 * (len(runs) + 1)),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for ax, truth in zip(axes, ("strict", "lineage"), strict=True):
        rows = [
            (
                "cheap scorer (v0)",
                runs[0] if truth == "strict" else runs[0]["lineage_truth"],
                "cheap_scorer",
                GREY,
                True,
            )
        ]
        for r in runs:
            block = r if truth == "strict" else r["lineage_truth"]
            model = r["model"].split(":", 1)[-1].split("/")[-1].split(":")[0]
            colour, filled = _model_style(model, r["prompt_version"])
            rows.append(
                (f"agent: {model}, {r['prompt_version'][-2:]}", block, "agent", colour, filled)
            )
        for i, (_name, block, who, colour, filled) in enumerate(rows):
            m = block[who]
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
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.set_ylim(len(rows) - 0.35, -0.6)
        ax.set_xlim(0.35, 1.08)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("balanced accuracy, 95% bootstrap CI")
        ax.set_title(f"{truth} ground truth")
    fig.savefig(out / "fig_agent.pdf")
    plt.close(fig)
    return True


def build(res: Path, out: Path) -> dict[str, bool]:
    out.mkdir(parents=True, exist_ok=True)
    return {
        "fig_retrieval": fig_retrieval(res, out),
        "fig_answers": fig_answers(res, out),
        "fig_lineage": fig_lineage(res, out),
        "fig_agent": fig_agent(res, out),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("paper/generated"))
    args = ap.parse_args()
    print(build(args.results, args.out))


if __name__ == "__main__":
    main()
