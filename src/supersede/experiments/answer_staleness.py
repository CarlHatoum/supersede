"""Experiment 1: how often does a RAG answer give the superseded answer?

For each change-aware question: retrieve, answer under each policy, then judge
the answer against the current and superseded references. Writes
``results/<name>.{json,md}``, plus every item to ``results/<name>_items.jsonl``.

Judge validation: ``--export-labeling N`` writes a seeded sample of N items to
label by hand. Fill each ``human_label`` with a value from ``judge.LABELS``,
then ``--kappa FILE`` reports judge-vs-human agreement.

Usage (LLM calls: run from a regular terminal, or add --offline):
    python -m supersede.experiments.answer_staleness --data data/rfc --retriever hybrid
    python -m supersede.experiments.answer_staleness --export-labeling 100
    python -m supersede.experiments.answer_staleness --kappa results/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from supersede import answer as answer_mod
from supersede import judge as judge_mod
from supersede.corpus import load_docs, read_jsonl, write_jsonl
from supersede.experiments.staleness import RETRIEVERS, build_retrievers
from supersede.lineage import LineageGraph
from supersede.llm import CacheMiss, make_llm
from supersede.metrics import bootstrap_ci, cohen_kappa

DEPTH = 10  # documents retrieved before the answer policy picks its context
CONTEXT_K = 5


def run(data: Path, retriever: str, llm, judge_llm, workers: int = 2, seed: int = 0) -> dict:
    docs = load_docs(data / "corpus.jsonl")
    by_id = {d.doc_id: d for d in docs}
    graph = LineageGraph(docs)
    questions = list(read_jsonl(data / "questions.jsonl"))
    index = build_retrievers([retriever], [(d.doc_id, d.text) for d in docs], data)[retriever]
    rankings = [[d for d, _ in h] for h in index.search_many([q["text"] for q in questions], DEPTH)]

    def work(job: tuple[dict, list[str], str]) -> dict:
        q, ranking, policy = job
        item = {"qid": q["qid"], "policy": policy, "question": q["text"]}
        item.update(answer_current=q["answer_current"], answer_stale=q["answer_stale"])
        try:
            a = answer_mod.answer(llm, q["text"], ranking, by_id, graph, policy, CONTEXT_K)
            v = judge_mod.judge(judge_llm, q, a.text)
        except CacheMiss:
            return {**item, "label": None}
        stale_in_ctx = any(d in set(q["stale"]) for d in a.context_ids)
        return {
            **item,
            "answer": a.text,
            "citations": list(a.citations),
            "context": list(a.context_ids),
            "stale_in_context": stale_in_ctx,
            "current_in_context": any(d in set(q["current"]) for d in a.context_ids),
            "label": v.label,
            "rationale": v.rationale,
        }

    pairs = zip(questions, rankings, strict=True)
    jobs = [(q, r, p) for q, r in pairs for p in answer_mod.POLICIES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        items = list(pool.map(work, jobs))
    done = [i for i in items if i["label"] is not None]

    rows = []
    for policy in answer_mod.POLICIES:
        labels = [i["label"] for i in done if i["policy"] == policy]
        row = {"policy": policy, "n": len(labels)}
        for name, hit in (
            ("stale_outcome", lambda lab: lab in judge_mod.STALE_OUTCOMES),
            ("current_outcome", lambda lab: lab in ("current", "mixed_flagged")),
            ("flagged_conflict", lambda lab: lab == "mixed_flagged"),
            ("abstain", lambda lab: lab == "abstain"),
        ):
            vals = [float(hit(lab)) for lab in labels]
            row[name] = sum(vals) / len(vals) if vals else float("nan")
            row[f"{name}_ci95"] = bootstrap_ci(vals, seed=seed)
        rows.append(row)

    return {
        "experiment": "answer_staleness",
        "retriever": index.name,
        "depth": DEPTH,
        "context_k": CONTEXT_K,
        "answer_backend": llm.name,
        "judge_backend": judge_llm.name,
        "prompt_versions": {"answer": answer_mod.PROMPT_VERSION, "judge": judge_mod.PROMPT_VERSION},
        "n_questions": len(questions),
        "n_missing": len(items) - len(done),
        "rows": rows,
        "usage": {"answer": llm.stats.as_dict(), "judge": judge_llm.stats.as_dict()},
        "items": items,
    }


def to_markdown(result: dict) -> str:
    lines = [
        f"{result['n_questions']} change-aware questions, retriever `{result['retriever']}`, "
        f"context top-{result['context_k']}. Answers by `{result['answer_backend']}`, "
        f"judged by `{result['judge_backend']}` (not yet validated against human labels). "
        "95% bootstrap CI in brackets.",
        "",
        "| policy | n | stale answer ↓ | current answer ↑ | conflict flagged | abstain |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        cells = [
            f"{r[m]:.3f} [{r[m + '_ci95'][0]:.3f}, {r[m + '_ci95'][1]:.3f}]"
            for m in ("stale_outcome", "current_outcome", "flagged_conflict", "abstain")
        ]
        lines.append(f"| {r['policy']} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def export_labeling(items_path: Path, n: int, out: Path, seed: int = 0) -> int:
    items = [i for i in read_jsonl(items_path) if i.get("label")]
    random.Random(seed).shuffle(items)
    # The judge's label is withheld so that it can't anchor the human rater.
    keep = ("qid", "policy", "question", "answer_current", "answer_stale", "answer")
    return write_jsonl(out, ({**{k: i[k] for k in keep}, "human_label": ""} for i in items[:n]))


def kappa(labels_path: Path, items_path: Path) -> dict:
    judged = {(i["qid"], i["policy"]): i["label"] for i in read_jsonl(items_path)}
    pairs = [
        (row["human_label"], judged[(row["qid"], row["policy"])])
        for row in read_jsonl(labels_path)
        if row.get("human_label") in judge_mod.LABELS
    ]
    human, model = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "kappa": cohen_kappa(human, model),
        "accuracy": sum(h == m for h, m in pairs) / len(pairs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/rfc"))
    ap.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    ap.add_argument("--backend", choices=("claude-cli", "ollama"), default="claude-cli")
    ap.add_argument("--model", help="answering model; defaults to the backend's default")
    ap.add_argument("--judge-model", help="judge model; defaults to --model")
    ap.add_argument("--cache", type=Path, default=Path("cache/llm.sqlite"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--export-labeling", type=int, metavar="N")
    ap.add_argument("--kappa", type=Path, metavar="LABELS_JSONL")
    args = ap.parse_args()

    name = f"rfc_answer_staleness_{args.retriever}"
    items_path = args.out / f"{name}_items.jsonl"
    if args.export_labeling:
        out = args.out / f"{name}_to_label.jsonl"
        print(f"wrote {export_labeling(items_path, args.export_labeling, out)} items to {out}")
        return
    if args.kappa:
        print(json.dumps(kappa(args.kappa, items_path), indent=2))
        return

    llm = make_llm(args.backend, args.model, args.cache, args.offline)
    judge_llm = make_llm(args.backend, args.judge_model or args.model, args.cache, args.offline)
    result = run(args.data, args.retriever, llm, judge_llm, args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(items_path, result.pop("items"))
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    md = to_markdown(result)
    (args.out / f"{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
