"""Closed-book baseline: what does the answering model say with no documents at all?

Models have likely read many RFCs in pre-training. If the model already knows the
current answer, a stale RAG answer means the retrieved context *overrode* correct
parametric knowledge; if it does not, retrieval is the only route to the current
answer. Same answering model, judge and judge prompt as ``answer_staleness``.

Reported:

* closed-book outcome rates (current, stale, abstain...), 95% bootstrap CI;
* for each RAG run (standard, blind) and answer policy, joined per question:
  ``override``: share of questions the model gets right closed-book whose RAG
  answer is stale; ``rescue``: share it does not get right closed-book whose RAG
  answer is current.

Usage (LLM calls: run from a regular terminal, or add --offline):
    python -m supersede.experiments.closed_book
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from supersede import judge as judge_mod
from supersede.corpus import read_jsonl, write_jsonl
from supersede.llm import CacheMiss, LLMError, extract_json, make_llm
from supersede.metrics import bootstrap_ci

PROMPT_VERSION = "cb-v1"
SYSTEM = (
    "You answer engineering questions from your own knowledge. "
    "You answer with a single JSON object and nothing else."
)
PROMPT = """Question: {question}

Answer from what you know about the relevant standards as they stand today. If you
do not know, say so.

Return: {{"answer": "..."}}"""

CURRENT = ("current", "mixed_flagged")
RAG_RUNS = {"standard": "rfc_answer_staleness_hybrid", "blind": "rfc_answer_staleness_hybrid_blind"}


def closed_book_answer(llm, question: str) -> str:
    completion = llm.complete(
        PROMPT.format(question=question), SYSTEM, prompt_version=PROMPT_VERSION
    )
    try:
        return str(extract_json(completion.text).get("answer", ""))
    except ValueError:
        return completion.text.strip()


def rate(labels: list[str], hit, seed: int = 0) -> dict:
    vals = [float(hit(lab)) for lab in labels]
    return {
        "rate": sum(vals) / len(vals) if vals else float("nan"),
        "ci95": bootstrap_ci(vals, seed=seed),
        "n": len(vals),
    }


def joint(cb: dict[str, str], rag_items: list[dict]) -> dict:
    """Override and rescue rates per answer policy, joined on question id."""
    out = {}
    for policy in sorted({i["policy"] for i in rag_items}):
        rag = {i["qid"]: i["label"] for i in rag_items if i["policy"] == policy and i.get("label")}
        common = [q for q in cb if q in rag]
        knows = [q for q in common if cb[q] in CURRENT]
        not_knows = [q for q in common if cb[q] not in CURRENT]
        out[policy] = {
            "n": len(common),
            "override": rate([rag[q] for q in knows], lambda lab: lab in judge_mod.STALE_OUTCOMES),
            "rescue": rate([rag[q] for q in not_knows], lambda lab: lab in CURRENT),
            "rag_stale": rate(
                [rag[q] for q in common], lambda lab: lab in judge_mod.STALE_OUTCOMES
            ),
        }
    return out


def run(data: Path, results: Path, llm, judge_llm, workers: int = 2) -> dict:
    questions = list(read_jsonl(data / "questions.jsonl"))

    def work(q: dict) -> dict:
        item = {"qid": q["qid"], "question": q["text"]}
        try:
            text = closed_book_answer(llm, q["text"])
            v = judge_mod.judge(judge_llm, q, text)
        except CacheMiss:
            return {**item, "label": None, "error": "cache_miss"}
        except LLMError as e:
            return {**item, "label": None, "error": str(e)[:300]}
        return {**item, "answer": text, "label": v.label, "rationale": v.rationale}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        items = list(pool.map(work, questions))
    done = [i for i in items if i["label"]]
    labels = [i["label"] for i in done]
    cb = {i["qid"]: i["label"] for i in done}
    rag = {}
    for name, stem in RAG_RUNS.items():
        path = results / f"{stem}_items.jsonl"
        if path.exists():
            rag[name] = joint(cb, list(read_jsonl(path)))
    return {
        "experiment": "closed_book",
        "answer_backend": llm.name,
        "judge_backend": judge_llm.name,
        "prompt_versions": {"answer": PROMPT_VERSION, "judge": judge_mod.PROMPT_VERSION},
        "n_questions": len(questions),
        "n_failed": len(items) - len(done),
        "labels": {lab: labels.count(lab) for lab in judge_mod.LABELS},
        "current": rate(labels, lambda lab: lab in CURRENT),
        "stale": rate(labels, lambda lab: lab in judge_mod.STALE_OUTCOMES),
        "abstain": rate(labels, lambda lab: lab == "abstain"),
        "rag": rag,
        "usage": {"answer": llm.stats.as_dict(), "judge": judge_llm.stats.as_dict()},
        "items": items,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/rfc"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--judge-model", default="claude-haiku-5-5")
    ap.add_argument("--cache", type=Path, default=Path("cache/llm.sqlite"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    llm = make_llm("claude-cli", args.model, args.cache, args.offline)
    judge_llm = make_llm("claude-cli", args.judge_model, args.cache, args.offline)
    result = run(args.data, args.results, llm, judge_llm, args.workers)
    write_jsonl(args.results / "rfc_closed_book_items.jsonl", result.pop("items"))
    (args.results / "rfc_closed_book.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"closed-book: current {result['current']['rate']:.2f}, stale "
        f"{result['stale']['rate']:.2f}, abstain {result['abstain']['rate']:.2f} "
        f"({result['n_failed']} failed)"
    )
    for name, per in result["rag"].items():
        for policy, d in per.items():
            print(
                f"  {name:8} {policy:13} override {d['override']['rate']:.2f} "
                f"(n={d['override']['n']})  rescue {d['rescue']['rate']:.2f} "
                f"(n={d['rescue']['n']})"
            )
    print(json.dumps(result["usage"]))


if __name__ == "__main__":
    main()
