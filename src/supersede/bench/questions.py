"""Change-aware questions: the answer differs between a document and its replacement.

For each supersession edge where both documents have an abstract, the LLM
sees the older and newer abstracts. It writes a question a practitioner might
ask, with the answer under each version and a verbatim evidence span for each.
Items are kept only if they pass these checks:

  * each evidence span is a verbatim substring of its abstract (whitespace and
    case normalised),
  * the two answers differ,
  * the question doesn't name an RFC number or say "this document".

Usage (run from a regular terminal, not inside Claude Code):
    python -m supersede.bench.questions --data data/rfc --limit 50
    python -m supersede.bench.questions --data data/rfc --offline   # cache only
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from supersede.corpus import Doc, load_docs, write_jsonl
from supersede.lineage import LineageGraph
from supersede.llm import CachedLLM, CacheMiss, extract_json, make_llm

PROMPT_VERSION = "q-v1"

SYSTEM = (
    "You build evaluation data for retrieval systems over technical standards. "
    "You answer with a single JSON object and nothing else."
)

PROMPT = """Two versions of a technical standard follow. The NEWER one replaces the OLDER one.

OLDER — {old_title}
{old_abstract}

NEWER — {new_title}
{new_abstract}

Write ONE question that an engineer could ask without knowing either document exists, such that \
the correct answer according to the NEWER document differs from the answer according to the OLDER \
one. Both answers must be stated in the abstracts above, not inferred from outside knowledge.

Rules:
- Do not mention RFC numbers, "this document", "the newer/older version", or publication dates.
- Name the protocol or topic explicitly so the question stands alone.
- Each evidence field is copied VERBATIM from the corresponding abstract (one short span).
- If the abstracts do not support such a question, return {{"skip": "<short reason>"}}.

Return:
{{"question": "...", "answer_current": "...", "answer_stale": "...", \
"evidence_current": "<verbatim span from NEWER>", \
"evidence_stale": "<verbatim span from OLDER>"}}"""

_RFC_REF = re.compile(r"\brfc\s*-?\s*\d+|\bthis (document|memo|specification)\b", re.I)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def validate(item: dict, old: Doc, new: Doc) -> str | None:
    """The reason to reject ``item``, or None if it passes."""
    if "skip" in item:
        return "skipped_by_model"
    fields = ("question", "answer_current", "answer_stale", "evidence_current", "evidence_stale")
    if any(not isinstance(item.get(f), str) or not item[f].strip() for f in fields):
        return "missing_field"
    if _norm(item["evidence_current"]) not in _norm(new.abstract):
        return "evidence_current_not_verbatim"
    if _norm(item["evidence_stale"]) not in _norm(old.abstract):
        return "evidence_stale_not_verbatim"
    if _norm(item["answer_current"]) == _norm(item["answer_stale"]):
        return "answers_identical"
    if _RFC_REF.search(item["question"]):
        return "question_leaks_reference"
    return None


def candidate_edges(docs: list[Doc], graph: LineageGraph) -> list[tuple[str, str]]:
    by_id = {d.doc_id: d for d in docs}
    return [
        (o, n)
        for o, n in graph.edges
        if by_id[o].abstract and by_id[n].abstract and by_id[o].abstract != by_id[n].abstract
    ]


def generate_one(llm: CachedLLM, old: Doc, new: Doc) -> tuple[dict | None, str]:
    prompt = PROMPT.format(
        old_title=old.title, old_abstract=old.abstract,
        new_title=new.title, new_abstract=new.abstract,
    )  # fmt: skip
    completion = llm.complete(prompt, SYSTEM, prompt_version=PROMPT_VERSION)
    try:
        item = extract_json(completion.text)
    except ValueError:
        return None, "unparseable"
    reason = validate(item, old, new)
    if reason:
        return None, reason
    item["_model"] = completion.model
    return item, "accepted"


def build(
    data: Path, llm: CachedLLM, limit: int | None, seed: int = 0, workers: int = 4
) -> tuple[list[dict], dict]:
    from supersede.bench.rfc import split_of

    docs = load_docs(data / "corpus.jsonl")
    by_id = {d.doc_id: d for d in docs}
    graph = LineageGraph(docs)
    edges = candidate_edges(docs, graph)
    random.Random(seed).shuffle(edges)
    if limit:
        edges = edges[:limit]

    def work(edge: tuple[str, str]) -> tuple[tuple[str, str], dict | None, str]:
        try:
            item, status = generate_one(llm, by_id[edge[0]], by_id[edge[1]])
        except CacheMiss:
            return edge, None, "cache_miss"
        return edge, item, status

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, edges))

    questions, reasons = [], {}
    for (old, new), item, status in results:
        reasons[status] = reasons.get(status, 0) + 1
        if item is None:
            continue
        comp = graph.component(old)
        questions.append(
            {
                "qid": f"rfc-change-{old}-{new}",
                "text": item["question"],
                "variant": "change_aware",
                "source_doc": new,
                "edge": [old, new],
                "current": sorted(graph.current_versions(old)),
                "stale": sorted(d for d in comp if graph.is_superseded(d)),
                "component": min(comp),
                "split": split_of(comp),
                "answer_current": item["answer_current"],
                "answer_stale": item["answer_stale"],
                "evidence_current": item["evidence_current"],
                "evidence_stale": item["evidence_stale"],
                "provenance": {"model": item["_model"], "prompt_version": PROMPT_VERSION},
            }
        )
    questions.sort(key=lambda q: q["qid"])
    attempted = len(edges) - reasons.get("cache_miss", 0)
    stats = {
        "prompt_version": PROMPT_VERSION,
        "backend": llm.name,
        "n_candidate_edges": len(candidate_edges(docs, graph)),
        "n_attempted": attempted,
        "n_accepted": len(questions),
        "acceptance_rate": len(questions) / attempted if attempted else 0.0,
        "outcomes": dict(sorted(reasons.items())),
        "usage": llm.stats.as_dict(),
    }
    return questions, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/rfc"))
    ap.add_argument("--backend", choices=("claude-cli", "ollama"), default="claude-cli")
    ap.add_argument("--model", help="model ID; defaults to the backend's default")
    ap.add_argument("--cache", type=Path, default=Path("cache/llm.sqlite"))
    ap.add_argument("--offline", action="store_true", help="read the cache only, never call")
    ap.add_argument("--limit", type=int, help="number of edges to try (seeded sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    llm = make_llm(args.backend, args.model, args.cache, offline=args.offline)
    questions, stats = build(args.data, llm, args.limit, args.seed, args.workers)
    write_jsonl(args.data / "questions.jsonl", questions)
    (args.data / "questions_manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
