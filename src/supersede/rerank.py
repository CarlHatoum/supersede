"""Version-aware post-processing of a ranked list.

These policies assume the lineage metadata is known (an oracle librarian).
They give the upper bound that an inferred lineage will be measured against.
"""

from __future__ import annotations

from collections.abc import Callable

from supersede.lineage import LineageGraph

Ranking = list[str]


def no_rerank(ranking: Ranking, graph: LineageGraph) -> Ranking:
    return list(ranking)


def filter_superseded(ranking: Ranking, graph: LineageGraph) -> Ranking:
    """Drop superseded documents. Recall drops if the successor was not retrieved."""
    return [d for d in ranking if not graph.is_superseded(d)]


def promote_current(ranking: Ranking, graph: LineageGraph) -> Ranking:
    """Put the current versions of each superseded hit just above it.

    The stale document is kept, ranked lower, so that an answer policy can still
    cite it as "previously ...".
    """
    out: list[str] = []
    seen: set[str] = set()
    for doc in ranking:
        if graph.is_superseded(doc):
            for head in sorted(graph.current_versions(doc)):
                if head not in seen:
                    out.append(head)
                    seen.add(head)
        if doc not in seen:
            out.append(doc)
            seen.add(doc)
    return out[: len(ranking)]


POLICIES: dict[str, Callable[[Ranking, LineageGraph], Ranking]] = {
    "none": no_rerank,
    "filter": filter_superseded,
    "promote": promote_current,
}


# ---------------------------------------------------------------------------
# Recency baselines: no lineage, only publication years.
#
# They answer the obvious objection "why not just prefer the newest document?".
# Both re-rank the retriever's scored candidates (deeper than the final top k,
# so a newer document just outside it can come in) and need only ``years``.
# A missing year is treated as old by ``newest_first`` and as the median age by
# ``time_decay``.

Scored = list[tuple[str, float]]


def newest_first(scored: Scored, years: dict[str, int | None], window: int, k: int) -> Ranking:
    """Within the top ``window`` hits, put later documents first (stable); then the rest."""
    head = [d for d, _ in scored[:window]]
    tail = [d for d, _ in scored[window:]]
    head.sort(key=lambda d: -(years.get(d) or 0))
    return (head + tail)[:k]


def time_decay(
    scored: Scored, years: dict[str, int | None], lam: float, k: int, ref_year: int | None = None
) -> Ranking:
    """Multiply each retrieval score by ``exp(-lam * age)``, age in years before ``ref_year``."""
    import math
    import statistics

    known = [y for y in years.values() if y]
    ref = ref_year or max(known)
    median_age = ref - statistics.median(known) if known else 0
    rescored = []
    for d, s in scored:
        y = years.get(d)
        age = (ref - y) if y else median_age
        rescored.append((d, s * math.exp(-lam * age)))
    rescored.sort(key=lambda x: -x[1])
    return [d for d, _ in rescored[:k]]


RECENCY: dict[str, tuple[Callable[..., Ranking], str, tuple]] = {
    "newest_first": (newest_first, "window", (2, 3, 5, 10, 20)),
    "time_decay": (time_decay, "lam", (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)),
}
