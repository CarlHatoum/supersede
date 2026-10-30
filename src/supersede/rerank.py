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
