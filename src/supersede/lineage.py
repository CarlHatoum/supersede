"""Supersession graph: which document replaces which, and what is current."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from supersede.corpus import Doc


class LineageGraph:
    """Directed graph old -> new over the documents of one corpus.

    Edges are taken from both sides of the declaration (``obsoletes`` on the new
    document, ``obsoleted_by`` on the old one), so a link declared on only one
    side is still honoured. Such one-sided links are counted in
    ``asymmetric_edges`` as a data-quality signal.
    """

    def __init__(self, docs: Iterable[Doc]):
        docs = list(docs)
        self.ids = {d.doc_id for d in docs}
        self.successors: dict[str, set[str]] = defaultdict(set)
        self.predecessors: dict[str, set[str]] = defaultdict(set)
        declared_forward: set[tuple[str, str]] = set()
        declared_backward: set[tuple[str, str]] = set()
        self.dangling_edges = 0

        for d in docs:
            for old in d.obsoletes:
                declared_forward.add((old, d.doc_id))
            for new in d.obsoleted_by:
                declared_backward.add((d.doc_id, new))

        for old, new in declared_forward | declared_backward:
            if old not in self.ids or new not in self.ids or old == new:
                self.dangling_edges += 1
                continue
            self.successors[old].add(new)
            self.predecessors[new].add(old)

        # Only links between two documents in the corpus count here; the rest are dangling.
        self.asymmetric_edges = sum(
            1
            for old, new in declared_forward ^ declared_backward
            if old in self.ids and new in self.ids
        )

    @property
    def edges(self) -> list[tuple[str, str]]:
        return sorted((o, n) for o, ns in self.successors.items() for n in ns)

    def is_superseded(self, doc_id: str) -> bool:
        return bool(self.successors.get(doc_id))

    def current_versions(self, doc_id: str) -> frozenset[str]:
        """Documents that are in force for the topic of ``doc_id``.

        Follows successors transitively and returns the heads (documents with no
        successor). A document can be split into several replacements, so the
        result can hold more than one ID. A pure cycle has no head, so
        ``doc_id`` itself is returned in that case.
        """
        if not self.is_superseded(doc_id):
            return frozenset({doc_id})
        heads, seen, queue = set(), {doc_id}, deque([doc_id])
        while queue:
            node = queue.popleft()
            nxt = self.successors.get(node, ())
            if not nxt:
                heads.add(node)
            for n in nxt:
                if n not in seen:
                    seen.add(n)
                    queue.append(n)
        return frozenset(heads or {doc_id})

    def component(self, doc_id: str) -> frozenset[str]:
        """All documents connected to ``doc_id`` by supersession, either direction."""
        seen, queue = {doc_id}, deque([doc_id])
        while queue:
            node = queue.popleft()
            for n in self.successors.get(node, set()) | self.predecessors.get(node, set()):
                if n not in seen:
                    seen.add(n)
                    queue.append(n)
        return frozenset(seen)

    def components(self) -> list[frozenset[str]]:
        seen: set[str] = set()
        out = []
        for old, _ in self.edges:
            if old not in seen:
                comp = self.component(old)
                seen |= comp
                out.append(comp)
        return out
