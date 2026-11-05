"""Candidate (old, new) pairs: near neighbours in embedding space, ordered by date."""

from __future__ import annotations

import re

from supersede.corpus import Doc

_DIGITS = re.compile(r"\d+")


def doc_order(d: Doc) -> tuple[int, int, str]:
    """Publication order: year, then the document number when there is one.

    Every corpus in the benchmark has dates, and RFC numbers are assigned in
    publication order, so this is metadata a real knowledge base would have.
    """
    m = _DIGITS.search(d.doc_id)
    return (d.year or 0, int(m.group()) if m else 0, d.doc_id)


def year_only_order(d: Doc) -> tuple[int, str]:
    """Publication order from the year alone, ties broken by a stable hash of the ID.

    For experiments where dates are unreliable: document numbers must not leak
    the true order.
    """
    import hashlib

    return (d.year or 0, hashlib.sha1(d.doc_id.encode()).hexdigest())


def neighbour_pairs(
    docs: list[Doc], emb, n_neighbours: int = 10, block: int = 1024, order_key=doc_order
):
    """``{(old, new): rank}``, where rank is 1-based in either doc's neighbour list (best kept).

    ``emb`` is an (n_docs, dim) array of unit vectors aligned with ``docs``;
    ``order_key`` decides which document of a pair is the older one.
    """
    import numpy as np

    order = [order_key(d) for d in docs]
    pairs: dict[tuple[str, str], int] = {}
    for start in range(0, len(docs), block):
        sims = emb[start : start + block] @ emb.T
        for row, i in enumerate(range(start, min(start + block, len(docs)))):
            sims[row, i] = -np.inf
            top = np.argpartition(-sims[row], n_neighbours)[:n_neighbours]
            top = top[np.argsort(-sims[row, top], kind="stable")]
            for rank, j in enumerate(top, 1):
                old, new = (i, j) if order[i] < order[j] else (j, i)
                key = (docs[old].doc_id, docs[new].doc_id)
                pairs[key] = min(pairs.get(key, rank), rank)
    return pairs
