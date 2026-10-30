"""Dependency-free BM25 baseline retriever."""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it its of on or that the this to with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, docs: Sequence[tuple[str, str]], k1: float = 1.2, b: float = 0.75):
        """``docs`` is a sequence of ``(doc_id, text)`` pairs."""
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, (doc_id, text) in enumerate(docs):
            tokens = tokenize(text)
            self.ids.append(doc_id)
            self.lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term].append((i, count))
        n = len(self.ids)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = 1 - self.b + self.b * self.lengths[i] / self.avgdl
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        # Ties are broken by corpus order so that runs are deterministic.
        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [(self.ids[i], s) for i, s in top]
