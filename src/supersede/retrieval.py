"""Retrievers: dependency-free BM25, dense embeddings, and hybrid rank fusion.

Every retriever exposes ``search(query, k)`` and ``search_many(queries, k)``,
both returning ``(doc_id, score)`` lists. Dense retrieval needs the ``dense``
extra (numpy, sentence-transformers); it is imported lazily so the core stays
dependency-free.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it its of on or that the this to with".split()
)


Hits = list[tuple[str, float]]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> Hits: ...

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[Hits]: ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, docs: Sequence[tuple[str, str]], k1: float = 1.2, b: float = 0.75):
        """``docs`` is a sequence of ``(doc_id, text)`` pairs."""
        self.k1, self.b = k1, b
        self.name = f"bm25(k1={k1},b={b})"
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

    def search(self, query: str, k: int = 10) -> Hits:
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

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[Hits]:
        return [self.search(q, k) for q in queries]


Encoder = Callable[[list[str]], Any]  # texts -> (n, dim) array of unit vectors


def sentence_transformer_encoder(model_name: str, batch_size: int = 64) -> Encoder:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]):
        return model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        )

    return encode


class DenseRetriever:
    """Exact cosine search over unit-normalised document embeddings.

    Document embeddings are cached in ``cache_dir`` under a key derived from the
    model name and the exact corpus texts, so a rerun reads them back instead of
    re-encoding, and any change to the corpus invalidates the cache.
    """

    # Query instruction recommended for the BGE v1.5 English models.
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        docs: Sequence[tuple[str, str]],
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Path | None = None,
        encoder: Encoder | None = None,
        query_prefix: str | None = None,
    ):
        import numpy as np

        self.np = np
        self.name = f"dense({model_name})"
        self.ids = [doc_id for doc_id, _ in docs]
        texts = [text for _, text in docs]
        if query_prefix is None:
            query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.query_prefix = query_prefix
        self._encoder = encoder
        self._model_name = model_name

        cache = None
        if cache_dir is not None:
            h = hashlib.sha256(model_name.encode())
            for t in texts:
                h.update(b"\0" + t.encode())
            cache = Path(cache_dir) / f"{model_name.replace('/', '__')}-{h.hexdigest()[:16]}.npy"
        if cache is not None and cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = np.asarray(self.encoder(texts), dtype=np.float32)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = sentence_transformer_encoder(self._model_name)
        return self._encoder

    def search(self, query: str, k: int = 10) -> Hits:
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[Hits]:
        np = self.np
        q = np.asarray(self.encoder([self.query_prefix + x for x in queries]), dtype=np.float32)
        scores = q @ self.emb.T
        out = []
        for row in scores:
            # Stable sort: ties keep corpus order, as in BM25.
            top = np.argsort(-row, kind="stable")[:k]
            out.append([(self.ids[i], float(row[i])) for i in top])
        return out


class HybridRRF:
    """Reciprocal rank fusion: ``score(d) = sum_r 1 / (rrf_k + rank_r(d))``."""

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, depth: int = 100):
        self.retrievers = list(retrievers)
        self.rrf_k, self.depth = rrf_k, depth
        self.name = "rrf(" + ",".join(r.name for r in self.retrievers) + f";k={rrf_k})"

    def search(self, query: str, k: int = 10) -> Hits:
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[Hits]:
        per_retriever = [r.search_many(queries, self.depth) for r in self.retrievers]
        out = []
        for qi in range(len(queries)):
            scores: dict[str, float] = defaultdict(float)
            best_rank: dict[str, int] = {}
            for hits in per_retriever:
                for rank, (doc_id, _) in enumerate(hits[qi], 1):
                    scores[doc_id] += 1.0 / (self.rrf_k + rank)
                    best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
            ranked = sorted(scores, key=lambda d: (-scores[d], best_rank[d], d))[:k]
            out.append([(d, scores[d]) for d in ranked])
        return out
