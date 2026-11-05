"""Pair scorers and edge selection, from cheap rules to a trained model."""

from __future__ import annotations

from collections.abc import Sequence

from supersede.librarian.features import FEATURES


def heuristic_score(f: dict[str, float]) -> float:
    """Hand-written rule: an explicit cue, or a near-identical title on a very close document."""
    if f["explicit_cue"]:
        return 1.0
    return float(f["title_cos"] >= 0.9 and f["text_cos"] >= 0.85 and not f["same_year"])


class LogisticRegression:
    """L2-regularised logistic regression on standardised features (numpy, full batch)."""

    def __init__(self, l2: float = 1e-2, lr: float = 0.5, steps: int = 2000, pos_weight=None):
        self.l2, self.lr, self.steps, self.pos_weight = l2, lr, steps, pos_weight

    def fit(self, rows: Sequence[dict[str, float]], y: Sequence[int]) -> LogisticRegression:
        import numpy as np

        X = np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        X = (X - self.mu) / self.sd
        pw = self.pos_weight or (len(y_arr) - y_arr.sum()) / max(y_arr.sum(), 1)
        w_sample = np.where(y_arr == 1, pw, 1.0)
        w_sample /= w_sample.sum()
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-(X @ self.w + self.b)))
            g = (p - y_arr) * w_sample
            self.w -= self.lr * (X.T @ g + self.l2 * self.w)
            self.b -= self.lr * g.sum()
        return self

    def predict_proba(self, rows: Sequence[dict[str, float]]) -> list[float]:
        import numpy as np

        X = np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float64)
        X = (X - self.mu) / self.sd
        return list(1 / (1 + np.exp(-(X @ self.w + self.b))))

    def weights(self) -> dict[str, float]:
        return {k: round(float(w), 3) for k, w in zip(FEATURES, self.w, strict=True)}


def select_edges(
    pairs: Sequence[tuple[str, str]], scores: Sequence[float], threshold: float
) -> list[tuple[str, str]]:
    """Keep, for each old document, its best-scoring successor above ``threshold``.

    One successor per document is a simplification. RFCs that are split into
    several replacements are a small minority, and the agentic librarian will
    handle them.
    """
    best: dict[str, tuple[float, str]] = {}
    for (old, new), s in zip(pairs, scores, strict=True):
        if s >= threshold and (old not in best or s > best[old][0]):
            best[old] = (s, new)
    return sorted((old, new) for old, (_, new) in best.items())


def edge_prf(predicted: set, gold: set) -> dict[str, float]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp}


def version_chains(
    pairs: Sequence[tuple[str, str]], sims: Sequence[float], tau: float, order: dict
) -> list[tuple[str, str]]:
    """Version-family baseline (in the style of Modani et al., 2021).

    Pairs at least ``tau`` similar are linked; each connected component is a
    version family, ordered by ``order`` (publication order), and consecutive
    members become edges old -> new.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), s in zip(pairs, sims, strict=True):
        if s >= tau:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    families: dict[str, list[str]] = {}
    for x in parent:
        families.setdefault(find(x), []).append(x)
    edges = []
    for members in families.values():
        members.sort(key=lambda d: order[d])
        edges.extend(zip(members, members[1:], strict=False))
    return sorted(edges)
