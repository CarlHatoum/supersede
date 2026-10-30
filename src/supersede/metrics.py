"""Staleness metrics for a ranked list, with the current and stale sets known."""

from __future__ import annotations

import random
from collections.abc import Callable, Collection, Sequence


def stale_at_1(ranking: Sequence[str], current: Collection[str], stale: Collection[str]) -> float:
    """1 if the top hit is a superseded document of the topic."""
    return float(bool(ranking) and ranking[0] in stale)


def current_at_k(ranking: Sequence[str], current: Collection[str], k: int) -> float:
    return float(any(d in current for d in ranking[:k]))


def reciprocal_rank(ranking: Sequence[str], current: Collection[str]) -> float:
    for i, d in enumerate(ranking, 1):
        if d in current:
            return 1.0 / i
    return 0.0


def stale_before_current(
    ranking: Sequence[str], current: Collection[str], stale: Collection[str]
) -> float:
    """1 if a stale version is ranked above every current version.

    A stale hit with no current version anywhere in the list also counts.
    """
    for d in ranking:
        if d in current:
            return 0.0
        if d in stale:
            return 1.0
    return 0.0


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = lambda v: sum(v) / len(v),
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    lo = stats[int(alpha / 2 * n_resamples)]
    hi = stats[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)
