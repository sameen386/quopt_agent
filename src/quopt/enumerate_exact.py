"""Exact reference solution by exhaustive enumeration.

At the problem size studied in the manuscript the feasible set contains only
C(16, 5) = 4368 equal-weighted portfolios, so the optimum can be computed
outright.  Reporting it is what turns the three-way tie between QuOpt-Agents,
GA, DE and SA from an unexplained coincidence into a measured saturation
effect: several strong searchers reach the same portfolio because it is the
best one available.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from .risk import sharpe_ratio

__all__ = ["EnumerationResult", "enumerate_optimum", "is_tractable",
           "TRACTABLE_LIMIT"]

TRACTABLE_LIMIT = 5_000_000


def is_tractable(n: int, b: int, limit: int = TRACTABLE_LIMIT) -> bool:
    return comb(n, b) <= limit


@dataclass
class EnumerationResult:
    x: np.ndarray
    sharpe: float
    n_candidates: int
    runtime_s: float


def enumerate_optimum(mu: np.ndarray, Sigma: np.ndarray, B: int,
                      rf: float = 0.0,
                      limit: int = TRACTABLE_LIMIT) -> EnumerationResult:
    """Best equal-weighted portfolio of exactly ``B`` assets, by brute force."""
    import time

    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    total = comb(n, B)
    if total > limit:
        raise MemoryError(
            f"C({n},{B}) = {total} exceeds the enumeration limit of {limit}; "
            f"report this instance as not enumerable rather than reporting a "
            f"heuristic value as the optimum")

    start = time.perf_counter()
    best_x, best_sharpe = None, -np.inf
    for support in combinations(range(n), B):
        x = np.zeros(n, dtype=int)
        x[list(support)] = 1
        value = sharpe_ratio(x, mu, Sigma, rf)
        if value > best_sharpe:
            best_sharpe, best_x = value, x
    runtime = time.perf_counter() - start

    return EnumerationResult(x=best_x, sharpe=float(best_sharpe),
                             n_candidates=total, runtime_s=runtime)
