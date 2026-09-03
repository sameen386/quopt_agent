"""Tier 3: classical local-search refinement.

A swap exchanges one selected asset for one unselected asset and is accepted
only when it raises the Sharpe ratio by more than ``epsilon``.  Because the
feasible set is finite, every accepted swap strictly improves a bounded
objective, and the objective is bounded above, the procedure terminates after
finitely many accepted swaps.  That is Proposition 1 of the manuscript, and it
is also the reason the proposed method's convergence trace is short: the
method stops when no improving swap exists, rather than exhausting a fixed
iteration budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .risk import EvalCounter

__all__ = ["LocalSearchResult", "refine", "EPSILON_DEFAULT"]

EPSILON_DEFAULT = 1e-5


@dataclass
class LocalSearchResult:
    x: np.ndarray
    sharpe: float
    n_accepted_swaps: int
    n_evaluations: int
    converged: bool


def refine(x0: np.ndarray, counter: EvalCounter,
           epsilon: float = EPSILON_DEFAULT,
           max_passes: int = 1000) -> LocalSearchResult:
    """Steepest-ascent pairwise swap refinement starting from ``x0``.

    Each pass evaluates every (selected, unselected) exchange and applies the
    single best improving swap.  ``converged`` is False only if the pass limit
    is reached, which would indicate a numerical problem rather than normal
    operation.
    """
    x = np.asarray(x0, dtype=int).copy()
    current = counter.sharpe(x)
    accepted = 0
    start_evals = counter.n_evals

    for _ in range(max_passes):
        selected = np.flatnonzero(x == 1)
        unselected = np.flatnonzero(x == 0)
        best_gain, best_move = 0.0, None

        for i in selected:
            for j in unselected:
                candidate = x.copy()
                candidate[i], candidate[j] = 0, 1
                value = counter.sharpe(candidate)
                gain = value - current
                if gain > best_gain:
                    best_gain, best_move = gain, (i, j, value)

        if best_move is None or best_gain <= epsilon:
            return LocalSearchResult(
                x=x, sharpe=float(current), n_accepted_swaps=accepted,
                n_evaluations=counter.n_evals - start_evals, converged=True)

        i, j, value = best_move
        x[i], x[j] = 0, 1
        current = value
        accepted += 1

    return LocalSearchResult(x=x, sharpe=float(current),
                             n_accepted_swaps=accepted,
                             n_evaluations=counter.n_evals - start_evals,
                             converged=False)
