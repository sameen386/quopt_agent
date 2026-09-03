"""Agent 2b: Risk Assessor.

Scores a candidate asset selection under equal weighting: expected portfolio
return, portfolio volatility and the Sharpe ratio that is the optimisation
objective throughout the study.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "portfolio_return",
    "portfolio_volatility",
    "sharpe_ratio",
    "evaluate_selection",
    "EvalCounter",
]


def _weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    b = x.sum()
    if b <= 0:
        raise ValueError("empty selection has no equal-weight portfolio")
    return x / b


def portfolio_return(x: np.ndarray, mu: np.ndarray) -> float:
    return float(_weights(x) @ np.asarray(mu, dtype=float))


def portfolio_volatility(x: np.ndarray, Sigma: np.ndarray) -> float:
    w = _weights(x)
    var = float(w @ np.asarray(Sigma, dtype=float) @ w)
    return float(np.sqrt(max(var, 0.0)))


def sharpe_ratio(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                 rf: float = 0.0) -> float:
    """Sharpe ratio of the equal-weighted portfolio on the given estimates.

    Returns ``-inf`` for a degenerate zero-volatility portfolio so that such a
    point can never win a maximisation.
    """
    vol = portfolio_volatility(x, Sigma)
    if vol <= 1e-12:
        return float("-inf")
    return (portfolio_return(x, mu) - rf) / vol


def evaluate_selection(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                       rf: float = 0.0) -> dict:
    """Full risk report for one selection.

    ``ret`` and ``vol`` mirror ``expected_return`` and ``volatility`` under the
    column names used by the results schema, so that ``_row(**...)`` records
    the annualised return and volatility of every selection (needed by the
    risk-return figure).
    """
    r = portfolio_return(x, mu)
    v = portfolio_volatility(x, Sigma)
    return {
        "expected_return": r,
        "volatility": v,
        "ret": r,
        "vol": v,
        "sharpe": sharpe_ratio(x, mu, Sigma, rf),
        "cardinality": int(np.asarray(x).sum()),
    }


class EvalCounter:
    """Counts objective evaluations and records the best-so-far trajectory.

    Every search method in this package scores candidates through an instance
    of this class, so the convergence figure can plot all methods against a
    common abscissa (cumulative objective evaluations) rather than against
    method-specific iteration indices.  That mismatch is what made the
    previously published convergence figure unreadable.
    """

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.rf = float(rf)
        self.n_evals = 0
        self.best_value = -np.inf
        self.best_x: np.ndarray | None = None
        self.history: list[tuple[int, float]] = []

    def sharpe(self, x: np.ndarray) -> float:
        self.n_evals += 1
        value = sharpe_ratio(x, self.mu, self.Sigma, self.rf)
        if value > self.best_value:
            self.best_value = value
            self.best_x = np.asarray(x, dtype=int).copy()
            self.history.append((self.n_evals, value))
        return value

    def trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Best-so-far trajectory as ``(eval_counts, best_values)``."""
        if not self.history:
            return np.array([0]), np.array([-np.inf])
        evals, values = zip(*self.history)
        return np.asarray(evals), np.asarray(values)
