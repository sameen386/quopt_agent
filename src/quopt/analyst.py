"""Agent 2a: Market Analyst.

Turns raw prices into the annualised expected-return vector and covariance
matrix used by every downstream method, plus the diagnostic tests reported in
the exploratory data analysis section.
"""

from __future__ import annotations

import numpy as np

from .generator import log_returns

__all__ = ["estimate_moments", "adf_test", "asset_sharpe_ratios",
           "TRADING_DAYS"]

TRADING_DAYS = 252


def estimate_moments(prices: np.ndarray | None = None, *,
                     returns: np.ndarray | None = None,
                     annualisation: int = TRADING_DAYS
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Annualised ``(mu_hat, Sigma_hat)`` from prices or from log-returns.

    Exactly one of ``prices`` or ``returns`` must be supplied.  The covariance
    uses the unbiased (ddof=1) estimator.
    """
    if (prices is None) == (returns is None):
        raise ValueError("supply exactly one of prices= or returns=")
    r = log_returns(prices) if returns is None else np.asarray(returns, float)
    if r.ndim != 2 or r.shape[0] < 2:
        raise ValueError("need at least two return observations")
    mu = r.mean(axis=0) * annualisation
    Sigma = np.cov(r, rowvar=False, ddof=1) * annualisation
    Sigma = np.atleast_2d(Sigma)
    # Symmetrise to remove floating-point asymmetry before Cholesky use.
    Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, Sigma


def adf_test(series: np.ndarray, max_lag: int = 1) -> dict:
    """Augmented Dickey-Fuller test with a constant, by ordinary least squares.

    Implemented directly so that the package does not depend on statsmodels.
    The returned ``t_stat`` is compared against the standard MacKinnon 5%
    critical value for the constant-only specification; ``stationary`` is True
    when the unit-root null is rejected.
    """
    y = np.asarray(series, dtype=float).ravel()
    n = y.size
    if n < max_lag + 10:
        raise ValueError("series too short for the requested lag order")
    dy = np.diff(y)
    y_lag = y[:-1]
    rows = n - 1 - max_lag
    X = [np.ones(rows), y_lag[max_lag:]]
    for lag in range(1, max_lag + 1):
        X.append(dy[max_lag - lag:-lag])
    X = np.column_stack(X)
    target = dy[max_lag:]
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = rows - X.shape[1]
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(s2 * XtX_inv[1, 1]))
    t_stat = float(beta[1] / se) if se > 0 else float("nan")
    critical_5pct = -2.86
    return {
        "t_stat": t_stat,
        "critical_5pct": critical_5pct,
        "stationary": bool(t_stat < critical_5pct),
        "n_obs": rows,
    }


def asset_sharpe_ratios(mu: np.ndarray, Sigma: np.ndarray,
                        rf: float = 0.0) -> np.ndarray:
    """Per-asset Sharpe ratios, the ranking signal used by Strong Greedy."""
    vol = np.sqrt(np.clip(np.diag(Sigma), 1e-24, None))
    return (np.asarray(mu, dtype=float) - rf) / vol
