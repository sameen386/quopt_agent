"""Agent 1: Synthetic Market Generator, and the real-data loader.

Synthetic prices follow correlated geometric Brownian motion with Gaussian
increments.  The generator therefore reproduces drift, volatility and
correlation structure by construction but *not* volatility clustering or
empirical fat tails; jump-diffusion and regime-switching dynamics are not
implemented anywhere in this package and are stated in the manuscript as
future extensions.

Regime parameters follow the definitions in the manuscript: bear is the
high-correlation regime and bull is the moderate-correlation regime.  Note the
ordering of the correlation parameter ``beta`` below, which corrects a
contradiction in the previous draft.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RegimeSpec", "REGIMES", "generate_prices", "log_returns",
           "REGIME_SCENARIO_COUNTS"]


@dataclass(frozen=True)
class RegimeSpec:
    """Stylised directional settings for one market regime.

    ``mu_range`` and ``sigma_range`` are sampled per asset; ``beta`` is the
    pairwise correlation of the driving Brownian increments.
    """
    name: str
    mu_range: tuple[float, float]
    sigma_range: tuple[float, float]
    beta: float
    beta_drift: float = 0.0   # linear change in beta over the horizon
    mu_drift: float = 0.0     # linear change in drift over the horizon


REGIMES: dict[str, RegimeSpec] = {
    # Positive drift, moderate volatility, moderate correlation.
    "bull": RegimeSpec("bull", (0.10, 0.30), (0.15, 0.25), beta=0.30),
    # Negative drift, high volatility, high correlation (synchronisation).
    "bear": RegimeSpec("bear", (-0.30, -0.10), (0.30, 0.45), beta=0.60),
    # Near-zero drift, moderate volatility, low or unstable correlation.
    "choppy": RegimeSpec("choppy", (-0.05, 0.05), (0.18, 0.28), beta=0.10),
    # Mixed drift, high volatility, time-varying correlation.
    "volatile": RegimeSpec("volatile", (-0.10, 0.10), (0.35, 0.50), beta=0.25,
                           beta_drift=0.20),
    # Drift rising from negative towards positive, correlations normalising.
    "recovery": RegimeSpec("recovery", (-0.10, 0.05), (0.20, 0.32), beta=0.45,
                           beta_drift=-0.25, mu_drift=0.30),
}

# Scenario counts per regime, matching the regime table of the manuscript.
REGIME_SCENARIO_COUNTS: dict[str, int] = {
    "bull": 14,
    "bear": 11,
    "choppy": 6,
    "volatile": 7,
    "recovery": 12,
}


def _correlation_matrix(n: int, beta: float) -> np.ndarray:
    """Equicorrelation matrix, clipped to remain positive definite."""
    beta = float(np.clip(beta, -1.0 / (n - 1) + 1e-6, 0.99))
    C = np.full((n, n), beta, dtype=float)
    np.fill_diagonal(C, 1.0)
    return C


def generate_prices(n_assets: int, regime: str, rng: np.random.Generator,
                    horizon: int = 252, dt: float | None = None,
                    s0: float = 100.0) -> tuple[np.ndarray, dict]:
    """Simulate one correlated GBM price path for the given regime.

    Returns ``(prices, params)`` where ``prices`` has shape
    ``(horizon + 1, n_assets)`` and ``params`` records the sampled drift,
    volatility and correlation settings so that any scenario can be audited.
    """
    if regime not in REGIMES:
        raise KeyError(f"unknown regime {regime!r}; expected one of "
                       f"{sorted(REGIMES)}")
    spec = REGIMES[regime]
    dt = 1.0 / 252.0 if dt is None else dt

    mu = rng.uniform(*spec.mu_range, size=n_assets)
    sigma = rng.uniform(*spec.sigma_range, size=n_assets)

    log_prices = np.empty((horizon + 1, n_assets), dtype=float)
    log_prices[0] = np.log(s0)

    for t in range(horizon):
        frac = t / max(horizon - 1, 1)
        beta_t = spec.beta + spec.beta_drift * frac
        mu_t = mu + spec.mu_drift * frac
        C = _correlation_matrix(n_assets, beta_t)
        L = np.linalg.cholesky(C)
        z = L @ rng.standard_normal(n_assets)
        drift = (mu_t - 0.5 * sigma ** 2) * dt
        log_prices[t + 1] = log_prices[t] + drift + sigma * np.sqrt(dt) * z

    params = {
        "regime": regime,
        "mu_annual_target": mu.tolist(),
        "sigma_annual_target": sigma.tolist(),
        "beta_start": spec.beta,
        "beta_end": spec.beta + spec.beta_drift,
        "mu_drift": spec.mu_drift,
        "horizon": horizon,
        "dt": dt,
    }
    return np.exp(log_prices), params


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log-returns of a price matrix shaped ``(time, assets)``."""
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices), axis=0)
