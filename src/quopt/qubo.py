"""QUBO construction for cardinality-constrained equal-weighted selection.

The objective minimised for a risk-aversion parameter q and penalty lambda is

    f_q(x) = (q / B^2) x' Sigma x  -  (1 / B) mu' x  +  lambda (1'x - B)^2

over x in {0,1}^N.  Written in the standard QUBO form
f_q(x) = x' Q x + c' x + const, this corresponds to

    Q_ij = (q / B^2) Sigma_ij + lambda      for all i, j (including i == j)
    c_i  = -mu_i / B - 2 lambda B
    const = lambda B^2

The expected-return vector appears once, and only in ``c``.  This matters:
because x_i^2 = x_i for binary variables, a term placed on the diagonal of Q
is algebraically indistinguishable from a linear term, so writing -mu_i into
Q_ii while +mu_i sits in c_i cancels the return contribution exactly and
leaves a return-blind objective.  ``tests/test_qubo.py`` guards against that
failure mode directly.

The identity lambda (1'x - B)^2 = lambda sum_ij x_i x_j - 2 lambda B sum_i x_i
+ lambda B^2 holds without invoking x_i^2 = x_i, which is why lambda is added
to every entry of Q rather than being split between Q and c.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_qubo",
    "qubo_energy",
    "objective_direct",
    "default_lambda",
    "qubo_to_ising",
]


def default_lambda(mu: np.ndarray, Sigma: np.ndarray, q: float, B: int,
                   kappa: float = 10.0) -> float:
    """Penalty coefficient large enough to make every infeasible point worse.

    On the feasible set the unpenalised objective is bounded by

        |(q/B^2) x'Sigma x - (1/B) mu'x|  <=  q max|Sigma| + max|mu|,

    since a feasible x selects exactly B assets.  Scaling that bound by
    ``kappa`` > 1 guarantees that violating the cardinality constraint by even
    one asset costs more than any attainable gain, because the penalty
    increment for a unit violation is exactly lambda.
    """
    bound = q * float(np.abs(Sigma).max()) + float(np.abs(mu).max())
    return float(kappa * max(bound, 1e-12))


def build_qubo(mu: np.ndarray, Sigma: np.ndarray, q: float, B: int,
               lam: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(Q, c, const)`` for the objective documented above."""
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    Q = (q / B ** 2) * Sigma + lam
    c = -mu / B - 2.0 * lam * B
    const = lam * B ** 2
    return Q, c, const


def qubo_energy(x: np.ndarray, Q: np.ndarray, c: np.ndarray,
                const: float) -> float:
    """Evaluate ``x' Q x + c' x + const``."""
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + c @ x + const)


def objective_direct(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                     q: float, B: int, lam: float) -> float:
    """Evaluate the objective directly, bypassing the QUBO coefficients.

    Used by the unit tests as an independent reference for ``qubo_energy``.
    """
    x = np.asarray(x, dtype=float)
    s = x.sum()
    risk = (q / B ** 2) * float(x @ Sigma @ x)
    ret = float(mu @ x) / B
    return risk - ret + lam * (s - B) ** 2


def qubo_to_ising(Q: np.ndarray, c: np.ndarray,
                  const: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Map the QUBO to an Ising Hamiltonian via x_i = (1 - z_i) / 2.

    Returns ``(J, h, offset)`` such that the energy of a spin configuration z
    equals ``z' J z + h' z + offset``.  Provided for readers who wish to
    reproduce the cost Hamiltonian of the QAOA circuit in spin variables; the
    simulator in :mod:`quopt.qaoa_sim` works directly with the diagonal
    binary energies and does not need this transformation.
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    J = 0.25 * Q
    h = -0.25 * (Q.sum(axis=0) + Q.sum(axis=1)) - 0.5 * c
    offset = const + 0.25 * Q.sum() + 0.5 * c.sum()
    return J, h, float(offset)
