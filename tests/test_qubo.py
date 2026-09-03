"""Guards for the QUBO formulation.

``test_return_term_survives_binary_encoding`` is the important one: it is a
direct regression test for the defect identified in review, where the expected
return appeared both on the diagonal of Q and in c and therefore cancelled for
binary variables, leaving a return-blind objective.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from quopt.qubo import (build_qubo, default_lambda, objective_direct,
                        qubo_energy, qubo_to_ising)

N, B = 8, 3


@pytest.fixture
def instance():
    rng = np.random.default_rng(7)
    mu = rng.normal(0.10, 0.15, size=N)
    A = rng.normal(size=(N, N))
    Sigma = (A @ A.T) / N * 0.05
    return mu, Sigma


def _feasible_points(n=N, b=B):
    for support in combinations(range(n), b):
        x = np.zeros(n, dtype=float)
        x[list(support)] = 1.0
        yield x


def test_qubo_matches_direct_objective_on_feasible_points(instance):
    mu, Sigma = instance
    for q in (0.01, 1.0, 20.0):
        lam = default_lambda(mu, Sigma, q, B)
        Q, c, const = build_qubo(mu, Sigma, q, B, lam)
        for x in _feasible_points():
            assert qubo_energy(x, Q, c, const) == pytest.approx(
                objective_direct(x, mu, Sigma, q, B, lam), abs=1e-12)


def test_qubo_matches_direct_objective_on_infeasible_points(instance):
    mu, Sigma = instance
    rng = np.random.default_rng(11)
    q, lam = 1.0, None
    lam = default_lambda(mu, Sigma, q, B)
    Q, c, const = build_qubo(mu, Sigma, q, B, lam)
    for _ in range(10_000):
        x = rng.integers(0, 2, size=N).astype(float)
        assert qubo_energy(x, Q, c, const) == pytest.approx(
            objective_direct(x, mu, Sigma, q, B, lam), abs=1e-12)


def test_return_term_survives_binary_encoding(instance):
    """The objective must depend on mu.  This is the reviewed defect.

    If the expected-return vector were written into both the diagonal of Q and
    the linear coefficient with opposite signs, the two contributions would
    cancel for every binary x and the objective would be identical for any mu.
    """
    mu, Sigma = instance
    q = 1.0
    lam = default_lambda(mu, Sigma, q, B)
    Q, c, const = build_qubo(mu, Sigma, q, B, lam)
    Q_zero, c_zero, const_zero = build_qubo(np.zeros_like(mu), Sigma, q, B,
                                            lam)

    differences = [
        abs(qubo_energy(x, Q, c, const)
            - qubo_energy(x, Q_zero, c_zero, const_zero))
        for x in _feasible_points()
    ]
    assert max(differences) > 1e-6, (
        "objective is invariant to the expected-return vector, i.e. the "
        "return term has cancelled out")


def test_return_term_enters_linearly(instance):
    """Doubling mu must change the linear part by exactly a factor of two."""
    mu, Sigma = instance
    q = 1.0
    lam = default_lambda(mu, Sigma, q, B)
    _, c1, _ = build_qubo(mu, Sigma, q, B, lam)
    _, c2, _ = build_qubo(2 * mu, Sigma, q, B, lam)
    penalty_part = -2.0 * lam * B
    assert np.allclose(c2 - penalty_part, 2.0 * (c1 - penalty_part))


def test_penalty_makes_infeasible_points_strictly_worse(instance):
    mu, Sigma = instance
    q = 1.0
    lam = default_lambda(mu, Sigma, q, B)
    Q, c, const = build_qubo(mu, Sigma, q, B, lam)
    best_feasible = min(qubo_energy(x, Q, c, const)
                        for x in _feasible_points())
    rng = np.random.default_rng(3)
    for _ in range(2_000):
        x = rng.integers(0, 2, size=N).astype(float)
        if x.sum() == B:
            continue
        assert qubo_energy(x, Q, c, const) > best_feasible


def test_ising_mapping_preserves_energies(instance):
    mu, Sigma = instance
    q = 1.0
    lam = default_lambda(mu, Sigma, q, B)
    Q, c, const = build_qubo(mu, Sigma, q, B, lam)
    J, h, offset = qubo_to_ising(Q, c, const)
    rng = np.random.default_rng(5)
    for _ in range(500):
        x = rng.integers(0, 2, size=N).astype(float)
        z = 1.0 - 2.0 * x
        assert float(z @ J @ z + h @ z + offset) == pytest.approx(
            qubo_energy(x, Q, c, const), abs=1e-9)
