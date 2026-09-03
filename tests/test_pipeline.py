"""End-to-end guards for the search components.

The saturation test is the one that matters for the review: on a small
instance every strong Tier-2 variant should reach the enumerated optimum, which
is precisely why the published Table 4 showed a three-decimal tie.
"""

from __future__ import annotations

import numpy as np
import pytest

from quopt.analyst import estimate_moments
from quopt.baselines import BASELINE_REGISTRY
from quopt.enumerate_exact import enumerate_optimum
from quopt.generator import generate_prices
from quopt.local_search import refine
from quopt.pipeline import PipelineConfig, run_pipeline
from quopt.qaoa_sim import CostOperator, QAOAConfig
from quopt.risk import EvalCounter, sharpe_ratio

N, B = 10, 3


@pytest.fixture(scope="module")
def moments():
    rng = np.random.default_rng(19)
    prices, _ = generate_prices(N, "bull", rng, horizon=252)
    return estimate_moments(prices)


def _fast_config():
    return PipelineConfig(
        B=B,
        risk_grid=(0.5, 2.0),
        qaoa=QAOAConfig(depth=2, shots=256, maxiter=12, n_restarts=1),
    )


def test_enumeration_is_an_upper_bound(moments):
    mu, Sigma = moments
    optimum = enumerate_optimum(mu, Sigma, B)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        x = np.zeros(N, dtype=int)
        x[rng.choice(N, size=B, replace=False)] = 1
        assert sharpe_ratio(x, mu, Sigma) <= optimum.sharpe + 1e-12


def test_local_search_terminates_and_never_worsens(moments):
    mu, Sigma = moments
    rng = np.random.default_rng(1)
    for _ in range(15):
        x0 = np.zeros(N, dtype=int)
        x0[rng.choice(N, size=B, replace=False)] = 1
        counter = EvalCounter(mu, Sigma)
        start = sharpe_ratio(x0, mu, Sigma)
        result = refine(x0, counter)
        assert result.converged
        assert result.sharpe >= start - 1e-12
        assert int(result.x.sum()) == B


def test_pipeline_respects_the_cardinality_budget(moments):
    mu, Sigma = moments
    result = run_pipeline(mu, Sigma, _fast_config(),
                          np.random.default_rng(2))
    assert int(result.x.sum()) == B
    assert result.best_q in _fast_config().risk_grid


def test_pipeline_reaches_the_optimum_on_a_small_instance(moments):
    mu, Sigma = moments
    optimum = enumerate_optimum(mu, Sigma, B)
    result = run_pipeline(mu, Sigma, _fast_config(),
                          np.random.default_rng(3))
    assert result.sharpe == pytest.approx(optimum.sharpe, abs=1e-9)


@pytest.mark.parametrize("variant", ["qaoa", "random", "sa", "greedy",
                                     "none"])
def test_tier2_variants_all_saturate_the_small_instance(moments, variant):
    """Saturation: the instance is too small to separate strong searchers."""
    mu, Sigma = moments
    optimum = enumerate_optimum(mu, Sigma, B)
    cfg = _fast_config()
    cfg.tier2 = variant
    result = run_pipeline(mu, Sigma, cfg, np.random.default_rng(4))
    assert result.sharpe <= optimum.sharpe + 1e-9
    if variant in ("qaoa", "random", "sa"):
        assert result.sharpe == pytest.approx(optimum.sharpe, abs=1e-9)


def test_baselines_return_feasible_selections(moments):
    mu, Sigma = moments
    for label, fn in BASELINE_REGISTRY.items():
        rng = np.random.default_rng(abs(hash(label)) % 1000)
        settings = {"iterations": 200} if label == "SA" else None
        x, counter = fn(mu, Sigma, B, rng, settings=settings)
        assert int(np.asarray(x).sum()) == B, label
        assert counter.n_evals >= 1, label


def test_cost_operator_energies_match_direct_evaluation(moments):
    mu, Sigma = moments
    from quopt.qubo import default_lambda, objective_direct

    op = CostOperator(mu, Sigma, B)
    q = 1.0
    lam = default_lambda(mu, Sigma, q, B)
    energies = op.energies(q, lam)
    rng = np.random.default_rng(6)
    for index in rng.integers(0, op.size, size=400):
        x = op.bits_of(int(index)).astype(float)
        assert energies[int(index)] == pytest.approx(
            objective_direct(x, mu, Sigma, q, B, lam), abs=1e-10)


def test_cost_operator_refuses_oversized_instances(moments):
    mu, Sigma = moments
    with pytest.raises(MemoryError, match="exceeds the configured ceiling"):
        CostOperator(np.zeros(30), np.eye(30), 8, max_qubits=22)


def test_qaoa_statevector_stays_normalised(moments):
    mu, Sigma = moments
    from quopt.qaoa_sim import _statevector
    from quopt.qubo import default_lambda

    op = CostOperator(mu, Sigma, B)
    energies = op.energies(1.0, default_lambda(mu, Sigma, 1.0, B))
    params = np.array([0.4, 0.7, np.pi / 4, np.pi / 8])
    psi = _statevector(energies, params, op.n, 2)
    assert float(np.sum(np.abs(psi) ** 2)) == pytest.approx(1.0, abs=1e-9)
