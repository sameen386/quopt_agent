"""Tier 2: classically simulated QAOA.

This is an exact state-vector simulation written directly in NumPy.  No
external quantum software development kit is used and no physical quantum
device is involved, so the manuscript must describe the simulator rather than
cite a vendor stack.  The full specification is:

* Encoding      one qubit per candidate asset; ``x_i = 1`` selects asset i.
* Cost operator diagonal in the computational basis, with entries given by the
                QUBO of :mod:`quopt.qubo`.  Because the operator is diagonal,
                the cost layer is an elementwise phase multiplication and no
                gate decomposition is required.
* Mixer         transverse field, ``exp(-i beta sum_j X_j)``, applied qubit by
                qubit as ``cos(beta) psi - i sin(beta) flip_j(psi)``.
* Initial state uniform superposition ``|+>^{tensor N}``.
* Depth         p layers, hence 2p variational parameters (gamma_1..gamma_p,
                beta_1..beta_p).
* Outer loop    COBYLA via ``scipy.optimize.minimize``, minimising the
                expectation of the cost operator.  With
                ``expectation_mode="shots"`` the expectation is estimated from
                the same shot budget used for the final draw, which matches
                the manuscript; ``"exact"`` uses the analytic expectation and
                is available for a noise-free reference.
* Readout       ``shots`` samples drawn from the final probability
                distribution.
* Selection     the four-step rule in :func:`select_portfolio` - feasibility
                filter, repair if nothing feasible was observed, Sharpe
                scoring of each distinct feasible bitstring, and seeding of
                Tier 3 with the best-scoring one.  The seed is the best
                observed feasible sample, not the modal bitstring and not the
                expectation value.

Memory grows as ``2**n_qubits`` complex amplitudes, so the simulator refuses
instances above ``max_qubits``.  That ceiling is a real limitation of
simulation-based study designs and is reported as such rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .qubo import build_qubo, default_lambda
from .risk import sharpe_ratio

__all__ = ["QAOAConfig", "QAOAResult", "CostOperator", "run_qaoa",
           "select_portfolio", "MAX_QUBITS_DEFAULT"]

MAX_QUBITS_DEFAULT = 22


@dataclass
class QAOAConfig:
    depth: int = 2
    shots: int = 1024
    max_qubits: int = MAX_QUBITS_DEFAULT
    expectation_mode: str = "shots"       # "shots" or "exact"
    optimizer: str = "COBYLA"
    maxiter: int = 60
    rhobeg: float = 0.30
    tol: float = 1e-4
    n_restarts: int = 2
    lambda_kappa: float = 10.0


@dataclass
class QAOAResult:
    seed_x: np.ndarray
    seed_sharpe: float
    params: np.ndarray
    expectation: float
    n_circuit_evals: int
    feasible_fraction: float
    n_distinct_feasible: int
    repaired: bool
    diagnostics: dict = field(default_factory=dict)


class CostOperator:
    """Diagonal cost operator over all ``2**N`` binary assignments.

    The heavy, q-independent part of the energy (the quadratic form and the
    linear return term) is computed once per instance; changing the risk
    parameter then only rescales precomputed vectors, which is what makes an
    eight-point risk grid affordable.
    """

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, B: int,
                 max_qubits: int = MAX_QUBITS_DEFAULT,
                 chunk: int = 1 << 16):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.n = self.mu.size
        self.B = int(B)
        if self.n > max_qubits:
            raise MemoryError(
                f"state-vector simulation of {self.n} qubits exceeds the "
                f"configured ceiling of {max_qubits}; report this instance as "
                f"not simulable rather than silently substituting a different "
                f"algorithm")
        size = 1 << self.n
        self.size = size
        quad = np.empty(size, dtype=float)
        lin = np.empty(size, dtype=float)
        card = np.empty(size, dtype=np.int16)
        powers = (1 << np.arange(self.n, dtype=np.int64))
        for start in range(0, size, chunk):
            stop = min(start + chunk, size)
            idx = np.arange(start, stop, dtype=np.int64)
            bits = ((idx[:, None] & powers) > 0).astype(float)
            quad[start:stop] = ((bits @ self.Sigma) * bits).sum(axis=1)
            lin[start:stop] = bits @ self.mu
            card[start:stop] = bits.sum(axis=1).astype(np.int16)
        self.quad = quad
        self.lin = lin
        self.cardinality = card
        self.feasible_mask = card == self.B

    def energies(self, q: float, lam: float) -> np.ndarray:
        """Diagonal energies for risk parameter ``q`` and penalty ``lam``."""
        dev = self.cardinality.astype(float) - self.B
        return ((q / self.B ** 2) * self.quad
                - self.lin / self.B
                + lam * dev ** 2)

    def bits_of(self, index: int) -> np.ndarray:
        powers = (1 << np.arange(self.n, dtype=np.int64))
        return ((int(index) & powers) > 0).astype(int)


def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply ``exp(-i beta sum_j X_j)`` in place-efficient qubit sweeps."""
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    for qubit in range(n):
        block = 1 << qubit
        view = psi.reshape(-1, 2, block)
        low = view[:, 0, :].copy()
        high = view[:, 1, :].copy()
        view[:, 0, :] = cos_b * low - 1j * sin_b * high
        view[:, 1, :] = cos_b * high - 1j * sin_b * low
        psi = view.reshape(-1)
    return psi


def _statevector(energies: np.ndarray, params: np.ndarray, n: int,
                 depth: int) -> np.ndarray:
    size = energies.size
    psi = np.full(size, 1.0 / np.sqrt(size), dtype=complex)
    gammas = params[:depth]
    betas = params[depth:]
    scale = np.max(np.abs(energies))
    scale = scale if scale > 0 else 1.0
    for layer in range(depth):
        psi *= np.exp(-1j * gammas[layer] * energies / scale)
        psi = _apply_mixer(psi, betas[layer], n)
    return psi


def run_qaoa(mu: np.ndarray, Sigma: np.ndarray, q: float, B: int,
             rng: np.random.Generator, config: QAOAConfig | None = None,
             cost_operator: CostOperator | None = None,
             lam: float | None = None, rf: float = 0.0) -> QAOAResult:
    """Optimise the QAOA parameters, sample, and return the Tier-3 seed."""
    config = config or QAOAConfig()
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    op = cost_operator or CostOperator(mu, Sigma, B,
                                       max_qubits=config.max_qubits)
    if lam is None:
        lam = default_lambda(mu, Sigma, q, B, kappa=config.lambda_kappa)
    energies = op.energies(q, lam)

    n_circuit_evals = 0

    def expectation(params: np.ndarray) -> float:
        nonlocal n_circuit_evals
        n_circuit_evals += 1
        psi = _statevector(energies, np.asarray(params, float), n,
                           config.depth)
        probs = np.abs(psi) ** 2
        probs /= probs.sum()
        if config.expectation_mode == "exact":
            return float(probs @ energies)
        counts = rng.multinomial(config.shots, probs)
        observed = counts > 0
        return float((counts[observed] * energies[observed]).sum()
                     / config.shots)

    best_params, best_value = None, np.inf
    for restart in range(max(1, config.n_restarts)):
        if restart == 0:
            x0 = np.concatenate([
                np.linspace(0.2, 0.8, config.depth),          # gammas
                np.linspace(np.pi / 4, np.pi / 8, config.depth),  # betas
            ])
        else:
            x0 = np.concatenate([
                rng.uniform(0.0, np.pi, size=config.depth),
                rng.uniform(0.0, np.pi / 2, size=config.depth),
            ])
        res = minimize(expectation, x0, method=config.optimizer,
                       options={"maxiter": config.maxiter,
                                "rhobeg": config.rhobeg,
                                "tol": config.tol})
        if res.fun < best_value:
            best_value, best_params = float(res.fun), np.asarray(res.x, float)

    psi = _statevector(energies, best_params, n, config.depth)
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    counts = rng.multinomial(config.shots, probs)

    selection = select_portfolio(counts, op, mu, Sigma, B, rf=rf)
    selection.params = best_params
    selection.expectation = best_value
    selection.n_circuit_evals = n_circuit_evals
    selection.diagnostics.update({
        "lambda": float(lam),
        "q": float(q),
        "depth": config.depth,
        "shots": config.shots,
        "expectation_mode": config.expectation_mode,
        "optimizer": config.optimizer,
        "n_restarts": config.n_restarts,
    })
    return selection


def select_portfolio(counts: np.ndarray, op: CostOperator, mu: np.ndarray,
                     Sigma: np.ndarray, B: int,
                     rf: float = 0.0) -> QAOAResult:
    """The four-step rule that turns measurement outcomes into one seed.

    1. Feasibility filter: keep only observed bitstrings with exactly B ones.
    2. Repair: if none was observed, take the B assets with the highest
       marginal selection frequency across all shots.
    3. Scoring: score every distinct feasible bitstring by Sharpe ratio.
    4. Seeding: return the best-scoring feasible bitstring.
    """
    shots = int(counts.sum())
    observed = np.flatnonzero(counts)
    feasible = observed[op.feasible_mask[observed]]
    feasible_fraction = (float(counts[feasible].sum()) / shots
                         if shots else 0.0)
    repaired = False

    if feasible.size == 0:
        repaired = True
        powers = (1 << np.arange(op.n, dtype=np.int64))
        marginal = np.zeros(op.n, dtype=float)
        for index in observed:
            bits = ((int(index) & powers) > 0).astype(float)
            marginal += bits * counts[index]
        chosen = np.argsort(-marginal)[:B]
        x = np.zeros(op.n, dtype=int)
        x[chosen] = 1
        return QAOAResult(
            seed_x=x,
            seed_sharpe=sharpe_ratio(x, mu, Sigma, rf),
            params=np.array([]),
            expectation=float("nan"),
            n_circuit_evals=0,
            feasible_fraction=feasible_fraction,
            n_distinct_feasible=0,
            repaired=repaired,
        )

    best_x, best_sharpe = None, -np.inf
    for index in feasible:
        x = op.bits_of(index)
        value = sharpe_ratio(x, mu, Sigma, rf)
        if value > best_sharpe:
            best_sharpe, best_x = value, x

    return QAOAResult(
        seed_x=best_x,
        seed_sharpe=float(best_sharpe),
        params=np.array([]),
        expectation=float("nan"),
        n_circuit_evals=0,
        feasible_fraction=feasible_fraction,
        n_distinct_feasible=int(feasible.size),
        repaired=repaired,
    )
