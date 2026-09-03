"""The QuOpt-Agents three-tier optimiser, with an interchangeable Tier 2.

Tier 1 sweeps a grid of risk-aversion parameters.  For each value, Tier 2
proposes a seed selection and Tier 3 refines it by local search; the risk
parameter whose refined portfolio has the highest in-sample Sharpe ratio wins.

``tier2`` selects the global-search block:

    "qaoa"    classically simulated QAOA (the proposed configuration)
    "random"  uniform sampling of the feasible set at the same draw budget
    "greedy"  the deterministic Strong Greedy seed
    "sa"      a simulated-annealing seed at the same evaluation budget
    "none"    a single random feasible seed, i.e. Tier 2 removed

Substituting only this block, with Tiers 1 and 3 and the draw budget held
fixed, is what isolates the contribution of the QAOA search.  ``"random"`` is
the decisive comparison, because it matches the QAOA arm's sample count
exactly while carrying no structure at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .baselines import simulated_annealing, strong_greedy
from .local_search import EPSILON_DEFAULT, refine
from .qaoa_sim import CostOperator, QAOAConfig, run_qaoa
from .risk import EvalCounter, sharpe_ratio

__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline",
           "DEFAULT_RISK_GRID", "TIER2_VARIANTS"]

DEFAULT_RISK_GRID = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

TIER2_VARIANTS = ("qaoa", "random", "greedy", "sa", "none")


@dataclass
class PipelineConfig:
    B: int = 5
    risk_grid: tuple[float, ...] = DEFAULT_RISK_GRID
    epsilon: float = EPSILON_DEFAULT
    rf: float = 0.0
    tier2: str = "qaoa"
    qaoa: QAOAConfig = field(default_factory=QAOAConfig)
    sa_budget: int = 1024


@dataclass
class PipelineResult:
    x: np.ndarray
    sharpe: float
    best_q: float
    seed_sharpe: float
    n_accepted_swaps: int
    n_evaluations: int
    runtime_s: float
    feasible_fraction: float
    n_distinct_feasible: int
    repaired_count: int
    trajectory: tuple[np.ndarray, np.ndarray]
    per_q: list[dict] = field(default_factory=list)


def _tier2_seed(variant: str, mu, Sigma, q, cfg: PipelineConfig,
                rng: np.random.Generator, counter: EvalCounter,
                cost_operator: CostOperator | None) -> dict:
    """Produce one seed selection and the diagnostics that describe it."""
    n = mu.size

    if variant == "qaoa":
        result = run_qaoa(mu, Sigma, q, cfg.B, rng, config=cfg.qaoa,
                          cost_operator=cost_operator, rf=cfg.rf)
        return {
            "x": result.seed_x,
            "seed_sharpe": result.seed_sharpe,
            "feasible_fraction": result.feasible_fraction,
            "n_distinct_feasible": result.n_distinct_feasible,
            "repaired": result.repaired,
        }

    if variant == "random":
        best_x, best_value = None, -np.inf
        for _ in range(cfg.qaoa.shots):
            x = np.zeros(n, dtype=int)
            x[rng.choice(n, size=cfg.B, replace=False)] = 1
            value = sharpe_ratio(x, mu, Sigma, cfg.rf)
            if value > best_value:
                best_value, best_x = value, x
        return {"x": best_x, "seed_sharpe": float(best_value),
                "feasible_fraction": 1.0,
                "n_distinct_feasible": cfg.qaoa.shots, "repaired": False}

    if variant == "greedy":
        x, _ = strong_greedy(mu, Sigma, cfg.B, rng, rf=cfg.rf)
        return {"x": x, "seed_sharpe": sharpe_ratio(x, mu, Sigma, cfg.rf),
                "feasible_fraction": 1.0, "n_distinct_feasible": 1,
                "repaired": False}

    if variant == "sa":
        x, sa_counter = simulated_annealing(
            mu, Sigma, cfg.B, rng, rf=cfg.rf,
            settings={"iterations": cfg.sa_budget})
        return {"x": x, "seed_sharpe": float(sa_counter.best_value),
                "feasible_fraction": 1.0,
                "n_distinct_feasible": cfg.sa_budget, "repaired": False}

    if variant == "none":
        x = np.zeros(n, dtype=int)
        x[rng.choice(n, size=cfg.B, replace=False)] = 1
        return {"x": x, "seed_sharpe": sharpe_ratio(x, mu, Sigma, cfg.rf),
                "feasible_fraction": 1.0, "n_distinct_feasible": 1,
                "repaired": False}

    raise ValueError(f"unknown tier2 variant {variant!r}; expected one of "
                     f"{TIER2_VARIANTS}")


def run_pipeline(mu: np.ndarray, Sigma: np.ndarray,
                 cfg: PipelineConfig | None = None,
                 rng: np.random.Generator | None = None,
                 skip_tier3: bool = False) -> PipelineResult:
    """Run the full three-tier pipeline and return the selected portfolio.

    ``skip_tier3`` reproduces the "no local refinement" ablation row without
    duplicating any code.
    """
    cfg = cfg or PipelineConfig()
    rng = rng or np.random.default_rng(42)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    counter = EvalCounter(mu, Sigma, cfg.rf)
    start = time.perf_counter()

    # The diagonal cost operator is independent of q apart from a rescaling,
    # so it is built once and reused across the whole risk grid.
    cost_operator = None
    if cfg.tier2 == "qaoa":
        cost_operator = CostOperator(mu, Sigma, cfg.B,
                                     max_qubits=cfg.qaoa.max_qubits)

    per_q: list[dict] = []
    total_swaps = 0
    repaired_count = 0
    feasible_fractions: list[float] = []
    distinct_counts: list[int] = []
    best = {"sharpe": -np.inf, "x": None, "q": None, "seed_sharpe": -np.inf}

    for q in cfg.risk_grid:
        seed = _tier2_seed(cfg.tier2, mu, Sigma, q, cfg, rng, counter,
                           cost_operator)
        feasible_fractions.append(seed["feasible_fraction"])
        distinct_counts.append(seed["n_distinct_feasible"])
        repaired_count += int(seed["repaired"])

        if skip_tier3:
            x = np.asarray(seed["x"], dtype=int)
            value = counter.sharpe(x)
            swaps = 0
        else:
            refined = refine(seed["x"], counter, epsilon=cfg.epsilon)
            x, value, swaps = refined.x, refined.sharpe, \
                refined.n_accepted_swaps
            total_swaps += swaps

        per_q.append({"q": float(q), "seed_sharpe": seed["seed_sharpe"],
                      "refined_sharpe": float(value), "n_swaps": swaps})

        if value > best["sharpe"]:
            best = {"sharpe": float(value), "x": np.asarray(x, dtype=int),
                    "q": float(q), "seed_sharpe": seed["seed_sharpe"]}

    runtime = time.perf_counter() - start

    return PipelineResult(
        x=best["x"], sharpe=best["sharpe"], best_q=best["q"],
        seed_sharpe=float(best["seed_sharpe"]),
        n_accepted_swaps=total_swaps, n_evaluations=counter.n_evals,
        runtime_s=runtime,
        feasible_fraction=float(np.mean(feasible_fractions)),
        n_distinct_feasible=int(np.mean(distinct_counts)),
        repaired_count=repaired_count,
        trajectory=counter.trajectory(), per_q=per_q)
