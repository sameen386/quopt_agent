"""Experiment drivers.  Every experiment appends rows to one results table.

All reported numbers in the manuscript come from ``results/results.csv``,
which this module writes.  Tables and figures are generated from that file and
never typed by hand, which is what prevents the class of inconsistency the
editor identified.

Row schema (long format, one row per experiment/instance/method):

    experiment      real | synthetic | tier2_ablation | tier_ablation | scaleup
    instance        window index, scenario index, or instance label
    regime          synthetic regime name, empty for real data
    method          method label
    n, B            problem size
    sharpe_is       in-sample Sharpe ratio of the selected portfolio
    sharpe_oos      out-of-sample Sharpe ratio (real data only)
    ret, vol        annualised expected return and volatility, in sample
    seed_sharpe     Sharpe ratio of the Tier-2 seed, before refinement
    n_swaps         accepted Tier-3 swaps
    n_evals         objective evaluations
    runtime_s       wall-clock seconds
    is_optimal      whether the selection equals the enumerated optimum
    selection       comma-separated indices of the selected assets
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from .analyst import estimate_moments
from .baselines import BASELINE_REGISTRY
from .data import load_returns, walk_forward_windows
from .enumerate_exact import enumerate_optimum, is_tractable
from .generator import REGIME_SCENARIO_COUNTS, generate_prices
from .pipeline import PipelineConfig, run_pipeline
from .qaoa_sim import QAOAConfig
from .risk import evaluate_selection, sharpe_ratio

__all__ = ["ExperimentSettings", "run_real", "run_synthetic",
           "run_tier2_ablation", "run_tier_ablation", "run_scaleup",
           "run_all", "RESULT_COLUMNS"]

RESULT_COLUMNS = ["experiment", "instance", "regime", "method", "n", "B",
                  "sharpe_is", "sharpe_oos", "ret", "vol", "seed_sharpe",
                  "n_swaps", "n_evals", "runtime_s", "is_optimal",
                  "selection"]

PROPOSED = "QuOpt-Agents"
ENUMERATION = "Exhaustive enumeration"


@dataclass
class ExperimentSettings:
    seed: int = 42
    B: int = 5
    rf: float = 0.0
    n_windows: int = 20
    lookback: int = 126
    horizon: int = 63
    n_scenarios: int = 50
    synthetic_horizon: int = 252
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    offline: bool = False
    qaoa: QAOAConfig = field(default_factory=QAOAConfig)
    scaleup_instances: tuple[tuple[int, int], ...] = ((16, 5), (20, 6),
                                                      (30, 8), (40, 10))
    trajectories: dict = field(default_factory=dict)


def _row(**kwargs) -> dict:
    row = {key: kwargs.get(key, "") for key in RESULT_COLUMNS}
    return row


def _selection_str(x: np.ndarray) -> str:
    return ",".join(str(int(i)) for i in np.flatnonzero(np.asarray(x) == 1))


def _same_support(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.flatnonzero(np.asarray(a) == 1),
                               np.flatnonzero(np.asarray(b) == 1)))


# --------------------------------------------------------------- real data
def run_real(settings: ExperimentSettings) -> list[dict]:
    """Walk-forward evaluation on real equities, including the exact optimum."""
    returns, tickers, _ = load_returns(settings.data_dir,
                                       offline=settings.offline)
    n_assets = returns.shape[1]
    windows = walk_forward_windows(returns.shape[0], settings.n_windows,
                                   settings.lookback, settings.horizon)
    rows: list[dict] = []
    print(f"  real data: {n_assets} assets, {returns.shape[0]} observations, "
          f"{len(windows)} windows")

    for window in windows:
        train = returns[window.train_start:window.train_stop]
        test = returns[window.test_start:window.test_stop]
        mu, Sigma = estimate_moments(returns=train)
        mu_oos, Sigma_oos = estimate_moments(returns=test)

        optimum = enumerate_optimum(mu, Sigma, settings.B, settings.rf)
        rows.append(_row(
            experiment="real", instance=window.index, method=ENUMERATION,
            n=n_assets, B=settings.B, sharpe_is=optimum.sharpe,
            sharpe_oos=sharpe_ratio(optimum.x, mu_oos, Sigma_oos,
                                    settings.rf),
            runtime_s=optimum.runtime_s, is_optimal=True,
            selection=_selection_str(optimum.x),
            **evaluate_selection(optimum.x, mu, Sigma, settings.rf)))

        for label, fn in BASELINE_REGISTRY.items():
            rng = np.random.default_rng(
                [settings.seed, window.index, abs(hash(label)) % 10_000])
            start = time.perf_counter()
            x, counter = fn(mu, Sigma, settings.B, rng, rf=settings.rf)
            runtime = time.perf_counter() - start
            settings.trajectories.setdefault(
                ("real", window.index, label), counter.trajectory())
            rows.append(_row(
                experiment="real", instance=window.index, method=label,
                n=n_assets, B=settings.B,
                sharpe_is=sharpe_ratio(x, mu, Sigma, settings.rf),
                sharpe_oos=sharpe_ratio(x, mu_oos, Sigma_oos, settings.rf),
                n_evals=counter.n_evals, runtime_s=runtime,
                is_optimal=_same_support(x, optimum.x),
                selection=_selection_str(x),
                **evaluate_selection(x, mu, Sigma, settings.rf)))

        rng = np.random.default_rng([settings.seed, window.index, 999])
        cfg = PipelineConfig(B=settings.B, rf=settings.rf, tier2="qaoa",
                             qaoa=settings.qaoa)
        result = run_pipeline(mu, Sigma, cfg, rng)
        settings.trajectories[("real", window.index, PROPOSED)] = \
            result.trajectory
        rows.append(_row(
            experiment="real", instance=window.index, method=PROPOSED,
            n=n_assets, B=settings.B, sharpe_is=result.sharpe,
            sharpe_oos=sharpe_ratio(result.x, mu_oos, Sigma_oos, settings.rf),
            seed_sharpe=result.seed_sharpe, n_swaps=result.n_accepted_swaps,
            n_evals=result.n_evaluations, runtime_s=result.runtime_s,
            is_optimal=_same_support(result.x, optimum.x),
            selection=_selection_str(result.x),
            **evaluate_selection(result.x, mu, Sigma, settings.rf)))
        print(f"    window {window.index + 1:>2}/{len(windows)} done")

    return rows


# ---------------------------------------------------------------- synthetic
def _scenario_plan(n_scenarios: int) -> list[str]:
    """Regime label for each scenario, matching the manuscript's counts."""
    plan: list[str] = []
    for regime, count in REGIME_SCENARIO_COUNTS.items():
        plan.extend([regime] * count)
    if len(plan) != n_scenarios:
        # Fall back to a balanced plan if a different scenario count is asked.
        regimes = list(REGIME_SCENARIO_COUNTS)
        plan = [regimes[i % len(regimes)] for i in range(n_scenarios)]
    return plan


def run_synthetic(settings: ExperimentSettings) -> list[dict]:
    """Controlled multi-regime stress test with all methods."""
    plan = _scenario_plan(settings.n_scenarios)
    n_assets = 16
    rows: list[dict] = []
    print(f"  synthetic: {len(plan)} scenarios across "
          f"{len(set(plan))} regimes")

    for index, regime in enumerate(plan):
        rng = np.random.default_rng([settings.seed, index])
        prices, _ = generate_prices(n_assets, regime, rng,
                                    horizon=settings.synthetic_horizon)
        mu, Sigma = estimate_moments(prices)

        optimum = enumerate_optimum(mu, Sigma, settings.B, settings.rf)
        rows.append(_row(
            experiment="synthetic", instance=index, regime=regime,
            method=ENUMERATION, n=n_assets, B=settings.B,
            sharpe_is=optimum.sharpe, runtime_s=optimum.runtime_s,
            is_optimal=True, selection=_selection_str(optimum.x),
            **evaluate_selection(optimum.x, mu, Sigma, settings.rf)))

        for label, fn in BASELINE_REGISTRY.items():
            method_rng = np.random.default_rng(
                [settings.seed, index, abs(hash(label)) % 10_000])
            start = time.perf_counter()
            x, counter = fn(mu, Sigma, settings.B, method_rng, rf=settings.rf)
            runtime = time.perf_counter() - start
            rows.append(_row(
                experiment="synthetic", instance=index, regime=regime,
                method=label, n=n_assets, B=settings.B,
                sharpe_is=sharpe_ratio(x, mu, Sigma, settings.rf),
                n_evals=counter.n_evals, runtime_s=runtime,
                is_optimal=_same_support(x, optimum.x),
                selection=_selection_str(x),
                **evaluate_selection(x, mu, Sigma, settings.rf)))

        pipeline_rng = np.random.default_rng([settings.seed, index, 999])
        cfg = PipelineConfig(B=settings.B, rf=settings.rf, tier2="qaoa",
                             qaoa=settings.qaoa)
        result = run_pipeline(mu, Sigma, cfg, pipeline_rng)
        rows.append(_row(
            experiment="synthetic", instance=index, regime=regime,
            method=PROPOSED, n=n_assets, B=settings.B,
            sharpe_is=result.sharpe, seed_sharpe=result.seed_sharpe,
            n_swaps=result.n_accepted_swaps, n_evals=result.n_evaluations,
            runtime_s=result.runtime_s,
            is_optimal=_same_support(result.x, optimum.x),
            selection=_selection_str(result.x),
            **evaluate_selection(result.x, mu, Sigma, settings.rf)))
        if (index + 1) % 10 == 0:
            print(f"    scenario {index + 1:>2}/{len(plan)} done")

    return rows


# ----------------------------------------------------------- Tier-2 ablation
TIER2_LABELS = {
    "qaoa": "Tier 2 = simulated QAOA (proposed)",
    "random": "Tier 2 = uniform random sampling",
    "greedy": "Tier 2 = Strong Greedy seed",
    "sa": "Tier 2 = simulated-annealing seed",
    "none": "Tier 2 removed (random feasible seed)",
}


def run_tier2_ablation(settings: ExperimentSettings) -> list[dict]:
    """Substitute only the Tier-2 block, holding Tiers 1 and 3 and the budget.

    This is the experiment requested in review: it is the only comparison that
    isolates the contribution of the QAOA search, because everything else in
    the pipeline is identical across arms.
    """
    returns, _, _ = load_returns(settings.data_dir, offline=settings.offline)
    n_assets = returns.shape[1]
    windows = walk_forward_windows(returns.shape[0], settings.n_windows,
                                   settings.lookback, settings.horizon)
    rows: list[dict] = []
    print(f"  tier-2 ablation: {len(TIER2_LABELS)} arms x {len(windows)} "
          f"windows")

    for window in windows:
        train = returns[window.train_start:window.train_stop]
        test = returns[window.test_start:window.test_stop]
        mu, Sigma = estimate_moments(returns=train)
        mu_oos, Sigma_oos = estimate_moments(returns=test)
        optimum = enumerate_optimum(mu, Sigma, settings.B, settings.rf)

        for variant, label in TIER2_LABELS.items():
            rng = np.random.default_rng(
                [settings.seed, window.index, abs(hash(variant)) % 10_000])
            cfg = PipelineConfig(B=settings.B, rf=settings.rf, tier2=variant,
                                 qaoa=settings.qaoa)
            result = run_pipeline(mu, Sigma, cfg, rng)
            rows.append(_row(
                experiment="tier2_ablation", instance=window.index,
                method=label, n=n_assets, B=settings.B,
                sharpe_is=result.sharpe,
                sharpe_oos=sharpe_ratio(result.x, mu_oos, Sigma_oos,
                                        settings.rf),
                seed_sharpe=result.seed_sharpe,
                n_swaps=result.n_accepted_swaps,
                n_evals=result.n_evaluations, runtime_s=result.runtime_s,
                is_optimal=_same_support(result.x, optimum.x),
                selection=_selection_str(result.x),
                **evaluate_selection(result.x, mu, Sigma, settings.rf)))
        print(f"    window {window.index + 1:>2}/{len(windows)} done")

    return rows


# ------------------------------------------------------------ tier ablation
def run_tier_ablation(settings: ExperimentSettings) -> list[dict]:
    """Module ablation on the synthetic suite (the existing ablation table)."""
    plan = _scenario_plan(settings.n_scenarios)
    n_assets = 16
    single_q = (0.5,)
    configurations = {
        "Full QuOpt-Agents": dict(tier2="qaoa", risk_grid=None,
                                  skip_tier3=False),
        "Tier 1 only (single q = 0.5)": dict(tier2="none", risk_grid=single_q,
                                             skip_tier3=True),
        "Tier 2 only (no local search)": dict(tier2="qaoa", risk_grid=single_q,
                                              skip_tier3=True),
        "Tier 3 only (random initialisation)": dict(tier2="none",
                                                    risk_grid=single_q,
                                                    skip_tier3=False),
        "Tiers 1+2 (no refinement)": dict(tier2="qaoa", risk_grid=None,
                                          skip_tier3=True),
        "Tiers 2+3 (fixed q)": dict(tier2="qaoa", risk_grid=single_q,
                                    skip_tier3=False),
    }
    rows: list[dict] = []
    print(f"  tier ablation: {len(configurations)} configurations x "
          f"{len(plan)} scenarios")

    for index, regime in enumerate(plan):
        rng = np.random.default_rng([settings.seed, index])
        prices, _ = generate_prices(n_assets, regime, rng,
                                    horizon=settings.synthetic_horizon)
        mu, Sigma = estimate_moments(prices)

        for label, spec in configurations.items():
            cfg_kwargs = dict(B=settings.B, rf=settings.rf,
                              tier2=spec["tier2"], qaoa=settings.qaoa)
            if spec["risk_grid"] is not None:
                cfg_kwargs["risk_grid"] = spec["risk_grid"]
            cfg = PipelineConfig(**cfg_kwargs)
            run_rng = np.random.default_rng(
                [settings.seed, index, abs(hash(label)) % 10_000])
            result = run_pipeline(mu, Sigma, cfg, run_rng,
                                  skip_tier3=spec["skip_tier3"])
            rows.append(_row(
                experiment="tier_ablation", instance=index, regime=regime,
                method=label, n=n_assets, B=settings.B,
                sharpe_is=result.sharpe, seed_sharpe=result.seed_sharpe,
                n_swaps=result.n_accepted_swaps,
                n_evals=result.n_evaluations, runtime_s=result.runtime_s,
                selection=_selection_str(result.x),
                **evaluate_selection(result.x, mu, Sigma, settings.rf)))
        if (index + 1) % 10 == 0:
            print(f"    scenario {index + 1:>2}/{len(plan)} done")

    return rows


# ------------------------------------------------------------ scale-up study
def run_scaleup(settings: ExperimentSettings) -> list[dict]:
    """Repeat the comparison at larger (N, B) to test whether the tie persists.

    State-vector simulation of the QAOA circuit needs 2**N amplitudes, so the
    proposed method is genuinely not simulable on the larger instances.  Those
    rows are recorded with a ``not_simulable`` marker rather than omitted,
    because the ceiling is itself a finding about simulation-based designs.
    """
    rows: list[dict] = []
    print(f"  scale-up: {len(settings.scaleup_instances)} instances")

    for n_assets, B in settings.scaleup_instances:
        label = f"N={n_assets},B={B}"
        n_feasible = comb(n_assets, B)
        rng = np.random.default_rng([settings.seed, n_assets, B])
        prices, _ = generate_prices(n_assets, "bull", rng,
                                    horizon=settings.synthetic_horizon)
        mu, Sigma = estimate_moments(prices)

        optimum = None
        if is_tractable(n_assets, B):
            optimum = enumerate_optimum(mu, Sigma, B, settings.rf)
            rows.append(_row(
                experiment="scaleup", instance=label, method=ENUMERATION,
                n=n_assets, B=B, sharpe_is=optimum.sharpe,
                runtime_s=optimum.runtime_s, is_optimal=True,
                selection=_selection_str(optimum.x),
                **evaluate_selection(optimum.x, mu, Sigma, settings.rf)))
        else:
            rows.append(_row(experiment="scaleup", instance=label,
                             method=ENUMERATION, n=n_assets, B=B,
                             sharpe_is="", is_optimal="not_enumerable"))

        for method_label, fn in BASELINE_REGISTRY.items():
            method_rng = np.random.default_rng(
                [settings.seed, n_assets, abs(hash(method_label)) % 10_000])
            start = time.perf_counter()
            x, counter = fn(mu, Sigma, B, method_rng, rf=settings.rf)
            runtime = time.perf_counter() - start
            rows.append(_row(
                experiment="scaleup", instance=label, method=method_label,
                n=n_assets, B=B,
                sharpe_is=sharpe_ratio(x, mu, Sigma, settings.rf),
                n_evals=counter.n_evals, runtime_s=runtime,
                is_optimal=(_same_support(x, optimum.x) if optimum else ""),
                selection=_selection_str(x),
                **evaluate_selection(x, mu, Sigma, settings.rf)))

        if n_assets <= settings.qaoa.max_qubits:
            pipeline_rng = np.random.default_rng([settings.seed, n_assets,
                                                  999])
            cfg = PipelineConfig(B=B, rf=settings.rf, tier2="qaoa",
                                 qaoa=settings.qaoa)
            result = run_pipeline(mu, Sigma, cfg, pipeline_rng)
            rows.append(_row(
                experiment="scaleup", instance=label, method=PROPOSED,
                n=n_assets, B=B, sharpe_is=result.sharpe,
                seed_sharpe=result.seed_sharpe,
                n_swaps=result.n_accepted_swaps,
                n_evals=result.n_evaluations, runtime_s=result.runtime_s,
                is_optimal=(_same_support(result.x, optimum.x) if optimum
                            else ""),
                selection=_selection_str(result.x),
                **evaluate_selection(result.x, mu, Sigma, settings.rf)))
        else:
            rows.append(_row(
                experiment="scaleup", instance=label, method=PROPOSED,
                n=n_assets, B=B, sharpe_is="",
                is_optimal="not_simulable"))
        print(f"    {label}: C(N,B) = {n_feasible:,} done")

    return rows


# ------------------------------------------------------------------- driver
EXPERIMENTS = {
    "synthetic": run_synthetic,
    "real": run_real,
    "tier2_ablation": run_tier2_ablation,
    "tier_ablation": run_tier_ablation,
    "scaleup": run_scaleup,
}


def run_all(settings: ExperimentSettings,
            which: list[str] | None = None) -> pd.DataFrame:
    """Run the requested experiments and write ``results/results.csv``."""
    which = which or list(EXPERIMENTS)
    rows: list[dict] = []
    for name in which:
        if name not in EXPERIMENTS:
            raise KeyError(f"unknown experiment {name!r}; expected any of "
                           f"{sorted(EXPERIMENTS)}")
        print(f"[experiment] {name}")
        started = time.perf_counter()
        rows.extend(EXPERIMENTS[name](settings))
        print(f"[experiment] {name} finished in "
              f"{time.perf_counter() - started:.1f} s")

    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / "results.csv"
    if out.exists() and which != list(EXPERIMENTS):
        existing = pd.read_csv(out)
        existing = existing[~existing["experiment"].isin(which)]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(out, index=False)
    print(f"[results] {len(frame)} rows written to {out}")
    return frame
