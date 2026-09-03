"""End-to-end smoke test of the whole pipeline at tiny budgets.

Runs every experiment, writes the tables and figures, and asserts that the
internal-consistency checks pass.  The real-data experiments are driven from a
temporary cache built out of synthetic returns, so the test needs neither
network access nor the ``yfinance`` package; the loader path exercised is
exactly the one used with real prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quopt.analyst import estimate_moments
from quopt.check_consistency import run_checks
from quopt.data import CACHE_NAME, DEFAULT_TICKERS, walk_forward_windows
from quopt.experiments import ExperimentSettings, run_all
from quopt.generator import generate_prices
from quopt.make_tables import build_tables
from quopt.qaoa_sim import QAOAConfig


@pytest.fixture
def workspace(tmp_path):
    """A repository-shaped temporary tree with a cached return matrix."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(123)
    prices, _ = generate_prices(16, "bull", rng, horizon=600)
    returns = np.diff(np.log(prices), axis=0)
    dates = pd.bdate_range("2018-01-02", periods=returns.shape[0])
    pd.DataFrame(returns, index=dates, columns=DEFAULT_TICKERS).to_csv(
        data_dir / CACHE_NAME)
    (tmp_path / "results").mkdir()
    (tmp_path / "figures").mkdir()
    return tmp_path


@pytest.fixture
def settings(workspace):
    return ExperimentSettings(
        seed=42, B=4, n_windows=3, n_scenarios=10,
        synthetic_horizon=252,
        data_dir=workspace / "data",
        results_dir=workspace / "results",
        offline=True,
        qaoa=QAOAConfig(depth=2, shots=64, maxiter=8, n_restarts=1),
        scaleup_instances=((16, 4), (24, 5)),
    )


def test_walk_forward_windows_do_not_leak_future_data():
    windows = walk_forward_windows(1000, n_windows=5, lookback=126,
                                   horizon=63)
    assert len(windows) == 5
    for window in windows:
        assert window.train_stop == window.test_start
        assert window.test_stop - window.test_start == 63
        assert window.train_stop - window.train_start == 126
        assert window.test_stop <= 1000


def test_full_pipeline_runs_and_is_self_consistent(settings, workspace):
    frame = run_all(settings)
    assert not frame.empty
    assert set(frame["experiment"]) == {"synthetic", "real", "tier2_ablation",
                                        "tier_ablation", "scaleup"}

    # Every selection respects the cardinality budget of its own instance;
    # the scale-up sweep varies B independently of the global setting.
    populated = frame[frame["selection"].astype(str) != ""]
    for _, row in populated.iterrows():
        assert len(str(row["selection"]).split(",")) == int(row["B"])

    tables = build_tables(frame)
    for name in ["table03_aggregate_performance",
                 "table04_real_data_performance",
                 "table10_tier2_ablation", "table11_scaleup"]:
        assert name in tables, name
        assert not tables[name][0].empty, name
        assert tables[name][1], f"{name} is missing its note"

    checks = run_checks(frame)
    failures = [c for c in checks if not c.passed]
    assert not failures, "; ".join(f"{c.name}: {c.detail}" for c in failures)


def test_scaleup_marks_instances_that_cannot_be_simulated(settings):
    settings.qaoa.max_qubits = 20
    settings.scaleup_instances = ((16, 4), (24, 5))
    frame = run_all(settings, which=["scaleup"])
    proposed = frame[(frame["method"] == "QuOpt-Agents")
                     & (frame["instance"] == "N=24,B=5")]
    assert not proposed.empty
    assert proposed["is_optimal"].iloc[0] == "not_simulable"


def test_enumeration_row_is_never_beaten(settings):
    frame = run_all(settings, which=["synthetic"])
    wide = frame.pivot_table(index="instance", columns="method",
                             values="sharpe_is", aggfunc="first")
    optimum = wide["Exhaustive enumeration"].to_numpy()
    for method in wide.columns:
        if method == "Exhaustive enumeration":
            continue
        assert np.all(wide[method].to_numpy() <= optimum + 1e-9), method
