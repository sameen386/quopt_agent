"""Command-line entry point.

    python -m quopt.cli all                 run everything, then tables,
                                            figures and consistency checks
    python -m quopt.cli run --only real tier2_ablation
    python -m quopt.cli tables
    python -m quopt.cli figures
    python -m quopt.cli check

``--quick`` reduces every budget so that the whole pipeline finishes in a
minute or so; use it to verify the installation, never to produce reported
numbers.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from .check_consistency import report
from .experiments import EXPERIMENTS, ExperimentSettings, run_all
from .make_figures import write_figures
from .make_tables import write_tables
from .qaoa_sim import QAOAConfig

TRAJECTORY_FILE = "trajectories.pkl"


def _settings(args) -> ExperimentSettings:
    root = Path(args.root)
    qaoa = QAOAConfig(
        depth=args.depth,
        shots=args.shots,
        maxiter=args.maxiter,
        n_restarts=args.restarts,
        expectation_mode=args.expectation,
        max_qubits=args.max_qubits,
    )
    settings = ExperimentSettings(
        seed=args.seed,
        B=args.budget,
        n_windows=args.windows,
        n_scenarios=args.scenarios,
        data_dir=root / "data",
        results_dir=root / "results",
        offline=args.offline,
        qaoa=qaoa,
    )
    if args.quick:
        settings.n_windows = min(settings.n_windows, 3)
        settings.n_scenarios = 10
        settings.qaoa.shots = 128
        settings.qaoa.maxiter = 12
        settings.qaoa.n_restarts = 1
        settings.scaleup_instances = ((16, 5), (20, 6))
        print("[quick] reduced budgets: verification only, not for reporting")
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quopt", description=__doc__)
    parser.add_argument("command",
                        choices=["all", "run", "tables", "figures", "check"])
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--only", nargs="*", default=None,
                        choices=sorted(EXPERIMENTS),
                        help="subset of experiments to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=5,
                        help="cardinality budget B")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--depth", type=int, default=2, help="QAOA depth p")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--maxiter", type=int, default=60,
                        help="COBYLA iteration limit")
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--expectation", default="shots",
                        choices=["shots", "exact"])
    parser.add_argument("--max-qubits", dest="max_qubits", type=int,
                        default=22)
    parser.add_argument("--offline", action="store_true",
                        help="use the cached return matrix only")
    parser.add_argument("--quick", action="store_true",
                        help="tiny budgets, for installation checks")
    args = parser.parse_args(argv)

    root = Path(args.root)
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_path = results_dir / "results.csv"
    settings = _settings(args)

    if args.command in ("all", "run"):
        run_all(settings, args.only)
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / TRAJECTORY_FILE, "wb") as handle:
            pickle.dump(settings.trajectories, handle)
        (results_dir / "run_settings.json").write_text(
            json.dumps({
                "seed": settings.seed, "B": settings.B,
                "n_windows": settings.n_windows,
                "n_scenarios": settings.n_scenarios,
                "lookback": settings.lookback, "horizon": settings.horizon,
                "qaoa": vars(settings.qaoa),
            }, indent=2), encoding="utf-8")

    if args.command in ("all", "tables"):
        print("[tables]")
        write_tables(results_path, results_dir)

    if args.command in ("all", "figures"):
        print("[figures]")
        trajectories = {}
        trajectory_path = results_dir / TRAJECTORY_FILE
        if trajectory_path.exists():
            with open(trajectory_path, "rb") as handle:
                trajectories = pickle.load(handle)
        write_figures(results_path, figures_dir, trajectories)

    if args.command in ("all", "check"):
        return report(results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
