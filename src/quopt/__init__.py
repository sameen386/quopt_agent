"""QuOpt-Agents: reproducibility package for the manuscript.

A hybrid quantum-classical multi-agent pipeline for cardinality-constrained
portfolio selection, together with the classical baselines, the exact
enumeration reference and the statistical machinery needed to reproduce every
table and figure in the paper.

All QAOA results in this package come from exact classical state-vector
simulation.  No physical quantum device is used anywhere, and no claim of
quantum advantage is made or supported by these experiments.
"""

__version__ = "1.0.0"

from .analyst import estimate_moments
from .enumerate_exact import enumerate_optimum
from .generator import REGIMES, generate_prices
from .pipeline import PipelineConfig, run_pipeline
from .qaoa_sim import QAOAConfig, run_qaoa
from .qubo import build_qubo, objective_direct, qubo_energy
from .risk import sharpe_ratio
from .stats import paired_comparison

__all__ = [
    "__version__",
    "REGIMES",
    "PipelineConfig",
    "QAOAConfig",
    "build_qubo",
    "enumerate_optimum",
    "estimate_moments",
    "generate_prices",
    "objective_direct",
    "paired_comparison",
    "qubo_energy",
    "run_pipeline",
    "run_qaoa",
    "sharpe_ratio",
]
