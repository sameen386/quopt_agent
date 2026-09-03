# QuOpt-Agents — Reproducibility Package

Submission bundle for *QuOpt-Agents: Quantum-Enhanced Multi-Agent Framework for Portfolio Optimization*.

## Contents

| Path | Description |
|------|-------------|
| `src/quopt/` | Python source (pipeline, QAOA simulator, baselines, stats) |
| `results/` | `results.csv`, table CSVs (Tables 3–12), `tables.md` |
| `figures/` | Manuscript figures (`.eps`) |
| `data/` | Cached return matrix for offline runs |
| `tests/` | Unit and smoke tests |
| `quopt-agents-revised.ipynb` | Supplementary Jupyter notebook |

## Quick start

```bash
pip install -e .
python -m quopt.cli check      # verify tables vs results.csv
python -m quopt.cli all        # full re-run (long; use --offline if no network)
python -m quopt.cli all --quick   # fast install check only
```

## Notes

- Table CSVs live under `results/` (not a separate `tables/` folder).
- Pre-generated outputs are included so reviewers can verify without re-running experiments.
- Requires Python ≥ 3.10.
