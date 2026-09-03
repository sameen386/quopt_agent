"""Generate every table in the manuscript from ``results/results.csv``.

Each table is written as CSV (for machine use), as Markdown (for reading), and
into a single ``tables.docx`` (for pasting into the manuscript).  Every table
carries the note explaining its bold convention, which was the subject of a
review comment.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import (mean_confidence_interval, paired_comparison,
                    scenariowise_improvement)

__all__ = ["build_tables", "write_tables"]

PROPOSED = "QuOpt-Agents"
ENUMERATION = "Exhaustive enumeration"
REGIME_ORDER = ["bull", "bear", "choppy", "volatile", "recovery"]


def _pivot(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return frame.pivot_table(index="instance", columns="method", values=value,
                             aggfunc="first")


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "\u2014"
    return f"{x:.{nd}f}"


def _table3(synth: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    wide = _pivot(synth, "sharpe_is")
    proposed = wide[PROPOSED].to_numpy()
    rows = []
    for baseline in ["Classical MVO", "Strong Greedy"]:
        cmp = paired_comparison(proposed, wide[baseline].to_numpy(),
                               f"{PROPOSED} vs {baseline}")
        rows.append({
            "Metric": baseline,
            "Mean Sharpe ratio": _fmt(cmp.mean_b),
            "Win rate of the baseline against QuOpt-Agents (%)":
                _fmt(100.0 - cmp.win_rate_pct, 0),
            "Aggregate improvement of QuOpt-Agents (%)":
                _fmt(cmp.improvement_agg_pct, 1),
            "p-value (paired t-test)":
                "< 0.001" if cmp.p_ttest < 1e-3 else f"{cmp.p_ttest:.4f}",
        })
    rows.append({
        "Metric": f"{PROPOSED} (ours)",
        "Mean Sharpe ratio": _fmt(float(np.mean(proposed))),
        "Win rate of the baseline against QuOpt-Agents (%)": "Baseline",
        "Aggregate improvement of QuOpt-Agents (%)": "\u2014",
        "p-value (paired t-test)": "\u2014",
    })
    note = ("Note: bold indicates the highest mean Sharpe ratio. The win rate "
            "column reports the percentage of the 50 scenarios in which the "
            "baseline attains a strictly higher Sharpe ratio than "
            "QuOpt-Agents; the complementary win rates of QuOpt-Agents are "
            "therefore the difference from 100%. Aggregate improvement is "
            "I_agg = (mean S_Q - mean S_B) / |mean S_B|, the ratio of means, "
            "and is not the mean of per-scenario ratios reported separately "
            "as I_sw.")
    return pd.DataFrame(rows), note


def _table4(real: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    grouped = real.groupby("method")
    rows = []
    order = [ENUMERATION, "Classical MVO", "Strong Greedy", "GA", "PSO", "DE",
             "SA", PROPOSED]
    for method in order:
        if method not in grouped.groups:
            continue
        block = grouped.get_group(method)
        oos = pd.to_numeric(block["sharpe_oos"], errors="coerce").dropna()
        hits = block["is_optimal"].astype(str).str.lower().isin(
            ["true", "1"]).mean() * 100.0
        rows.append({
            "Method": method,
            "Mean OOS Sharpe": _fmt(oos.mean()),
            "Std": _fmt(oos.std(ddof=1)),
            "Optimality hit rate (%)": _fmt(hits, 0),
            "Runtime (s)": _fmt(pd.to_numeric(block["runtime_s"],
                                              errors="coerce").mean()),
        })
    note = ("Note: bold identifies the proposed method, not the best value. "
            "Optimality hit rate is the percentage of walk-forward windows in "
            "which the selected portfolio coincides with the in-sample "
            "optimum found by enumerating all feasible equal-weighted "
            "portfolios; where several methods reach 100%, no method is "
            "superior on solution quality at this problem size. Runtime is "
            "the mean per window on a classical CPU; no quantum hardware was "
            "used.")
    return pd.DataFrame(rows), note


def _table_regime(synth: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    for regime in REGIME_ORDER:
        block = synth[synth["regime"] == regime]
        if block.empty:
            continue
        wide = _pivot(block, "sharpe_is")
        proposed = wide[PROPOSED].to_numpy()
        rows.append({
            "Regime": regime.capitalize(),
            "Scenarios": int(proposed.size),
            "Win rate vs Classical MVO (%)":
                _fmt(100.0 * np.mean(proposed > wide["Classical MVO"]
                                     .to_numpy()), 1),
            "Win rate vs Strong Greedy (%)":
                _fmt(100.0 * np.mean(proposed > wide["Strong Greedy"]
                                     .to_numpy()), 1),
            "QuOpt-Agents mean Sharpe": _fmt(float(np.mean(proposed))),
        })
    note = ("Note: bold indicates the highest and lowest regime-wise mean "
            "Sharpe ratio. Scenario counts sum to the total number of "
            "synthetic scenarios, and the scenario-weighted mean of the "
            "regime means reproduces the overall mean reported in the "
            "aggregate performance table; this identity is asserted "
            "automatically by check_consistency.py.")
    return pd.DataFrame(rows), note


def _table7_8(synth: pd.DataFrame) -> tuple[pd.DataFrame, str,
                                            pd.DataFrame, str]:
    wide = _pivot(synth, "sharpe_is")
    proposed = wide[PROPOSED].to_numpy()
    tests, intervals = [], []

    for baseline in ["Classical MVO", "Strong Greedy"]:
        cmp = paired_comparison(proposed, wide[baseline].to_numpy(),
                               f"{PROPOSED} vs {baseline}")
        tests.append({
            "Test": "Paired t-test",
            "Comparison": cmp.label,
            "Statistic": f"t = {cmp.t_stat:.3f}",
            "p-value": "< 0.001" if cmp.p_ttest < 1e-3
                       else f"{cmp.p_ttest:.4f}",
            "Effect size": f"Cohen's d = {cmp.cohens_d:.3f}",
            "Significant (alpha = 0.05)": "Yes" if cmp.p_ttest < 0.05
                                          else "No",
        })
        tests.append({
            "Test": "Wilcoxon signed-rank",
            "Comparison": cmp.label,
            "Statistic": (f"W+ = {cmp.wilcoxon_w_plus:.0f} "
                          f"(max {cmp.wilcoxon_upper_bound:.0f}), "
                          f"Z = {cmp.wilcoxon_z:.2f}"),
            "p-value": ("< 0.001" if cmp.wilcoxon_p_approx < 1e-3
                        else f"{cmp.wilcoxon_p_approx:.2e}"),
            "Effect size": "\u2014",
            "Significant (alpha = 0.05)": "Yes"
                if cmp.wilcoxon_p_approx < 0.05 else "No",
        })
        intervals.append({
            "Comparison": cmp.label,
            "Mean difference": _fmt(cmp.mean_diff),
            "SD of differences": _fmt(cmp.sd_diff),
            "SE": _fmt(cmp.sem_diff, 4),
            "95% CI of the paired difference":
                f"[{cmp.diff_ci_lower:.3f}, {cmp.diff_ci_upper:.3f}]",
            "Excludes zero?": "Yes" if cmp.diff_ci_lower > 0
                              or cmp.diff_ci_upper < 0 else "No",
        })

    for method in [PROPOSED, "Classical MVO", "Strong Greedy"]:
        mean, lower, upper = mean_confidence_interval(wide[method].to_numpy())
        intervals.append({
            "Comparison": f"{method} (marginal interval, descriptive only)",
            "Mean difference": _fmt(mean),
            "SD of differences": "\u2014",
            "SE": "\u2014",
            "95% CI of the paired difference":
                f"[{lower:.3f}, {upper:.3f}]",
            "Excludes zero?": "\u2014",
        })

    note7 = ("Note: bold indicates statistical significance at alpha = 0.05. "
             "Cohen's d is the mean paired difference divided by the standard "
             "deviation of the paired differences; |d| >= 0.8 denotes a large "
             "effect. W+ is the sum of positive signed ranks and is bounded "
             "above by M(M+1)/2, a bound attained only when every paired "
             "difference is positive; the maximum is printed alongside the "
             "statistic so that it can be checked against the win rate.")
    note8 = ("Note: bold indicates the confidence interval of the paired "
             "difference, which is the interval that supports a paired "
             "conclusion. The marginal intervals for individual means are "
             "descriptive only and may overlap even when the paired "
             "difference is significant, because they retain the "
             "between-scenario variance that pairing removes.")
    return pd.DataFrame(tests), note7, pd.DataFrame(intervals), note8


def _table9(ablation: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    wide = _pivot(ablation, "sharpe_is")
    full_label = "Full QuOpt-Agents"
    full = wide[full_label].to_numpy()
    full_mean = float(np.mean(full))
    rows = []
    for method in wide.columns:
        values = wide[method].to_numpy()
        mean = float(np.mean(values))
        rows.append({
            "Configuration": method,
            "Mean Sharpe": _fmt(mean),
            "Win rate vs full (%)": ("Baseline" if method == full_label
                                     else _fmt(100.0 * np.mean(values > full),
                                               0)),
            "Degradation (%)": ("\u2014" if method == full_label
                                else _fmt(100.0 * (mean - full_mean)
                                          / full_mean, 1)),
        })
    rows.sort(key=lambda r: r["Configuration"] != full_label)
    note = ("Note: bold indicates the full configuration, which serves as the "
            "reference. Degradation is (S_cfg - S_full) / S_full and is "
            "therefore negative for every reduced configuration.")
    return pd.DataFrame(rows), note


def _table10(tier2: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    order = ["Tier 2 = simulated QAOA (proposed)",
             "Tier 2 = uniform random sampling",
             "Tier 2 = Strong Greedy seed",
             "Tier 2 = simulated-annealing seed",
             "Tier 2 removed (random feasible seed)"]
    grouped = tier2.groupby("method")
    for method in order:
        if method not in grouped.groups:
            continue
        block = grouped.get_group(method)
        hits = block["is_optimal"].astype(str).str.lower().isin(
            ["true", "1"]).mean() * 100.0
        rows.append({
            "Tier-2 configuration (Tiers 1 and 3 unchanged)": method,
            "Mean seed Sharpe (before Tier 3)":
                _fmt(pd.to_numeric(block["seed_sharpe"],
                                   errors="coerce").mean()),
            "Mean final OOS Sharpe":
                _fmt(pd.to_numeric(block["sharpe_oos"],
                                   errors="coerce").mean()),
            "Optimality hit rate (%)": _fmt(hits, 0),
            "Mean Tier-3 swaps":
                _fmt(pd.to_numeric(block["n_swaps"], errors="coerce").mean(),
                     2),
            "Mean runtime (s)":
                _fmt(pd.to_numeric(block["runtime_s"], errors="coerce")
                     .mean()),
        })
    note = ("Note: bold is not used, because no configuration attains a "
            "higher final Sharpe ratio than any other on this instance. The "
            "discriminating columns are the seed Sharpe ratio, which measures "
            "the Tier-2 block in isolation, and the Tier-3 swap count, which "
            "measures how much refinement work the seed saves. Every "
            "stochastic arm receives the same number of candidate draws per "
            "risk parameter as the QAOA arm.")
    return pd.DataFrame(rows), note


def _table11(scaleup: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    for instance, block in scaleup.groupby("instance", sort=False):
        n = int(block["n"].iloc[0])
        b = int(block["B"].iloc[0])
        wide = block.set_index("method")
        proposed = wide.loc[PROPOSED] if PROPOSED in wide.index else None
        enumeration = (wide.loc[ENUMERATION] if ENUMERATION in wide.index
                       else None)

        def value(row):
            if row is None:
                return "\u2014"
            raw = row["sharpe_is"]
            if raw == "" or pd.isna(raw):
                marker = str(row.get("is_optimal", ""))
                return {"not_simulable": "Not simulable",
                        "not_enumerable": "Not enumerable"}.get(marker,
                                                                "\u2014")
            return _fmt(float(raw))

        best_classical = max(
            (float(r["sharpe_is"]) for m, r in wide.iterrows()
             if m not in (PROPOSED, ENUMERATION)
             and r["sharpe_is"] != "" and not pd.isna(r["sharpe_is"])),
            default=None)
        rows.append({
            "Instance": instance,
            "C(N, B)": f"{comb(n, b):,}",
            "Exact optimum": value(enumeration),
            "Best classical heuristic": _fmt(best_classical),
            "QuOpt-Agents": value(proposed),
        })
    note = ("Note: state-vector simulation of the QAOA circuit requires 2**N "
            "complex amplitudes, so the proposed method is genuinely not "
            "simulable on the larger instances; those entries are marked "
            "rather than omitted, because the ceiling is a limitation of "
            "simulation-based study designs and not of the algorithm. "
            "\u201cNot enumerable\u201d marks instances whose feasible set is "
            "too large for exhaustive search.")
    return pd.DataFrame(rows), note


def _table12(synth: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    wide = _pivot(synth, "sharpe_is")
    proposed = wide[PROPOSED].to_numpy()
    rows = []
    for baseline in ["Classical MVO", "Strong Greedy"]:
        stats_ = scenariowise_improvement(proposed, wide[baseline].to_numpy())
        rows.append({
            "Comparison": f"{PROPOSED} vs {baseline}",
            "I_sw mean (%)": _fmt(stats_["mean_pct"], 2),
            "I_sw median (%)": _fmt(stats_["median_pct"], 2),
            "I_sw 10% trimmed mean (%)": _fmt(stats_["trimmed_mean_pct"], 2),
            "SEM (%)": _fmt(stats_["sem_pct"], 2),
            "Scenarios used": stats_["n_used"],
            "Excluded (baseline near zero)":
                stats_["n_excluded_near_zero"],
        })
    note = ("Note: I_sw is the mean of the per-scenario relative improvements, "
            "(S_Q - S_B) / |S_B|, and differs from the aggregate measure "
            "I_agg because the mean of ratios is not the ratio of means. The "
            "estimator is unstable when a baseline Sharpe ratio approaches "
            "zero, which occurs in the bear regime, so the median and a "
            "trimmed mean are reported alongside it and scenarios with a "
            "near-zero baseline are excluded and counted.")
    return pd.DataFrame(rows), note


def build_tables(results: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str]]:
    """Return ``{name: (table, note)}`` for every table available in the data."""
    tables: dict[str, tuple[pd.DataFrame, str]] = {}
    synth = results[results["experiment"] == "synthetic"]
    real = results[results["experiment"] == "real"]
    tier2 = results[results["experiment"] == "tier2_ablation"]
    tiers = results[results["experiment"] == "tier_ablation"]
    scaleup = results[results["experiment"] == "scaleup"]

    if not synth.empty:
        tables["table03_aggregate_performance"] = _table3(synth)
        tables["table05_regime_performance"] = _table_regime(synth)
        t7, n7, t8, n8 = _table7_8(synth)
        tables["table07_significance_tests"] = (t7, n7)
        tables["table08_confidence_intervals"] = (t8, n8)
        tables["table12_scenariowise_improvement"] = _table12(synth)
    if not real.empty:
        tables["table04_real_data_performance"] = _table4(real)
    if not tiers.empty:
        tables["table09_tier_ablation"] = _table9(tiers)
    if not tier2.empty:
        tables["table10_tier2_ablation"] = _table10(tier2)
    if not scaleup.empty:
        tables["table11_scaleup"] = _table11(scaleup)
    return tables


def write_tables(results_path: Path, out_dir: Path) -> dict:
    """Write every table as CSV, Markdown and a combined Word document."""
    results = pd.read_csv(results_path)
    tables = build_tables(results)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = ["# Tables generated from results.csv", ""]
    for name, (table, note) in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
        markdown.append(f"## {name}")
        markdown.append("")
        try:
            markdown.append(table.to_markdown(index=False))
        except ImportError:
            # tabulate is optional; a fixed-width dump is adequate for reading.
            markdown.append("```")
            markdown.append(table.to_string(index=False))
            markdown.append("```")
        markdown.append("")
        markdown.append(f"_{note}_")
        markdown.append("")
    (out_dir / "tables.md").write_text("\n".join(markdown), encoding="utf-8")

    try:
        _write_docx(tables, out_dir / "tables.docx")
    except ImportError:
        print("  python-docx not installed; skipped tables.docx")

    print(f"  {len(tables)} tables written to {out_dir}")
    return tables


def _write_docx(tables: dict, path: Path) -> None:
    import docx
    from docx.shared import Pt

    doc = docx.Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(9)
    doc.add_heading("Tables generated from results.csv", level=1)

    for name, (table, note) in tables.items():
        doc.add_heading(name.replace("_", " "), level=2)
        grid = doc.add_table(rows=1, cols=len(table.columns))
        grid.style = "Table Grid"
        for i, column in enumerate(table.columns):
            cell = grid.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(column))
            run.bold = True
            run.font.size = Pt(8)
        for _, row in table.iterrows():
            cells = grid.add_row().cells
            for i, value in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(value))
                run.font.size = Pt(8)
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(note)
        run.italic = True
        run.font.size = Pt(8)
    doc.save(path)
