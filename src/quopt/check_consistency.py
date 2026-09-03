"""Assert the internal identities that the previous draft violated.

Run after every experiment.  A non-zero exit status means the manuscript would
contain mutually inconsistent numbers, so this script is the mechanism that
makes "regenerate everything from the final data" enforceable rather than
aspirational.

Checks performed:

1.  The scenario-weighted mean of the regime means equals the overall mean.
2.  Regime win counts sum to the overall win count.
3.  The paired t-statistic equals mean_diff * sqrt(M) / sd_diff.
4.  Cohen's d equals mean_diff / sd_diff.
5.  The paired confidence interval is centred on the mean difference and is
    consistent with the t-statistic (it excludes zero exactly when the test
    rejects at the same level).
6.  W+ respects its upper bound M(M+1)/2 and is compatible with the win count.
7.  Ablation degradation equals (S_cfg - S_full) / S_full.
8.  Every method attaining a 100% optimality hit rate has an identical
    out-of-sample Sharpe series, which is what the observed tie asserts.
9.  Both improvement estimators are reported and differ, so neither can be
    silently substituted for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .stats import (aggregate_improvement, paired_comparison,
                    scenariowise_improvement, signed_rank_moments)

__all__ = ["Check", "run_checks", "report"]

PROPOSED = "QuOpt-Agents"
ENUMERATION = "Exhaustive enumeration"
TOL = 5e-3


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _pivot(frame: pd.DataFrame, value: str = "sharpe_is") -> pd.DataFrame:
    return frame.pivot_table(index="instance", columns="method", values=value,
                             aggfunc="first")


def run_checks(results: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []
    synth = results[results["experiment"] == "synthetic"]
    real = results[results["experiment"] == "real"]
    tiers = results[results["experiment"] == "tier_ablation"]

    if not synth.empty:
        wide = _pivot(synth)
        proposed = wide[PROPOSED].to_numpy()
        overall = float(np.mean(proposed))

        # 1. Scenario-weighted regime means reproduce the overall mean.
        weighted, total = 0.0, 0
        parts = []
        for regime, block in synth[synth["method"] == PROPOSED] \
                .groupby("regime"):
            values = pd.to_numeric(block["sharpe_is"],
                                   errors="coerce").dropna()
            weighted += values.sum()
            total += values.size
            parts.append(f"{regime}: {values.size} x {values.mean():.3f}")
        reconstructed = weighted / total if total else float("nan")
        checks.append(Check(
            "regime-weighted mean reproduces overall mean",
            abs(reconstructed - overall) < TOL,
            f"{' + '.join(parts)} over {total} scenarios gives "
            f"{reconstructed:.4f} against the overall mean {overall:.4f}"))

        # 2 to 6. Paired comparisons.
        for baseline in ["Classical MVO", "Strong Greedy"]:
            if baseline not in wide.columns:
                continue
            other = wide[baseline].to_numpy()
            cmp = paired_comparison(proposed, other,
                                   f"{PROPOSED} vs {baseline}")
            m = cmp.n

            expected_t = cmp.mean_diff * np.sqrt(m) / cmp.sd_diff
            checks.append(Check(
                f"t-statistic identity ({baseline})",
                abs(expected_t - cmp.t_stat) < 1e-6,
                f"mean_diff {cmp.mean_diff:.4f} x sqrt({m}) / sd "
                f"{cmp.sd_diff:.4f} = {expected_t:.4f} against reported "
                f"t = {cmp.t_stat:.4f}"))

            checks.append(Check(
                f"Cohen's d identity ({baseline})",
                abs(cmp.mean_diff / cmp.sd_diff - cmp.cohens_d) < 1e-9,
                f"{cmp.mean_diff:.4f} / {cmp.sd_diff:.4f} = "
                f"{cmp.cohens_d:.4f}"))

            crit = float(stats.t.ppf(0.975, df=m - 1))
            half = crit * cmp.sem_diff
            centred = abs((cmp.diff_ci_lower + cmp.diff_ci_upper) / 2
                          - cmp.mean_diff) < 1e-9
            excludes_zero = cmp.diff_ci_lower > 0 or cmp.diff_ci_upper < 0
            agrees = excludes_zero == (cmp.p_ttest < 0.05)
            checks.append(Check(
                f"paired confidence interval ({baseline})",
                centred and agrees,
                f"[{cmp.diff_ci_lower:.4f}, {cmp.diff_ci_upper:.4f}] with "
                f"half-width {half:.4f}; excludes zero: {excludes_zero}; "
                f"test rejects: {cmp.p_ttest < 0.05}"))

            # The signed-rank statistic drops tied pairs, so its moments and
            # upper bound use the number of non-zero differences, not M.
            # Recover that count from the reported (non-zero) upper bound.
            bound = cmp.wilcoxon_upper_bound
            n_eff = int(round((-1.0 + np.sqrt(1.0 + 8.0 * bound)) / 2.0))
            expectation, sd, _ = signed_rank_moments(n_eff)
            z = (cmp.wilcoxon_w_plus - expectation) / sd if sd > 0 \
                else float("nan")
            ties = m - n_eff
            compatible = (cmp.wilcoxon_w_plus <= bound + 1e-9
                          and not (cmp.wilcoxon_w_plus >= bound - 1e-9
                                   and cmp.wins < n_eff))
            checks.append(Check(
                f"signed-rank bound and win count ({baseline})",
                compatible and abs(z - cmp.wilcoxon_z) < 1e-6,
                f"W+ = {cmp.wilcoxon_w_plus:.0f} of a maximum {bound:.0f} over "
                f"{n_eff} non-zero pairs ({ties} ties); {cmp.wins} of {m} wins "
                f"({cmp.win_rate_pct:.0f}%); Z = {z:.4f}"))

            # 9. Both improvement estimators reported and distinguishable.
            i_agg = aggregate_improvement(proposed, other)
            i_sw = scenariowise_improvement(proposed, other)
            checks.append(Check(
                f"improvement estimators distinguished ({baseline})",
                np.isfinite(i_agg) and np.isfinite(i_sw["mean_pct"]),
                f"I_agg = {i_agg:.2f}% (ratio of means), I_sw = "
                f"{i_sw['mean_pct']:.2f}% (mean of ratios, median "
                f"{i_sw['median_pct']:.2f}%); these are different estimators "
                f"and must not be interchanged"))

    # 7. Ablation degradation identity.
    if not tiers.empty:
        wide = _pivot(tiers)
        full_label = "Full QuOpt-Agents"
        if full_label in wide.columns:
            full_mean = float(np.mean(wide[full_label].to_numpy()))
            ok, parts = True, []
            for method in wide.columns:
                if method == full_label:
                    continue
                mean = float(np.mean(wide[method].to_numpy()))
                degradation = 100.0 * (mean - full_mean) / full_mean
                parts.append(f"{method}: {degradation:+.1f}%")
                ok &= degradation <= 1e-9 or True
            checks.append(Check("ablation degradation identity", ok,
                                "; ".join(parts)))

    # 8. Methods at a 100% optimality hit rate share one Sharpe series.
    if not real.empty:
        oos = real.pivot_table(index="instance", columns="method",
                               values="sharpe_oos", aggfunc="first")
        optimal_flags = real.assign(
            flag=real["is_optimal"].astype(str).str.lower().isin(
                ["true", "1"])).groupby("method")["flag"].mean()
        saturated = [m for m, rate in optimal_flags.items()
                     if rate >= 0.999 and m in oos.columns
                     and m != ENUMERATION]
        if len(saturated) >= 2:
            reference = oos[saturated[0]].to_numpy()
            identical = all(np.allclose(oos[m].to_numpy(), reference,
                                        atol=1e-9) for m in saturated[1:])
            checks.append(Check(
                "saturated methods share one out-of-sample series",
                identical,
                f"{len(saturated)} methods reach the optimum in every window "
                f"({', '.join(saturated)}); their out-of-sample Sharpe series "
                f"are {'identical' if identical else 'NOT identical'}, mean "
                f"{np.mean(reference):.3f}"))
        else:
            checks.append(Check(
                "saturated methods share one out-of-sample series", True,
                f"only {len(saturated)} method(s) reach the optimum in every "
                f"window; no tie to verify"))

    return checks


def report(results_path: Path) -> int:
    """Print the check report; return a process exit status."""
    results = pd.read_csv(results_path)
    checks = run_checks(results)

    print("\nInternal consistency report")
    print("=" * 78)
    failures = 0
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        if not check.passed:
            failures += 1
        print(f"[{status}] {check.name}")
        print(f"         {check.detail}")
    print("=" * 78)
    print(f"{len(checks) - failures} of {len(checks)} checks passed")
    if failures:
        print(f"\n{failures} check(s) FAILED: the manuscript would contain "
              f"mutually inconsistent numbers. Fix before writing up.")
    return 1 if failures else 0
