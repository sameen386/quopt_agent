"""Statistical validation: paired t-test, Wilcoxon signed-rank, intervals.

Two points deserve emphasis, because both were reported incorrectly in the
previous draft.

The signed-rank statistic is bounded: ``W+ <= M(M+1)/2``, and that bound is
attained only when *every* paired difference is positive.  For M = 50 the
bound is 1275, so reporting W = 1275 alongside a 98% win rate is internally
contradictory.  :func:`wilcoxon_test` returns the statistic from SciPy and
:func:`check_signed_rank_bound` asserts the relationship against the observed
win count.

The variance of the statistic is ``M(M+1)(2M+1)/24``, not
``M(M+1)(2M+1)/6``.  The latter is the sum of squared ranks and is four times
too large, which would halve every z-score.

For a paired design the interval that supports a paired conclusion is the
interval of the paired difference, not the overlap or otherwise of the two
marginal intervals.  Both are computed here, and the reporting code labels
them distinctly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

__all__ = ["PairedComparison", "paired_comparison", "wilcoxon_test",
           "mean_confidence_interval", "signed_rank_moments",
           "check_signed_rank_bound", "win_rate", "aggregate_improvement",
           "scenariowise_improvement"]


def signed_rank_moments(m: int) -> tuple[float, float, float]:
    """Return ``(expectation, standard deviation, upper bound)`` of ``W+``."""
    expectation = m * (m + 1) / 4.0
    variance = m * (m + 1) * (2 * m + 1) / 24.0
    upper_bound = m * (m + 1) / 2.0
    return expectation, float(np.sqrt(variance)), upper_bound


def win_rate(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    """Number and percentage of entries where ``a`` strictly exceeds ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    wins = int(np.sum(a > b))
    return wins, 100.0 * wins / a.size


def aggregate_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """``I_agg``: ratio of the two means, as a percentage.

    This is the headline estimator, used in the aggregate performance table
    and in the abstract.
    """
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    if abs(mean_b) < 1e-12:
        return float("nan")
    return 100.0 * (mean_a - mean_b) / abs(mean_b)


def scenariowise_improvement(a: np.ndarray, b: np.ndarray,
                             floor: float = 1e-6) -> dict:
    """``I_sw``: mean of the per-scenario ratios, as a percentage.

    Reported separately from ``I_agg`` because the mean of ratios is not the
    ratio of means.  The estimator is unstable whenever a baseline Sharpe
    ratio approaches zero, which happens in the bear regime, so the median and
    a 10% trimmed mean are returned alongside it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.abs(b)
    usable = denom > floor
    ratios = 100.0 * (a[usable] - b[usable]) / denom[usable]
    return {
        "mean_pct": float(np.mean(ratios)),
        "median_pct": float(np.median(ratios)),
        "trimmed_mean_pct": float(stats.trim_mean(ratios, 0.10)),
        "std_pct": float(np.std(ratios, ddof=1)),
        "sem_pct": float(np.std(ratios, ddof=1) / np.sqrt(ratios.size)),
        "n_used": int(ratios.size),
        "n_excluded_near_zero": int(a.size - ratios.size),
    }


def mean_confidence_interval(values: np.ndarray,
                             alpha: float = 0.05) -> tuple[float, float, float]:
    """Student-t interval for a mean; returns ``(mean, lower, upper)``."""
    values = np.asarray(values, dtype=float)
    m = values.size
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(m))
    crit = float(stats.t.ppf(1 - alpha / 2, df=m - 1))
    return mean, mean - crit * sem, mean + crit * sem


def wilcoxon_test(a: np.ndarray, b: np.ndarray,
                  zero_method: str = "wilcox") -> dict:
    """Wilcoxon signed-rank test, with both exact and normal p-values.

    ``statistic_plus`` is the sum of positive signed ranks, recomputed
    explicitly so that it can be checked against the win count and against the
    theoretical upper bound.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    m = diff.size

    nonzero = diff[diff != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))

    expectation, sd, upper_bound = signed_rank_moments(nonzero.size)
    z = (w_plus - expectation) / sd if sd > 0 else float("nan")

    approx = stats.wilcoxon(a, b, zero_method=zero_method,
                            alternative="two-sided", method="approx")
    try:
        exact = stats.wilcoxon(a, b, zero_method=zero_method,
                               alternative="two-sided", method="exact")
        p_exact = float(exact.pvalue)
    except (ValueError, MemoryError):
        p_exact = float("nan")

    return {
        "statistic_scipy": float(approx.statistic),
        "statistic_plus": w_plus,
        "statistic_minus": w_minus,
        "expectation": expectation,
        "sd": sd,
        "upper_bound": upper_bound,
        "z": float(z),
        "p_approx": float(approx.pvalue),
        "p_exact": p_exact,
        "n_pairs": m,
        "n_nonzero": int(nonzero.size),
        "zero_method": zero_method,
    }


def check_signed_rank_bound(wilcoxon_result: dict, wins: int) -> None:
    """Raise if ``W+`` contradicts the bound or the observed win count.

    ``W+`` equals its maximum only when every difference is positive, so a
    maximal statistic combined with a sub-100% win rate is impossible.  This is
    the inconsistency the editor identified, expressed as an assertion.
    """
    w_plus = wilcoxon_result["statistic_plus"]
    bound = wilcoxon_result["upper_bound"]
    n = wilcoxon_result["n_nonzero"]

    if w_plus > bound + 1e-9:
        raise AssertionError(
            f"W+ = {w_plus} exceeds its upper bound M(M+1)/2 = {bound}")
    if w_plus >= bound - 1e-9 and wins < n:
        raise AssertionError(
            f"W+ = {w_plus} equals its maximum, which requires all {n} "
            f"differences to be positive, but only {wins} wins were observed")


@dataclass
class PairedComparison:
    label: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    sd_diff: float
    sem_diff: float
    t_stat: float
    p_ttest: float
    cohens_d: float
    diff_ci_lower: float
    diff_ci_upper: float
    wins: int
    win_rate_pct: float
    improvement_agg_pct: float
    wilcoxon_w_plus: float
    wilcoxon_z: float
    wilcoxon_p_approx: float
    wilcoxon_p_exact: float
    wilcoxon_upper_bound: float

    def as_dict(self) -> dict:
        return asdict(self)


def paired_comparison(a: np.ndarray, b: np.ndarray, label: str,
                      alpha: float = 0.05,
                      zero_method: str = "wilcox") -> PairedComparison:
    """Complete paired analysis of method ``a`` against baseline ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs equal-length arrays")

    diff = a - b
    m = diff.size
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    sem_diff = sd_diff / np.sqrt(m)
    crit = float(stats.t.ppf(1 - alpha / 2, df=m - 1))

    ttest = stats.ttest_rel(a, b)
    wilcox = wilcoxon_test(a, b, zero_method=zero_method)
    wins, win_pct = win_rate(a, b)
    check_signed_rank_bound(wilcox, wins)

    return PairedComparison(
        label=label,
        n=m,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        mean_diff=mean_diff,
        sd_diff=sd_diff,
        sem_diff=float(sem_diff),
        t_stat=float(ttest.statistic),
        p_ttest=float(ttest.pvalue),
        cohens_d=float(mean_diff / sd_diff) if sd_diff > 0 else float("nan"),
        diff_ci_lower=float(mean_diff - crit * sem_diff),
        diff_ci_upper=float(mean_diff + crit * sem_diff),
        wins=wins,
        win_rate_pct=win_pct,
        improvement_agg_pct=aggregate_improvement(a, b),
        wilcoxon_w_plus=wilcox["statistic_plus"],
        wilcoxon_z=wilcox["z"],
        wilcoxon_p_approx=wilcox["p_approx"],
        wilcoxon_p_exact=wilcox["p_exact"],
        wilcoxon_upper_bound=wilcox["upper_bound"],
    )
