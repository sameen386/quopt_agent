"""Guards for the statistical reporting.

The signed-rank tests encode the inconsistency identified in review: for
M = 50 the statistic W+ cannot exceed 1275, and it equals 1275 only when all
50 paired differences are positive, so a maximal statistic is incompatible with
a 98% win rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from quopt.stats import (check_signed_rank_bound, mean_confidence_interval,
                         paired_comparison, signed_rank_moments,
                         wilcoxon_test)


def test_signed_rank_moments_use_the_correct_variance():
    """Var(W+) = M(M+1)(2M+1)/24, not /6."""
    m = 50
    expectation, sd, bound = signed_rank_moments(m)
    assert expectation == pytest.approx(637.5)
    assert sd == pytest.approx(np.sqrt(m * (m + 1) * (2 * m + 1) / 24))
    assert sd == pytest.approx(103.5917, abs=1e-3)
    assert bound == pytest.approx(1275.0)
    # The published formula would have been four times the variance.
    assert sd != pytest.approx(np.sqrt(m * (m + 1) * (2 * m + 1) / 6))


def test_signed_rank_upper_bound_is_attained_only_by_a_clean_sweep():
    rng = np.random.default_rng(0)
    b = rng.normal(size=50)
    a = b + np.abs(rng.normal(size=50)) + 0.1     # every difference positive
    result = wilcoxon_test(a, b)
    assert result["statistic_plus"] == pytest.approx(1275.0)
    check_signed_rank_bound(result, wins=50)


def test_maximal_statistic_with_a_loss_is_rejected():
    """A maximal W+ combined with fewer than M wins must raise."""
    fabricated = {"statistic_plus": 1275.0, "upper_bound": 1275.0,
                  "n_nonzero": 50}
    with pytest.raises(AssertionError, match="equals its maximum"):
        check_signed_rank_bound(fabricated, wins=49)


def test_statistic_above_the_bound_is_rejected():
    fabricated = {"statistic_plus": 1300.0, "upper_bound": 1275.0,
                  "n_nonzero": 50}
    with pytest.raises(AssertionError, match="exceeds its upper bound"):
        check_signed_rank_bound(fabricated, wins=50)


def test_one_loss_reduces_the_statistic_below_the_bound():
    """49 wins out of 50 must give W+ strictly below 1275."""
    rng = np.random.default_rng(1)
    b = rng.normal(size=50)
    diff = np.abs(rng.normal(size=50)) + 0.5
    diff[0] = -0.001                      # smallest magnitude, negative
    a = b + diff
    result = wilcoxon_test(a, b)
    wins = int(np.sum(a > b))
    assert wins == 49
    assert result["statistic_plus"] < 1275.0
    check_signed_rank_bound(result, wins)


def test_paired_t_statistic_identity():
    rng = np.random.default_rng(2)
    b = rng.normal(size=40)
    a = b + rng.normal(0.4, 0.3, size=40)
    cmp = paired_comparison(a, b, "test")
    expected = cmp.mean_diff * np.sqrt(cmp.n) / cmp.sd_diff
    assert cmp.t_stat == pytest.approx(expected, rel=1e-9)
    assert cmp.cohens_d == pytest.approx(cmp.mean_diff / cmp.sd_diff)


def test_paired_interval_agrees_with_the_test():
    rng = np.random.default_rng(3)
    b = rng.normal(size=60)
    a = b + rng.normal(0.5, 0.4, size=60)
    cmp = paired_comparison(a, b, "test")
    excludes_zero = cmp.diff_ci_lower > 0 or cmp.diff_ci_upper < 0
    assert excludes_zero == (cmp.p_ttest < 0.05)
    midpoint = (cmp.diff_ci_lower + cmp.diff_ci_upper) / 2
    assert midpoint == pytest.approx(cmp.mean_diff)


def test_marginal_intervals_may_overlap_while_the_pairing_is_significant():
    """The methodological point behind the corrected interval claim.

    With large between-scenario variance and a consistent small advantage, the
    marginal intervals overlap heavily while the paired difference is
    unambiguous.  Marginal overlap is therefore the wrong criterion.
    """
    rng = np.random.default_rng(4)
    common = rng.normal(1.0, 1.2, size=50)      # shared scenario effect
    b = common
    # Consistent advantage plus a little idiosyncratic noise, so the paired
    # differences are not numerically identical.
    a = common + 0.28 + rng.normal(0.0, 0.01, size=50)
    _, a_lo, a_hi = mean_confidence_interval(a)
    _, b_lo, b_hi = mean_confidence_interval(b)
    assert a_lo < b_hi and b_lo < a_hi, "marginal intervals should overlap"
    cmp = paired_comparison(a, b, "test")
    assert cmp.diff_ci_lower > 0
    assert cmp.p_ttest < 1e-9
