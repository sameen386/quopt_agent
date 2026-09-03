# Tables generated from results.csv

## table03_aggregate_performance

| Metric              |   Mean Sharpe ratio | Win rate of the baseline against QuOpt-Agents (%)   | Aggregate improvement of QuOpt-Agents (%)   | p-value (paired t-test)   |
|:--------------------|--------------------:|:----------------------------------------------------|:--------------------------------------------|:--------------------------|
| Classical MVO       |               1.147 | 30                                                  | 40.5                                        | < 0.001                   |
| Strong Greedy       |               1.594 | 60                                                  | 1.1                                         | 0.0017                    |
| QuOpt-Agents (ours) |               1.612 | Baseline                                            | —                                           | —                         |

_Note: bold indicates the highest mean Sharpe ratio. The win rate column reports the percentage of the 50 scenarios in which the baseline attains a strictly higher Sharpe ratio than QuOpt-Agents; the complementary win rates of QuOpt-Agents are therefore the difference from 100%. Aggregate improvement is I_agg = (mean S_Q - mean S_B) / |mean S_B|, the ratio of means, and is not the mean of per-scenario ratios reported separately as I_sw._

## table05_regime_performance

| Regime   |   Scenarios |   Win rate vs Classical MVO (%) |   Win rate vs Strong Greedy (%) |   QuOpt-Agents mean Sharpe |
|:---------|------------:|--------------------------------:|--------------------------------:|---------------------------:|
| Bull     |          14 |                            35.7 |                            64.3 |                      3.097 |
| Bear     |          11 |                           100   |                            18.2 |                     -0.066 |
| Choppy   |           6 |                            66.7 |                            66.7 |                      1.957 |
| Volatile |           7 |                           100   |                            14.3 |                      1.254 |
| Recovery |          12 |                            66.7 |                            33.3 |                      1.453 |

_Note: bold indicates the highest and lowest regime-wise mean Sharpe ratio. Scenario counts sum to the total number of synthetic scenarios, and the scenario-weighted mean of the regime means reproduces the overall mean reported in the aggregate performance table; this identity is asserted automatically by check_consistency.py._

## table07_significance_tests

| Test                 | Comparison                    | Statistic                    | p-value   | Effect size       | Significant (alpha = 0.05)   |
|:---------------------|:------------------------------|:-----------------------------|:----------|:------------------|:-----------------------------|
| Paired t-test        | QuOpt-Agents vs Classical MVO | t = 6.227                    | < 0.001   | Cohen's d = 0.881 | Yes                          |
| Wilcoxon signed-rank | QuOpt-Agents vs Classical MVO | W+ = 630 (max 630), Z = 5.16 | < 0.001   | —                 | Yes                          |
| Paired t-test        | QuOpt-Agents vs Strong Greedy | t = 3.326                    | 0.0017    | Cohen's d = 0.470 | Yes                          |
| Wilcoxon signed-rank | QuOpt-Agents vs Strong Greedy | W+ = 210 (max 210), Z = 3.92 | < 0.001   | —                 | Yes                          |

_Note: bold indicates statistical significance at alpha = 0.05. Cohen's d is the mean paired difference divided by the standard deviation of the paired differences; |d| >= 0.8 denotes a large effect. W+ is the sum of positive signed ranks and is bounded above by M(M+1)/2, a bound attained only when every paired difference is positive; the maximum is printed alongside the statistic so that it can be checked against the win rate._

## table08_confidence_intervals

| Comparison                                          |   Mean difference | SD of differences   | SE     | 95% CI of the paired difference   | Excludes zero?   |
|:----------------------------------------------------|------------------:|:--------------------|:-------|:----------------------------------|:-----------------|
| QuOpt-Agents vs Classical MVO                       |             0.465 | 0.528               | 0.0747 | [0.315, 0.615]                    | Yes              |
| QuOpt-Agents vs Strong Greedy                       |             0.018 | 0.039               | 0.0055 | [0.007, 0.029]                    | Yes              |
| QuOpt-Agents (marginal interval, descriptive only)  |             1.612 | —                   | —      | [1.175, 2.049]                    | —                |
| Classical MVO (marginal interval, descriptive only) |             1.147 | —                   | —      | [0.618, 1.676]                    | —                |
| Strong Greedy (marginal interval, descriptive only) |             1.594 | —                   | —      | [1.161, 2.027]                    | —                |

_Note: bold indicates the confidence interval of the paired difference, which is the interval that supports a paired conclusion. The marginal intervals for individual means are descriptive only and may overlap even when the paired difference is significant, because they retain the between-scenario variance that pairing removes._

## table12_scenariowise_improvement

| Comparison                    |   I_sw mean (%) |   I_sw median (%) |   I_sw 10% trimmed mean (%) |   SEM (%) |   Scenarios used |   Excluded (baseline near zero) |
|:------------------------------|----------------:|------------------:|----------------------------:|----------:|-----------------:|--------------------------------:|
| QuOpt-Agents vs Classical MVO |          176.04 |             20.03 |                       53.03 |     95.72 |               50 |                               0 |
| QuOpt-Agents vs Strong Greedy |            1.13 |              0    |                        0.49 |      0.36 |               50 |                               0 |

_Note: I_sw is the mean of the per-scenario relative improvements, (S_Q - S_B) / |S_B|, and differs from the aggregate measure I_agg because the mean of ratios is not the ratio of means. The estimator is unstable when a baseline Sharpe ratio approaches zero, which occurs in the bear regime, so the median and a trimmed mean are reported alongside it and scenarios with a near-zero baseline are excluded and counted._

## table04_real_data_performance

| Method                 |   Mean OOS Sharpe |   Std |   Optimality hit rate (%) |   Runtime (s) |
|:-----------------------|------------------:|------:|--------------------------:|--------------:|
| Exhaustive enumeration |             1.716 | 2.128 |                       100 |         0.251 |
| Classical MVO          |             1.577 | 2.045 |                        25 |         0.028 |
| Strong Greedy          |             1.48  | 1.956 |                        20 |         0     |
| GA                     |             1.716 | 2.128 |                       100 |         0.807 |
| PSO                    |             1.822 | 2.062 |                        70 |         0.246 |
| DE                     |             1.716 | 2.128 |                       100 |         0.66  |
| SA                     |             1.716 | 2.128 |                       100 |         0.191 |
| QuOpt-Agents           |             1.716 | 2.128 |                       100 |        58.622 |

_Note: bold identifies the proposed method, not the best value. Optimality hit rate is the percentage of walk-forward windows in which the selected portfolio coincides with the in-sample optimum found by enumerating all feasible equal-weighted portfolios; where several methods reach 100%, no method is superior on solution quality at this problem size. Runtime is the mean per window on a classical CPU; no quantum hardware was used._

## table09_tier_ablation

| Configuration                       |   Mean Sharpe | Win rate vs full (%)   | Degradation (%)   |
|:------------------------------------|--------------:|:-----------------------|:------------------|
| Full QuOpt-Agents                   |         1.612 | Baseline               | —                 |
| Tier 1 only (single q = 0.5)        |         0.176 | 0                      | -89.1             |
| Tier 2 only (no local search)       |         1.355 | 0                      | -16.0             |
| Tier 3 only (random initialisation) |         1.612 | 0                      | 0.0               |
| Tiers 1+2 (no refinement)           |         1.529 | 0                      | -5.1              |
| Tiers 2+3 (fixed q)                 |         1.612 | 0                      | 0.0               |

_Note: bold indicates the full configuration, which serves as the reference. Degradation is (S_cfg - S_full) / S_full and is therefore negative for every reduced configuration._

## table10_tier2_ablation

| Tier-2 configuration (Tiers 1 and 3 unchanged)   |   Mean seed Sharpe (before Tier 3) |   Mean final OOS Sharpe |   Optimality hit rate (%) |   Mean Tier-3 swaps |   Mean runtime (s) |
|:-------------------------------------------------|-----------------------------------:|------------------------:|--------------------------:|--------------------:|-------------------:|
| Tier 2 = simulated QAOA (proposed)               |                              2.238 |                   1.716 |                       100 |               12.55 |             59.213 |
| Tier 2 = uniform random sampling                 |                              2.461 |                   1.716 |                       100 |                7.35 |              0.874 |
| Tier 2 = Strong Greedy seed                      |                              2.442 |                   1.716 |                       100 |                7.6  |              0.051 |
| Tier 2 = simulated-annealing seed                |                              2.531 |                   1.716 |                       100 |                0    |              1.445 |
| Tier 2 removed (random feasible seed)            |                              0.657 |                   1.716 |                       100 |               29.05 |              0.123 |

_Note: bold is not used, because no configuration attains a higher final Sharpe ratio than any other on this instance. The discriminating columns are the seed Sharpe ratio, which measures the Tier-2 block in isolation, and the Tier-3 swap count, which measures how much refinement work the seed saves. Every stochastic arm receives the same number of candidate draws per risk parameter as the QAOA arm._

## table11_scaleup

| Instance   |     C(N, B) | Exact optimum   |   Best classical heuristic | QuOpt-Agents   |
|:-----------|------------:|:----------------|---------------------------:|:---------------|
| N=16,B=5   |       4,368 | 2.592           |                      2.592 | 2.592          |
| N=20,B=6   |      38,760 | 2.292           |                      2.292 | 2.292          |
| N=30,B=8   |   5,852,925 | Not enumerable  |                      1.478 | Not simulable  |
| N=40,B=10  | 847,660,528 | Not enumerable  |                      4.55  | Not simulable  |

_Note: state-vector simulation of the QAOA circuit requires 2**N complex amplitudes, so the proposed method is genuinely not simulable on the larger instances; those entries are marked rather than omitted, because the ceiling is a limitation of simulation-based study designs and not of the algorithm. “Not enumerable” marks instances whose feasible set is too large for exhaustive search._
