"""Fixed settings of the W1 repair round (2026-10-12, after the independent verification).

Everything the repaired 2026 uncertainty, its coverage check and the label depend on lives
here, so the A1 declaration (rep_prereg.py) can hash it before any coverage is computed.
Existing W1 modules are not edited: published manifests and pre-registrations hash them.

Verifier major issue 1 (full/v2): the 2026 sd was the RMSE of the chain against the flag-5
Denton series (a model, not an observation), the 200 draws held only 11 distinct
trajectories, their mean differed from the shipped value by 0.169 / 0.337 x 10^8 m3 for
Jan-Sep, structural-zero January carried an sd while every draw was 0, and nothing was
coverage-checked or labelled "uncalibrated" (plan 4.7). The repaired error model:

1. Residual trajectories against observation-constrained truths. For a target year t and each
   pool year y (the chain's evaluation years except t), r_{y,k}(m) = D_k(y, m) - c_y(m), where
   D_k are the 200 full/v1 state-space posterior draws (their annual sums equal the observed
   Jiji annual totals exactly) and c_y is the chain prediction for y from a fold that excludes
   y (and t: double leave-one-year-out when t is an evaluation year). The annual sum of every
   residual trajectory is therefore the observed annual error of the chain.
2. Bias folded into the spread, not corrected: the shipped value stays the registered chain
   prediction; f = RMSE / SD of the pool's annual residuals (both with ddof 0).
3. Small-sample predictive inflation: g = t_{0.95, n} / z_{0.95}, n = number of pool years
   (the error variance is estimated from n residuals with the mean fixed at zero).
4. Smoothed bootstrap (continuous support): each member adds a year-level Gaussian shock
   h * eps * pi(m) with h = 1.06 n^(-1/5) (Silverman factor) and pi(m) = s_m / sum(s) * SD
   (s_m the pool's per-month residual sd), then divides by sqrt(1 + h^2) to keep the annual
   variance.
5. Members are centred on their own mean (pre-truncation mean = value exactly), scaled by
   f * g, added to the value and truncated at 0; structural-zero months (winter weight 0)
   are exactly 0 in every member and carry sd 0.
6. sd of a 2026 month = standard deviation (ddof 0) of the shipped members.

Coverage check (design-time units; declared, not a fresh test): for each evaluation year t,
the ensemble built from the other evaluation years (double leave-one-year-out) gives a 90 %
interval (5th-95th percentile of the member sums) of the held-out annual district total;
hits over 11 years x 2 districts, clustered by year, through harness.coverage2.coverage_gate.
Jan-May (2017-2021, the briefing table) and the inflow regression (monthly, years whose
composite is observed, flag <= 1) are reported, not gated. The 2012-2025 rows keep full/v1's
state-space draws; their label follows the A1 gate-2 re-evaluation of full/v1's existing
Jan-May leave-one-year-out coverage (LABEL_HIST_TEMPLATE).
"""
from __future__ import annotations

from w1_canal import ext_config as E

REPAIR_TAG = "W1 repair round 2026-10-12"

# ----------------------------------------------------------------------------- error model
N_DRAWS = 200                      # shipped members per district (same as full/v1)
DRAW_SEED = 20261014               # members of the 2026 product
COVERAGE_SEED = 20261015           # + target year: shocks of the coverage ensembles
LEVEL = 0.90                       # central interval of the member sums
PRED_QUANTILE = 0.95               # g = t_{PRED_QUANTILE, n} / z_{PRED_QUANTILE}
SILVERMAN = 1.06                   # h = SILVERMAN * n ** (-1/5)
WINTER_ALT = 0.5                   # structural alternative for winter-weight-0 months

# ----------------------------------------------------------------------------- coverage
#: Declared rule: the annual-scale error model is "coverage-checked" if the A1 gate-2
#: interval (harness.coverage2.coverage_gate, clusters = years, level 0.95, Student-t
#: quantile) of the 22 annual hits contains 0.90; otherwise every 2026 interval is labelled
#: "uncalibrated (annual-scale coverage check failed)". Monthly and Jan-Sep intervals have no
#: observed target of the same kind, so they are labelled "uncalibrated" at those scales
#: whatever the annual result. Clopper-Pearson on n_eff = number of years is reported for
#: continuity with W1's earlier coverage rule.
COVERAGE_TARGET = 0.90
JANMAY_YEARS = (2017, 2018, 2019, 2020, 2021)     # evaluation years with a briefing value
INFLOW_Z = 1.6448536269514722                     # 90 % normal quantile for the inflow sd
INFLOW_MAX_FLAG = 1                                # composite days that count as observed

# ----------------------------------------------------------------------------- labels
LABEL_2026_PASS = ("annual-scale error model coverage-checked on the design-time evaluation "
                   "years (A1 gate-2 interval contains 0.90); monthly and Jan-Sep intervals "
                   "uncalibrated (no observed target at those scales)")
LABEL_2026_FAIL = ("uncalibrated (annual-scale coverage check failed); monthly and Jan-Sep "
                   "intervals uncalibrated")
#: Historical rows (2012-2025, full state space): the existing Jan-May leave-one-year-out
#: coverage (statespace_loo_coverage.csv, 2015-2021 x 2 districts) is re-evaluated with the A1
#: gate-2 interval (clusters = years); the label wording follows its verdict. Monthly values have
#: no observed target and stay "uncalibrated" at the monthly scale.
LABEL_HIST_TEMPLATE = ("Jan-May aggregate {verdict} (full state space, leave-one-year-out "
                       "2015-2021: {hits}/{n} hits; A1 gate-2 interval [{lo:.2f}, {hi:.2f}] "
                       "{rel} 0.90); monthly scale uncalibrated")
LABEL_STRUCTURAL = ("structural zero (winter weight 0): draws and sd are 0; the winter-0.5 "
                    "alternative is in structural_alt_value (a structural, not a sampling, "
                    "uncertainty)")

# ----------------------------------------------------------------------------- years
EXT_YEAR = E.EXT_YEAR
FIT_YEARS = E.FIT_YEARS
