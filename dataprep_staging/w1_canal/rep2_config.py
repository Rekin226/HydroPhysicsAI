"""Fixed settings of the second W1 repair round (2026-10-13, after the second verification).

The verifier found that the first repair round labelled only district_monthly.csv and
inflow_predicted_monthly.csv, while other published intervals carried no label (plan 4.7):
the daily inflow composite sd (preorigin/v1 and full/v1-v3), the basin rain index sd, the
percolation basis sd, the grid sd and the draws. This round publishes new versions
(preorigin/v2, full/v4) with the same values and members, in which every sd, percentile and
member array carries either a coverage-checked label or "uncalibrated", plus an inventory of
every interval in the manifest (``interval_labels``). Nothing is re-fitted. Existing W1
modules are not edited: published manifests and pre-registrations hash them.

Other changes (verifier minor issues):
* the 2026 label leads with "uncalibrated at every published scale"; the annual check covers
  12-month sums that the product does not publish;
* the 2012-2025 labels state the weak coverage evidence (hits, n_eff, direction of misses);
* ``member_mean`` and ``value_minus_member_mean`` make the different centres of the Denton
  value and the state-space draws explicit;
* preorigin/v2: structural-zero months get sd 0 and ``structural_alt_value`` as full/v3 did;
  ``prior_width_input`` becomes forbidden (no sd of the product is coverage-checked at a scale
  a prior width would use); the gate carries the design-time caveat; validation.json holds the
  parse QC of 2012-2022 only (re-parsed from L0), with no post-origin yearbook;
* a post hoc diagnostic of the serial dependence of the chain's annual residuals (Yunlin
  2020-2025 all negative), reported, not used to change anything.
"""
from __future__ import annotations

ROUND = "repair_r2"
ROUND_TAG = "W1 repair round 2 (2026-10-13)"
PRE_YEARS = tuple(range(2012, 2023))           # pre-origin yearbook years (U3)
DIAG_RECENT_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)   # verifier's run (post hoc)

# ----------------------------------------------------------------------------- labels
LABEL_HIST = ("monthly interval uncalibrated (no monthly observation exists); the Jan-May "
              "aggregate of the state-space draws passes the plan 4.4 coverage rule only weakly: "
              "leave-one-year-out 2015-2021 {hits}/{n} = {cov:.2f} hits, A1 gate-2 interval "
              "[{lo:.2f}, {hi:.2f}] {rel} 0.90 at n_eff = {neff} design-time years; misses: "
              "{misses}")
LABEL_2026 = ("uncalibrated at every published scale (monthly values and their Jan-Sep sums "
              "have no observed target); the error model's 12-month annual sums, which this "
              "product does not publish, passed the declared check W1.ext2026_uncertainty.r1 on "
              "{n_years} design-time years ({hits}/{n}, A1 gate-2 [{lo:.2f}, {hi:.2f}]), "
              "declared after the chain's leave-one-year-out errors were known{serial}")
LABEL_2026_SERIAL = ("; the Yunlin annual residuals of 2020-2025 are all negative (post hoc "
                     "runs diagnostic), so the pooled residuals may mis-centre the 2026 Yunlin "
                     "interval")
LABEL_COMPOSITE = ("uncalibrated: heuristic relative sd {rel:.3f} = 1.4826 x MAD of the log "
                   "annual weir-inflow / gauge-sum scale factors over the fit years "
                   "{fy0}-{fy1}, doubled on donor-filled days (flag 3); the composite's daily "
                   "error is not observable (the weir publishes only annual inflow), so no "
                   "coverage check is possible; not a prior width")
LABEL_RAIN = ("uncalibrated: standard error of the area-weighted station mean from the "
              "between-record scatter of the normal ratios (sd / sqrt(n)), times the index; no "
              "areal-rain truth to check it against; not propagated into the chain or its "
              "intervals")
LABEL_RAIN_NONE = "no interval (index missing: fewer than 8 valid records)"
LABEL_GAUGES = "no interval published (sd empty; the flag columns carry the QC state)"
LABEL_INFLOW_SUFFIX = (" (design-time years, double leave-one-year-out; rule declared in "
                       "W1.ext2026_uncertainty.r1 after the chain's errors were known)")
LABEL_CHECK_TABLE = ("coverage-check table: its intervals are the held-out predictions being "
                     "checked (column 'hit'); not a forcing interval")
LABEL_NO_INTERVAL = "no interval in this output"

SD_BASIS_COMPOSITE = {1: "relative heuristic sd (rescaled gauge sum, all gauges observed)",
                      4: "relative heuristic sd (a gauge interpolated over <= 15 days)",
                      3: "2 x relative heuristic sd (a gauge donor-filled or excluded)"}
SD_BASIS_RAIN = "between-record standard error of the normal-ratio index"

#: uses removed from the pre-origin product (plan 4.7: no sd of it is checked at a scale a
#: prior width would use)
PRE_REMOVED_USES = ("prior_width_input",)
