"""Release definition: which product versions enter the first L3 release, how, and why others do not.

Everything here was fixed before the first build of ``release_20261012`` and is read, never
computed, by the builder, so two builds from scratch see the same specification (plan 2.8
item 5). Inclusion rule (task and plan 2.1, 2.3, 2.8):

* a product version enters only if its WP's final status is ``passed``, or the WP is
  ``partial`` and this specific sub-product passed its own gate (stated per component);
* its manifest validates, its gate passed and every input/code/output hash reproduces
  (``common.guard.assert_releasable``), and its validation report meets the plan's section-3
  acceptance for what it contains (plan 2.8 item 2);
* the builder obtains every byte through ``common.guard.open_product`` with a declared use
  (and a time window for time-sliced products), so the release never holds a row the guard
  would not grant; release uses are a subset of what the upstream guard grants;
* per-pump and per-meter tables (well numbers, coordinates) are never carried.

The release ships every arm and member of the included products and chooses none (U7).
"""
from __future__ import annotations

RELEASE = "release_20261012"
RELEASE_DIR = "results/dataprep/model_ready"         # repo-relative parent of the release
ORIGIN = "2022-12"
GRID_SOURCE = "results/twin_forward/datum_gate_innov.npz"
GRID_POLYGON = "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json"
GRID_CODE = "hydrophysics/twin/grid.py"
LEDGER_WP = "L3"

# --------------------------------------------------------------------------------- uses
PRE_FORCING_USES = ["forcing_frozen", "model_design", "prior", "design_time_evidence",
                    "validator", "after_the_fact_analysis"]
FULL_FORCING_USES = ["forcing_recorded_arm", "statistical_layer_history", "validator",
                     "after_the_fact_analysis"]
GRID_USES = ["model_design", "forcing_frozen", "forcing_recorded_arm", "initial_conditions",
             "idw_field", "validator", "statistical_layer_history", "after_the_fact_analysis",
             "design_time_evidence"]

SCORING_NOTE_H1A = ("scoring_target only for pre-origin folds, held-out sites and secondary "
                    "scores; the primary 2023-26 score and the registered prospective scorer "
                    "keep their registered QC (plan 4.6; heads/L1_clean/v3 card)")

# --------------------------------------------------------------------------------- upstream WPs
WP_STATUS = {
    "C0": "passed (open issues)",
    "P2abc": "passed",
    "P2def": "partial",
    "H1a": "passed (open issues)",
    "H2": "partial (open issues)",
    "S2": "partial",
    "W1": "partial (open issues)",
    "W2W4": "passed",
    "P6": "passed",
}

INCLUSION_BASIS = {
    "pump_electricity/preorigin/v3": (
        "WP C0 passed (open issues). Gate passed (pre-registered C0.census_product.v3: R1, R2, "
        "R4, R5, R6, T5). Verifier open issue: the December-2022 fill sd was labelled "
        "'calibrated' under a rule looser than plan 4.4 (coverage 0.808, A1 gate-3 interval "
        "excludes 0.90); the release labels that sd uncalibrated and does not grant "
        "prior_width_input."),
    "pump_electricity/full/v3": (
        "WP C0 passed (open issues). Gate passed (C0.census_product.v3: R1, R4, R5, R6, T5). "
        "Recorded-forcing arm (U4)."),
    "yunlin_electricity/preorigin/v2": (
        "WP P2abc passed. Gate passed (P2r3b.k1.construction_preorigin, "
        "P2r3b.c1.zero_shipped_yunlin). Covers every in-grid county despite its name. Several "
        "sds are uncalibrated (Changhua and Chiayi Chow-Lin, the December-2022 fill, the "
        "synchronized-zero fill), so the release does not grant prior_width_input."),
    "yunlin_electricity/full/v2": (
        "WP P2abc passed. Gate passed (P2r3c.k1.construction_full, "
        "P2r3b.c1.zero_shipped_yunlin). Recorded-forcing arm (U4)."),
    "census_extension/full/v1": (
        "WP P2def partial; this sub-product passed its gate (P2e.x1 annual RMS Yunlin 0.070, "
        "Changhua 0.054 against 0.10; P2e.k1 construction). Not included from P2def: "
        "census_post2016 (gate failed) and layer_response (evidence product)."),
    "heads/L1_clean/v3": (
        "WP H1a passed (open issues). Gate passed (H1a.qc_acceptance.v2, 7/7 registered "
        "criteria, recomputed on v3). Open issue (events.csv unreadable through the guard's flag "
        "filter) does not concern the carried tables: events.csv is not carried."),
    "heads/L2_filled_preorigin/v4": (
        "WP H2 partial (open issues); this sub-product passed its gate: calibration-target rows "
        "(flags 0-2) by the v1 packaging rule (H2.fresh.d, replicated by H2.fresh3.d: coverage "
        "0.908 [0.885, 0.926], RMSE 0.436 m) and the January-2012 IC field by H2.repair3.v1 "
        "(H2.ic.twin 'confirmed', 6.27 vs 8.42 m, interval [-4.37, -0.08] m). Not included "
        "from H2: the complete gap fill (gate failed in both rounds), so no IDW head field and "
        "no flag 3-4 head fill exists."),
    "heads/L2_filled_full/v2": (
        "WP H2 partial (open issues); this sub-product passed its gate (H2.fresh.d and "
        "H2.fresh3.d under the H2.repair3.v1 packaging rule): daily-bridge component of the "
        "full variant."),
    "deep_benchmarks_printed/v2": (
        "WP S2 partial; this sub-product passed its gate (printed 007001 values two-read "
        "verified 0.975, integrated 007003 values 1.000; >= 0.9). S2 names it the validator "
        "and scoring source for piles. Not included: deep_benchmarks/v2 (feed-based; plan "
        "acceptance failed, 58/61 units within 2 mm), so the 2019-2021 printed months are not "
        "corroborated by the feeds under any tested convention."),
    "et0/v1": (
        "WP W2W4 passed. Gate passed (W2.et0_overlap: T0-T3, completeness >= 99 %; 2022 "
        "overlap max MAE 0.063 mm/d < 0.1)."),
    "surface_water_v4/preorigin/v2": (
        "WP W1 partial (open issues); this sub-product passed its gate and the plan's Jan-May "
        "acceptance (RMSE 0.359/0.339 vs climatology 0.661/0.563 and v3 0.839/1.062 x 10^8 m3, "
        "all four 90 % intervals below zero), weakly: 7 design-time years, Yunlin-vs-climatology "
        "bound -0.002, supporting head check failed in Yunlin layer 1 (-0.0087). W1 is partial "
        "for the forward archive (blocked), not for this product."),
    "surface_water_v4/full/v4": (
        "WP W1 partial (open issues); gate passed by the plan's ship rule (point RMSE below v3 "
        "and climatology); acceptance not met (2/4 intervals), so it ships labelled 'not "
        "distinguishable from climatology on the benchmark', as plan W1 prescribes. 2026 months "
        "flag 6, uncalibrated at every published scale."),
    "public_wells/v2": (
        "WP P6 passed. Gate passed (R1: inventory 775 wells / 8,793.85 x 10^4 m3/yr against the "
        "annex's 774 / 8,793.9, within 5 %; T1, D1, M1). Sensitivity member only."),
}

# --------------------------------------------------------------------------------- components
# kind: copy (byte copy through the guard), repack (re-written npz), table (guarded read_table)
# Each read lists (use, window_end, files); a file has a mode and an optional rename.
COMPONENTS: list[dict] = [
    # ------------------------------------------------------------------ pre-origin variant
    {
        "variant": "preorigin", "name": "electricity_census",
        "title": "Cleaned pump-electricity census on the grid (C0), 2007-01..2022-12",
        "upstream": "pump_electricity/preorigin/v3", "wp": "C0",
        "reads": [{"use": "forcing_frozen", "window_end": None, "files": [
            {"name": "cells_by_class.npz", "mode": "copy", "kind": "grid"},
            {"name": "cells_total.npz", "mode": "copy", "kind": "grid"},
            {"name": "county_class_monthly.parquet", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": PRE_FORCING_USES,
        "fit_window_end": "inherit",
        "personal_data": ("none (cell-level and county-level aggregates; the upstream per-pump "
                          "arrays and meter tables are not carried)"),
        "sd_status": ("sd only on flag-4 (December-2022 fill of Dec-Jan-paired meters) and "
                      "flag-9 aggregates; UNCALIBRATED (C0 verifier: class x cell coverage "
                      "0.808, A1 gate-3 interval [0.762, 0.847] excludes 0.90)"),
        "notes": [
            "kWh/month in model units (cleaned census, pumping.clean_census cap 1.0); arrays "
            "(C, A, T) by class and (K, C, A, T) by county; cells_total.npz holds all classes "
            "and the domestic-excluded sensitivity member (value_nodomestic).",
            "Arrays carry every flag: consumers apply flags themselves (forcing_frozen: 0-5). "
            "Flag 4 = December-2022 fill; flag 9 = aggregate whose missing readings exceed 10 % "
            "of its installed kept HP; flag 8 = missing.",
            "Monthly values of bimonthly meters are TPC's even split: no monthly-disaggregation "
            "test may score against them.",
            "Pre-origin uses run through the two inherited exemptions (2026 census attribute "
            "table; undetectable December-2022 halves).",
        ],
    },
    {
        "variant": "preorigin", "name": "electricity_billing_arms",
        "title": "Billing arms and members of the census forcing (P2abc), 2007-01..2022-12",
        "upstream": "yunlin_electricity/preorigin/v2", "wp": "P2abc",
        "reads": [{"use": "forcing_frozen", "window_end": None, "files": [
            {"name": "kwh_by_class_recorded.npz", "mode": "copy", "kind": "grid"},
            {"name": "kwh_by_class_chowlin.npz", "mode": "copy", "kind": "grid"},
            {"name": "kwh_by_class_recorded_syncfill.npz", "mode": "copy", "kind": "grid"},
            {"name": "kwh_by_class_chowlin_syncfill.npz", "mode": "copy", "kind": "grid"},
            {"name": "county_class_monthly.parquet", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": PRE_FORCING_USES,
        "fit_window_end": "inherit",
        "personal_data": ("none (cell-level and county-level aggregates; per-pump arrays, "
                          "meter phase, zero-class and sync-fill tables are not carried)"),
        "sd_status": ("Yunlin Chow-Lin sd calibrated (P2abc); Changhua and Chiayi Chow-Lin, "
                      "the December-2022 fill and the synchronized-zero fill sds UNCALIBRATED"),
        "notes": [
            "Four arms: recorded (equal to electricity_census to 3e-11 kWh, same flags), "
            "chowlin (bimonthly irrigation bills re-split by a fan-wide profile, flag 3), and "
            "the sensitivity members recorded_syncfill and chowlin_syncfill (synchronized-zero "
            "meter-months filled, flag 4).",
            "The arm and member choice must be declared before any result is read, on "
            "pre-origin evidence only (U7). Chow-Lin is a timing sensitivity, not a proven "
            "improvement at cell scale.",
        ],
    },
    {
        "variant": "preorigin", "name": "surface_water",
        "title": "Monthly Jiji-weir canal deliveries v4 on the grid (W1), 2012-01..2022-12",
        "upstream": "surface_water_v4/preorigin/v2", "wp": "W1",
        "reads": [{"use": "forcing_frozen", "window_end": None, "files": [
            {"name": "grid_sw_v4.npz", "mode": "repack", "kind": "grid",
             "repack": "dates_from_district_monthly"},
            {"name": "district_draws.npz", "mode": "repack", "kind": "members",
             "repack": "dates_from_district_monthly"},
            {"name": "district_monthly.csv", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": ["forcing_frozen", "model_design", "design_time_evidence",
                         "after_the_fact_analysis"],
        "fit_window_end": "inherit",
        "personal_data": "none",
        "sd_status": ("monthly district sd and members UNCALIBRATED (no monthly observation); "
                      "the Jan-May aggregate of the draws passes the coverage rule only weakly "
                      "(11/14, 7 design-time years); structural-zero months have sd 0"),
        "notes": [
            "m/day of the cell (mean daily rate of the month), delivered canal water, not "
            "recharge. Grid allocation from LUIMAP 2024 (inherited exemption); district series "
            "and draws are not affected by it.",
            "A grid member = share_d (x) district draw x 1e8 / dx^2 / days of the month "
            "(share_changhua, share_yunlin in grid_sw_v4.npz; draws in district_draws.npz). The "
            "draws are centred on member_mean, not on the Denton value: use members as members.",
            "Upstream 'dates' arrays are pickled object arrays; they were not unpickled. The "
            "release writes dates (YYYY-MM) reconstructed from district_monthly.csv and checks "
            "their count against the arrays.",
            "Evidence is weak: Jan-May acceptance on 7 design-time years; the supporting head "
            "check failed in Yunlin layer 1 (-0.0087). Physical A/B test is I1 rung R3.",
        ],
    },
    {
        "variant": "preorigin", "name": "public_wells_member",
        "title": "Public-supply deep-well closure-ramp sensitivity member (P6), 2007-01..2022-12",
        "upstream": "public_wells/v2", "wp": "P6",
        "reads": [{"use": "forcing_frozen", "window_end": None, "files": [
            {"name": "sensitivity_member.npz", "mode": "repack", "kind": "grid",
             "repack": "member_value_flag"}]}],
        "allowed_uses": ["forcing_frozen", "model_design", "prior", "design_time_evidence",
                         "validator", "after_the_fact_analysis"],
        "fit_window_end": "inherit",
        "personal_data": "none (cohort weights on cells; no well names or coordinates)",
        "sd_status": ("no sd (NaN): the product represents uncertainty by its variants (plan "
                      "4.8 'an sd or members'); uncalibrated"),
        "notes": [
            "SENSITIVITY MEMBER ONLY: it never enters the primary model (plan P6 step 3).",
            "m3/day of abstraction (positive = abstraction; pass -rate as the engine's "
            "q_extra), physical (WRA) volumes, (4 twin layers, A, T); model units = physical / "
            "2.58. value = rate_phys_m3d__primary, flag 5 everywhere; every variant and the "
            "components pending_tw_changhua and extra_tuku_huwei_2013 are carried unchanged.",
            "Volumes are agency-credited cuts, not metered pumping; locations are proxies "
            "(overlap with the primary 0.19-0.93 across location variants): run the variants.",
        ],
        "accepted_exceptions": [
            {"check": "validate_grid:sensitivity_member.npz",
             "contains": "non-observed cells without finite sd",
             "reason": ("public_wells/v2 gives its member no sd by design (flag 5; 'no sd "
                        "(uncalibrated); the variants span the assumptions'); plan 4.8 lets a "
                        "forcing product carry 'an sd or members'. The release writes sd as NaN "
                        "rather than invent one and ships every variant and component.")}],
    },
    {
        "variant": "preorigin", "name": "et0",
        "title": "Daily reference ET0 at the twin's 61 points (W2), 2012-01-01..2022-12-31",
        "upstream": "et0/v1", "wp": "W2W4",
        "reads": [{"use": "forcing_frozen", "window_end": None, "files": [
            {"name": "et0_daily_2012_2022_preorigin.npz", "mode": "copy", "kind": "points"}]}],
        "allowed_uses": ["forcing_frozen", "statistical_layer_history",
                         "after_the_fact_analysis"],
        "fit_window_end": "2022-12",
        "personal_data": "none (reanalysis at public well points)",
        "sd_status": "no sd (reanalysis values, flag 0)",
        "notes": [
            "Point forcing at its native support: the twin IDWs the monthly mean of these 61 "
            "series to the cells inside its recharge builder (calibrate_flow."
            "_load_recharge_field), so the file keeps the keys that loader reads (et0, dates, "
            "well_ids). Identical to the calibration cache results/et/openmeteo_et0_2012_2022.npz "
            "(GMT days).",
        ],
    },
    {
        "variant": "preorigin", "name": "heads_observed",
        "title": "Cleaned monthly heads, observed months (H1a L1 v3), 2012-01..2022-12",
        "upstream": "heads/L1_clean/v3", "wp": "H1a",
        "reads": [
            {"use": "design_time_evidence", "window_end": "2022-12", "files": [
                {"name": "heads_monthly.parquet", "mode": "table", "kind": "heads_monthly"}]},
            {"use": "model_design", "window_end": None, "files": [
                {"name": "station_meta_preorigin.parquet", "mode": "table",
                 "kind": "stations", "rename": "stations.parquet"}]}],
        "allowed_uses": ["calibration_target", "datum_fit", "baseline", "initial_conditions",
                         "idw_field", "model_design", "selection", "design_time_evidence",
                         "scoring_target", "validator", "statistical_layer_history",
                         "after_the_fact_analysis"],
        "fit_window_end": "2022-12",
        "personal_data": ("none (public monitoring-station ids, names and coordinates; no "
                          "contact fields)"),
        "sd_status": ("observed months: sd 0 (no fill), measurement plus representativeness "
                      "error in sigma_obs (U2 rho(k), pre-origin values)"),
        "notes": [
            "Rows: the L1 v3 monthly means the guard grants a pre-origin read with window_end "
            "2022-12 (flags 0-1; flag-8 missing and flag-9 suspect months are not carried).",
            "stations.parquet: H1a's pre-origin station table (repaired coordinates, layer "
            "codes) with the active-cell index on the twin grid; GroundHeight is not carried "
            "(unreliable at 159 of 344 stations; H1b's repair waits on W3's gate).",
            SCORING_NOTE_H1A + ".",
        ],
    },
    {
        "variant": "preorigin", "name": "heads_targets",
        "title": "Calibration-target heads with daily-bridged months (H2 L2 v4), 2012-01..2022-12",
        "upstream": "heads/L2_filled_preorigin/v4", "wp": "H2",
        "reads": [{"use": "calibration_target", "window_end": None, "files": [
            {"name": "fill.parquet", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": ["calibration_target"],
        "fit_window_end": "2022-12",
        "personal_data": "none (public monitoring-station ids)",
        "sd_status": ("flag 2 (daily bridge) sd by bridged-day share 0.181 / 0.288 / 0.554 m, "
                      "covered on fresh tests (H2.fresh3.d 0.908); observed rows sd 0"),
        "notes": [
            "Calibration targets: flags 0-2, likelihood variance sigma_obs_m^2 + head_sd_m^2 "
            "(equivalently u2_weight on sigma_obs_m). Never scoring targets, baselines or datum "
            "fits (U1).",
        ],
    },
    {
        "variant": "preorigin", "name": "heads_initial_conditions",
        "title": "January-2012 initial head field, 4 layers (H2 L2 v4, gp_bf)",
        "upstream": "heads/L2_filled_preorigin/v4", "wp": "H2",
        "reads": [{"use": "initial_conditions", "window_end": None, "files": [
            {"name": "ic_field_2012-01.npz", "mode": "repack", "kind": "grid",
             "repack": "ic_field"}]}],
        "allowed_uses": ["initial_conditions"],
        "fit_window_end": "2022-12",
        "personal_data": "none (public monitoring-station ids)",
        "sd_status": "gp_bf_sd: native 90 % coverage 0.905 in the registered test H2.ic.twin",
        "notes": [
            "value (L=4, A, T=1) = gp_bf, sd = gp_bf_sd, flag 3 (field from other sites) on "
            "every cell; the upstream keys are carried unchanged.",
            "The registered gain over the twin's construction is narrow (6.27 vs 8.42 m; 90 % "
            "interval [-4.37, -0.08] m) and comes from fewer large errors; at late-starting "
            "wells design-time evidence favours the twin's construction (3.79 vs 4.33 m). The "
            "12-month burn-in stays.",
        ],
    },
    {
        "variant": "preorigin", "name": "piles_printed",
        "title": "Printed deep-benchmark (pile) compression, monthly (S2), to 2022-12",
        "upstream": "deep_benchmarks_printed/v2", "wp": "S2",
        "reads": [{"use": "design_time_evidence", "window_end": "2022-12", "files": [
            {"name": "printed_monthly.parquet", "mode": "table", "kind": "piles"}]}],
        "allowed_uses": ["calibration_target", "datum_fit", "baseline", "design_time_evidence",
                         "validator", "scoring_target", "after_the_fact_analysis"],
        "fit_window_end": "2022-12",
        "personal_data": "none (printed station names and depths of public benchmarks)",
        "sd_status": ("rounding error of one printed decimal only; UNCALIBRATED (its nominal "
                      "95 % interval covers 0.80 of re-printing differences)"),
        "notes": [
            "Monthly compression in mm (one row per sensor-month, primary printing): use "
            "increments, not levels.",
            "2019-2021 printed months are not reproduced by the feeds under any tested "
            "convention (S2): treat them as uncertain by several mm. 21 pre-origin "
            "sensor-months are known only from a 2023 re-printing (vintage_month).",
        ],
    },
    # ------------------------------------------------------------------ full variant
    {
        "variant": "full", "name": "electricity_census",
        "title": "Cleaned pump-electricity census on the grid (C0), 2007-01..2026-07",
        "upstream": "pump_electricity/full/v3", "wp": "C0",
        "reads": [{"use": "forcing_recorded_arm", "window_end": None, "files": [
            {"name": "cells_by_class.npz", "mode": "copy", "kind": "grid"},
            {"name": "cells_by_class_evensplit.npz", "mode": "copy", "kind": "grid"},
            {"name": "cells_total.npz", "mode": "copy", "kind": "grid"},
            {"name": "county_class_monthly.parquet", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": FULL_FORCING_USES,
        "fit_window_end": "inherit",
        "personal_data": ("none (cell-level and county-level aggregates; per-pump arrays not "
                          "carried)"),
        "sd_status": "sd only on flag-9 aggregates (indicator); no calibrated sd",
        "notes": [
            "Recorded-forcing diagnostic arm only (U4). 2026-07 is a partial month (flag 8/9). "
            "From 2025-07, 12-33 % of bimonthly pairs post the whole bill to the second month: "
            "cells_by_class_evensplit.npz re-splits them evenly (flag 1).",
        ],
    },
    {
        "variant": "full", "name": "electricity_billing_arms",
        "title": "Billing arms and members of the census forcing (P2abc), 2007-01..2026-07",
        "upstream": "yunlin_electricity/full/v2", "wp": "P2abc",
        "reads": [{"use": "forcing_recorded_arm", "window_end": None, "files": [
            {"name": "kwh_by_class_recorded.npz", "mode": "copy", "kind": "grid"},
            {"name": "kwh_by_class_chowlin.npz", "mode": "copy", "kind": "grid"},
            {"name": "kwh_by_class_recorded_syncfill.npz", "mode": "copy", "kind": "grid"},
            {"name": "kwh_by_class_chowlin_syncfill.npz", "mode": "copy", "kind": "grid"},
            {"name": "county_class_monthly.parquet", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": FULL_FORCING_USES,
        "fit_window_end": "inherit",
        "personal_data": ("none (aggregates; per-pump arrays and the per-meter July-2026 "
                          "imputation table are not carried)"),
        "sd_status": ("Yunlin Chow-Lin sd calibrated; Changhua and Chiayi Chow-Lin, July-2026 "
                      "imputation and synchronized-zero fill sds UNCALIBRATED"),
        "notes": [
            "Recorded-forcing diagnostic arm only (U4). M0 meters' July 2025 is flag 9 (650 of "
            "2148 total cells), values kept. The per-meter July-2026 imputation table is not "
            "carried (per-meter rows).",
        ],
    },
    {
        "variant": "full", "name": "electricity_extension",
        "title": "Census extension to 2026-07..2026-08 from TPC farm-sector sales (P2def)",
        "upstream": "census_extension/full/v1", "wp": "P2def",
        "reads": [{"use": "forcing_recorded_arm", "window_end": None, "files": [
            {"name": "census_extension.npz", "mode": "copy", "kind": "grid"},
            {"name": "county_extension.csv", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": ["forcing_recorded_arm", "after_the_fact_analysis"],
        "fit_window_end": "inherit",
        "personal_data": "none",
        "sd_status": "flag 5, sd UNCALIBRATED (monthly log RMS of the backtest)",
        "notes": [
            "Recorded-forcing arm only (U4). The Yunlin extension's recent mean error is +8.8 % "
            "(expect 5-10 % high). For 2026-07 P2def prefers P2abc's per-meter census "
            "imputation (6.2 % lower in Yunlin), which this release does not carry at cell "
            "level; use the extension for 2026-08.",
        ],
    },
    {
        "variant": "full", "name": "surface_water",
        "title": "Monthly canal deliveries v4 on the grid (W1), 2012-01..2026-09",
        "upstream": "surface_water_v4/full/v4", "wp": "W1",
        "reads": [{"use": "forcing_recorded_arm", "window_end": None, "files": [
            {"name": "grid_sw_v4.npz", "mode": "repack", "kind": "grid",
             "repack": "dates_from_district_monthly"},
            {"name": "district_draws.npz", "mode": "repack", "kind": "members",
             "repack": "dates_from_district_monthly"},
            {"name": "district_monthly.csv", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": ["forcing_recorded_arm", "statistical_layer_history",
                         "after_the_fact_analysis"],
        "fit_window_end": "inherit",
        "personal_data": "none",
        "sd_status": "every interval uncalibrated; 2026 intervals uncalibrated at every scale",
        "notes": [
            "Recorded-forcing arm only (U4). 2012-2025 flag 5 'not distinguishable from "
            "climatology on the benchmark'; 2026-01..09 flag 6 (rain-driven extension; never "
            "calibration or scoring). Yunlin 2026 is probably biased high (post hoc: 2020-2025 "
            "annual residuals all negative).",
            "Upstream 'dates' arrays are pickled; the release writes reconstructed YYYY-MM dates.",
        ],
    },
    {
        "variant": "full", "name": "et0",
        "title": "Daily reference ET0 at the twin's 61 points (W2), 2012-01-01..2026-09-30",
        "upstream": "et0/v1", "wp": "W2W4",
        "reads": [{"use": "forcing_recorded_arm", "window_end": None, "files": [
            {"name": "et0_daily_2012_2026.npz", "mode": "copy", "kind": "points"}]}],
        "allowed_uses": ["forcing_recorded_arm", "statistical_layer_history",
                         "after_the_fact_analysis"],
        "fit_window_end": "inherit",
        "personal_data": "none (reanalysis at public well points)",
        "sd_status": "no sd",
        "notes": [
            "Recorded-forcing arm only (U4). Primary 'et0' is level-matched to the cache from "
            "2023-01 (flag 1); 'et0_raw_splice' is the free sensitivity member. Days "
            "2026-07-01..09-30 are provisional. 2023-26 ET0 runs +0.249 mm/d above the "
            "2012-2021 climatology.",
        ],
    },
    {
        "variant": "full", "name": "heads_observed",
        "title": "Cleaned monthly heads, observed months (H1a L1 v3), 2012-01..2026-09",
        "upstream": "heads/L1_clean/v3", "wp": "H1a",
        "reads": [{"use": "validator", "window_end": None, "files": [
            {"name": "heads_monthly.parquet", "mode": "table", "kind": "heads_monthly"},
            {"name": "station_meta_clean.parquet", "mode": "table", "kind": "stations",
             "rename": "stations.parquet"}]}],
        "allowed_uses": ["scoring_target", "validator", "statistical_layer_history",
                         "after_the_fact_analysis"],
        "fit_window_end": "inherit",
        "personal_data": ("none (public monitoring-station ids, names and coordinates; no "
                          "contact fields)"),
        "sd_status": ("observed months sd 0 (error in sigma_obs); the 2 flag-3 months carry no "
                      "sd upstream (NaN)"),
        "notes": [
            "Rows: the L1 v3 monthly means the guard grants to 'validator' (flags 0-6; flag 8 "
            "and 9 not carried). Includes the 2 flag-3 months of 09100211 (2024-09, 2024-10; "
            "step correction corroborated by neighbour regression only, sd not given upstream).",
            SCORING_NOTE_H1A + ".",
        ],
        "accepted_exceptions": [
            {"check": "validate_long_table:heads_monthly.parquet",
             "contains": "non-observed rows (flags 2-6) without sd",
             "reason": ("heads/L1_clean/v3 is an L1 product and gives no sd for its 2 flag-3 "
                        "months (09100211, 2024-09 and 2024-10); the release does not invent "
                        "one. Flag 3 never enters scoring targets (U1); these rows serve the "
                        "validator and after-the-fact uses only.")}],
    },
    {
        "variant": "full", "name": "heads_filled",
        "title": "Monthly heads with daily-bridged months (H2 L2 full v2), 2012-01..2026-09",
        "upstream": "heads/L2_filled_full/v2", "wp": "H2",
        "reads": [{"use": "statistical_layer_history", "window_end": None, "files": [
            {"name": "fill.parquet", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": ["statistical_layer_history", "after_the_fact_analysis"],
        "fit_window_end": "inherit",
        "personal_data": "none (public monitoring-station ids)",
        "sd_status": "flag 2 sd by bridged-day share (covered on fresh tests)",
        "notes": [
            "Statistical-layer history across the Feb-Mar 2026 outage (287 bridged model-well "
            "months) and after-the-fact analysis only: its bridge uses 2023-26 deviation curves.",
        ],
    },
    {
        "variant": "full", "name": "piles_printed",
        "title": "Printed deep-benchmark (pile) compression, monthly (S2), 2019-05..2026-01",
        "upstream": "deep_benchmarks_printed/v2", "wp": "S2",
        "reads": [{"use": "validator", "window_end": None, "files": [
            {"name": "printed_monthly.parquet", "mode": "copy", "kind": "table"},
            {"name": "integrated_monthly.parquet", "mode": "copy", "kind": "table"}]}],
        "allowed_uses": ["validator", "scoring_target", "after_the_fact_analysis"],
        "fit_window_end": "inherit",
        "personal_data": "none (printed station names and depths of public benchmarks)",
        "sd_status": "rounding only, UNCALIBRATED",
        "notes": [
            "Validator and scoring source for twin subsidence 2023-26 (never fitted, never in a "
            "gate or choice; U7). printed_monthly (007001, to 2025-11) and integrated_monthly "
            "(007003, 2024-01..2026-01; includes the Pingtung pile DLIO, outside the domain).",
        ],
    },
]

# --------------------------------------------------------------------------------- exclusions
# Every product version found under results/dataprep at design time that is not a component,
# plus WP-level exclusions without a product. 'category' is one of: gate_failed, superseded,
# acceptance_failed, not_model_data, privacy, phase1, not_a_product.
EXCLUDED: list[dict] = [
    {"product": "county_electricity/census_check", "versions": ["v1", "v2"], "wp": "P1",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate false (construction check C9: one irrigation pump missing from AMP_V2); "
               "after_the_fact_analysis only; T10/T11 are after-the-fact checks; v1 superseded "
               "by v2"},
    {"product": "county_electricity/preorigin", "versions": ["v1", "v2"], "wp": "P1",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed (T1_BV1: Yunlin RMS 5.19 %, bias -5.04 %; T8 coverage 0.826/0.840); "
               "superseded as a Changhua forcing source by C0; v1 superseded by v2"},
    {"product": "county_electricity/full", "versions": ["v1"], "wp": "P1",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed (T8)"},
    {"product": "county_electricity/tpc_observed", "versions": ["v1"], "wp": "P1",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed on G3 (4 of 1024 Changhua/Yunlin rows); validator use refused"},
    {"product": "census_post2016/preorigin", "versions": ["v1"], "wp": "P2def",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed: registered P2d back-test, Yunlin median township count error 0.579 "
               "(threshold 0.30) and in-grid kWh +30.4 % (threshold 20 %); Changhua count error "
               "0.332; no post-2016 pump layer is in the release"},
    {"product": "census_post2016/full", "versions": ["v1"], "wp": "P2def",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed (same P2d back-test)"},
    {"product": "layer_response", "versions": ["v1"], "wp": "P2def",
     "wp_final_status": "partial", "category": "not_model_data",
     "reason": "evidence product with no acceptance threshold of its own (gate = the declared "
               "analysis ran); its model-side ratios rest on the provisional batch-8 cwa theta; "
               "it is the acceptance evidence of I1 rung R4, read in place"},
    {"product": "changhua_abstraction/engine", "versions": ["v1", "v2"], "wp": "P40",
     "wp_final_status": "passed", "category": "not_model_data",
     "reason": "the P4.0 linear-response engine (kernels, operator fields; layer L3 by its own "
               "manifest): a model component built on the batch-8 theta, not data; "
               "assert_releasable refuses layer L3; v1 superseded by v2"},
    {"product": "changhua_forcing_diagnostics/preorigin", "versions": ["v1", "v2"], "wp": "P4m",
     "wp_final_status": "passed (open issues)", "category": "not_model_data",
     "reason": "diagnostic evidence (layer L3): in-sample design-time evidence on the cwa theta, "
               "not skill; allowed uses model_design, design_time_evidence, "
               "after_the_fact_analysis; not a data product; v1 superseded by v2"},
    {"product": "changhua_forcing_diagnostics/recorded_arm", "versions": ["v1"], "wp": "P4m",
     "wp_final_status": "passed (open issues)", "category": "not_model_data",
     "reason": "after-the-fact diagnostics only (layer L3); not a data product"},
    {"product": "heads/L1_clean", "versions": ["v1", "v2"], "wp": "H1a",
     "wp_final_status": "passed (open issues)", "category": "superseded",
     "reason": "v1 gate failed (spurious step corrections); v2 is the registered build kept for "
               "provenance only (its 8 co-located step corrections unsupported); both superseded "
               "by v3"},
    {"product": "heads/L1_station_ground", "versions": ["v1", "v2"], "wp": "H1b",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate false: built on ground_elevation/v3, whose gate failed (W3); waits for the "
               "user's decision on W3"},
    {"product": "heads/L2_filled_preorigin", "versions": ["v1", "v3"], "wp": "H2",
     "wp_final_status": "partial (open issues)", "category": "gate_failed",
     "reason": "complete gap fills; gates failed (12/14 and 11/14 strata after the family-E "
               "correction); no IDW head field or flag 3-4 fill can enter a model"},
    {"product": "heads/L2_filled_preorigin", "versions": ["v2"], "wp": "H2",
     "wp_final_status": "partial (open issues)", "category": "superseded",
     "reason": "superseded by v4 (same rows; card and allowed uses corrected)"},
    {"product": "heads/L2_filled_full", "versions": ["v1"], "wp": "H2",
     "wp_final_status": "partial (open issues)", "category": "gate_failed",
     "reason": "complete full fill; gate failed"},
    {"product": "heads/recovered_wells", "versions": ["v3"], "wp": "H3",
     "wp_final_status": "partial", "category": "acceptance_failed",
     "reason": "plan H3 acceptance part 1 FAILED: 3 of 15 admitted wells pass H3.d_wells (plan: "
               "each recovered well passes the H2 coverage strata); gate.plan_acceptance_passed "
               "is false and the manifest gate passes only by H3's registered release rule. Plan "
               "2.8 item 2 requires a release product's validation report to meet its section-3 "
               "acceptance. I1 rung R5 (user decision D9: separate rung) can still open "
               "heads/recovered_wells/v3 through the guard"},
    {"product": "heads/recovered_wells", "versions": ["v1", "v2"], "wp": "H3",
     "wp_final_status": "partial", "category": "superseded",
     "reason": "superseded by v3"},
    {"product": "ground_elevation", "versions": ["v1", "v2", "v3"], "wp": "W3",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed: median held-out bias -0.842 m (90 % -0.908..-0.783) against "
               "|bias| <= 0.30 m; the twin keeps SRTM; v1 and v2 also fail the checker"},
    {"product": "gnss_clean/preorigin", "versions": ["v1", "v2"], "wp": "S3",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed twice (v2: A1 recovery 0.863 WRA / 0.863 NGL against 0.90; A2 same "
               "sign 0.50); v1's station set is selection-contaminated"},
    {"product": "gnss_clean/full", "versions": ["v1", "v2"], "wp": "S3",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed (as the pre-origin variant)"},
    {"product": "insar_069A", "versions": ["v1", "v2"], "wp": "S4b",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed: G1 13.2 mm against 10 mm on run P"},
    {"product": "insar_licsar_subsets", "versions": ["v1"], "wp": "S4a",
     "wp_final_status": "partial", "category": "not_model_data",
     "reason": "range-read LiCSAR interferogram subsets (download-integrity gate): inputs of "
               "insar_069A, whose gate failed; not a deformation product"},
    {"product": "rings_bridged", "versions": ["v1", "v2"], "wp": "S1a",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed: (e) worst window 2.79 cm against 1.5 cm; (a) power 0.721 against "
               "0.8"},
    {"product": "deep_benchmarks", "versions": ["v1", "v2"], "wp": "S2",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "feed-based series; plan acceptance failed (A: 58/61 units within 2 mm, max "
               "3.33 mm); v1 superseded by v2"},
    {"product": "deep_benchmarks_printed", "versions": ["v1"], "wp": "S2",
     "wp_final_status": "partial", "category": "superseded",
     "reason": "superseded by v2 (one primary printing per sensor-month)"},
    {"product": "subsidence_fused", "versions": ["v1", "v2", "v3"], "wp": "S5",
     "wp_final_status": "partial", "category": "gate_failed",
     "reason": "gate failed: T1 ratio 1.042 with coverage 0.631; T2 coverage 0.940"},
    {"product": "subgrid_operator", "versions": ["v1", "v2"], "wp": "H4",
     "wp_final_status": "failed", "category": "gate_failed",
     "reason": "gate failed: F2 (shuffled-position placebo p 0.090) and C2 (site-held-out Q4 "
               "excess +1.11 m2)"},
    {"product": "head_signatures", "versions": ["v1"], "wp": "P5",
     "wp_final_status": "passed", "category": "not_model_data",
     "reason": "validator product (rank and timing checks of model pumping; never a prior, "
               "forcing or target); read in place by I1; outside the L3 scope of grid forcing "
               "arrays and observation series"},
    {"product": "natural_experiments", "versions": ["v1"], "wp": "N1",
     "wp_final_status": "passed", "category": "not_model_data",
     "reason": "validator table of I1's registered N1 response tests, read in place; its "
               "pre-origin slice carries facts published in 2023 (2022 drought-well inventory) "
               "that need U3 exemptions for model use"},
    {"product": "rain_check", "versions": ["v1"], "wp": "W2W4",
     "wp_final_status": "passed", "category": "not_model_data",
     "reason": "report only: kriging not adopted (fan RMSE ratio 0.989 against 0.90; worst "
               "gauge 1.83 against 1.20), so no rainfall field exists; the twin keeps masked IDW "
               "on its own rain inputs"},
    {"product": "validation_units", "versions": ["v1", "v2", "v3"], "wp": "A1",
     "wp_final_status": "passed", "category": "not_model_data",
     "reason": "harness registry of design-time and fresh validation units (selection, "
               "validator), opened in place by the harness; v1 and v2 superseded"},
    {"product": "public_wells", "versions": ["v1"], "wp": "P6",
     "wp_final_status": "passed", "category": "superseded",
     "reason": "superseded by v2 (card error; data byte-identical; v1 fails the checker on its "
               "build code hash)"},
    {"product": "pump_electricity/census_L1", "versions": ["v1", "v2", "v3"], "wp": "C0",
     "wp_final_status": "passed (open issues)", "category": "privacy",
     "reason": "per-pump L1 census table (pump coordinates and well numbers; personal_data "
               "declared); the release carries the cell-level arrays built from v3; v1 and v2 "
               "superseded"},
    {"product": "pump_electricity/preorigin", "versions": ["v1", "v2"], "wp": "C0",
     "wp_final_status": "passed (open issues)", "category": "superseded",
     "reason": "v1 superseded and leaks: its 2022-12 holds half of bills read in January 2023; "
               "v2 gate failed (an R4 check defect and a T5 product defect)"},
    {"product": "pump_electricity/full", "versions": ["v1", "v2"], "wp": "C0",
     "wp_final_status": "passed (open issues)", "category": "superseded",
     "reason": "v1 superseded by v3; v2 gate failed"},
    {"product": "pump_electricity/model_census", "versions": ["v1"], "wp": "I1",
     "wp_final_status": "Phase 1 (not a Phase-0 WP)", "category": "phase1",
     "reason": "built by dataprep_staging/i1/build_model_census.py for a Phase-1 rung, not by "
               "a Phase-0 WP; reads kWh to 2023-01 and keeps the paired meters' December-2022 "
               "half bills (its exemption R1.model_census.dec2022_halfbill), unlike C0 v3's "
               "pre-origin fill; the release wires nothing for Phase 1"},
    {"product": "surface_water_v4/model_npz", "versions": ["v1"], "wp": "I1",
     "wp_final_status": "Phase 1 (not a Phase-0 WP)", "category": "phase1",
     "reason": "layer-L3 model input built by dataprep_staging/i1/build_sw_v4_model_npz.py for "
               "calibrate_flow; not a Phase-0 WP product"},
    {"product": "zone_county_map", "versions": ["v1", "v2"], "wp": "I1",
     "wp_final_status": "Phase 1 (not a Phase-0 WP)", "category": "phase1",
     "reason": "built with code in a Phase-1 worktree (hydrophysics/twin/zone_county.py outside "
               "this checkout); not a Phase-0 WP product; v1 superseded (polygon shifted about "
               "850 m)"},
    {"product": "surface_water_v4/preorigin", "versions": ["v1"], "wp": "W1",
     "wp_final_status": "partial (open issues)", "category": "superseded",
     "reason": "superseded by v2 (same values; interval labels; prior_width_input forbidden)"},
    {"product": "surface_water_v4/full", "versions": ["v1", "v2", "v3"], "wp": "W1",
     "wp_final_status": "partial (open issues)", "category": "superseded",
     "reason": "superseded by v4 (v2's 2026 sd and draws must not be used)"},
    {"product": "yunlin_electricity/preorigin", "versions": ["v1"], "wp": "P2abc",
     "wp_final_status": "passed", "category": "superseded",
     "reason": "superseded by v2 (unvalidated zero table)"},
    {"product": "yunlin_electricity/full", "versions": ["v1"], "wp": "P2abc",
     "wp_final_status": "passed", "category": "superseded",
     "reason": "superseded by v2 (unvalidated zero table; no M0 July-2025 flag)"},
    {"product": "l0_registry", "versions": ["v1"], "wp": "A0",
     "wp_final_status": "passed", "category": "not_a_product",
     "reason": "L0 registry (layer 'registry'): hashes of pre-convention L0 files, every use "
               "forbidden"},
    {"product": "subsidence_fused", "versions": ["v9999"], "wp": "S5",
     "wp_final_status": "partial", "category": "not_a_product",
     "reason": "synthetic dry run in reports/S5_evidence/repair/dryrun_product_c, not a "
               "product"},
    {"product": "(WP P3, no product)", "versions": [], "wp": "P3",
     "wp_final_status": "passed", "category": "not_a_product",
     "reason": "methods-validation report (land-use and crop transfer tested against the "
               "Changhua census); no L2 product; evidence under reports/P3_evidence"},
    {"product": "_pilots/*", "versions": ["2026-10-12.r2"], "wp": "A0",
     "wp_final_status": "passed", "category": "not_a_product",
     "reason": "pilot evidence (layer 'pilot'): design-time evidence only, never releasable"},
]

# --------------------------------------------------------------------------------- card text
CARD_LIMITS = {
    ("preorigin", "electricity_census"): (
        "December-2022 fill (flag 4) sd uncalibrated; bimonthly meters are TPC even-split "
        "halves; Changhua's domestic class (23 % of in-grid energy) may be household load, so "
        "value_nodomestic is the sensitivity member; two inherited exemptions."),
    ("preorigin", "electricity_billing_arms"): (
        "recorded arm equals electricity_census (to 3e-11 kWh); Chow-Lin (flag 3) is a timing "
        "sensitivity; the syncfill members (flag 4) bound the unexplained synchronized zeros; "
        "choose the arm before results (U7)."),
    ("preorigin", "surface_water"): (
        "weak evidence (7 design-time years; Yunlin layer-1 head check failed); monthly sd and "
        "draws uncalibrated; LUIMAP 2024 allocation (exemption); m/day of delivered water, not "
        "recharge."),
    ("preorigin", "public_wells_member"): (
        "sensitivity member only, never the primary model; agency-credited cuts at proxy "
        "locations; no sd, so run the variants; physical m3/day (model units = / 2.58)."),
    ("preorigin", "et0"): (
        "61 native points (the twin IDWs them to cells); identical to the calibration cache."),
    ("preorigin", "heads_observed"): (
        "flags 0-1 only (flag 8 and 9 months not carried); upstream proxy: about 29 % of "
        "flag-9 episode days may be good data; GroundHeight not carried; scoring_target only "
        "for pre-origin folds, held-out sites and secondary scores."),
    ("preorigin", "heads_targets"): (
        "calibration targets only (U2); bridged months (flag 2) carry sd 0.18-0.55 m by "
        "bridged share; use sigma_obs_m as the noise of every observed row."),
    ("preorigin", "heads_initial_conditions"): (
        "narrow registered gain over the twin's construction (6.27 vs 8.42 m); weaker at "
        "late-starting wells (design-time evidence); keep the 12-month burn-in."),
    ("preorigin", "piles_printed"): (
        "increments, not levels; sd is rounding only (uncalibrated); 2019-2021 printed months "
        "are not corroborated by the feeds."),
    ("full", "electricity_census"): (
        "recorded arm only (U4); 2026-07 partial (flags 8/9); second-month postings from "
        "2025-07 (evensplit alternative, flag 1)."),
    ("full", "electricity_billing_arms"): (
        "recorded arm only (U4); M0 July 2025 flagged 9 in 650 cells; the per-meter July-2026 "
        "imputation is not carried."),
    ("full", "electricity_extension"): (
        "recorded arm only (U4); flag 5, sd uncalibrated; Yunlin probably 5-10 % high; P2def "
        "prefers the census imputation for 2026-07."),
    ("full", "surface_water"): (
        "recorded arm only (U4); 'not distinguishable from climatology'; 2026 months flag 6 "
        "and uncalibrated; Yunlin 2026 probably high."),
    ("full", "et0"): (
        "recorded arm only (U4); level-matched from 2023-01 (flag 1), raw splice as the "
        "alternative; 2026-07..09 provisional."),
    ("full", "heads_observed"): (
        "scoring_target only for held-out sites and secondary scores (the registered scorer "
        "keeps its QC); 2 flag-3 months without sd."),
    ("full", "heads_filled"): (
        "statistical-layer history and after-the-fact analysis only (the bridge uses 2023-26 "
        "deviation curves)."),
    ("full", "piles_printed"): (
        "validator and scoring source for 2023-26 subsidence; sd rounding only; the "
        "integrated table includes the Pingtung pile DLIO (outside the domain)."),
}

KNOWN_LIMITS_GENERAL = [
    "Subsidence side: only the printed deep piles are model-ready. Rings (S1a), GNSS (S3), "
    "InSAR (S4b), the fused field (S5) and the DTM (W3) failed their gates and are absent; the "
    "twin keeps SRTM ground elevation.",
    "No post-2016 pump layer (P2d gate failed), no public or industrial deep wells in the "
    "census (P6: missing forcing in both counties; only a sensitivity member), no complete head "
    "gap fill, no IDW head field, no recovered wells (H3 acceptance failed).",
    "No included product's values depend on the provisional batch-8 theta (C0 reads the "
    "model's cached forcing only for its R3 comparison; H2 reads the stage-3 recipe, not "
    "theta).",
    "Forcing arrays (.npz) carry every flag; the guard filters flags only in tables, so array "
    "consumers apply the per-use flags of common.uses.FLAGS_BY_USE themselves.",
    "Arm, member and variant choices (recorded vs Chow-Lin, synchronized-zero members, "
    "domestic exclusion, public-well variants, W1 draws) must be fixed before any result is "
    "read, on pre-origin evidence only (U7); the release chooses none.",
    "Most sds are uncalibrated (C0 fill, Changhua and Chiayi Chow-Lin, W1 monthly, P2e "
    "extension, pile rounding); plan 4.8 propagation must not take them at face value.",
]
