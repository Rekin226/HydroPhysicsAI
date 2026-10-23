"""Write the WP P4m repair-round declaration (harness.prereg3) before any repair number is
computed.

    cd dataprep_staging
    python -m p4_changhua.prereg_p4m_repair

``P4m.repair_audit.v1`` fixes, before they are computed, the readings of the repair round
that answers the independent verification of P4m (2026-10-17):

* RA-RC: validity rules for the three independent indicators behind rule T5 (BE sign, E9
  sign agreement, E8 robustness and physical direction), and RD the audit reading of T5 that
  follows from them (post-look: the 2023-25 recorded arm had been read);
* RE: cluster-bootstrap intervals for the AR(1)-GLS secondary of the T3 mean channel;
* RF: whether the amplitude-matched scale s_AM can identify the magnitude of the metered
  series separately from timing or location error;
* RG: E9 as declared in P4m.metered_diagnostics.v1 (site_id, station layer codes 1 and 2) and
  with the vendor rain cache as a sensitivity.

Every head read here is a calibration target of the frozen cwa theta (2012-22, design-time
data), so this is a declaration, as P4m's first two were; looks go to the A1 look log.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg3  # noqa: E402
from harness.prereg import not_applicable as NA  # noqa: E402

V1 = "results/dataprep/changhua_forcing_diagnostics/preorigin/v1/"
H1A = "results/dataprep/heads/L1_clean/v3/"
C0 = "results/dataprep/pump_electricity/preorigin/v3/"

DOC = {
    "id": "P4m.repair_audit.v1",
    "wp": "P4m",
    "title": "P4m repair round: validity audit of the T5 indicators, GLS intervals for the T3 "
             "mean channel, identifiability of the amplitude-matched scale, E9 as declared",
    "kind": "declaration",
    "plan_section": "plan 2.5 U7, 4.1, 4.5, E8, E9, E10; P4m verifier findings 2026-10-17 "
                    "(major 1: T5 indicators; major 2: item (2) interpretation; minor: E9 "
                    "definition, GLS fragility, pilot tide input)",
    "tests": NA("in-sample audit on the theta's own calibration heads 2012-22 and on P4m "
                "preorigin/v1 outputs (design-time data); no fresh unit is scored"),
    "zones": "model zones zone3 as calibrated; counties Changhua / Yunlin as in P4m v1",
    "priors": NA("no prior"),
    "burn_in": "as P4m.metered_diagnostics.v1: 12 months for scores; T3 variants V1-V4 as "
               "declared there",
    "folds": NA("no folds; in-sample"),
    "fresh_units": NA("none (the 158 calibration wells 2012-22 are the theta's fit data and "
                      "were already read by P4m v1)"),
    "gate_scores": NA("no gate: the repair round's readings are reported with the fixed "
                      "interpretations below; numerical replication checks are acceptance "
                      "rows"),
    "primary_model": "frozen cwa theta (sha256 789d0e45...), sparse twin, arms A0/A1 of P4m v1",
    "inputs": [
        V1 + "MANIFEST.json", V1 + "evidence_e8_be_e9.json", V1 + "e8_tides_preorigin.csv",
        V1 + "e9_nested_slopes.csv", V1 + "diagnostics_t1_t3.json",
        V1 + "theta_refit_t2b.json", V1 + "recommendation_preorigin.json",
        V1 + "well_arrays.npz", V1 + "scale_curves.npz",
        H1A + "MANIFEST.json", H1A + "barometric_efficiency.csv",
        H1A + "heads_monthly.parquet", H1A + "station_meta_preorigin.parquet",
        C0 + "MANIFEST.json", C0 + "cells_total.npz", C0 + "cells_by_class.npz",
        "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json",
        "results/dataprep/changhua_abstraction/engine/v2/MANIFEST.json",
        "chou-shui-data/data/rf_timeseries.csv",
        "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv",
        "chou-shui-data/data/rf_stations.csv",
    ],
    "thresholds": {
        "RA_BE_validity": "per county x model layer: n accepted pre-origin epochs, n with "
                          "BE_long <= 0, n with BE_long >= 1, median. A stratum is physically "
                          "valid iff its median lies in (0, 1) and n >= 3 (BE = -A_K; a "
                          "negative BE is impossible). The BE indicator points the same way "
                          "only if both counties' pump-layer (model layer 1) strata are valid "
                          "and BE_YL / BE_CH > 1.25; if the pump layer is invalid, the "
                          "fallback is the ratio of medians over epochs with 0 < BE < 1 pooled "
                          "over layers, used only with n >= 5 per county, same threshold. "
                          "Otherwise the indicator is invalid (does not point either way)",
        "RB_E9_validity": "primary radius 3 km, Changhua pump-layer wells (v1 definition: model "
                          "layer 1; as-declared definition RG: station layer code 2): counts of "
                          "wells with b_obs < 0 and > 0 and with t_obs < -2 and > +2. The "
                          "magnitude comparison is valid only if median b_obs < 0 (the sign of "
                          "the model's median slope) and at least 2/3 of the wells with a "
                          "slope have b_obs < 0; if valid, the indicator points the same way "
                          "if 0 < median b_obs / median b_mod < 0.67. Otherwise it is invalid "
                          "(sign reversal is not evidence of over-response). Yunlin reported "
                          "as context with the same counts",
        "RC_E8_robustness_and_direction": "(a) the declared OLS (log A_M2 on easting, county, "
                                          "layer dummies; v1 numbers); (b) the same OLS "
                                          "without easting; (c) the pump-layer (model layer 1) "
                                          "raw median ratio Changhua / Yunlin; (b) and (c) with "
                                          "the v1 cluster bootstrap (10 km, 2000 draws, seed "
                                          "2027). Robust only if all three point estimates "
                                          "exceed 1 and the 5 % bounds of (a), (b), (c) exceed "
                                          "1. Direction: a larger M2 strain response implies a "
                                          "smaller specific storage (plan E8), i.e. a stronger "
                                          "and faster pumping response, the opposite of the "
                                          "damping that s_theta < 0.67 calls for; so under the "
                                          "physical reading E8 cannot point the same way, and "
                                          "it is recorded as 'not supporting' whether robust "
                                          "or not (plan E8 also forbids freeing a Changhua T "
                                          "multiplier on tidal evidence alone)",
        "RD_T5_audit_reading": "post-look (the 2023-25 recorded arm of P4m v1 was read before "
                               "this declaration): the Changhua-specific multiplier is "
                               "supported only if s_theta < 0.67 AND at least one indicator is "
                               "valid under RA-RC and points the same way. The declared T5 "
                               "rule's mechanical result (fired) stays on record unchanged; "
                               "the audit reading replaces it in the recommendation. If no "
                               "indicator is valid and supporting: no multiplier in R1, no "
                               "R1m rung recommended on P4m evidence",
        "RE_GLS_intervals": "AR(1)-GLS estimator of P4m v1 (gls_free_datum, phi as declared "
                            "per variant) with cluster-bootstrap 90 % intervals on the same "
                            "draws as the OLS (L.boot_indices, 10 km clusters, 2000 draws, "
                            "seed 2027) in V1, V2, V4; V3 added (whitened per-well datum and "
                            "trend projected out; new). Replication: OLS and GLS point "
                            "estimates of V1, V2, V4 equal v1's within 1e-9 and the OLS "
                            "intervals equal v1's within 1e-9. Reading: if the GLS 90 % "
                            "interval of theta_mean lies entirely below 0 in every variant "
                            "V1-V4, the E10 verdict is labelled fragile (the OLS primary covers "
                            "0, the GLS secondary does not) and the level channel is logged "
                            "for plan 7.3 as a possible misspecification; the declared primary "
                            "(OLS) verdict is not changed. Otherwise the verdict is robust to "
                            "the secondary in the variants where GLS covers 0",
        "RF_s_AM_identifiability": "variants of the metered series E_CH (cells x 131 months, "
                                   "2012-02..2022-12): LAG+2 (each cell delayed 2 months, "
                                   "circular), LAG-2 (advanced 2 months, circular), PERM "
                                   "(cell series permuted among the cells with positive mean, "
                                   "numpy default_rng(2027)), NOSEAS (per-cell month-of-year "
                                   "seasonal part removed). For each, P4m v1's scale_family "
                                   "(exact stepwise rollouts on S_GRID + bisection): s_LS "
                                   "(all, Changhua), s_AM, Changhua median correlation and "
                                   "anomaly RMSE at s = 1. Replication: the metered series "
                                   "reproduces v1's s_LS_all and s_AM within 1e-6. Reading: a "
                                   "timing/location variant (LAG+2, LAG-2, PERM) is "
                                   "informative if it lowers the Changhua median correlation "
                                   "at s = 1 by >= 0.02. If every informative variant keeps "
                                   "s_AM within a factor [0.8, 1.25] of the metered s_AM, "
                                   "amplitude matching does not separate magnitude from timing "
                                   "or location error at this correlation, and the v1 "
                                   "statement 'the metered magnitude fits' is withdrawn "
                                   "('consistent with, not identified'). If any informative "
                                   "variant moves s_AM outside that factor, s_AM carries some "
                                   "timing/location information and the statement is "
                                   "weakened to 'consistent with', with the variant numbers",
        "RG_E9_as_declared": "sites = station_meta_preorigin site_id (H1a v3, model_design, no "
                             "window: static pre-origin metadata) among the 158 calibration "
                             "wells, with station layer codes 1 and 2 at the same site (code "
                             "2 = pump layer); otherwise the v1 regression unchanged (H1a v3 "
                             "heads_monthly flags 0-1 2012-01..2022-12, C0 preorigin/v3 "
                             "cells_total value_all, nearest rf gauge with > 90 % complete "
                             "months from chou-shui-data/data/rf_timeseries.csv, which is the "
                             "CWA CODIS rebuild installed 2026-10-11, sha256 42308c8d...); "
                             "sensitivity: the same with the vendor cache "
                             "rf_timeseries_vendor_2012_2022.csv. Reported: the RB counts and "
                             "medians for both definitions and both rain files",
        "other_rules": "Thresholds are not changed after results are seen. Results are "
                       "reported whichever way they fall. New product versions only (v1 is "
                       "never overwritten). Everything here is in-sample design-time "
                       "evidence or an audit of it.",
    },
    "interpretation": "RA-RC decide whether any independent indicator validly supports a "
                      "Changhua-specific response multiplier (RD); RE says whether the T3 "
                      "'no plan 7.3 entry' verdict survives the GLS secondary; RF says whether "
                      "s_AM ~ 1 can be read as heads accepting the metered magnitude; RG "
                      "replaces E9's undeclared site definition with the declared one.",
    "prior_exposure": [
        "Seen before this declaration: every P4m v1 result (products preorigin/v1 and "
        "recorded_arm/v1, report P4m.md), including the 2023-25 recorded-arm scores (T4); the "
        "v1 evidence file (BE county x layer medians including the negative layer-0/1 "
        "medians, E8 county x layer medians and the OLS, E9 slope medians at 2, 3 and 5 km and "
        "layer ratios); the verifier's findings (layer-1 BE medians -0.362 / -0.496; 10 of 13 "
        "Changhua pump-layer wells with positive observed slope at 3 km; E8 raw pump-layer "
        "medians 4.97 vs 5.87 mm; refit Changhua RMSE 1.112 at s = 0.5 vs 1.153 at s = 1; GLS "
        "point estimates -1.8 to -2.0).",
        "Not seen: per-epoch BE sign counts, E8 without easting, E9 under the site_id "
        "definition or with the vendor rain, GLS bootstrap intervals, any scale under the "
        "LAG/PERM/NOSEAS variants.",
    ],
}


def main() -> int:
    path, sha = prereg3.write_prereg(DOC)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
