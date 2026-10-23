"""Write the WP P4m declarations (harness.prereg3) before any P4m number is computed.

    cd dataprep_staging
    python -m p4_changhua.prereg_p4m

Two declarations, written once each:

* ``P4m.metered_diagnostics.v1`` - items (1)-(3) and (5) of WP P4m and the evidence of item
  (2): pre-origin, in-sample diagnostics of the frozen cwa theta with the C0 metered Changhua
  (and in-grid Chiayi) forcing added, on the calibration heads 2012-02..2022-12.
* ``P4m.recorded_arm_2023_25.v1`` - item (4): the recorded-forcing diagnostic arm
  2023-01..2025-06 (U4), reported only, never gating; written separately so that its
  2023+ heads cannot touch the pre-origin record.

Neither declaration scores fresh units: every head the pre-origin diagnostics read is a
calibration target of the theta (design-time data, plan 4.1/4.5), and the recorded arm reads
post-origin heads as a validator only. They are therefore declarations (fixed cases, metrics,
thresholds and interpretations), as P4.0's were; looks are logged with record_look.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg3  # noqa: E402
from harness.prereg import not_applicable as NA  # noqa: E402

THETA = "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
ENGINE = "results/dataprep/changhua_abstraction/engine/v2/MANIFEST.json"
C0_PRE = "results/dataprep/pump_electricity/preorigin/v3/MANIFEST.json"
C0_FULL = "results/dataprep/pump_electricity/full/v3/MANIFEST.json"
H1A = "results/dataprep/heads/L1_clean/v3/MANIFEST.json"
TIDES = "results/dataprep/_pilots/head_pumping_signatures/out/tide_coherence.csv"
ENGINE_CODE = ["dataprep_staging/p4_changhua/engine.py",
               "dataprep_staging/p4_changhua/engine_tlm.py",
               "dataprep_staging/p4_changhua/profiles_v2.py",
               "dataprep_staging/p4_changhua/inputs_p4.py"]
RF = "chou-shui-data/data/rf_timeseries.csv"
RF_ST = "chou-shui-data/data/rf_stations.csv"
FX_FORCING = "results/twin/forcing_experiment/forcing.npz"
FX_OBS = "results/twin/new_data_challenge/observations.csv"
FX_PROTOCOL = "results/twin/forcing_experiment/protocol.json"

COMMON_SETUP = {
    "theta": THETA + " (batch 8's merged cwa theta, sha256 789d0e45...; engine/v2 confirms it "
             "is the merged theta, not a provisional one). The deliverable page still uses the "
             "registered datum_gate model (commit 709f660); every P4m statement names the cwa "
             "theta.",
    "engine": "p4_changhua.engine.SparseTwin built from the theta with forward.build_model on "
              "CPU (CUDA hidden), inputs from p4_changhua.inputs_p4.load_inputs (explicit census "
              "columns); the base rollout must equal engine/v2 H_hindcast.",
    "steps": "131 monthly steps 2012-02..2022-12 from h0 (2012-01), as the calibration",
    "arms": {
        "A0_zero_changhua": "the frozen model as calibrated (model forcing E: the Yunlin census "
                            "of the deliverable, Changhua zero)",
        "A1_metered": "E plus E_CH = sum over the 6 classes of C0 pump_electricity/preorigin/v3 "
                      "cells_by_class value_by_county[changhua] + value_by_county[chiayi] "
                      "(kWh/month per cell; opened with guard use 'model_design'; values as "
                      "delivered, the model's convention: missing post-install readings count "
                      "as zero, the flag-4 December-2022 fill included), months 2012-02..2022-12, "
                      "added as E_extra (the calibrated spread kernel and the lift feedback act "
                      "on it, stepwise rollout, as P4.0 asks)",
        "A2_metered_nodomestic": "as A1 without the domestic class (C0's sensitivity member for "
                                 "possible household load)",
    },
    "observations": "the calibration targets: TwinInputs.obs_h (raw monthly means of "
                    "build_head_field), 158 wells, observed months only (no back-fill)",
    "scored_window": "2013-01..2022-12 (12-month burn-in 2012-02..2012-12 excluded; plan P4 "
                     "fact 5)",
    "datum": "training-mean datum: each well's mean of obs - pred over its observed scored "
             "months (anomaly RMSE, harness.scorers.anomaly_rmse agg 'rms')",
    "groups": "wells by county (Changhua 74, Yunlin 84; engine/v2 well_county), by model zone "
              "(zone3 of the well's cell: proximal / mid / distal) x county, and by model layer "
              "(obs_layer 0-3) x county; wells need >= 24 observed scored months",
    "clusters": "10 km clusters of well coordinates (floor(x/10 km), floor(y/10 km)); nested "
                "wells share a cluster; cluster bootstrap 2000 draws, seed 2027, 90 % "
                "percentile intervals; every such interval is in-sample and UNCALIBRATED",
}

DIAG = {
    "id": "P4m.metered_diagnostics.v1",
    "wp": "P4m",
    "title": "P4m: Changhua metered-forcing diagnostics of the frozen cwa theta (pre-origin, "
             "in-sample), the heads-accepted scale of the true Changhua series and its "
             "decomposition, the E10-style start-up / trend / mean-channel check, and the "
             "rules of the Phase-1 rung-R1 recommendation",
    "kind": "declaration",
    "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md revision note (P4m replaces "
                    "P4.1-P4.2), P4 facts 4-8, E8-E10, 4.1-4.5, I1 ladder; task WP P4m items "
                    "(1), (2), (3), (5)",
    "tests": NA("in-sample diagnostics on the theta's own calibration heads (design-time "
                "data, plan 4.1/4.5); no fresh unit is scored. The cases, metrics, thresholds "
                "and interpretations are fixed in 'thresholds'"),
    "zones": "model zones zone3 (proximal / mid / distal) as calibrated (zone_boundaries "
             "205,182,208); no zonation is chosen",
    "priors": NA("no prior; the theta refit emulation of T2b re-optimises the theta's 28 "
                 "interior entries without priors, as P4.0's reference refit did"),
    "burn_in": "12 months (2012-02..2012-12) for every score; E10 variants add a 24-month "
               "burn-in and a 2007-2011 spin-up; the theta-refit emulation (T2b) uses the "
               "calibration's own window 2012-02..2022-12 without burn-in, like fit_flow",
    "folds": NA("no folds; in-sample"),
    "fresh_units": NA("none: the 158 calibration wells 2012-22 are the theta's fit data; "
                      "A1's fresh units are also already blocked for 2012-22 head scopes by "
                      "P4.0's in-sample looks (P40 report)"),
    "primary_model": "frozen cwa theta, sparse twin; arms A0 (zero Changhua) and A1 (metered "
                     "Changhua + in-grid Chiayi); A2 (no domestic) as the sensitivity member",
    "inputs": [THETA, ENGINE, C0_PRE, H1A, TIDES, RF, RF_ST] + ENGINE_CODE,
    "setup": COMMON_SETUP,
    "thresholds": {
        "engine_checks (pass/fail, define the WP's numerical acceptance)": {
            "EC1": "sparse base rollout (A0) vs engine/v2 H_hindcast: max |dh| <= 1e-9 m",
            "EC2": "the arm's E_CH summed over 2012-01..2022-12 equals C0 preorigin/v3 "
                   "county_class_monthly.parquet in-grid 'all' totals of Changhua + Chiayi "
                   "(an independent output of C0; about 187.36 + 0.63 GWh/yr) to 1e-9 "
                   "relative, and A2's equals the 'all_nodomestic' totals",
            "EC3": "A1 sparse rollout vs forward.rollout (CPU torch, CUDA hidden) on the same "
                   "forcing (E + E_CH): max |dh| <= 1e-5 m (P4.0's A1 threshold)",
            "EC4": "lift clipped (<= 2 m) in how many A1 cell-steps: reported; if > 0 the "
                   "tangent-linear responses of T3 are flagged one-sided there",
        },
        "T1_skill_change (reported; interpretation fixed)": {
            "metrics": "per group and arm: anomaly RMSE (rms over wells), amplitude ratio "
                       "(harness.scorers.amplitude_ratio, median std(pred)/std(obs), min 24 "
                       "months), correlation (median over wells of Pearson r on observed "
                       "scored months), and A1 - A0 / A2 - A0 differences with the cluster "
                       "bootstrap 90 % interval of the anomaly-RMSE difference (Changhua, "
                       "Yunlin, all)",
            "interpretation": "Changhua interval of dRMSE (A1 - A0) entirely below 0: the "
                              "metered timing and amplitude are consistent with the heads "
                              "even under the zero-Changhua theta. Entirely above 0: the frozen "
                              "theta over- or mis-responds to the true forcing (read with the "
                              "amplitude ratio: > 1 = over-response), which the R1 "
                              "recalibration must resolve. Interval covering 0: no detectable "
                              "change. Yunlin: an anomaly-RMSE rise of more than 2 % flags "
                              "cross-border coupling (plan V3's Yunlin tolerance). Everything "
                              "here is in-sample (the theta was fitted on these heads with zero "
                              "Changhua pumping): design-time evidence, never skill.",
        },
        "T2_heads_accepted_scale (reported)": {
            "family": "pred(s): stepwise rollout with E_extra = s x E_CH (A1 series), exact "
                      "(lift feedback), s in [0, 3]",
            "s_LS (labelled attenuated)": "argmin over s of the sum over all 158 wells and "
                                          "observed scored months of the squared anomaly "
                                          "residual (training-mean datum), evaluated on the grid "
                                          "{0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3} and "
                                          "refined by bounded scalar minimisation with exact "
                                          "rollouts (xatol 1e-3); '>3' if at the bound",
            "s_LS secondary": "Changhua wells only; and first-order cross-checks with the "
                              "tangent-linear response about A0 as regressor: OLS with a free "
                              "datum and engine.solve_with_datum (fit_flow's datum prior, sd "
                              "5 m)",
            "s_AM (amplitude-matched)": "smallest s in [0, 3] where the Changhua amplitude "
                                        "ratio (median over Changhua wells of std(pred)/std(obs)) "
                                        "reaches 1, by grid and bisection with exact rollouts; "
                                        "0 if it is >= 1 at s = 0; '>3' if never",
            "uncertainty": "cluster bootstrap of s_LS from per-well objective curves on the "
                           "grid (cubic spline), 90 % interval, UNCALIBRATED",
            "far_below_1": "s_LS < 0.5 or s_AM < 0.67",
            "decomposition (computed whatever s is)": {
                "a_shape": "s_LS and s_AM for a Yunlin-shaped series with E_CH's per-cell "
                           "2012-22 mean (shape = the model's total Yunlin energy / its "
                           "mean): the pilots' borrowed-shape convention. Seasonal CV "
                           "(month-of-year profile sd / mean) of the in-grid Changhua and "
                           "Yunlin series, irrigation and all classes, reported beside it",
                "b_theta_bias": "CPU emulation of the R1 recalibration: P4.0's reference refit "
                                "(bounded Gauss-Newton of the 28 interior entries, 7 pinned "
                                "held, central-difference Jacobian each iteration, trust 0.5, "
                                "tol 1e-6, 15 iterations; fit_flow level loss, observed months "
                                "2012-02..2022-12, datum profiled at kappa from theta-hat) run "
                                "with E_extra = s x E_CH at s = 0, 0.5, 1. s_theta = vertex of "
                                "the parabola through L(0), L(0.5), L(1) (refit objectives); "
                                "s_frozen_cal = the frozen-theta argmin of the same objective. "
                                "theta-bias share of the gap = (s_theta - s_frozen_cal) / (1 - "
                                "s_frozen_cal) when s_frozen_cal < 1. dtheta = refit(1) - "
                                "refit(0) per entry is the expected direction of the R1 shift. "
                                "Labelled: CPU emulation, unconverged references (P4.0: "
                                "predicted remaining ~1 %), level loss at fixed kappa, pinned "
                                "entries held, in-sample",
                "c_response_conversion": "central-difference sensitivity (step 0.1 in each "
                                         "entry's own units) of A_resp = rms over Changhua wells "
                                         "and scored months of the anomaly of pred(A1) - "
                                         "pred(A0) to each of the 35 theta entries (28 interior, "
                                         "7 pinned evaluated across their bound) and to log_eta "
                                         "and log_head_extra (the fixed conversion): e_j = d ln "
                                         "A_resp / d theta_j; the single-entry change that would "
                                         "scale A_resp by s_AM is ln(s_AM) / e_j (flagged if it "
                                         "leaves the bounds or exceeds 2 units)",
                "d_independent_evidence": {
                    "E8_tides (design-time evidence, pilot output, not rebuilt)": TIDES + ": "
                    "wells with M2 coherence > 0.8; median M2 amplitude by county and layer, "
                    "and OLS of log A_M2 on easting (km), county and layer dummies; the county "
                    "coefficient is reported as context only (plan E8: cannot free a Changhua "
                    "T multiplier)",
                    "BE (H1a v3 barometric_efficiency.csv, accepted pre-origin epochs)": "median "
                    "BE_long by county and layer; implied Ss ratio Changhua/Yunlin ~ "
                    "BE_Yunlin / BE_Changhua at equal porosity; context only",
                    "E9_nested_sites (H1a v3 heads_monthly flags 0-1, window 2012-01..2022-12, "
                    "guard use model_design)": "sites (station_meta_preorigin site_id) with layer "
                    "codes 1 and 2 in active cells; per well, OLS of the high-passed head "
                    "(minus a centred 25-month mean) on the high-passed local metered energy "
                    "(C0 preorigin/v3 cells_total value_all within r of the well's cell, r = 2, "
                    "3, 5 km; GWh/month), high-passed regional energy (fan total) and "
                    "high-passed rain (nearest rf gauge, lags 0-2); site ratio = slope(layer "
                    "1) / slope(layer 2); median by county, observed vs the A1 model heads at "
                    "the same cells and layers (same regression); primary r = 3 km",
                },
                "attribution_rule": "the gap 1 - s is attributed: theta bias = the T2b share; "
                                    "shape/seasonality = reported for the pilots' borrowed-"
                                    "shape gap only (the metered series has no borrowed "
                                    "shape); the remainder is response/conversion, read with "
                                    "T2c (which entries could close it, by how much) and the "
                                    "independent evidence. If T2b's refits do not converge "
                                    "(stop reason not 'converged') the theta-bias share is "
                                    "reported as approximate",
            },
        },
        "T3_E10_check (reported; plan E10 interpretation)": {
            "transient": "per well, mean residual (obs - pred) over 2012-02..2013-12 minus mean "
                         "over 2015-01..2022-12; mean over Changhua wells with cluster bootstrap "
                         "90 % interval, A0 vs A1 (and under the spin-up IC)",
            "trend": "per well OLS slope of the residual over 2013-01..2022-12 (m/yr); mean over "
                     "Changhua wells with cluster bootstrap 90 % interval; share of wells with "
                     "|t| > 2 (naive); A0 vs A1",
            "mean_channel": "E_CH split per cell into mean, month-of-year seasonal and "
                            "anomaly parts over 2012-02..2022-12; head responses by the "
                            "tangent-linear model about the A0 path (primary; about A1 "
                            "secondary); OLS of the A0 residual on the three responses with a "
                            "free per-well datum, all 158 wells; theta_k = scale of part k (1 = "
                            "metered); 90 % cluster bootstrap; AR(1)-GLS (phi from the A0 "
                            "residual) as the pilot-convention secondary",
            "variants": "V1 12-month burn-in; V2 24-month burn-in (2014-01..2022-12); V3 V1 + a "
                        "per-well linear trend nuisance; V4 V1 from a consistent spin-up IC: "
                        "60 steps 2007-01..2011-12 from h0 (apex pinned to h0) with C0 "
                        "preorigin/v3 pumping of the arm (A0: Yunlin; A1: Yunlin + Changhua + "
                        "Chiayi) and the model recharge's 2012-22 month-of-year climatology; "
                        "channels recomputed about the spun-up A0 path",
            "interpretation": "transient / trend 'disappear' under A1 if their A1 interval "
                              "covers 0 while the A0 interval excludes 0. Mean channel: if "
                              "theta_mean's interval lies entirely below 0 in V1-V4, the level "
                              "channel is misspecified (plan E10: no fixed-datum or level "
                              "variant; logged for plan 7.3); if it covers or exceeds 0 in one "
                              "variant, that treatment is recommended for the design",
        },
        "T5_R1_recommendation_rules (written before any 2023+ head is read; U7)": {
            "free": "R1 frees the R0 recipe's zonal parameters unchanged (as I1's declaration "
                    "I1.R1.changhua_census.v1 does); a Changhua-specific response multiplier is "
                    "recommended only if s_theta < 0.67 (shared zonal parameters cannot absorb "
                    "the metered series) AND at least one independent indicator (E8, BE, E9) "
                    "points the same way; otherwise not",
            "bounds": "a bound-pinned entry is recommended for a widened-bound sensitivity rung "
                      "if, at the s = 1 refit, the bounded GN step pushes against it (gradient "
                      "points outward)",
            "direction": "expected R1 theta shift = sign of dtheta (T2b) for entries with "
                         "|dtheta| > 0.05",
            "acceptance": "proposed for I1 to register (not registered here): the plan I1 "
                          "binding criteria, plus the P4m diagnostics rerun on R1's theta",
            "u7": "the recommendation file is written and hashed before the recorded-arm "
                  "declaration's first score; no 2023+ observation enters it",
        },
        "other_rules": "Thresholds are not changed after results are seen. Results are "
                       "reported whichever way they fall. Everything here is pre-origin "
                       "design-time evidence for the Phase-1 ladder.",
    },
    "gate_scores": NA("no gate: EC1-EC3 are the WP's numerical acceptance; T1-T3 are "
                      "reported with fixed interpretations"),
    "interpretation": "See thresholds. In short: T1 says whether the metered Changhua forcing "
                      "improves or degrades the frozen model's Changhua and Yunlin anomaly "
                      "skill; T2 gives the scale of the TRUE series the heads accept, least "
                      "squares (attenuated) and amplitude-matched, and splits any gap below 1 "
                      "into theta bias (CPU refit emulation), shape (for the pilots' gap) and "
                      "response/conversion; T3 says whether the E10 artefacts survive metered "
                      "forcing; T5 maps these to the R1 recommendation.",
    "prior_exposure": [
        "Read before this declaration: the plan's pilot numbers (heads-accepted least-squares "
        "scales 0.209 / 0.156 / 0.099 of a borrowed-shape TPC series, amplitude-matched about "
        "0.50 / 0.42 / 0.59; mean channel -0.187; Changhua CV anomaly RMSE 1.203 at zero "
        "pumping); P4.0's report (in-sample R2 0.865, reference refits and pinned entries; the "
        "per-entry shifts of its reference refit were not read); C0's report and the "
        "forcing totals and seasonal CVs of preorigin/v3 (forcing only: Changhua 187.36, Chiayi "
        "0.63 GWh/yr in grid; month-of-year CV irrigation 0.135 vs Yunlin 0.224); the audit's "
        "frozen-challenge county scores (Changhua 1.280 vs climatology 1.207 m) and the "
        "2023-25 forcing experiment report (datum_gate theta, Yunlin pumping only).",
        "Not read: any head score with Changhua forcing (I1's running rung R1, "
        "results/twin_runs/stage3_datum_gate_chc, has not been opened; P2's head checks not "
        "read). No P4m number has been computed.",
    ],
}

RECORDED = {
    "id": "P4m.recorded_arm_2023_25.v1",
    "wp": "P4m",
    "title": "P4m item (4): recorded-forcing diagnostic arm 2023-01..2025-06 with Changhua "
             "metered forcing added (U4; reported only, never gating)",
    "kind": "declaration",
    "plan_section": "plan 2.5 U4/U7, 4.2, I1 re-scoring protocol step 4, 7.1 (recorded-forcing "
                    "arm); task WP P4m item (4)",
    "tests": NA("post-origin diagnostic: 2023-25 heads are validators of a recorded-forcing "
                "run (U4) and never enter any gate, threshold, member choice or the P4m "
                "recommendation (U7)"),
    "zones": NA("no zonation"),
    "priors": NA("no fit"),
    "burn_in": NA("hindcast 2012-02..2022-12 as calibrated; the forecast starts from the "
                  "hindcast state nudged to the 2022-12 observations"),
    "folds": NA("no folds"),
    "fresh_units": NA("post-origin validator run; no pre-origin fresh unit is touched"),
    "primary_model": "frozen cwa theta, sparse twin (CPU), one member (the experiment used the "
                     "datum_gate theta with 36 members on GPU: the CPU arms are a new, cwa-theta "
                     "baseline, not a replication of its numbers)",
    "inputs": [THETA, ENGINE, C0_PRE, C0_FULL, FX_FORCING, FX_OBS, FX_PROTOCOL] + ENGINE_CODE,
    "setup": {
        "protocol": "hydrophysics.twin.forcing_experiment as registered (protocol.json), on the "
                    "sparse twin: hindcast 2012-02..2022-12 with the arm's pre-origin forcing; "
                    "state at 2022-12 nudged to the 2022-12 observations "
                    "(forward.nudge_to_observations, gain 1, taper 5 km, the theta's well "
                    "datum, noise-free initial field only); rollout 2023-01..2025-06 (30 steps, "
                    "apex pinned to h0); prediction = model + per-well offset (mean of obs - "
                    "hindcast over 2012-01..2022-12, the experiment's convention); recharge = "
                    "climatology (forward.future_forcing) in every arm",
        "arms": {
            "R_clim_zero": "climatology pumping (Yunlin) and zero Changhua; hindcast A0",
            "R_clim_ch": "climatology pumping plus a Changhua+Chiayi climatology "
                         "(month-of-year mean 2012-01..2022-12 of preorigin/v3, the "
                         "forward.climatology rule); hindcast A1",
            "R_obs_zero": "observed pumping = the experiment's cached recorded Yunlin energy "
                          "(forcing.npz 'energy', frozen 2012-22 kept set), Changhua zero; "
                          "hindcast A0 (the experiment's observed_pumping arm on the cwa theta)",
            "R_obs_ch": "R_obs_zero plus Changhua+Chiayi recorded energy 2023-01..2025-06 from "
                        "C0 pump_electricity/full/v3 (guard use forcing_recorded_arm; kept set "
                        "frozen at the pre-origin duty), all classes; hindcast A1",
            "R_obs_ch_nodomestic": "as R_obs_ch without the domestic class; hindcast A2",
        },
        "scoring": "observations.csv (the experiment's, registered QC); forcing_experiment."
                   "paired_metrics convention (wells with >= 12 matched months across every "
                   "arm and the baseline): equal-well RMSE (rms of per-well MSE; and mean of "
                   "per-well RMSE, the audit's 1.519 convention) and bias, by county (Changhua, "
                   "Yunlin) and all; baseline 'clim' = drift_diag.fair_baselines on the "
                   "2012-22 observations (month-of-year means); ratio to clim",
        "numerical_check": "the sparse forecast from the nudged state vs forward.rollout (CPU "
                           "torch) for R_obs_ch: max |dh| <= 1e-5 m",
    },
    "thresholds": {
        "reported_only": "every number of this declaration is reported and never gates "
                         "anything (U4, U7); the P4m R1 recommendation is written and hashed "
                         "before the first score of this declaration",
        "interpretation": "R_obs_ch vs R_obs_zero at Changhua wells shows whether metered "
                          "Changhua pumping moves the recorded-forcing arm toward climatology "
                          "(plan 7.1 expected a modest Changhua improvement; no pilot "
                          "quantified it); Yunlin wells must not degrade by more than 2 % "
                          "(plan V3 tolerance, reported only).",
    },
    "gate_scores": NA("no gate (U4, U7)"),
    "interpretation": "A diagnostic of the frozen cwa theta under recorded forcing; it does "
                      "not validate a recalibrated model and is not skill of a forecast.",
    "prior_exposure": [
        "The experiment's own report (datum_gate theta, 36 members, Yunlin pumping only): "
        "climatology arm 2.067, observed_pumping 1.845, clim baseline 1.630 m (equal-well rms), "
        "and the audit's mean-of-RMSE table (1.772 / 1.519 / 1.422). No cwa-theta or "
        "Changhua-forced 2023-25 number has been computed by P4m."],
}


def main() -> int:
    for doc in (DIAG, RECORDED):
        path, sha = prereg3.write_prereg(doc)
        print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
