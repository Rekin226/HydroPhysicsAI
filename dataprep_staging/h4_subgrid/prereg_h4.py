"""WP H4: write the pre-registration H4.subgrid_operator.v1 (harness.prereg3), BEFORE any head
data is read by H4 code. Usage: python prereg_h4.py [--dry] (--dry validates and prints the
blocking check without writing)."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
import pandas as pd  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

ID = "H4.subgrid_operator.v1"
UNIT = "H4:wells158_2020-2022"
CRIT = {
    "A1": "T1 held-out years: Q4 excess falls (corrected < base, strict)",
    "A2": "T2 held-out sites and years: Q4 excess falls (corrected < base, strict)",
    "A3": "T1: Q4 excess reduction beats the shuffled-position placebo, p <= 0.05 (39 draws)",
    "A4": "T1: Q4 excess reduction beats the >15 km placebo, p <= 0.05 (39 draws)",
    "A5": "T2: Q4 excess reduction beats the shuffled-position placebo, p <= 0.05 (39 draws)",
    "A6": "T2: Q4 excess reduction beats the >15 km placebo, p <= 0.05 (39 draws)",
    "A7": "T1: Q1, Q2, Q3 equal-well RMSE corrected / base <= 1.02 in each quartile",
    "A8": "T2: Q1, Q2, Q3 equal-well RMSE corrected / base <= 1.02 in each quartile",
}


def inputs() -> list[str]:
    code = ["dataprep_staging/h4_subgrid/h4lib.py", "dataprep_staging/h4_subgrid/r1_registered.py",
            "dataprep_staging/h4_subgrid/prereg_h4.py", "dataprep_staging/common/guard.py",
            "dataprep_staging/common/paths.py", "dataprep_staging/harness/prereg3.py"]
    data = [L.ENGINE / "MANIFEST.json", L.ENGINE / "engine_arrays.npz",
            L.UNITS / "MANIFEST.json", L.UNITS / "units.json", L.UNITS / "wells.csv",
            L.ELEC / "MANIFEST.json", L.ELEC / "pumps_yunlin.npz", L.ELEC / "pumps_changhua.npz",
            L.ELEC / "pumps_chiayi_ingrid.npz",
            L.CENSUS / "MANIFEST.json", L.CENSUS / "census_pumps.parquet",
            L.HEADS / "MANIFEST.json", L.HEADS / "heads_monthly.parquet",
            DATAPREP / "pump_electricity/preorigin/v3/MANIFEST.json",
            DATAPREP / "pump_electricity/preorigin/v3/pumps_yunlin.npz",
            DATAPREP / "pump_electricity/preorigin/v3/pumps_changhua.npz",
            DATAPREP / "pump_electricity/preorigin/v3/pumps_chiayi_ingrid.npz"]
    return code + [rel_to_repo(p) for p in data]


def doc() -> dict:
    wells = pd.read_csv(L.UNITS / "wells.csv", dtype={"sid": str})
    ids = sorted(wells["sid"].astype(str))
    if len(ids) != 158:
        raise SystemExit(f"expected 158 wells, got {len(ids)}")
    cfg = L.PRIMARY.as_dict()
    ext = {UNIT: {
        "description": "observed monthly heads (heads/L1_clean/v3, flags 0-1) of the 158 "
                       "calibration wells in 2020-01..2022-12: the H4 held-out years; also the "
                       "held-out sites of T2 (every well is in exactly one test fold of the A1 "
                       "site-fold partition). Never used for an H4 design choice; looked at "
                       "before by others (twin theta fitted 2012-22; P40 in-sample looks; H2 "
                       "scores; the pilots' design-time block dt.block.2020_2022)",
        "ids": ids, "window": "2020-01..2022-12"}}
    common_metric = (
        "Operator h4lib (config PRIMARY): c_i(t) = a_s sum_j g_s(r_ij) q_j^s(t) + a_l sum_j "
        "g_l(r_ij) Qbar_j(t) over kept Yunlin census pumps with coordinates within r_max 2 km "
        "(r floored at 100 m); Q_j = Chow-Lin kWh (yunlin_electricity/preorigin/v2) x the twin's "
        "lift conversion (eta 0.5, head_extra 40 m, dt 30 d, hindcast previous-month layer-1 "
        "head at the pump's cell); Qbar_j trailing 12-month mean, q_j^s = Q_j - Qbar_j; "
        "g_s = w/(2 pi T)(|K0(r k sqrt(i))| - |K0(R k sqrt(i))|), k = sqrt(omega S/T), omega = "
        "2 pi/365.25 d; g_l = w/(2 pi T) ln(R/r); T, S of the well's model layer at its cell "
        "(engine/v2); w = f_shallow (layer 0), 1 - f_shallow (layer 1), 0 (layers 2-3). "
        "Prediction h_cell - c with h_cell the engine/v2 hindcast at the well's cell and layer. "
        "a_s, a_l >= 0 by NNLS on fit-window (2012-02..2019-12) residuals obs - h_cell, "
        "per-well means removed, each well weighted 1/n_i, wells with >= 24 fit months. "
        "Score per well over its observed 2020-01..2022-12 months (flags 0-1, >= 7 months, "
        ">= 24 fit months): MSE of obs - pred - datum with the training-mean datum (fit window) "
        "for the base (c = 0) and corrected model; climatology = per-well month-of-year mean "
        "over the fit window (a calendar month without data falls back to the well's fit-window "
        "mean). Quartiles Q1-Q4 by the sd of the observed fit-window heads over the wells with "
        ">= 24 fit months (cuts at the 25/50/75 % quantiles). Q4 excess = sum over scored Q4 "
        "wells of (MSE_model - MSE_clim); reduction = base excess - corrected excess. Placebos "
        "(fitted and scored identically, 39 draws each, the same draws for T1 and T2): shuffle = "
        "pump coordinates permuted among the Yunlin pumps (rng default_rng(20261014 + r)), each "
        "pump keeps its Q series; distant = for each well with real kernel support a donor "
        "point drawn uniformly from the Yunlin pump positions >= 15 km + r_max from the well "
        "(default_rng(20261114 + r)) and the well's kernel evaluated at the donor (only pumps "
        "> 15 km from the well; wells without real support stay at zero); p = (1 + #{placebo "
        "reduction >= real reduction}) / 40. Q1-Q3 check: sqrt(mean corrected MSE / mean base "
        "MSE) over the scored wells of each quartile.")
    tests = [
        {"test_id": "H4.T1_heldout_years", "data_scope": "heads:2020-01..2022-12",
         "fresh_units": [UNIT], "external_units": ext,
         "metric": common_metric + " T1: multipliers fitted on every well (158 calibration "
                                   "wells) and applied to all.",
         "threshold": "A1, A3, A4, A7 (see thresholds)"},
        {"test_id": "H4.T2_heldout_sites", "data_scope": "heads:2020-01..2022-12",
         "fresh_units": [UNIT], "external_units": ext,
         "metric": common_metric + " T2: the A1 site-fold partition (fresh.folds.wells_all of "
                                   "validation_units/v3, read as data: sites within 100 m, "
                                   "complete-linkage 10 km clusters, 5 folds seed 2027, 5 km "
                                   "buffer); per fold, multipliers fitted on the fold's train "
                                   "wells (test and buffer wells excluded) and applied to its "
                                   "test wells; per-well MSEs pooled over the 5 folds.",
         "threshold": "A2, A5, A6, A8 (see thresholds)"}]
    return {
        "id": ID, "wp": "H4", "kind": "test",
        "title": "WP H4: sub-grid periodic-Theis / Thiem observation operator from census pumps "
                 "within 2 km; Q4 excess on held-out years and held-out sites vs two placebos",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP H4; 2.5 (U1-U7); 4.3-4.7",
        "tests": tests,
        "zones": {"description": "T and S of the well's model layer at its cell: the calibrated "
                                 "cwa theta's zone values as assembled in "
                                 "changhua_abstraction/engine/v2 (3 zones); not profiled (see "
                                 "interpretation); sensitivity t_bound_half halves T where it "
                                 "sits at the 20000 m2/d bound"},
        "priors": prereg3.not_applicable("no priors: least-squares multipliers, constrained >= 0"),
        "burn_in": {"description": "2012-01 is the hindcast's initial condition and is excluded; "
                                   "fit window 2012-02..2019-12"},
        "folds": {"description": "T2 uses the A1 site-fold partition fresh.folds.wells_all "
                                 "(validation_units/v3) as data; the registered test unit is the "
                                 "external unit H4:wells158_2020-2022 (the registry unit carries "
                                 "other WPs' looks on 2012-22 and cannot be registered fresh)"},
        "fresh_units": [UNIT],
        "thresholds": {k: v for k, v in CRIT.items()},
        "gate_scores": [{"name": k, "rule": v, "weight": 0.125} for k, v in CRIT.items()],
        "gate_rule": "subgrid_operator/v1 gate passes only if all of A1-A8 pass. Pass: allowed "
                     "uses model_design (secondary-model observation hook for I1, refitted after "
                     "any recalibration that changes the hindcast), validator, "
                     "design_time_evidence, after_the_fact_analysis. Fail: design_time_evidence "
                     "and after_the_fact_analysis only, and plan 7.3 states that the local cones "
                     "are not recovered by this operator on this grid. Thresholds are not "
                     "changed after results; no 2023+ observation enters (U7).",
        "primary_model": "h4lib.PRIMARY: " + json.dumps(cfg, sort_keys=True),
        "reported_not_gated": [
            "sensitivity members (each T1 and T2 without placebos): recorded_arm (C0 "
            "preorigin/v3 even split), r_max_3km, all_ingrid_counties (Yunlin + Changhua + "
            "in-grid Chiayi pumps), one_multiplier (a_s = a_l), layer_weight_ones (w = 1 in "
            "every layer), c20_months_only, t_bound_half",
            "T2 spatial-only variant scored on the fit window (in-sample datum and climatology)",
            "per-quartile RMSE and ratio to climatology, Q4 amplitude ratio (median sd(pred) / "
            "sd(obs)), multipliers (T1 and per fold), placebo multipliers"],
        "design_time_units": [
            "H4:d0_design_check (no head data: pumps within 1-2 km, kernel shapes, unit-"
            "multiplier correction sd 0.05 m median at the 61 supported Yunlin wells, conversion "
            "check, placebo cost)",
            "H4:t0_smoke (synthetic heads only: pipeline smoke test and multiplier recovery)",
            "audit 2026-10-11 section 1-2 (2023-26 quartile analysis that motivated H4; not used "
            "in any H4 threshold or choice beyond the plan's wording)",
            "P2a / head_pumping_signatures pilots: partial R2 of local kWh +0.01..+0.05 at 1-2 km "
            "(design-time; same wells 2012-22)"],
        "inputs": inputs(),
        "interpretation": (
            "The operator is a data-driven correction of the observation operator with no solver "
            "change. The base hindcast (theta calibrated on 2012-22 at the same wells) is in "
            "sample in 2020-22: the held-out years are held out for the operator's two "
            "multipliers only. Quartiles are defined on 2012-19 sd, not on the audit's 2023-26 "
            "quartiles (U7). Only Yunlin pumps enter the primary because the hindcast carries "
            "Yunlin pumping only: a local kernel on Changhua pumps would also absorb the missing "
            "Changhua cell-mean drawdown and bias a fan-wide multiplier (sensitivity "
            "all_ingrid_counties reports it). With the twin's layer split only Yunlin wells in "
            "layers 0-1 get a correction (61 wells, design-time count). T is not profiled: it "
            "enters the kernel as 1/(2 pi T), absorbed by the fitted multiplier, and the shape "
            "dependence is weak at r <= 2 km (|K0| / ln within 0.91-0.98 at the median wells, "
            "design-time); 4 supported wells sit at the T bound (sensitivity t_bound_half). At "
            "multiplier 1 the operator is the model-unit Thiem/periodic-Theis cone (correction "
            "sd 0.05 m at the median supported well, design-time), far below the Q4 swing; a "
            "fitted multiplier far above 1 means the correction carries something besides the "
            "model-unit cone (physical volumes above model units, a lower local T, or the "
            "regional amplitude deficit), and the placebos decide whether pump location "
            "matters. p <= 0.05 needs the real reduction to exceed at least 38 of 39 draws. "
            "The Q1-Q3 check is per quartile (stricter than pooled). Coverage is not part of H4 "
            "(no interval is published; multiplier spreads are labelled uncalibrated)."),
        "registered_on": _dt.date.today().isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    dd = doc()
    errs = prereg3.validate_prereg(dd)
    if errs:
        raise SystemExit("invalid: " + "; ".join(errs))
    if a.dry:
        idx = prereg3.membership_index()
        for t in dd["tests"]:
            for u, v in t["external_units"].items():
                idx.members[idx.canon(u)] = prereg3._declared_members(v)
        now = _dt.datetime.now(_dt.timezone.utc)
        blk = prereg3.blocking_looks(dd["tests"], None, idx=idx, test_time=now)
        print(json.dumps({"valid": True, "blocking": blk[:5], "n_blocking": len(blk),
                          "n_inputs": len(dd["inputs"])}, indent=1, default=str))
        return
    path, sha = prereg3.write_prereg(dd)
    print(json.dumps({"written": rel_to_repo(path), "sha256": sha}, indent=1))


if __name__ == "__main__":
    main()
