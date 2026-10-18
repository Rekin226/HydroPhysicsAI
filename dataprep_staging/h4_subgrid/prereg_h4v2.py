"""WP H4 repair round: write the pre-registration H4.subgrid_operator.v2 (harness.prereg3) BEFORE
any v2 code reads a head value. Usage: python prereg_h4v2.py [--dry]
(--dry validates and prints the blocking check without writing)."""
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
import h4v2lib as V  # noqa: E402
import h4v2run as RN  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

ID = "H4.subgrid_operator.v2"
UNIT = "H4r:heldout_wells_2012_2022"
D1A = DATAPREP / "reports/H4_evidence/repair/design_time/d1_availability.json"
CRIT = {
    "F1": "held-out wells: high-swing (Q4 by the v1 sd cuts if >= 5 scored held-out wells with "
          "a kernel are in Q4, else Q3+Q4) summed MSE reduction (base - corrected) > 0, strict",
    "F2": "held-out wells: F1 reduction beats the shuffled-position placebo, p <= 0.05 "
          "(199 draws)",
    "F3": "held-out wells: F1 reduction beats the >15 km donor placebo, p <= 0.05 (199 draws)",
    "F4": "held-out wells: equal-well RMSE corrected / base <= 1.02 pooled over the "
          "non-high-swing quartiles and in each of them with >= 5 scored wells",
    "C1": "[design-time necessary condition; v1 unit, post-look] T1 held-out years 2020-22: Q4 "
          "excess falls (strict)",
    "C2": "[design-time necessary condition; v1 unit, post-look] T2 A1 site folds 2020-22: Q4 "
          "excess falls (strict)",
    "C3": "[design-time necessary condition; v1 unit, post-look] T1: Q1, Q2, Q3 RMSE ratio "
          "<= 1.02 each",
    "C4": "[design-time necessary condition; v1 unit, post-look] T2: Q1, Q2, Q3 RMSE ratio "
          "<= 1.02 each",
}


def heldout_members() -> dict:
    d = json.loads(D1A.read_text(encoding="utf-8"))
    rows = [r for r in d["heldout_wells"] if r["n_fit"] + r["n_test"] >= RN.F_MIN_MONTHS]
    return {"ids": sorted(r["sid"] for r in rows),
            "cells": sorted({int(r["cell"]) for r in rows}),
            "window": f"{RN.F_WINDOW[0]}..{RN.F_WINDOW[1]}"}


def inputs() -> list[str]:
    code = ["dataprep_staging/h4_subgrid/h4lib.py", "dataprep_staging/h4_subgrid/h4v2lib.py",
            "dataprep_staging/h4_subgrid/h4v2run.py",
            "dataprep_staging/h4_subgrid/r2_registered.py",
            "dataprep_staging/h4_subgrid/prereg_h4v2.py", "dataprep_staging/common/guard.py",
            "dataprep_staging/common/paths.py", "dataprep_staging/harness/prereg3.py"]
    data = [L.ENGINE / "MANIFEST.json", L.ENGINE / "engine_arrays.npz",
            L.UNITS / "MANIFEST.json", L.UNITS / "units.json", L.UNITS / "wells.csv",
            L.ELEC / "MANIFEST.json", L.ELEC / "pumps_yunlin.npz", L.ELEC / "pumps_changhua.npz",
            L.ELEC / "pumps_chiayi_ingrid.npz",
            L.CENSUS / "MANIFEST.json", L.CENSUS / "census_pumps.parquet",
            L.HEADS / "MANIFEST.json", L.HEADS / "heads_monthly.parquet",
            L.HEADS / "station_meta_preorigin.parquet",
            DATAPREP / "pump_electricity/preorigin/v3/MANIFEST.json",
            DATAPREP / "pump_electricity/preorigin/v3/pumps_yunlin.npz",
            DATAPREP / "pump_electricity/preorigin/v3/pumps_changhua.npz",
            DATAPREP / "pump_electricity/preorigin/v3/pumps_chiayi_ingrid.npz",
            V.GRID_NPZ, D1A]
    return code + [rel_to_repo(p) for p in data]


def doc() -> dict:
    mem = heldout_members()
    cfg = V.PRIMARY.as_dict()
    ext = {UNIT: {
        "description": "observed monthly heads (heads/L1_clean/v3, flags 0-1) in 2012-02..2022-12 "
                       "of the held-out wells: stations of station_meta_preorigin that are not "
                       "among the 158 calibration wells (the twin's theta was never fitted to "
                       "them), in an active model cell, layer code 1-4 not a proposal "
                       "(layer_flag < 5), >= 24 observed months in the window (count from the "
                       "logged availability look; no value read). Never used by H4 for a design "
                       "choice. Looked at before by others: H3 (recovered wells: daily bridge "
                       "tests and the recovered_wells product, 15 wells), H2 gap tests (if any "
                       "covered them), H1a QC; the H4 repair read only their month counts.",
        **mem}}
    metric = (
        "Operator h4v2lib.PRIMARY (sub-grid, cell-mean kernel subtracted): c_i(t) = a_s sum_j "
        "g_s,ij q_j^s(t) + a_l sum_j g_l,ij Qbar_j(t) over kept census pumps of Yunlin, "
        "Changhua and in-grid Chiayi with coordinates within r_max 2 km of well i; g_ij = "
        "w_i/(2 pi T_i) (f(|x_i - x_j|) - mean_{x in cell(i)} f(|x - x_j|)), f = |K0(r k "
        "sqrt(i))| (seasonal, k = sqrt(omega S/T), omega = 2 pi/365.25 d) or -ln r (slow), r "
        "floored at 100 m in both terms, cell(i) the 1 km model cell containing the well "
        "(cell means tabulated on a 20 m grid, trapezoid rule, bilinear lookup); Q_j = Chow-Lin "
        "kWh (yunlin_electricity/preorigin/v2) x the twin's lift conversion (h4lib, hindcast "
        "previous-month layer-1 head at the pump's cell), Qbar_j trailing 12-month mean, q^s = "
        "Q - Qbar; T, S, w of the well's layer at its cell (engine/v2; w = f_shallow layer 0, "
        "1 - f_shallow layer 1, 0 layers 2-3). Prediction h_cell - c with h_cell the engine/v2 "
        "hindcast at the well's cell and layer (held-out wells: layer = layer_code - 1, the "
        "rule that maps all 158 calibration wells). a_s, a_l >= 0 by NNLS on the 158 "
        "calibration wells' 2012-02..2019-12 residuals obs - h_cell (per-well means removed, "
        "each well weighted 1/n_i, >= 24 fit months; h4lib.fit_multipliers). F (held-out "
        "wells, >= 24 observed months 2012-02..2022-12): per-well MSE of obs - pred minus the "
        "well's own mean over its scored months (same free datum for base and corrected); sd = "
        "sd of its observed heads over the scored months; quartile by the fixed v1 cuts "
        "0.87588/1.49986/2.35346 m (158-well 2012-19 sd); high-swing set = Q4 if >= 5 scored "
        "held-out wells with a kernel fall in Q4, else Q3+Q4; reduction = sum over the "
        "high-swing set of (MSE_base - MSE_corrected). Placebos (199 draws each, fitted on the "
        "calibration wells and scored on F identically): shuffle = pump coordinates permuted "
        "within each county (default_rng(20261017 + r)), the same permutation for both well "
        "sets, each pump keeps its Q series; distant = for each well with real kernel support "
        "a donor point drawn uniformly from the census pump positions >= 15 km + r_max from "
        "the well (default_rng(20261117 + r), calibration wells drawn first), kernel and cell "
        "mean evaluated at the donor and the donor's own cell; p = (1 + #{placebo >= real}) / "
        "200. No-harm: equal-well RMSE ratio sqrt(mean corrected MSE / mean base MSE) over "
        "every scored held-out well (with or without kernel) of the non-high-swing quartiles, "
        "pooled and per quartile with >= 5 scored wells.")
    tests = [{"test_id": "H4v2.F_heldout_wells", "data_scope": "heads:2012-02..2022-12",
              "fresh_units": [UNIT], "external_units": ext, "metric": metric,
              "threshold": "F1-F4 (see thresholds)"}]
    return {
        "id": ID, "wp": "H4", "kind": "test",
        "title": "WP H4 repair round: sub-grid observation operator (cell-mean kernel "
                 "subtracted) from census pumps within 2 km; high-swing MSE at held-out wells vs "
                 "two placebos, with design-time no-harm conditions on the v1 unit",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP H4; 2.5 (U1-U7); 4.3-4.7",
        "tests": tests,
        "zones": {"description": "T and S of the well's model layer at its cell from the cwa "
                                 "theta (engine/v2); not profiled in the primary; member "
                                 "t_profile_up profiles T x {1, 2, 4} at the wells whose T sits "
                                 "at the 2e4 m2/d upper bound by the 2012-19 fit loss; no S value "
                                 "is at a bound (1e-6..0.3; theta S 1.5e-4..7.5e-3)"},
        "priors": prereg3.not_applicable("no priors: least-squares multipliers, constrained >= 0"),
        "burn_in": {"description": "2012-01 is the hindcast's initial condition and is excluded"},
        "folds": {"description": "F uses no folds (multipliers from the 158 calibration wells; "
                                 "the held-out wells never enter a fit). The v1-unit conditions "
                                 "C2/C4 use the A1 site-fold partition fresh.folds.wells_all as "
                                 "data, as in v1"},
        "fresh_units": [UNIT],
        "thresholds": dict(CRIT),
        "gate_scores": [{"name": k, "rule": v, "weight": 0.125} for k, v in CRIT.items()],
        "gate_rule": (
            "subgrid_operator/v2 gate passes only if all of F1-F4 and C1-C4 pass. F1-F4 are the "
            "registered test on the held-out wells (fresh for H4). C1-C4 are computed on the v1 "
            "unit (158 wells, 2020-22), which H4 v1 already scored: they are post-look "
            "design-time evidence and enter only as necessary conditions (they can fail the "
            "gate, never pass it on their own). Pass: allowed uses model_design (secondary-model "
            "observation hook for I1, refitted after any recalibration), validator, "
            "design_time_evidence, after_the_fact_analysis. Fail: design_time_evidence and "
            "after_the_fact_analysis only. Thresholds are not changed after results; no 2023+ "
            "observation enters (U7)."),
        "primary_model": "h4v2lib.PRIMARY: " + json.dumps(cfg, sort_keys=True),
        "reported_not_gated": [
            "placebo p for T1 and T2 on the v1 unit (post-look)",
            "F2: held-out wells with >= 24 months 2012-02..2019-12 and >= 7 months 2020-22, "
            "training-mean datum and climatology, scored 2020-22 (held-out sites and years)",
            "sensitivity members (each F, T1, T2 without placebos): yunlin_only, r_max_3km, "
            "centroid_reference (Peaceman r_e = 0.198 dx cell-centred reference), "
            "one_multiplier, recorded_arm (C0 preorigin/v3 even split), t_profile_up",
            "Monte-Carlo sd of every p, concentration of the reductions (top-5 share), counts "
            "of improved / degraded wells, multipliers and fitted correction sd"],
        "design_time_units": [
            "H4r:d1_operator_no_heads (no head data: v1 kernel split into cell-mean and "
            "sub-grid parts, median corr 0.994, sub-grid sd 0.197 of the point sd; sub-grid "
            "unit sd 0.0071 m median at the 116 supported calibration wells)",
            "H4r:heldout_availability (month counts of the held-out candidates, no values)",
            "H4r:t1_synthetic_heads (power and bias on synthetic heads, no head data)",
            "H4:wells158_2020-2022 (v1 registered results, the v1 report and the verifier's "
            "findings: the motivation of v2; post-look for this unit)"],
        "inputs": inputs(),
        "interpretation": (
            "v1 subtracted the whole model-unit cone from the cell head, but the 1 km hindcast "
            "already carries the cell-average cone (verifier: the v1 kernel averaged over the "
            "well's cell correlates 0.99 with the point value), so v1 tested mostly a "
            "cell-scale re-scaling. v2 keeps only the within-cell (sub-grid) part, whose mean "
            "over the cell is zero: a pass would show location-specific sub-grid cones; a fail "
            "would say that the census-pump sub-grid term at model units, scaled by fan-wide "
            "multipliers, does not improve high-swing wells at held-out sites. Changhua and "
            "in-grid Chiayi pumps now enter: the zero-cell-mean kernel cannot absorb the "
            "missing Changhua cell-mean drawdown (v1's reason to drop them). At multiplier 1 "
            "the sub-grid correction is tiny (median sd 0.007 m); the multipliers are free (>= "
            "0) and a large value is a statistical scale, not a physical cone strength. The "
            "sub-grid sum is not converged in r_max (unit sd +50 % at 3 km; the far-field "
            "gradient term), so r_max 2 km is the plan's lower value, with r_max_3km reported. "
            "The held-out-years requirement of the plan can only be checked on the v1 unit, "
            "which is post-look: it enters as necessary conditions (C1-C4) and F2 reports the "
            "few held-out wells with 2020-22 data. Design-time power on synthetic heads (real "
            "availability masks, 20 replicates, 99 placebo draws; t1_power.json): truth 0 with "
            "an unmodelled seasonal residual -> joint F pass 0/20; a real sub-grid cone with "
            "median sd 0.25 m at supported wells -> joint pass 0.60 with the seasonal residual "
            "(a_s biased low: 14.6 vs 29.9) and 1.00 without; the same cone at shuffled pump "
            "positions -> 0/20. A fail therefore does not exclude cones of about 0.25 m sd."),
        "registered_on": _dt.datetime.now(_dt.timezone.utc).date().isoformat(),
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
        unv = prereg3.unverifiable_looks(dd["tests"], None, idx=idx)
        print(json.dumps({"valid": True, "n_blocking": len(blk),
                          "blocking": [{"pairs": b["pairs"][:4], "look": b["look"]}
                                       for b in blk[:10]],
                          "n_unverifiable": len(unv), "n_inputs": len(dd["inputs"]),
                          "n_ids": len(dd["fresh_units"]) and len(
                              dd["tests"][0]["external_units"][UNIT]["ids"])},
                         indent=1, default=str, ensure_ascii=False))
        return
    path, sha = prereg3.write_prereg(dd)
    print(json.dumps({"written": rel_to_repo(path), "sha256": sha}, indent=1))


if __name__ == "__main__":
    main()
