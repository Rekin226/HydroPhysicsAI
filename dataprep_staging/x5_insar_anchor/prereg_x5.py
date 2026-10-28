"""Write the WP X5 pre-registrations (harness.prereg3). Run each under the shared lock:

    flock results/dataprep/preregistration/.x5.lock python -I prereg_x5.py syn --write
    flock ... python -I prereg_x5.py t1 --write     (after S0 passed and predict_t1.py ran)
    flock ... python -I prereg_x5.py e1 --write     (before any post-2022 WRA or pile value)

Without --write the document is validated and the blocking looks are listed (read-only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import x5const as K  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
CODE = ["x5const.py", "x5lib.py", "synth.py", "predict_t1.py", "score_t1.py", "prereg_x5.py"]
PRED_DIR = K.OUT / "t1_predictions"
S0 = K.OUT / "synthetic" / "S0_result.json"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_inputs(names=CODE) -> list[str]:
    return [rel_to_repo(HERE / n) for n in names]


# ----------------------------------------------------------------------------- S0
def doc_syn() -> dict:
    return {
        "id": K.PREREG_SYN, "wp": K.WP, "kind": "declaration",
        "title": "X5 S0: synthetic recovery of leveling increments through per-epoch GNSS "
                 "anchoring of InSAR with a +-22-32 mm quarterly common mode",
        "tests": NA("declaration: synthetic recovery check, no real unit"),
        "checks": {
            "S0": "synth.py (module docstring): 20 seeds (101-120) of a known field sampled "
                  "through the T1 geometry (run P2 epochs and valid-pixel pattern, anchor "
                  "stations and their daily availability, the 801 benchmarks) with the "
                  "audit's corruption (common mode +-22-32 mm per quarter, random sign, 5 mm "
                  "jitter; 3 mm smooth per-epoch field; 4 mm pixel noise; 3 mm/sqrt(yr) "
                  "station-site random walk; 2 stations drifting +-40 mm/yr; GNSS 5 mm; "
                  "leveling 3 mm with +-20 d survey dates), recovered by the T1 pipeline.",
            "pass_criterion": K.SYN_PASS,
            "criterion_meaning": "C1 median over seeds of OLS slope / lambda in [0.95, 1.05], "
                                 "lambda = var(true leveling increment) / var(synthetic "
                                 "leveling increment) on the scored benchmarks (the known "
                                 "errors-in-variables attenuation of OLS); C2 median RMSE "
                                 "<= 5 mm; C3 T1 rule in >= 19 of 20 seeds",
            "consequence": "T1 is registered and scored only if S0 passes; a failed S0 is "
                           "reported and T1 is not run on that pipeline"},
        "zones": NA("whole InSAR grid"), "priors": NA("none"), "burn_in": NA("none"),
        "folds": NA("none"), "fresh_units": NA("synthetic draws; no real unit is scored"),
        "thresholds": K.SYN_PASS, "gate_scores": NA("declaration"),
        "primary_model": NA("declaration"),
        "inputs": code_inputs(["x5const.py", "x5lib.py", "synth.py"]),
        "interpretation": "a pass shows that the anchoring + May-window pipeline removes a "
                          "time-varying common mode of the audited size from the increments; "
                          "it says nothing about real InSAR error structure",
        "attempts_before_declaration": [
            "engineering smoke 1 (2026-10-25): crash (3-D valid mask passed to a 2-D helper); "
            "fixed in x5lib.point_matrix",
            "engineering smoke 2 (seed 101 only): anchored OLS slope 0.937, RMSE 4.66 mm, "
            "bias -1.79 mm; raw slope 0.937, RMSE 35.2 mm. The first draft criterion (median "
            "OLS slope in [0.95, 1.05]) was replaced, before this declaration, by slope / "
            "lambda, because the smoke run showed that the raw OLS slope tests the known "
            "attenuation by leveling noise, not the anchoring; disclosed here",
            "engineering smoke 3 (seed 101, import-order re-check after lint): same numbers; "
            "lambda 0.961, slope / lambda 0.974, RMSE against the true increments 3.73 mm. "
            "Seed 101 is one of the 20 declared seeds; it is kept (dropping it after seeing "
            "it would be a choice made on a result)"],
        "no_prior_look_statement": "No leveling, InSAR or GNSS value enters the synthetic "
                                   "data; only epochs, validity patterns and positions.",
        "written_by": rel_to_repo(Path(__file__)),
    }


# ----------------------------------------------------------------------------- T1
def t1_unit() -> dict:
    import x5lib as X
    b = X.bench_geometry()
    ids = sorted(b["sid"].astype(str))
    return {"description": "WRA leveling benchmarks with a survey in 2020-05 and in 2021-05 "
                           "(ls-wra-lsp-obs choushui panel): the 2020-05 -> 2021-05 "
                           "increments, scored against anchored InSAR alone. ids are given "
                           "bare (validation_units form) and with the 'lev:' prefix used by "
                           "S5, so membership overlaps with either form are detected",
            "ids": ids + [f"lev:{s}" for s in ids], "window": "2020-05..2021-05"}


def doc_t1() -> dict:
    pred = PRED_DIR / "predictions.csv"
    s0 = json.loads(S0.read_text(encoding="utf-8"))
    if not s0.get("pass"):
        raise SystemExit("S0 did not pass; T1 is not registered on this pipeline")
    unit = t1_unit()
    rule = (f"PASS iff n >= {K.T1_MIN_N} scored benchmarks AND the OLS slope of the anchored "
            f"InSAR increment (column p2_offset_scaled) on the leveling increment lies in "
            f"[{K.T1_SLOPE[0]}, {K.T1_SLOPE[1]}] AND RMSE(InSAR - leveling) <= {K.T1_RMSE_MM} "
            f"mm (bias included)")
    return {
        "id": K.PREREG_T1, "wp": K.WP, "kind": "test",
        "title": "X5 T1: anchored InSAR alone against the 2020-05 -> 2021-05 leveling "
                 "increments (801 benchmarks)",
        "tests": [{
            "test_id": K.T1_TEST_ID, "data_scope": "leveling:2020-05..2021-05",
            "fresh_units": [K.T1_UNIT], "external_units": {K.T1_UNIT: unit},
            "what": "Do per-epoch GNSS-anchored InSAR May-to-May increments reproduce the "
                    "leveling increments of the 2020-05 and 2021-05 campaigns?",
            "decision_rule": rule,
            "label": "registered test, scored once (predictions frozen and hashed below)"}],
        "fresh_units": [K.T1_UNIT],
        "registered_predictions": {"path": rel_to_repo(pred), "sha256": sha(pred),
                                   "anchors": {"path": rel_to_repo(PRED_DIR / "anchors.csv"),
                                               "sha256": sha(PRED_DIR / "anchors.csv")}},
        "primary_model": {
            "series": "insar_069A v3 method, run P2 (2020-01..2022-12 pairs only; U7: the "
                      "full-product run F2 contains 2023+ interferograms in its LiCSBAS15 mask "
                      "and gamma constraint, so it does not gate)",
            "why_not_raw_licsar": "the LiCSAR interferograms are the L0 input of P2; the "
                                  "LiCSBAS network inversion, loop-error nullification and "
                                  "off-fan reference of v3 are needed to form a time series",
            "anchoring": "median polish of D_s(t) = GNSS_s(t) - InSAR_s(t) into station "
                         "constants + epoch offsets c(t); anchors: gnss_clean/full/v4 WRA "
                         "'member' and NGL 'fan box' / 'on grid' stations, flags 0-1, 7-day "
                         "median, +-3 d match, 3 x 3 pixels >= 4 valid; epochs < 5 stations "
                         "dropped",
            "point_value": "3 x 3 multilooked pixels around the benchmark (<= 150 m), >= 4 "
                           "valid (flag 0 chain)",
            "may_value": "mean over epochs within May 15 +-30 d; increment scaled to 365 d by "
                         "the mean epoch dates (the 2021 window holds only 2021-04-18 and "
                         "2021-04-30: no acquisition 2021-05-01..2021-08-15)",
            "estimator": "OLS slope of InSAR on leveling with intercept; RMSE of InSAR - "
                         "leveling (bias included)",
            "screens": {"remark_disturbed": list(K.LEV_BAD_REMARK),
                        "gross_abs_increment_mm": K.LEV_MAX_ABS_MM,
                        "scored": "finite primary prediction"}},
        "declared_alternatives_not_gating": {
            "A1": "p2_offset_unscaled (no 365-d scaling)",
            "A2": "p2_plane_scaled (offset + east/north plane per epoch, >= 6 stations)",
            "A3": "v3_offset_scaled (full product F2 + second frame; contains 2023+ data "
                  "in its processing: report-only under U7)",
            "D": "p2_raw_scaled (no GNSS anchoring): diagnostic of what anchoring changes",
            "also_reported": "Deming, Theil-Sen and reverse-OLS slopes; county split "
                             "(Changhua / rest / off-grid); spatial 5 km block bootstrap 90 % "
                             "intervals (2000 draws, seed 20261025)"},
        "thresholds": {"slope_band": list(K.T1_SLOPE), "rmse_max_mm": K.T1_RMSE_MM,
                       "n_min": K.T1_MIN_N},
        "gate_scores": [{"name": "T1 pass/fail (slope band and RMSE, n >= 200)",
                         "weight": 1.0}],
        "zones": NA("one fan-wide test; county splits reported"),
        "priors": NA("nothing is fitted to leveling"), "burn_in": NA("no fit"),
        "folds": NA("single held-out row"),
        "synthetic_check": {"prereg": K.PREREG_SYN, "result": rel_to_repo(S0),
                            "sha256": sha(S0), "pass": bool(s0["pass"]),
                            "median_slope_over_lambda": s0["median_slope_over_lambda"],
                            "median_rmse_mm": s0["median_rmse_mm"],
                            "seeds_passing_t1_rule": s0["seeds_passing_t1_rule"]},
        "inputs": code_inputs() + [rel_to_repo(PRED_DIR / f) for f in
                                   ("predictions.csv", "anchors.csv", "summary.json")]
        + [rel_to_repo(S0)],
        "exposure_disclosure": [
            "The 2020-05 and 2021-05 leveling rows are not new to the project. S5 used them: "
            "the 2020-2021 surveys were the scored targets of S5.T1 (fused field of InSAR run "
            "P + GNSS + rings + piles + leveling <= 2019; scored 2026-10-17, ratio 1.042, "
            "coverage 0.631; unit s5.lev.t1_targets, 585 benchmarks, ids prefixed 'lev:') "
            "and entered S5's hyperparameter fits and fused product (s5.lev.t2_blocks). W3 "
            "and A1 looked at leveling 2011-2021 (DTM bias; design-time folds).",
            "No InSAR-alone comparison with this row exists in looks.jsonl or the reports: "
            "S4b ran G1/C1/Y only; docs/DATA_REMEDIATION_STATUS_2026-10.md lists the S4 "
            "'leveling 2020-21 withheld' row as 'not covered' and notes that using it needs "
            "an A1 ruling. Under prereg3's BLOCKING_RULE those earlier looks (design-time or "
            "external units, before this declaration) do not block; this test is therefore "
            "registered as a test with this disclosure, and the ruling is left to the lead.",
            "What the S5 exposure could have leaked into this design: nothing is fitted to "
            "leveling here; the predictions are frozen before registration; the method "
            "choices (P2, offset anchoring, 3 x 3 pixels, May windows) follow S4b's rules "
            "and the audit, not the S5.T1 residuals, which this WP has not read.",
            "This WP read no leveling elevation before registration (positions, survey "
            "presence and remarks only). The audit agents computed InSAR-GNSS biases "
            "(anchors, not targets) and left this row uncomputed on purpose."],
        "interpretation": {
            "pass": "anchored InSAR alone carries the leveling's annual increments in 2020-21 "
                    "to within the declared slope band and 10 mm: it can serve as a "
                    "post-2022 spatial reconstruction once its anchors exist (GNSS to "
                    "2025-03, piles to 2026-01), with the caveat that 2023+ anchoring has "
                    "fewer and drifting stations",
            "fail": "anchoring per epoch does not make InSAR a leveling substitute at the "
                    "annual-increment level; the report states which part failed (slope = "
                    "spatial scale/LOS conversion, RMSE/bias = level)"},
        "attempt_policy": "engineering retries (crash, NaN, missing file) before any metric "
                          "is seen are free and logged; the score runs once; at most one "
                          "exploratory follow-up after a failure, labelled post-failure",
        "written_by": rel_to_repo(Path(__file__)),
    }


# ----------------------------------------------------------------------------- E1
def doc_e1() -> dict:
    return {
        "id": K.PREREG_E1, "wp": K.WP, "kind": "declaration",
        "title": "X5 E1: anchored InSAR field 2020-2025 (product v1) against the WRA county "
                 "leveling summaries 2022-23, 2023-24, 2024-25 (exploration, U7)",
        "tests": NA("exploration: 2023+ data never gate (U7); cannot promote"),
        "checks": {
            "product_v1": "insar_069A/v3 merged ml field (F2 + tied second frame; valid = "
                          "finite, not bridged, source 1-3) anchored per epoch by the T1 "
                          "median polish on the T1 GNSS anchors to 2025-03-31; after that, "
                          "pile pseudo-stations at TKJS, STES, SJES, NTUH: G*(t) = Q(t_L) + "
                          "beta (t - t_L) - P(t), with P the cumulative printed compression "
                          "of the deepest sensor (flag 0), Q = GNSS + P monthly, beta its OLS "
                          "rate over 2020-06..2024-12 and Q(t_L) the mean of the last 3 GNSS "
                          "months; epochs after 2025-03-31 need >= 3 pile anchors",
            "intervals": [list(i) for i in K.E1_INTERVALS],
            "statistics": "May-window increments (May 15 +-30 d, scaled to 365 d) per pixel; "
                          "1-km twin cells = median of >= 5 valid pixels; per county "
                          "(zone_county_map v2: Changhua vs rest of the twin domain): max "
                          "sinking rate over cells (cm/yr), 99th percentile, area of cells "
                          "sinking > 2 and > 3 cm/yr (1 km2 each), coverage; alternative: "
                          "pixel areas",
            "comparison": "against subsidence_public_reports/postorigin/v1 county_intervals "
                          "(max_rate_cm_yr, area_gt2_km2, area_gt3_km2) for Changhua and "
                          "Yunlin: differences, ratios, and whether the year-to-year "
                          "direction of each statistic agrees (sign of the change)",
            "loso": "leave-one-station-out: per GNSS station, c(t) recomputed without it; "
                    "anchored InSAR increment at the station vs its GNSS increment for "
                    "2022-23 and 2023-24 (2024-25 has no GNSS at 2025-05)",
            "label": "post-2022 exploration; never validation; cannot promote; the WRA "
                     "areas are county-wide while the field covers the twin domain only"},
        "zones": NA("counties"), "priors": NA("none"), "burn_in": NA("none"),
        "folds": NA("none"), "fresh_units": NA("exploration; no fresh unit"),
        "thresholds": NA("reported, no gate"), "gate_scores": NA("declaration"),
        "primary_model": NA("declaration"),
        "inputs": code_inputs(["x5const.py", "x5lib.py"]),
        "interpretation": "describes how far an anchored InSAR field reproduces the only "
                          "post-2022 leveling truth (county aggregates); a match or mismatch "
                          "is context for Q6, never a skill claim",
        "prior_knowledge_disclosure": "the WRA county values were quoted in the 2026-10-25 "
                                      "audit report read by this WP; no product or "
                                      "statistic choice above uses them",
        "written_by": rel_to_repo(Path(__file__)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=["syn", "t1", "e1"])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    d = {"syn": doc_syn, "t1": doc_t1, "e1": doc_e1}[a.which]()
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    if errs:
        return 1
    if d["kind"] == "test":
        import datetime as dt
        idx = prereg3.membership_index()
        for _, ext in prereg3._external_dicts(d):
            for u, v in ext.items():
                m = prereg3._declared_members(v)
                if m is not None and not idx.known(u):
                    idx.members[idx.canon(u)] = m
        now = dt.datetime.now(dt.timezone.utc)
        blk = prereg3.blocking_looks(d["tests"], None, idx=idx, test_time=now)
        unv = prereg3.unverifiable_looks(d["tests"], None, idx=idx)
        print("blocking looks:", json.dumps(blk, ensure_ascii=False, default=str)[:2000])
        print("unverifiable prior looks:", len(unv),
              json.dumps(unv, ensure_ascii=False, default=str)[:1500])
    if not a.write:
        return 0
    path, h = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
