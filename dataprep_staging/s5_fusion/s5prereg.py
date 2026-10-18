"""WP S5: write the pre-registration ``S5.subsidence_fusion.v1`` with ``harness.prereg3``
BEFORE any fit of the fusion to real data (hyperparameters included) and before any score.

    python -I dataprep_staging/s5_fusion/s5prereg.py [--dry-run]

Registers the model, the pre-origin hyperparameter procedure, the product gate (T1 and T2) and
two report-only tests (T3 pre-origin; R2 after the fact, U7: never a gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5looks as Lk  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

CODE = ["s5const.py", "s5model.py", "s5data.py", "s5fit.py", "s5hyper.py", "s5tests.py",
        "s5insarP.py", "s5looks.py", "s5prereg.py", "tests/test_s5model.py"]


def t1_target_ids(lev: pd.DataFrame) -> list[str]:
    """Benchmarks with a survey in 2020-2021 and >= 3 surveys in 2012-2019 spanning >= 3 yr
    (dates only)."""
    out = []
    for sid, g in lev.groupby("sid"):
        pre = g[g.date <= C.T1_BASE_WIN[1]]
        new = g[(g.date >= C.T1_WITHHELD[0]) & (g.date <= C.T1_WITHHELD[1])]
        if len(new) and len(pre) >= C.T1_BASE_MIN_SURVEYS and \
                pre.t.max() - pre.t.min() >= C.T1_BASE_MIN_SPAN_YR:
            out.append(sid)
    return sorted(out)


def doc() -> dict:
    na = prereg3.not_applicable
    g = D.grid()
    lev, _ = D.leveling(g)
    gpre, _ = D.gnss(g, "pre")
    gfull, _ = D.gnss(g, "full")
    lev_ids = sorted(lev.sid.unique().tolist())
    t1_ids = t1_target_ids(lev)
    gpre_ids = sorted({f"gnss:{f}:{s}" for f, s in zip(gpre.family, gpre.station, strict=True)})
    gfull_ids = sorted({f"gnss:{f}:{s}" for f, s in zip(gfull.family, gfull.station,
                                                         strict=True)})
    ins_ids = Lk.insar_block_ids()
    cmp_ids = Lk.cmp_ids()
    ext1 = {"s5.lev.t1_targets": {
        "description": "leveling benchmarks in active cells (>= 2 surveys 2012-2021, no reset "
                       "> 0.5 m/yr) with a survey in 2020-01..2021-12 and >= 3 surveys in "
                       "2012-2019 spanning >= 3 yr: the withheld 2020-2021 surveys",
        "ids": t1_ids, "window": "2020-01..2021-12"}}
    ext2 = {"s5.lev.t2_blocks": {
        "description": "every S5 leveling benchmark (active cells, >= 2 surveys 2012-2021, no "
                       "reset), left out one 10 km complete-linkage block at a time",
        "ids": lev_ids, "window": "2012-01..2021-12"}}
    ext3 = {"s5.gnss.preorigin": {"description": "gnss_clean/preorigin/v2 stations within 3 km "
                                                 "of an active cell (flags 0-1)",
                                  "ids": gpre_ids, "window": "2012-01..2022-12"},
            "s5.cmp.preorigin": {"description": "rings (L0, <= 2021) and printed piles "
                                                "(<= 2022-12)", "ids": cmp_ids,
                                 "window": "2012-01..2022-12"},
            "s5.insar.runP": {"description": "InSAR run P (pre-origin pairs) 4 km blocks",
                              "ids": ins_ids, "window": "2020-01..2022-12"}}
    ext4 = {"s5.post2022.families": {
        "description": "post-2022 observations of gnss_clean/full/v2 (WRA, NGL), insar_069A/v2 "
                       "(run F), rings (L0, S1 interval) and printed piles",
        "ids": gfull_ids + ins_ids + cmp_ids, "window": "2023-01..2026-01"}}
    common_proc = (
        "Model: u(x,t) = mu_v (t - 2012) + predictive-process Matern(nu 3/2, range ell) field on "
        f"a {C.LATTICE_KM:g} km node lattice (nodes within {C.LATTICE_REACH_KM:g} km of an "
        "active cell) x integrated random walk on velocity at annual knots (u(2012-01-01) = "
        "0, v0 ~ N(0, sig_v0^2 K), intensity q) + seasonal pair ~ sig_s^2 K; separable "
        "precision; exact Gaussian inference (dense Cholesky). Operators as in s5fit.py "
        "(leveling: diffuse offset + random velocity sig_rb + white sig_lev, eliminated; GNSS: "
        "offset, station rate sig_d, family frame rate, family-month common mode sig_cm, "
        "white sig_gnss^2 + sd^2; InSAR 4 km block x network segment offsets, free plane per "
        "month projected out, APS exp(-d/ell_aps) sig_aps and white sig_ins over epochs "
        "averaged; rings and piles f_s u + per-segment offsets + random velocity sig_rc + "
        "white sig_cmp). Hyperparameters: procedure H (s5hyper.py), pre-origin data only, "
        "stages A-D with one |z| > 5 screening pass per family; f_s from the A-C fit.")
    tests = [
        {"test_id": "S5.T1_leveling_2020_2021_withheld",
         "data_scope": "leveling:2020-01..2021-12", "fresh_units": sorted(ext1),
         "external_units": ext1,
         "procedure": common_proc + (
             " T1 (mirror of the real post-2021 gap): training = every pre-origin instrument "
             "(GNSS <= 2022-12, InSAR run P, piles <= 2022-12, rings <= 2021) plus the leveling "
             "surveys <= 2019-12-31 (screened); stage-A hyperparameters re-estimated on that "
             "training leveling (s5hyper.py t1; B-D and the screening flags from the main "
             "pre-origin run), f re-estimated from the training fit without rings and piles. "
             "Targets: each withheld 2020-2021 survey of a benchmark with >= 3 training "
             "surveys spanning >= 3 yr, as the change from its last training survey. Fused "
             "predictive (exact; the benchmark's own offset and random velocity conditioned "
             "on its training surveys) and a leveling-only fit with the same hyperparameters. "
             "VR = 1 - Var_fused / Var_levonly of the field increment at the benchmark over the "
             "target interval; scored targets VR >= 0.5 (plan: fraction fixed before any "
             "score). Baseline (plan): continue the leveling-era rate, i.e. the benchmark's "
             "least-squares rate over its 2012-2019 surveys times the interval. Coverage of "
             "the fused 90 % predictive intervals by harness.coverage3 (rule gate-3) with "
             "clusters = the T2 blocks."),
         "threshold": ("PASS iff (a) >= 20 benchmarks scored, (b) RMSE_fused < RMSE_baseline "
                       "on the scored targets, (c) coverage3 gate passes on the scored "
                       "targets")},
        {"test_id": "S5.T2_leveling_block_out", "data_scope": "leveling:2012-01..2021-12",
         "fresh_units": sorted(ext2), "external_units": ext2,
         "procedure": common_proc + (
             " T2: blocks = complete-linkage clusters of the screened benchmarks with diameter "
             "<= 10 km (scipy fcluster, criterion distance, t = 10). Each block's leveling is "
             "left out (every other instrument and every other benchmark kept; main pre-origin "
             "hyperparameters and f, which saw every benchmark: disclosed). Targets: "
             "consecutive-survey increments of the held-out benchmarks; predictive = field "
             "increment + dt^2 sig_rb^2 + 2 sig_lev^2. Baseline: IDW (power 2, within 10 km, "
             ">= 3 neighbours outside the block) of the neighbours' least-squares rates "
             "2012-2021 (>= 3 surveys spanning >= 3 yr) times the interval."),
         "threshold": ("PASS iff (a) coverage3 gate passes on all targets (clusters = blocks) "
                       "and (b) RMSE_fused < RMSE_IDW on the targets with a baseline")},
        {"test_id": "S5.T3_family_out_preorigin",
         "data_scope": "subsidence_preorigin:2012-01..2022-12",
         "fresh_units": sorted(ext3), "external_units": ext3,
         "procedure": common_proc + (
             " T3 (report only): pre-origin; GNSS, rings+piles and InSAR left out in turn. GNSS "
             "and rings/piles: each series de-trended (offset + rate per station, per sensor "
             "segment) against the same projection of the fused prediction (zero baseline = "
             "constant rate); coverage of 90 % intervals (clusters = sites). InSAR: per-unit "
             "rate relative to the segment median against the fused and the leveling-only "
             "predictions (correlation, RMSE)."),
         "threshold": "report only (no pass/fail)"},
        {"test_id": "S5.R2_post2022_family_holdout",
         "data_scope": "subsidence_post2022:2023-01..2026-01",
         "fresh_units": sorted(ext4), "external_units": ext4,
         "procedure": common_proc + (
             " R2 (after the fact; U7: never a gate): full data, pre-origin hyperparameters and "
             "f; leveling screening from the pre-origin run, GNSS and ring/pile screening by "
             "one pass on the full fit. For each family (WRA GNSS, NGL GNSS, InSAR, piles, "
             "rings) its observations from 2023-01-01 on are withheld and predicted. GNSS: "
             "change from the station's 2022 mean (baseline: its own 2020-2022 rate "
             "continued); rings and piles: change within a post-2022 segment from its first "
             "value (baseline: own pre-2023 rate continued); InSAR: within-segment change from "
             "the unit's first 2023+ month, de-planed per month (baseline: the leveling-only "
             "field, i.e. the leveling-era velocity continued). Scored where VR >= 0.5 against "
             "the leveling-only fit."),
         "threshold": "report only (no pass/fail; U7)"}]
    fresh = sorted(set(ext1) | set(ext2) | set(ext3) | set(ext4))
    inputs = [rel_to_repo(HERE / c) for c in CODE] + [
        rel_to_repo(C.LEVELING_PANEL), rel_to_repo(C.RING_STATIONS),
        rel_to_repo(C.GNSS_PRE / "MANIFEST.json"), rel_to_repo(C.GNSS_FULL / "MANIFEST.json"),
        rel_to_repo(C.INSAR_V2 / "MANIFEST.json"), rel_to_repo(C.PILES / "MANIFEST.json"),
        rel_to_repo(C.RINGS_BRIDGED / "MANIFEST.json"),
        rel_to_repo(C.WORK / "insar_runP_cells.npz"), rel_to_repo(C.WORK / "insar_runP_cells.json"),
        rel_to_repo(C.UNITS_V3), rel_to_repo(C.TWIN_GRID),
        "dataprep_staging/s1_rings/s1lib.py", "dataprep_staging/s1_rings/s1const.py",
        "dataprep_staging/harness/coverage3.py", "dataprep_staging/common/guard.py"]
    return {
        "id": C.PREREG_ID, "wp": C.WP, "kind": "test",
        "title": "Space-time subsidence fusion (leveling, GNSS, InSAR, rings, piles): product "
                 "gate T1 (2020-2021 leveling withheld, VR >= 0.5 mask, continue-rate "
                 "baseline) and T2 (10 km leveling blocks out); report-only T3 (family out, "
                 "pre-origin) and R2 (post-2022 family hold-out, after the fact)",
        "tests": tests,
        "zones": na("no zones; spatial units are leveling benchmarks and instrument sites"),
        "priors": {"latent": "separable Matern x IRW (s5model.py)",
                   "mu_v": f"N(0, {C.MU_V_SD}^2) mm/yr",
                   "gnss_frame_rate": f"N(0, {C.FRAME_RATE_SD}^2) mm/yr",
                   "offsets": "diffuse (restricted likelihood or sd 1e4)",
                   "hyperparameters": "type-II ML (no hyperprior); bounds in s5const.py"},
        "burn_in": na("no model spin-up: u(2012-01-01) = 0 by definition"),
        "folds": {"T1": "temporal block: leveling 2020-01..2021-12 withheld",
                  "T2": "leave one 10 km complete-linkage benchmark block out",
                  "T3": "leave one instrument family out (pre-origin)",
                  "R2": "post-2022 observations of one family withheld (after the fact)"},
        "fresh_units": fresh,
        "thresholds": {
            "VR_min": C.VR_MIN, "coverage_level": C.COVERAGE_LEVEL,
            "T1": "evaluable (>= 20 scored benchmarks) and ratio RMSE_fused/RMSE_baseline < 1 "
                  "and coverage3 pass",
            "T2": "coverage3 pass and ratio RMSE_fused/RMSE_IDW < 1",
            "product_gate": "subsidence_fused/v1 gate = T1 and T2 (pre-origin data only, U7). "
                            "Whatever the gate, the product's allowed uses are "
                            "after_the_fact_analysis only while its GNSS, InSAR and ring "
                            "inputs are gate-failed products (S3, S4b, S1a); validator use in "
                            "VR >= 0.5 cells needs those user decisions and a new version.",
            "report_only": ["T3", "R2", "T1 and T2 rows on all targets and leveling-only "
                            "predictions", "cluster-bootstrap 90 % intervals of the RMSE "
                            "ratios (2000 draws, seed 5005)"]},
        "gate_scores": na("conjunctive pass/fail of T1 and T2; no weighted score"),
        "primary_model": {"lattice_km": C.LATTICE_KM, "lattice_reach_km": C.LATTICE_REACH_KM,
                          "matern_nu": C.MATERN_NU, "knots_full": list(C.KNOTS_FULL),
                          "knots_preorigin": list(C.KNOTS_PRE), "insar_block": C.INSAR_BLOCK,
                          "hyper_start": C.HYPER_START, "hyper_bounds": C.HYPER_BOUNDS,
                          "stages": {k: list(v) for k, v in C.STAGES.items()},
                          "maxfev": C.NM_MAXFEV, "outlier_z": C.OUTLIER_Z,
                          "code": [rel_to_repo(HERE / c) for c in CODE]},
        "inputs": inputs,
        "interpretation": (
            "Disclosures. (1) Upstream gates: gnss_clean (S3), insar_069A (S4b, its G1 failed "
            "on run P itself) and rings_bridged (S1a) failed their gates; they are opened for "
            "after_the_fact_analysis only and the S5 product inherits that restriction. The "
            "printed piles (S2) passed. Rings enter from L0 with free offsets per segment, so "
            "S1's bridge and scale correction never enter; InSAR enters only through "
            "within-segment changes (free offsets per network segment), so LiCSBAS-bridged "
            "gap offsets never enter. (2) U7: hyperparameters, f, the gate and every "
            "threshold use pre-origin data only (InSAR run P, gnss_clean/preorigin/v2, piles "
            "<= 2022-12, rings <= 2021); R2 uses 2023+ observations and is report only. (3) "
            "Design-time status of the leveling: every benchmark was a hold-out unit of the "
            "2026-09 zonal-column CV and was scored by W3 (A1: dt.folds.leveling_*); A1 rules "
            "that no leveling unit is fresh, so T1 and T2 are S5-registered tests whose "
            "results are design-time evidence in A1's sense. They were not used to design "
            "S5: before this registration S5 only built the data tables (look "
            "'s5.dt.data_build', structure only) and ran the code on synthetic values. The "
            "R2 data (2023+ GNSS, InSAR, rings, piles) were looked at by S1b, S3 and S4b. "
            "(4) T1 re-estimates the leveling hyperparameters on its training data; T2 uses "
            "the main pre-origin hyperparameters, which saw every benchmark (mild optimism, "
            "disclosed). (5) The 0.5 VR fraction is the plan's; the skill thresholds (ratio "
            "< 1) and coverage rule (gate-3) are fixed here. Thresholds are not changed after "
            "results. (6) Expectation: InSAR (2020-2022) and GNSS should cut the variance of "
            "2020-2021 increments in much of Yunlin; northern Changhua lies outside the InSAR "
            "footprint and has few GNSS stations, so it may fall outside the VR mask."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs or a.dry_run:
        print(json.dumps(errs or {"ok": True, "n_inputs": len(d["inputs"]),
                                  "fresh_units": d["fresh_units"],
                                  "t1_targets": len(d["tests"][0]["external_units"][
                                      "s5.lev.t1_targets"]["ids"])}, indent=1))
        return 1 if errs else 0
    path, sha = prereg3.write_prereg(d)
    print(f"wrote {path} sha256 {sha}")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
