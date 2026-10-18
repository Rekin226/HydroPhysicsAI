"""WP S5 repair round 2: the registration ``S5.repair_r2.v1`` (harness.prereg3), written after
the diagnosis (reports/S5.repair.md Part 1, hashed 2026-10-18T02:52Z) and the method design
(Part 2), BEFORE any real fit of the repaired model and before any score.

One registered (gating) test on FRESH units: T1F, the plan's temporal block at a pseudo-origin
of 2020-01 (leveling 2018-01..2019-12 withheld; never an S5 temporal target). The second looks
on spent units (T1, T2, T2 with buffer, T3) are declared here and logged before each is
computed; they cannot open the gate.

    python -I s5r2_prereg.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5r2_const as K  # noqa: E402
import s5r2_tests as RT  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import from_repo, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

UNIT = "s5r2.lev.t1f_2018_2019"
ROUND1_CODE = ["s5const.py", "s5model.py", "s5data.py", "s5fit.py", "s5tests.py"]
R1_CODE = ["s5r_core.py", "s5r_vconst.py"]
R2_CODE = ["s5r2_const.py", "s5r2_model.py", "s5r2_data.py", "s5r2_hyper.py", "s5r2_tests.py",
           "s5r2_insarP2.py", "s5r2_prereg.py", "s5r2_dryrun.py", "tests/test_s5r2.py"]
SHARED = ["dataprep_staging/s1_rings/s1lib.py", "dataprep_staging/s1_rings/s1const.py",
          "dataprep_staging/common/guard.py", "dataprep_staging/common/hashing.py",
          "dataprep_staging/common/paths.py", "dataprep_staging/harness/coverage3.py",
          "dataprep_staging/harness/prereg3.py", "dataprep_staging/s4_insar/s4b_lib.py",
          "dataprep_staging/s4_insar/s4br_lib.py", "dataprep_staging/s4_insar/s4br_config.py"]
PART1_SHA = "9340a55819a454c5640091cd148e732e3ce138517e1385c8e6f30394384add3d"


def code_paths() -> list[str]:
    return [rel_to_repo(HERE / n) for n in ROUND1_CODE + R1_CODE + R2_CODE] + SHARED


def data_inputs() -> list[str]:
    from s1_rings import s1const as K1
    f = [C.LEVELING_PANEL, C.RING_STATIONS, C.TWIN_GRID,
         K.GNSS_PRE_V4 / "MANIFEST.json", K.GNSS_PRE_V4 / "stations.csv",
         K.GNSS_PRE_V4 / "wra" / "monthly.parquet", K.GNSS_PRE_V4 / "ngl" / "monthly.parquet",
         K.GNSS_PRE_V4 / "wra" / "daily.parquet", K.GNSS_PRE_V4 / "ngl" / "daily.parquet",
         K.GNSS_FULL_V4 / "MANIFEST.json", K.GNSS_FULL_V4 / "stations.csv",
         C.PILES / "MANIFEST.json", C.PILES / "printed_monthly.parquet",
         K.INSAR_P2_CELLS, K.INSAR_P2_CELLS.with_suffix(".json"),
         C.EVIDENCE / "repair_r2" / "snapshots" / "part1_diagnosis.md",
         C.EVIDENCE / "repair_r2" / "snapshots" / "part2_design.md"]
    f += [C.RING_DIR / f"{s}.parquet" for s in K1.SITES]
    return [rel_to_repo(p) for p in f]


def unit_members() -> list[str]:
    """Benchmarks with a survey dated 2018-01..2019-12 in an active cell (dates and positions
    only; no elevation is read)."""
    g = D.grid()
    df = pd.read_parquet(C.LEVELING_PANEL, columns=["sid", "datetime", "x_3826", "y_3826"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    pos = df.sort_values("datetime").groupby("sid")[["x_3826", "y_3826"]].first()
    w = df[(df.datetime >= K.T1F_WITHHELD[0]) & (df.datetime <= K.T1F_WITHHELD[1])]
    sids = sorted(w.sid.unique())
    cell = D.cell_of(g, pos.loc[sids, "x_3826"].to_numpy(), pos.loc[sids, "y_3826"].to_numpy())
    return [f"lev:{s}" for s, c in zip(sids, cell, strict=True) if c >= 0]


MODEL = (
    "Repaired model (s5r2_model.py; Part 2 of reports/S5.repair.md): round-1 space-time "
    "structure (u = mu_v (t - 2012) + predictive-process Matern 3/2 field on the 4 km lattice x "
    "temporal basis at annual knots, cubic Hermite between knots, seasonal pair, separable "
    "prior, exact Gaussian inference) with R1 the temporal prior v_k = b + beta_k, u_k = sum "
    "d_j (b + (beta_{j-1} + beta_j)/2) + eps_k (b ~ N(0, sig_v0^2), beta ~ N(0, sig_be^2), eps ~ "
    "N(0, sig_ep^2), u_0 = 0) replacing the IRW; R2 leveling campaign offsets (county x survey "
    "month) ~ N(0, sig_camp^2) on top of the round-1 leveling operator (diffuse offset, random "
    "velocity sig_rb, white sig_lev); R3 GNSS white (sig_gnss^2 + sd^2) plus station-level "
    "sig_gc^2 exp(-|dt|/tau_g), with offset, excess rate sig_d, family frame rate, family-month "
    "common mode sig_cm and offset segments at daily-record gaps > 30 d (repair round 1); R4 "
    "rings and piles f_s u + per-segment diffuse offsets + covariance sig_cmp^2 I + sig_rc^2 "
    "tau tau' + sig_cc^2 exp(-|dt|/tau_c), f_s from the fit without rings and piles (round-1 "
    "rule, bounds 0.2-1.2); R5 InSAR per-unit random rate ~ N(0, sig_ir^2) added to the round-1 "
    "operator (unit offsets, free plane per month projected out, exponential APS + white); R6 "
    "inputs gnss_clean/preorigin/v4 and run P2 cells. Hyperparameters: procedure H2 "
    "(s5r2_hyper.py; stages A-D, Nelder-Mead budgets and restarts in s5r2_const.py, one |z| > 5 "
    "screening pass per family, best evaluated point kept).")

T1F_PROC = (
    "T1F (s5r2_tests.py t1f): pseudo-origin 2020-01. Hyperparameters by H2 chain 't1f' "
    "(s5r2_hyper.py t1f): leveling <= 2017-12-31 only (the 2018-2019 surveys are not loaded), "
    "GNSS (preorigin/v4) <= 2019-12, rings (interval from 2017-2019 coverage) and printed piles "
    "<= 2019-12, no InSAR (none before 2020); stage C skipped; knots 2012..2018 for stage A, "
    "2012..2020 for B, D and the test fits. Fits: fused (leveling + GNSS + rings + piles) and "
    "leveling-only, same hyperparameters, chain screening flags and depth fractions. Targets: "
    "every leveling survey dated 2018-01-01..2019-12-31 of a benchmark (active cell, round-1 "
    "leveling QC) with >= 3 kept training surveys spanning >= 3 yr, as the change from its last "
    "training survey; prediction exact as in round-1 T1 (benchmark offset and random velocity "
    "conditioned on its training surveys) plus the campaign term (training campaigns are "
    "parameters; a withheld campaign adds sig_camp^2). VR = 1 - Var_fused / Var_levonly of the "
    "field increment over the target interval; scored where VR >= 0.5. Baseline: the "
    "benchmark's least-squares rate over its 2012-2017 training surveys times the interval. "
    "Coverage of the 90 % predictive intervals by harness.coverage3 (gate-3), clusters = 10 km "
    "complete-linkage blocks of the training benchmarks; cluster bootstrap of the RMSE ratio "
    "(2000 draws, seed 5005), report only.")

SECOND_LOOKS = {
    "T1_second_look": ("registered T1 design of S5.subsidence_fusion.v1 (2020-21 withheld, "
                       "training leveling <= 2019-12 with stage A re-estimated on it, B-D from "
                       "H2 main, f re-estimated without rings and piles, VR >= 0.5, continue-"
                       "rate baseline 2012-2019, coverage3) with the repaired model and inputs; "
                       "spent units: a pass is conditional (second look), never gating"),
    "T2_second_look": ("registered T2 design (10 km complete-linkage blocks, no buffer, H2 main "
                       "hyperparameters, IDW baseline within 10 km), repaired model; second "
                       "look"),
    "T2b_second_look": ("T2 with plan 4.3's >= 10 km buffer: benchmarks within 10 km of a "
                        "held-out block also left out of the fit, IDW of neighbours beyond the "
                        "buffer within 20 km; corrected design (the registered T2 lacked the "
                        "buffer the plan requires), same thresholds; its round-1 values were "
                        "seen after the fact in repair round 1, so second look"),
    "T3_report": ("leave one instrument family out, pre-origin, report only (as registered)"),
}


def doc() -> dict:
    na = prereg3.not_applicable
    ext = {UNIT: {"description": ("leveling surveys dated 2018-01..2019-12 of benchmarks in "
                                  "active cells: the withheld targets of S5.r2.T1F (never an "
                                  "S5 temporal target)"),
                  "ids": unit_members(), "window": "2018-01..2019-12"}}
    code = code_paths()
    return {
        "id": K.PREREG_ID, "wp": C.WP, "kind": "test",
        "title": ("S5 repair round 2: repaired subsidence fusion (temporal prior, correlated "
                  "instrument noise, campaign offsets, InSAR unit rates, repaired inputs) - "
                  "fresh temporal-block test T1F (2018-2019 leveling withheld, pseudo-origin "
                  "2020-01) and declared second looks"),
        "tests": [{"test_id": RT.TEST_T1F, "data_scope": "leveling:2018-01..2019-12",
                   "fresh_units": [UNIT], "external_units": ext}],
        "fresh_units": [UNIT],
        "zones": na("no zones; units are leveling benchmarks"),
        "burn_in": na("no spin-up: u(2012-01-01) = 0 by definition"),
        "folds": {"T1F": "temporal block at the pseudo-origin 2020-01 (leveling 2018-19 "
                         "withheld; every instrument after 2019-12 and every leveling survey "
                         "after 2017-12 excluded from the fit)"},
        "priors": {"hyper_start": K.HYPER_START, "hyper_bounds": K.HYPER_BOUNDS,
                   "stages": K.STAGES, "nm_maxfev": K.NM_MAXFEV, "nm_restart": K.NM_RESTART,
                   "mu_v": f"N(0, {C.MU_V_SD}^2)", "gnss_frame_rate":
                   f"N(0, {C.FRAME_RATE_SD}^2)", "offsets": "diffuse"},
        "thresholds": {
            "T1F": ("PASS iff (a) >= 20 benchmarks scored at VR >= 0.5, (b) RMSE_fused < "
                    "RMSE_baseline on the scored targets, (c) coverage3 gate-3 passes on the "
                    "scored targets (clusters = 10 km blocks); identical to the registered T1 "
                    "thresholds of S5.subsidence_fusion.v1"),
            "VR_min": C.VR_MIN, "coverage_level": C.COVERAGE_LEVEL,
            "product_gate": ("subsidence_fused/v4 gate = T1F (fresh) AND T2 (registered design; "
                             "only a second look exists, so a T1F pass with a passing T2 second "
                             "look is reported as 'conditional (second look)'). Whatever the "
                             "gate, allowed uses stay after_the_fact_analysis while the GNSS, "
                             "InSAR and ring inputs are gate-failed products"),
            "second_looks_same_thresholds": ("T1: as T1F; T2 and T2b: coverage3 gate-3 on all "
                                             "targets (clusters = blocks) and RMSE_fused < "
                                             "RMSE_IDW")},
        "gate_scores": na("conjunctive pass/fail"),
        "primary_model": {"model": MODEL, "test_T1F": T1F_PROC, "second_looks": SECOND_LOOKS,
                          "code": code, "test_id_T1F": RT.TEST_T1F},
        "inputs": code + data_inputs(),
        "interpretation": (
            "Repair round 2 of WP S5 after the failed gate of S5.subsidence_fusion.v1 (T1 ratio "
            "1.042 and coverage 0.631; T2 coverage 0.940). Diagnosis (S5.repair.md Part 1, "
            f"snapshot sha256 {PART1_SHA}): D1 the IRW temporal prior is mis-specified (rate "
            "anomalies revert: lag-1 -0.36), D2 the instrument error models treat "
            "autocorrelated GNSS and ring/pile residuals as white and give InSAR no block rate "
            "error (rings/piles tip T1 from 11.06 to 12.24 mm and rms z from 1.74 to 2.47), D3 "
            "inputs (GNSS v2 GFES step; run P per-cell gaps), D4 T2 lacks the plan's buffer and "
            "leveling campaign errors are missing. Fresh units: the only leveling window never "
            "used as an S5 temporal target is 2018-2019; it is used here at a pseudo-origin "
            "2020-01 that mirrors the real gap structurally (leveling ends, other instruments "
            "continue) but without InSAR (none before 2020), so T1F tests the temporal prior "
            "and the GNSS, ring and pile operators, not the InSAR operator. Disclosures: (1) "
            "A1 rules that no leveling unit is fresh (every benchmark was a 2026-09 zonal-column "
            "hold-out and W3 scored them); the 2018-19 surveys were training data of round-1 "
            "procedure H and of the round-1 T1 stage A, and their consecutive increments were "
            "pooled targets of round-1 T2 (leave-block-out, spatial) and of repair round 1's "
            "buffered T2. No S5 look in this round read them: the diagnosis looks of "
            "2026-10-18 cover leveling 2012-2017 and 2020-2021 only, and the dry runs used "
            "synthetic values. T1F is therefore 'fresh window, design-time evidence in A1's "
            "sense'. (2) The method changes were designed on spent T1/T3 evidence and on "
            "2012-2017 leveling; none was tuned against T1F's window. (3) The leveling QC "
            "(reset rule over 2012-2021) is round 1's and decides benchmark membership before "
            "the split. (4) Upstream products gnss_clean v4 and insar_069A run P2 failed their "
            "own gates; S5 inherits after_the_fact_analysis. (5) Thresholds are those of the "
            "round-1 registration and are not changed."),
        "design_time_exposure": [
            "S5 round-1 and repair-round-1 evidence (T1, T2, T3, R2, M1, diagnostics, product "
            "v1-v3) and the verifier's findings were seen",
            "diagnosis looks 2026-10-18T02:43:26Z (leveling 2012-2017, 2020-2021; GNSS, InSAR, "
            "rings, piles pre-origin) and the P2 cell build look 02:55:21Z",
            "dry runs of H2 and T1F on SYNTHETIC values (look 02:58:41Z; "
            "S5_evidence/repair_r2/dryrun): code paths, run times and the budgets were set "
            "from them; no real value was used"],
        "code_at_registration": {p: sha256_file(from_repo(p)) for p in code},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    if a.dry_run:
        errs = prereg3.validate_prereg(d)
        print(json.dumps({"errors": errs, "inputs": len(d["inputs"]),
                          "members": len(d["tests"][0]["external_units"][UNIT]["ids"])},
                         indent=1))
        return 0
    path, sha = prereg3.write_prereg(d)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
