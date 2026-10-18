"""WP S5 repair round: the registration ``S5.monthly_field.v1`` (test M1 of the monthly
component), harness.prereg3; written BEFORE the stage-2 fit and before any M1 score.

    python -I s5r_prereg_m1.py [--dry-run]

Written after the verifier's review of subsidence_fused/v1 (2026-10-17); the registered gate
tests T1 and T2 of S5.subsidence_fusion.v1 are NOT re-run, re-scored or changed. This file and
every module it lists are registered inputs: they stay unchanged after the registration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5r_const as R  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

V1_PREREG_SHA = "6b56e526c4c334ac0b58648da93c64be0b409ab838963c60ade35a860c579a29"
ROUND1_CODE = ["s5const.py", "s5model.py", "s5data.py", "s5fit.py", "s5hyper.py"]
REPAIR_CODE = ["s5r_const.py", "s5r_stage2.py", "s5r_mtest.py", "s5r_prereg_m1.py",
               "tests/test_s5r_stage2.py"]
SHARED_CODE = ["dataprep_staging/s1_rings/s1lib.py", "dataprep_staging/s1_rings/s1const.py",
               "dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
               "dataprep_staging/common/hashing.py", "dataprep_staging/common/paths.py",
               "dataprep_staging/common/uses.py", "dataprep_staging/harness/coverage3.py",
               "dataprep_staging/harness/prereg3.py"]


def code_paths(names) -> list[str]:
    return [rel_to_repo(HERE / n) for n in names]


def pre_data_inputs() -> list[str]:
    from s1_rings import s1const as K1
    files = [C.LEVELING_PANEL, C.RING_STATIONS, C.TWIN_GRID,
             C.GNSS_PRE / "MANIFEST.json", C.GNSS_PRE / "stations.csv",
             C.GNSS_PRE / "wra" / "monthly.parquet", C.GNSS_PRE / "ngl" / "monthly.parquet",
             C.GNSS_FULL / "MANIFEST.json", C.GNSS_FULL / "stations.csv",
             C.PILES / "MANIFEST.json", C.PILES / "printed_monthly.parquet",
             C.WORK / "insar_runP_cells.npz", C.WORK / "insar_runP_cells.json",
             C.EVIDENCE / "hyper_pre.json"]
    files += [C.RING_DIR / f"{s}.parquet" for s in K1.SITES]
    return [rel_to_repo(f) for f in files]


def hashes(paths: list[str]) -> dict:
    from common.paths import from_repo
    return {p: sha256_file(from_repo(p)) for p in paths}


def m1_station_ids() -> list[str]:
    g = D.grid()
    gpre, _ = D.gnss(g, "pre")
    w = gpre[gpre.family == R.M1_FAMILY]
    return sorted({f"gnss:{R.M1_FAMILY}:{s}" for s in w.station.unique()})


# ============================================================================ test M1
def test_doc() -> dict:
    na = prereg3.not_applicable
    unit = "s5m.gnss.wra_monthly_2020_2022"
    ext = {unit: {
        "description": ("WRA GNSS stations of gnss_clean/preorigin/v2 within 3 km of an active "
                        "cell (flags 0-1, n_days >= 10): monthly heights 2020-01..2022-12, the "
                        "targets of S5.M1 (GNSS is withheld from the annual fit and from stage "
                        "2)"),
        "ids": m1_station_ids(), "window": "2020-01..2022-12"}}
    proc = (
        "Stage 2 (s5r_stage2.py, s5r_mtest.py hyper): annual fit = the registered round-1 "
        "model (S5.subsidence_fusion.v1 code, hyper_pre.json, f from hyper_pre.json, "
        "pre-origin screening flags) on pre-origin data WITHOUT GNSS: leveling 2012-2021, "
        "InSAR run P, rings <= 2021, printed piles <= 2022-12, knots 2012..2023. Residuals of "
        "its InSAR run-P month groups (posterior means of field incl. seasonal pair and unit "
        "offsets removed) over 2020-01..2022-12 (36 months). Model: r_e = Phi_e a_m + plane_e "
        "+ aps + white; a_m on the annual model's 4 km node lattice, Matern nu 3/2 range ell_a "
        "x stationary AR(1) in months (rho, marginal sd sig_a); plane per month group projected "
        "out; exponential APS (sig_aps2, ell_aps2) + white (sig_ins2) divided by the epochs "
        "averaged. Exact Gaussian inference (block-tridiagonal precision), restricted marginal "
        "likelihood. Hyperparameters: (i) noise-only model (sig_ins2, sig_aps2, ell_aps2) by "
        "Nelder-Mead (maxfev 600) from the annual model's sig_ins, sig_aps, ell_aps_km; (ii) the "
        "full model by Nelder-Mead (maxfev 800, then one restart from its optimum with a fresh "
        "simplex of step 0.2, maxfev 400) from the noise-only optimum with sig_a 3.0, ell_a = "
        "ell_km, rho 0.5; log space (rho: logit of rho / 0.99), xatol 1e-3, fatol 0.05, first "
        "simplex step 0.35, bounds by clipping with a quadratic penalty; the returned values "
        "are the best evaluated point. Bounds: " + json.dumps(R.S2_BOUNDS) + ". Support of the "
        "anomaly: month footprint F_m = cells inside a 4 km unit with an InSAR value that month "
        "(>= 30 cells), plane-free operator psi = phi - x~'(X_F'X_F)^-1 X_F' Phi_F (x~ = 1, "
        "east, north centred on F_m), VR_dt = 1 - psi' Sigma_m psi / psi' S psi; supported "
        "where the point is in F_m and VR_dt >= 0.5. Report only: the likelihood ratio of the "
        "full against the noise-only model.")
    test_txt = (
        "M1 (s5r_mtest.py score): targets = WRA monthly GNSS (flags 0-1, n_days >= 10, "
        "hyper_pre screening) 2020-01..2022-12 at stations with >= 12 months, in months with "
        ">= 7 such stations. g = y - u_annual (GNSS-out annual posterior mean at the station "
        "point and month mid, incl. seasonal pair and mean velocity); p = stage-2 posterior-mean "
        "anomaly at the station point. One operator T = I - X pinv(X), X = station offset + "
        "station rate (t - station mean t) per station and a plane (1, east, north; centred "
        "coordinates) per month; z = T g, Tp = T p, e = z - Tp. Scored station-months: the "
        "station's twin cell lies in the month's run-P footprint AND VR_dt at the station point "
        ">= 0.5. Evaluable: >= 100 scored station-months at >= 8 stations. Gate (skill): "
        "RMSE(e) / RMSE(z) < 1 on the scored station-months AND the station-cluster bootstrap "
        "of that ratio (2000 draws, seed 5006) gives P(ratio >= 1) <= 0.10. sd label: hits "
        "|e| <= 1.645 sd_e, sd_e^2 = diag(T (S_annual + S_anomaly + diag(sig_gnss^2 + "
        "sd_month^2) + sig_cm^2 [same month]) T'), coverage3 gate-3 with station clusters: "
        "pass -> 'calibrated at monthly resolution on pre-origin WRA station-months', else "
        "'uncalibrated'. Report only: all station-months, by year, corr(z, Tp), the stage-2 "
        "likelihood ratio against noise only.")
    return {
        "id": R.MTEST_ID, "wp": C.WP,
        "title": ("S5 repair round: InSAR-supported monthly anomaly of the subsidence fusion "
                  "(stage 2) - registered pre-origin test M1 against withheld WRA GNSS months "
                  "2020-2022"),
        "kind": "test",
        "tests": [{"test_id": R.MTEST_TEST, "data_scope": "gnss_wra_monthly:2020-01..2022-12",
                   "fresh_units": [unit], "external_units": ext}],
        "zones": na("no zones; units are GNSS stations"),
        "priors": {"stage2_bounds": R.S2_BOUNDS,
                   "stage2_start": "noise from the annual model (sig_ins, sig_aps, "
                                   "ell_aps_km), sig_a 3.0, ell_a = ell_km, rho 0.5",
                   "annual": "S5.subsidence_fusion.v1 (hyper_pre.json)"},
        "burn_in": na("no spin-up; the AR(1) anomaly is stationary"),
        "folds": {"M1": "GNSS (every family) withheld from the annual fit and from stage 2; "
                        "one score over the WRA station-months"},
        "fresh_units": [unit],
        "thresholds": {"M1_evaluable": f">= {R.M1_MIN_TARGETS} scored station-months at >= "
                                       f"{R.M1_MIN_STATIONS} stations",
                       "M1_skill": "RMSE(T(g - p)) / RMSE(T g) < 1 and bootstrap "
                                   "P(ratio >= 1) <= 0.10 (2000 station-cluster draws)",
                       "support_vr": R.VR_A_MIN,
                       "sd_label": "coverage3 gate-3 pass (station clusters)",
                       "coverage_level": 0.90},
        "gate_scores": na("conjunctive: evaluable and skill"),
        "primary_model": {"procedure_stage2": proc, "test_M1": test_txt,
                          "code": code_paths(ROUND1_CODE + REPAIR_CODE) + SHARED_CODE,
                          "constants": {k: getattr(R, k) for k in (
                              "S2_NAMES", "S2_NOISE", "S2_MAXFEV_NULL", "S2_MAXFEV",
                              "S2_MAXFEV_RESTART", "S2_START", "S2_JITTER", "S2_MIN_FOOTPRINT",
                              "VR_A_MIN", "M1_WINDOW", "M1_FAMILY", "M1_PLANE_MIN_OTHERS",
                              "M1_MIN_MONTHS", "M1_MIN_TARGETS", "M1_MIN_STATIONS", "M1_SEED",
                              "M1_BOOT_P_MAX", "Z90")}},
        "inputs": code_paths(ROUND1_CODE + REPAIR_CODE) + SHARED_CODE + pre_data_inputs(),
        "interpretation": (
            "Plan WP S5 asks for a monthly field where InSAR supports it; v1 delivered none "
            "(verifier major issue). Stage 2 adds an InSAR-only monthly anomaly to the annual "
            "fusion; M1 asks whether it predicts withheld GNSS monthly residuals better than "
            "the annual model with its constant seasonal harmonic, after removing what InSAR "
            "cannot see (per-month planes, incl. the GNSS common mode) and the GNSS station "
            "nuisances (offset, excess rate). Pre-origin only (U7). Disclosures: (1) the WRA "
            "GNSS units are design-exposed: procedure H (S5.subsidence_fusion.v1, stage B) "
            "fitted sig_gnss, sig_d and sig_cm on them, T3 scored them (GNSS out, pre-origin), "
            "S4b's G1 compared InSAR with GNSS at these stations (14-month changes) and S3/S1b "
            "built and inspected them; the annual hyperparameters of stages C and D were fitted "
            "with GNSS in the model. M1's monthly-anomaly comparison itself has not been "
            "computed before this registration: the repair round looked only at GNSS daily "
            "dates and flags (look 2026-10-17T03:23Z) and ran the procedure on synthetic values "
            "(s5r_dryrun.py, look 2026-10-17T03:35Z). M1 is therefore a registered test on "
            "design-exposed units (design-time evidence in A1's sense, like T1 and T2). (2) Two-"
            "stage approximation: the annual posterior mean is fixed in stage 2; S_annual and "
            "S_anomaly are added as independent. (3) The product subsidence_fused/v2 applies "
            "the same stage 2 with these hyperparameters to residuals of the full-data v2 fit "
            "(run F, 2020-01..2025-12); its monthly values are delivered where the support rule "
            "holds ONLY IF M1 passes; if M1 fails or is not evaluable, no monthly value is "
            "delivered (InSAR does not support monthly detail beyond the annual model) and the "
            "trend and anomaly arrays stay available for diagnosis. (4) No threshold is changed "
            "after results."),
        "design_time_exposure": [
            "subsidence_fused/v1, its validation and the verifier's findings were seen",
            "S4b G1 (InSAR vs GNSS 14-month changes at WRA stations, run P) was seen",
            "s5r_dryrun.py on SYNTHETIC values (variants anomaly and null; look "
            "2026-10-17T03:35Z) was run before this registration, twice. Round a (capped at 150 "
            "evaluations; S5_evidence/repair/dryrun_monthly_a) showed the full-anomaly VR is "
            "dominated by the unobservable plane (no supported station-month in either "
            "variant), so the support rule was moved to the plane-free anomaly, and the "
            "optimiser did not converge, so the noise-only start, the larger budgets, the "
            "restart and the best-evaluated rule were added. Round b (final code, "
            "S5_evidence/repair/dryrun_monthly): anomaly variant ratio 0.955 on 533 supported "
            "station-months at 18 stations, bootstrap P(ratio >= 1) 0.057, coverage 0.876; "
            "null variant no supported station-month (VR_dt median 0.29). The bootstrap clause "
            "P(ratio >= 1) <= 0.10 of the gate was chosen after round b (point ratio < 1 alone "
            "would pass the null variant's all-station-month ratio of 0.994); this is design on "
            "synthetic values only",
            "rho was estimated at 0 in both synthetic variants (the annual fit absorbs the "
            "slow part of the synthetic anomaly), so stage 2 cannot separate a temporally white "
            "anomaly from APS by its time correlation alone: M1 is what separates them"],
        "code_at_registration": hashes(code_paths(ROUND1_CODE + REPAIR_CODE) + SHARED_CODE),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    doc = test_doc()
    if a.dry_run:
        errs = prereg3.validate_prereg(doc)
        print(json.dumps({"errors": errs, "inputs": len(doc["inputs"])}, indent=1))
        print(json.dumps({k: v for k, v in doc.items() if k not in ("inputs",
                                                                     "code_at_registration")},
                         indent=1, ensure_ascii=False)[:6000])
        return 0
    path, sha = prereg3.write_prereg(doc)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
