"""WP S4b: build ``results/dataprep/insar_069A/v<N>/`` (a new version directory, never an
overwrite) from the full-network run F, with the registered test results of run P.

    python -I s4b_build_product.py      (hydro env, one CPU slot)

Outputs:
  vertical_cells.npz       (A, T) vertical-equivalent displacement on the twin's active cells:
                           value / sd / flag, dates, valid-pixel counts, per-cell U
  vertical_ml.npz          (T, L, W) the same on the 2 x 2 multilooked grid (LiCSBAS15 mask),
                           with pixel centres (lon, lat, EPSG:3826), cell index, U, E, hgt
  reference_epochs.csv     per-epoch stable-area fit (coefficients, residual sd, flag)
  gnss_tie_offsets.csv     per-epoch mean (InSAR - WRA GNSS) over the G1 stations (an optional
                           offset-only tie; NOT applied; using it spends those stations)
  coherence_summer_preorigin.npz, zone_coverage.json      C1 data and per-zone verdicts
  seasonal_amplitude_preorigin.npz                         run P annual amplitude (Y failed)
  decomposition_2025.npz   ascending/descending 2025 rate decomposition (after the fact)
  validation.json, PRODUCT_CARD.md
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_evaluate as Ev  # noqa: E402
import s4b_lib as Lb  # noqa: E402
from common import manifest  # noqa: E402
from common.hashing import grid_hash  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

CODE = ["s4b_config.py", "s4b_lib.py", "s4b_geom.py", "s4b_run_licsbas.sh", "s4b_coherence.py",
        "s4b_battery.py", "s4b_gnss_inventory.py", "s4b_evaluate.py", "s4b_prereg.py",
        "s4b_after.py", "s4b_looks.py", "s4b_build_product.py", "make_licsbas_workdir.py",
        "licsbas_env.sh"]
PREORIGIN_USES = ["calibration_target", "datum_fit", "baseline", "initial_conditions",
                  "idw_field", "prior", "prior_width_input", "model_design", "selection",
                  "forcing_frozen", "design_time_evidence"]


def tests() -> dict:
    t = {k: json.loads((C.EVID / f"test_{k}.json").read_text())
         for k in ("gnss", "coherence", "seasonal")}
    t["after"] = json.loads((C.EVID / "after.json").read_text())
    return t


def main() -> int:
    T = tests()
    d = manifest.new_version_dir(C.PRODUCT)
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    A = int(np.asarray(z["mask"]).sum())
    ghash = grid_hash(int(z["nx"]), int(z["ny"]), float(z["dx"]), float(z["x0"]),
                      float(z["y0"]), z["mask"])
    run, g, v, tab = Ev.processed("F")
    dates = [x.strftime("%Y-%m-%d") for x in run["dates"]]
    val, se, nval = Lb.to_cells(v, run["mask"], g, A)
    Uc, _, _ = Lb.to_cells(g["U"], np.isfinite(g["U"]), g, A)
    ref_sd = tab["resid_sd_los_mm"].to_numpy(float)[:, None] / Uc[None]
    sd = np.sqrt(np.nan_to_num(se) ** 2 + np.nan_to_num(ref_sd) ** 2).astype(np.float32)
    sd[~np.isfinite(val)] = np.nan
    eflag = tab["flag"].to_numpy()[:, None]
    flag = np.where(np.isfinite(val), np.where(eflag == 9, 9, 0), 8).astype(np.uint8)
    np.savez_compressed(
        d / "vertical_cells.npz", value=val.T.astype(np.float32), sd=sd.T, flag=flag.T,
        dates=np.array(dates), n_valid_pixels=nval.astype(np.int32),
        U_cell=Uc.astype(np.float32), nx=z["nx"], ny=z["ny"], dx=z["dx"], x0=z["x0"],
        y0=z["y0"], grid_hash=np.array(ghash),
        units=np.array("mm; vertical-equivalent = LOS (towards satellite) / U; positive up; "
                       "relative to the first epoch and to the stable off-fan area"),
        sd_note=np.array("UNCALIBRATED: sqrt(se_cell^2 + (stable-pixel residual sd / U)^2); "
                         "the calibrated accuracy statement is the G1 change RMS"))
    vm = np.where(run["mask"][None], v, np.nan).astype(np.float32)
    np.savez_compressed(d / "vertical_ml.npz", value=vm, dates=np.array(dates),
                        lon=g["lon"], lat=g["lat"], x=g["x"], y=g["y"], cell=g["cell"],
                        U=g["U"].astype(np.float32), E=g["E"].astype(np.float32),
                        hgt=g["hgt"].astype(np.float32), licsbas15_mask=run["mask"],
                        stable=g["stable_geom"] & run["mask"])
    tab.to_csv(d / "reference_epochs.csv", index=False)
    # optional offset-only GNSS tie (mean over the G1 stations of InSAR - GNSS, demeaned)
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    st = T["gnss"]["primary"]["stations"]
    ins, gn = Ev.gnss_tables(run, g, v, inv, st, "exclude")
    dd = ins - gn
    off = (dd - dd.mean()).mean(axis=1)
    pd.DataFrame({"date": dates, "offset_mm": off.to_numpy(),
                  "n_stations": dd.notna().sum(axis=1).to_numpy()}).to_csv(
        d / "gnss_tie_offsets.csv", index=False)
    shutil.copyfile(C.RUN_DIR / "coh_P_summer_preorigin.npz", d / "coherence_summer_preorigin.npz")
    cov = {"test": T["coherence"], "time_series_coverage_runF": {}}
    U = Ev.units()
    for zid in C.CH_ZONES:
        cells = np.array(U[zid]["cells"], int)
        cov["time_series_coverage_runF"][zid] = float(np.mean(nval[cells] >= C.CELL_MIN_PIXELS))
    yl = sorted(set().union(*[set(U[u]["cells"]) for u in C.YUNLIN_UNITS]))
    cov["time_series_coverage_runF"]["yunlin"] = float(np.mean(nval[yl] >= C.CELL_MIN_PIXELS))
    cov["time_series_coverage_runF"]["all_active"] = float(np.mean(nval >= C.CELL_MIN_PIXELS))
    Lb.jdump(cov, d / "zone_coverage.json")
    amp_c, info = Ev.amplitude_cells("P")
    h = info["pixel"]
    np.savez_compressed(d / "seasonal_amplitude_preorigin.npz", amp_cells=amp_c,
                        amp=h["amp"].astype(np.float32), phase_doy=h["phase_doy"],
                        rate=h["rate"].astype(np.float32), resid_sd=h["resid_sd"],
                        valid=info["valid"],
                        status=np.array("NOT a validator: registered test Y failed (Y2 0/8)"
                                        if not T["seasonal"]["passed"] else
                                        "rank validator in Changhua (Y passed)"))
    shutil.copyfile(C.EVID / "decomp_2025_cells.npz", d / "decomposition_2025.npz")
    g1, c1, y = T["gnss"], T["coherence"], T["seasonal"]
    gate = {"passed": bool(g1["passed"]),
            "criteria": (f"pre-registered {C.PREREG_ID}: G1 WRA GNSS leave-one-station-out "
                         f"14-month change RMS <= {C.GNSS_LOSO_RMS_MAX_MM} mm on the pre-origin "
                         f"run P ({g1['primary']['rms_mm']:.1f} mm, "
                         f"{g1['primary']['n_evaluable']} stations)"),
            "tests": {"G1": bool(g1["passed"]),
                      "C1_zone_covered": {k: v["covered"] for k, v in c1["zones"].items()},
                      "Y_seasonal_validator": bool(y["passed"])}}
    validation = {"prereg": C.PREREG_ID, "gate": gate, "G1": g1, "C1": c1, "Y": y,
                  "after_the_fact": T["after"],
                  "licsbas_runs": {k: list(v) for k, v in C.RUNS.items()},
                  "licsbas_commit": C.LICSBAS_COMMIT}
    Lb.jdump(validation, d / "validation.json")
    write_card(d, gate, g1, c1, y, T["after"], cov, nval, A, len(dates), dates)
    outputs = []
    for p in sorted(x.name for x in d.iterdir()):
        rec = {"path": p}
        if p == "coherence_summer_preorigin.npz":
            rec["fit_window_end"] = "2022-09"
        if p == "seasonal_amplitude_preorigin.npz":
            rec["fit_window_end"] = "2022-12"
        outputs.append(rec)
    inputs = [rel_to_repo(C.S4A_PRODUCT / "MANIFEST.json"), rel_to_repo(C.S4A_NOTES),
              rel_to_repo(C.S4A_RUN / f"pairs_{C.PRIMARY}.csv"),
              rel_to_repo(C.S4A_RUN / f"pairs_{C.DESC}.csv"),
              "data_fetch_api/new_measurements/licsar_069A_06694/fetch_2026-10-12_s4a/SOURCE.json",
              "data_fetch_api/new_measurements/licsar_105D_06628/fetch_2026-10-12_s4a/SOURCE.json",
              rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz"),
              rel_to_repo(C.RUN_DIR / f"geom_{C.DESC}.npz"),
              rel_to_repo(C.RUN_DIR / "gnss_inventory.json"),
              rel_to_repo(C.RUN_DIR / "battery.json"),
              rel_to_repo(C.RUN_DIR / "coh_P_all_preorigin.npz"),
              rel_to_repo(C.UNITS_V3), rel_to_repo(C.TWIN_GRID),
              "results/dataprep/pump_electricity/preorigin/v1/MANIFEST.json",
              "results/twin_runs/stage3_datum_gate/coupled_leveling/vep_zonal_leveling.json",
              "chou-shui-data/data/ls_cache/ls-wra-gnss-obs__meta.parquet",
              "results/dataprep/preregistration/S4b.insar_acceptance.v1.json"]
    for nm in ("P", "F", "D"):
        ts = C.RUN_DIR / nm / "TS_GEOCml2"
        inputs += [rel_to_repo(ts / "cum.h5"), rel_to_repo(ts / "results" / "mask"),
                   rel_to_repo(ts / "info" / "13parameters.txt"),
                   rel_to_repo(ts / "info" / "15parameters.txt")]
    inputs += [rel_to_repo(p) for st, p in Lb.wra_files().items() if st in
               set(T["gnss"]["report_only"]["steps_corrected"]["stations"])]
    m = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2", inputs=inputs,
        code=[rel_to_repo(HERE / c) for c in CODE],
        parameters={"licsbas_commit": C.LICSBAS_COMMIT, "licsbas_steps": C.LICSBAS_STEPS,
                    "runs": {k: list(v) for k, v in C.RUNS.items()}, "product_run": "F",
                    "gate_run": "P", "atmosphere": "none (D11)",
                    "stable_area": {"hgt_m": [C.STABLE_MIN_HGT_M, C.STABLE_MAX_HGT_M],
                                    "min_dist_km": C.STABLE_MIN_DIST_KM},
                    "reference_terms": list(C.REF_TERMS), "vertical": C.VERTICAL,
                    "cell_min_pixels": C.CELL_MIN_PIXELS, "prereg": C.PREREG_ID},
        fit_window_end="2025-12", validation_report="validation.json", gate=gate,
        allowed_uses=["validator", "after_the_fact_analysis"],
        forbidden_uses=PREORIGIN_USES + ["scoring_target", "prospective_scorer",
                                         "forcing_recorded_arm", "statistical_layer_history"],
        personal_data="none", outputs=outputs,
        extra={"licence": C.LICENCE, "zone_coverage": {k: v["covered"]
                                                       for k, v in c1["zones"].items()},
               "notes": "InSAR is a validator of twin subsidence 2022-2026 only (U7): never a "
                        "column target, never a gate, threshold or member choice input. The "
                        "gate failed: the guard refuses 'validator' until a user decision; "
                        "'after_the_fact_analysis' stays open."})
    manifest.write_manifest(d, m)
    print(json.dumps({"product": str(d), "gate": gate["passed"], "n_outputs": len(outputs)}))
    return 0


def write_card(d, gate, g1, c1, y, after, cov, nval, A, T, dates) -> None:
    z = c1["zones"]
    y1 = y["Y1_all_yunlin"]
    lines = [
        f"# {C.PRODUCT}/{d.name} (WP S4b)", "",
        f"Gate: **{'PASSED' if gate['passed'] else 'FAILED'}** ({gate['criteria']}).", "",
        "LiCSAR frame 069A_06694_131402 (ascending) processed with LiCSBAS "
        f"(commit {C.LICSBAS_COMMIT[:12]}, defaults, 2 x 2 multilook, no atmospheric "
        "correction, D11) over 2020-01..2025-12 (run F, "
        f"{T} epochs {dates[0]}..{dates[-1]}). Referenced per epoch to a stable off-fan area "
        "(land, LiCSAR DEM 100-600 m, >= 2.5 km from every active cell centre, LiCSBAS15 "
        "mask) with offset + east/north plane + height term; vertical-equivalent = LOS / U "
        "(per-pixel U, 0.80-0.85). No GNSS, ring, pile or leveling value enters the product.",
        "",
        "Files: vertical_cells.npz (value/sd/flag (A, T) mm, positive up, relative to the "
        "first epoch), vertical_ml.npz (multilooked grid), reference_epochs.csv, "
        "gnss_tie_offsets.csv (optional offset-only tie, not applied), "
        "coherence_summer_preorigin.npz and zone_coverage.json, "
        "seasonal_amplitude_preorigin.npz, decomposition_2025.npz, validation.json.", "",
        "Flags: 0 observed (cell median of >= 5 valid multilooked pixels); 9 epoch whose "
        "stable set has < 200 pixels; 8 missing (outside the frame footprint, masked, or < 5 "
        "valid pixels). sd is UNCALIBRATED (cell standard error and the stable-pixel residual "
        "sd in quadrature).", "",
        "## Registered tests (pre-origin run P, S4b.insar_acceptance.v1)", "",
        f"- G1 GNSS leave-one-station-out 14-month change ({C.GNSS_T0} -> {C.GNSS_T1}): RMS "
        f"{g1['primary']['rms_mm']:.1f} mm over {g1['primary']['n_evaluable']} WRA stations "
        f"(threshold {C.GNSS_LOSO_RMS_MAX_MM} mm): {'pass' if g1['passed'] else 'FAIL'}.",
        "- C1 Changhua summer coherence (cells with median coherence >= 0.3, median over 57 "
        "pairs; threshold 0.30): " + "; ".join(
            f"{k.split('.')[-1]} {v['median_frac_coherent']:.3f} "
            f"({'covered' if v['covered'] else 'NOT covered'})" for k, v in z.items()) + ".",
        f"- Y seasonal amplitude as a pumping validator (Yunlin): Y1 rho {y1['rho_primary']:.3f}"
        f" (bootstrap p05 {y1['boot_p05']:.3f}) "
        f"{'pass' if y1['passed'] else 'fail'}; Y2 wins "
        f"{y['Y2_battery']['wins']}/8 against the placebo and thickness proxies: "
        f"{'pass' if y['passed'] else 'FAIL'} -> the amplitude is "
        f"{'a rank validator' if y['passed'] else 'NOT a validator'} in Changhua.", "",
        "## After the fact (report only; not a gate)", "",
        f"- Run F (full network), same G1 stations and window: RMS "
        f"{after['F_g1']['loso']['rms_mm']:.1f} mm; the pilot's post-origin window "
        f"2021-12-05 -> 2023-02-20: {after['pilot_window_F']['loso']['rms_mm']:.1f} mm.",
        f"- Run P vs run F, cell change 2020-01-12 -> {after['p_vs_f']['last']}: RMS "
        f"difference {after['p_vs_f']['rms_diff_mm']:.1f} mm.",
        f"- 2025 asc/desc decomposition: vertical-equivalent minus decomposed vertical rate, "
        f"median {after['decomp_2025']['median_vert_eq_minus_up_mm_yr']:.1f} mm/yr, RMS "
        f"{after['decomp_2025']['rms_vert_eq_minus_up_mm_yr']:.1f} mm/yr (east-west leakage).",
        "",
        "## Coverage", "",
        "Active cells with >= 5 valid pixels (run F): " + "; ".join(
            f"{k} {v:.2f}" for k, v in cov["time_series_coverage_runF"].items()) +
        ". The primary footprint misses the northern band of Changhua (about 9 % of active "
        "cells).", "",
        "## Uses", "",
        "Allowed: validator (refused by the guard while the gate fails), "
        "after_the_fact_analysis. Every pre-origin use is forbidden (U7: InSAR never enters "
        "a calibration, gate, threshold or member choice). Licence: " + C.LICENCE]
    (d / "PRODUCT_CARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
