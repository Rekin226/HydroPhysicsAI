"""Build ``results/dataprep/ground_elevation/v<N>/`` (WP W3): the MOI 20 m DTM aggregated to the
twin's 1 km cells with a representativeness sd, epoch-corrected with leveling, validated on
benchmarks held out in 10 km blocks, plus a well-point table for H1b.

Usage (run through the CPU-slot wrapper, with ``python -I``):

    cpu_slot.sh python -I build.py --dev OUTDIR     # development: no benchmark comparison,
                                                    # outputs to a scratch dir, no manifest
    cpu_slot.sh python -I build.py                  # the product: needs the pre-registration

The product run refuses to start unless the pre-registration
``results/dataprep/preregistration/W3.dtm_heldout_bias.json`` exists, matches the sha256
recorded in the W3 ledger fragment and declares the same design constants as this code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never the GPU (Phase-0 rule)
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))           # dataprep_staging
sys.path.insert(0, str(HERE.parents[2]))           # repository (read-only imports)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import hashing, ledger, manifest  # noqa: E402
from common.paths import L0_NEW, PREREG_DIR, REPO, TWIN_GRID_NPZ, rel_to_repo  # noqa: E402
from w3_dtm import leveling, mosaic, validation  # noqa: E402

PRODUCT = "ground_elevation"
PREREG_ID = "W3.dtm_heldout_bias"
DTM_DIR = L0_NEW / "moi_dtm20m_2025"
ARCHIVES = [DTM_DIR / f"{a}_20MDEM_2025.zip"
            for a in ("changhua", "yunlin", "nantou", "chiayi_county")]
LEVELING = REPO / "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"
SRTM_NPZ = REPO / "results/twin/basemap.npz"
STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"
STATION_COLS = ["sid", "LocationByTWD97", "GroundHeight", "WellElevation", "BaseElevation"]
T_REF = 2017.5                       # 2017-07-01, midpoint of the 2012-01..2022-12 window
EPOCHS = [float(y) for y in range(2012, 2024)]   # 1 January 2012 .. 1 January 2023
LOCAL_HALF = 2                       # 5 x 5 lattice window (100 m) for the point sd
CODE = [HERE, HERE.parent / "dtm_io.py", HERE.parent / "mosaic.py",
        HERE.parent / "leveling.py", HERE.parent / "validation.py",
        REPO / "hydrophysics/twin/heads.py"]

DESIGN = {   # must equal the pre-registration's "design_constants"
    "block_m": validation.BLOCK_M, "buffer_m_secondary": validation.BUFFER_M,
    "threshold_m": validation.THRESHOLD_M, "val_years": [min(validation.VAL_YEARS),
                                                         max(validation.VAL_YEARS)],
    "disp_years": [min(validation.DISP_YEARS), max(validation.DISP_YEARS)],
    "main_months": list(leveling.MAIN_MONTHS), "reset_threshold_m": leveling.RESET_THRESHOLD_M,
    "idw_k": leveling.IDW_K, "idw_power": leveling.IDW_POWER,
    "n_last_rate": leveling.N_LAST_RATE, "boot_n": validation.BOOT_N,
    "boot_seed": validation.BOOT_SEED, "max_acquisition": mosaic.MAX_ACQUISITION,
    "lattice_extent": list(mosaic.EXTENT), "t_ref": T_REF,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_grid() -> dict:
    z = np.load(TWIN_GRID_NPZ, allow_pickle=False)
    mask = z["mask"].astype(bool)
    g = {"mask": mask, "nx": int(z["nx"]), "ny": int(z["ny"]), "dx": float(z["dx"]),
         "x0": float(z["x0"]), "y0": float(z["y0"])}
    r, c = np.nonzero(mask)
    g["rows"], g["cols"] = r, c
    g["centroids"] = np.column_stack([g["x0"] + (c + 0.5) * g["dx"],
                                      g["y0"] + (r + 0.5) * g["dx"]])
    g["mask_sha256"] = hashing.mask_hash(mask)
    g["grid_sha256"] = hashing.grid_hash(g["nx"], g["ny"], g["dx"], g["x0"], g["y0"], mask)
    gd = json.loads(str(z["gate"]))
    g["model_wells"] = sorted(gd.get("well_datum", {}).keys())
    return g


def station_points() -> pd.DataFrame:
    """Station coordinates (only non-personal columns are read). Coordinates stored as
    "y x" are detected by range and swapped locally (H1a owns the official repair)."""
    from hydrophysics.twin.heads import _station_xy
    st = pd.read_parquet(STATIONS, columns=STATION_COLS)
    rec = []
    for _, row in st.iterrows():
        p = _station_xy(row)
        status = "ok"
        if p is None:
            status = "missing"
            try:
                a, b = (float(v) for v in str(row["LocationByTWD97"]).split()[:2])
                if 140000 <= b <= 240000 and 2580000 <= a <= 2700000:
                    p, status = (b, a), "swapped_repaired_locally"
            except (ValueError, TypeError):
                pass
        rec.append({"sid": str(row["sid"]), "x": p[0] if p else np.nan,
                    "y": p[1] if p else np.nan, "xy_status": status,
                    "ground_height_meta": row["GroundHeight"],
                    "well_elevation_meta": row["WellElevation"],
                    "base_elevation_meta": row["BaseElevation"]})
    return pd.DataFrame(rec)


def local_sd(mos: mosaic.Mosaic, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i, (xx, yy) in enumerate(zip(x, y, strict=True)):
        if not (np.isfinite(xx) and np.isfinite(yy)):
            continue
        j = int(round((xx - mos.extent[0]) / mos.step))
        k = int(round((yy - mos.extent[2]) / mos.step))
        w = mos.H[max(k - LOCAL_HALF, 0):k + LOCAL_HALF + 1,
                  max(j - LOCAL_HALF, 0):j + LOCAL_HALF + 1].astype("float64")
        w = w[np.isfinite(w)]
        if len(w) > 1:
            out[i] = float(w.std(ddof=1))
    return out


def check_prereg() -> tuple[Path, str, dict]:
    p = PREREG_DIR / f"{PREREG_ID}.json"
    if not p.is_file():
        raise SystemExit(f"pre-registration {p} missing: write it (prereg.py) before the run")
    sha = hashing.sha256_file(p)
    rows = [r for r in ledger.read_all() if r["number_id"] == "W3.preregistration"
            and r["source"] == rel_to_repo(p)]
    if not rows or f"sha256={sha}" not in rows[0]["note"]:
        raise SystemExit("pre-registration sha256 differs from the ledger record")
    pr = json.loads(p.read_text(encoding="utf-8"))
    if pr["design_constants"] != json.loads(json.dumps(DESIGN)):
        raise SystemExit(f"design constants differ from the pre-registration:\n"
                         f"{pr['design_constants']}\n{DESIGN}")
    return p, sha, pr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", help="development output dir (no validation, no manifest)")
    ap.add_argument("--note", default=None, help="build note stored in the manifest (for "
                    "example what a new version changes relative to the previous one)")
    a = ap.parse_args(argv)
    if a.dev:
        out = Path(a.dev)
        out.mkdir(parents=True, exist_ok=True)
        prereg = None
    else:
        prereg = check_prereg()
        out = manifest.new_version_dir(PRODUCT)
    log(f"output: {out}")

    g = load_grid()
    mos = mosaic.build_mosaic([str(p) for p in ARCHIVES], log=log)
    blocks = mosaic.cell_blocks(mos, g["x0"], g["y0"], g["dx"], g["rows"], g["cols"])
    cells = mosaic.aggregate_cells(mos, blocks)
    if (cells["n_points"] == 0).any():
        raise SystemExit("some active cells have no DTM point")
    k_act = np.unique(np.concatenate([mos.sheet[si, sj].ravel() for si, sj in blocks]))
    act_dates = mos.sheets.loc[k_act[k_act >= 0], "acquisition_date"].astype(str)
    mos.active_acq = (act_dates.min(), act_dates.max(), int((k_act >= 0).sum()))
    if mos.excluded_late is not None and len(mos.excluded_late):
        log(f"{len(mos.excluded_late)} sheets acquired after {mosaic.MAX_ACQUISITION} were "
            f"excluded")
    with np.load(SRTM_NPZ) as z:
        srtm = np.asarray(z["dem"], dtype="float64")
    if len(srtm) != len(cells):
        raise SystemExit("basemap SRTM does not match the active cells")

    obs = leveling.load_panel(str(LEVELING))
    camp = leveling.campaigns(obs, validation.DISP_YEARS)
    dm = leveling.DisplacementModel(camp, validation.DISP_YEARS)
    used = obs[obs["month"].isin(leveling.MAIN_MONTHS)
               & obs["year"].isin(list(validation.DISP_YEARS))]
    ym = (used["year"] * 100 + used["month"]).max()
    lev_last_ym = f"{ym // 100}-{ym % 100:02d}"

    # ---------------------------------------------------------------- validation (held out)
    val, res, res_buf = None, None, None
    a_epoch = float("nan")
    sd_pt_mad = float("nan")
    if prereg is not None:
        res = validation.run(camp=camp, cells=cells, centroids=g["centroids"],
                             mask=g["mask"], x0=g["x0"], y0=g["y0"], dx=g["dx"], mos=mos,
                             srtm=srtm, log=log)
        res_buf = validation.run(camp=camp, cells=cells, centroids=g["centroids"],
                                 mask=g["mask"], x0=g["x0"], y0=g["y0"], dx=g["dx"], mos=mos,
                                 srtm=srtm, buffer=True, log=log)
        ee = validation.epoch_error_slope(res)
        ee["coverage_leave_one_block_out"] = validation.epoch_error_cv(res)
        a_epoch = ee["a_m_per_yr"]
        val = summarize_validation(res, res_buf, ee, cells, dm, g, prereg)
        sd_pt_mad = val["secondary"]["point_dtm_bilinear"]["per_benchmark"]["mad_sd"]

    # ---------------------------------------------------------------- product arrays
    C = g["centroids"]
    ep_list, sh_list = list(cells["epochs"]), list(cells["epoch_shares"])
    corr = np.column_stack([dm.correction(C, np.full(len(C), t), ep_list, sh_list)
                            for t in EPOCHS])                                  # (A, n_t)
    corr_ref = dm.correction(C, np.full(len(C), T_REF), ep_list, sh_list)
    zmean = cells["mean"].to_numpy()
    zmed = cells["median"].to_numpy()
    sdw = cells["sd"].to_numpy()
    nps = cells["n_points"].to_numpy()
    epm = cells["epoch_mean"].to_numpy()
    dt = np.array(EPOCHS)[None, :] - epm[:, None]
    sd_epoch = np.abs(dt) * a_epoch
    sd_epoch_ref = np.abs(T_REF - epm) * a_epoch
    z_t = zmean[:, None] + corr
    z_ref = zmean + corr_ref
    sd_mean = np.sqrt(sd_epoch ** 2 + (sdw ** 2 / nps)[:, None])
    sd_point = np.sqrt(sd_epoch ** 2 + (sdw ** 2)[:, None])
    T_last = float(dm.T[-1])
    T_first = float(dm.T[0])
    flag_t = np.where((np.array(EPOCHS) > T_last) | (np.array(EPOCHS) < T_first), 6, 3)
    flag_t = np.broadcast_to(flag_t, z_t.shape).astype(np.uint8)
    flag_sheet = np.zeros(len(zmean), dtype=np.uint8)
    flag_ref = np.full(len(zmean), 3, dtype=np.uint8)
    rate = dm.rate(C, 2012.0, 2021.0)

    np.savez_compressed(
        out / "ground_elevation_grid.npz",
        mask=g["mask"], nx=g["nx"], ny=g["ny"], dx=g["dx"], x0=g["x0"], y0=g["y0"],
        mask_sha256=g["mask_sha256"], grid_sha256=g["grid_sha256"],
        epochs=np.array(EPOCHS), epoch_labels=np.array([f"{int(t)}-01-01" for t in EPOCHS]),
        value=z_t.astype("float64"), sd=sd_mean, sd_point=sd_point, sd_epoch=sd_epoch,
        flag=flag_t, t_ref=T_REF, value_ref=z_ref, sd_ref=np.sqrt(sd_epoch_ref ** 2
                                                                   + sdw ** 2 / nps),
        sd_point_ref=np.sqrt(sd_epoch_ref ** 2 + sdw ** 2), flag_ref=flag_ref,
        value_sheet_mean=zmean, value_sheet_median=zmed, flag_sheet=flag_sheet,
        sd_within=sdw, n_points=nps, coverage=cells["coverage"].to_numpy(),
        epoch_mean=epm, epoch_min=cells["epoch_min"].to_numpy(),
        epoch_max=cells["epoch_max"].to_numpy(), rate_2012_2021_m_per_yr=rate,
        srtm_basemap=srtm, epoch_error_a_m_per_yr=a_epoch)
    cdf = pd.DataFrame({
        "cell": np.arange(len(zmean)), "row": g["rows"], "col": g["cols"],
        "x": C[:, 0], "y": C[:, 1], "n_points": nps, "coverage": cells["coverage"],
        "z_sheet_mean": zmean, "z_sheet_median": zmed, "sd_within": sdw,
        "z_min": cells["min"], "z_max": cells["max"], "z_p05": cells["p05"],
        "z_p95": cells["p95"], "epoch_mean": epm, "epoch_min": cells["epoch_min"],
        "epoch_max": cells["epoch_max"], "n_epochs": [len(e) for e in ep_list],
        "rate_2012_2021_m_per_yr": rate, "corr_ref": corr_ref, "z_ref": z_ref,
        "sd_epoch_ref": sd_epoch_ref, "sd_point_ref": np.sqrt(sd_epoch_ref ** 2 + sdw ** 2),
        "flag_ref": flag_ref, "srtm_basemap": srtm, "srtm_minus_z_ref": srtm - z_ref,
    })
    cdf.to_parquet(out / "ground_elevation_cells.parquet", index=False)

    # ---------------------------------------------------------------- mosaic and sheets
    np.savez_compressed(out / "dtm20m_mosaic.npz", H=mos.H, epoch=mos.epoch,
                        sheet=mos.sheet, extent=np.array(mos.extent), step=mos.step)
    sh = mos.sheets.copy()
    sh["archive"] = [Path(x).name for x in sh["archive"]]
    keep = ["file_sheet_id", "sheet_id", "sheet_name", "archive", "member_grd", "grd_sha256",
            "plane_datum", "height_datum", "dx", "dy", "n_cols", "n_rows", "n_points",
            "sw_e", "sw_n", "e_min", "e_max", "n_min", "n_max", "n_read", "n_inside",
            "n_sentinel", "n_off_lattice", "z_min", "z_max", "method_code",
            "production_equipment", "acquisition_equipment", "acquisition_date",
            "acquisition_year", "acquisition_agency", "dtm_date", "dtm_producer"]
    sh[keep].to_csv(out / "sheets.csv", index=False)

    # ---------------------------------------------------------------- well points (H1b)
    st = station_points()
    xs, ys = st["x"].to_numpy(), st["y"].to_numpy()
    hp, hep = mos.bilinear(xs, ys)
    lsd = local_sd(mos, xs, ys)
    okp = np.isfinite(hp)
    corr_w = np.full(len(st), np.nan)
    rate_w = np.full(len(st), np.nan)
    if okp.any():
        P = np.column_stack([xs[okp], ys[okp]])
        corr_w[okp] = dm.correction(P, np.full(okp.sum(), T_REF),
                                    [[e] for e in hep[okp]], [[1.0]] * int(okp.sum()))
        rate_w[okp] = dm.rate(P, 2012.0, 2021.0)
    cell_w = validation.cell_of_point(np.nan_to_num(xs, nan=-1), np.nan_to_num(ys, nan=-1),
                                      g["mask"], g["x0"], g["y0"], g["dx"])
    sd_w = np.sqrt(lsd ** 2 + sd_pt_mad ** 2 + (a_epoch * np.abs(T_REF - hep)) ** 2)
    st["model_well"] = st["sid"].isin(set(g["model_wells"]))
    st["ground_height_is_zero"] = st["ground_height_meta"] == 0.0
    st["dtm_point_sheet_m"] = hp
    st["dtm_point_epoch"] = hep
    st["dtm_local_sd_m"] = lsd
    st["epoch_corr_ref_m"] = corr_w
    st["ground_elev_dtm_ref_m"] = hp + corr_w
    st["sd_m"] = sd_w
    st["rate_2012_2021_m_per_yr"] = rate_w
    st["cell"] = cell_w
    st["cell_z_ref_m"] = np.where(cell_w >= 0, z_ref[np.clip(cell_w, 0, None)], np.nan)
    st["flag"] = np.where(np.isfinite(st["ground_elev_dtm_ref_m"]), 5, 8).astype(np.uint8)
    st["method"] = np.where(np.isfinite(st["ground_elev_dtm_ref_m"]),
                            "dtm20m_bilinear_epoch_corrected_to_2017.5", "no_coordinates")
    st["t_ref"] = T_REF
    st.to_parquet(out / "well_ground_elevation.parquet", index=False)

    summary = product_summary(cells, cdf, st, mos, g, dm, corr, z_t, srtm, flag_t, a_epoch,
                              sd_pt_mad, T_last)
    log(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    if prereg is None:
        (out / "dev_summary.json").write_text(json.dumps(summary, indent=1, default=float))
        return 0

    res.to_parquet(out / "benchmark_residuals_heldout.parquet", index=False)
    res_buf.to_parquet(out / "benchmark_residuals_heldout_buffer10km.parquet", index=False)
    val["product_summary"] = summary
    (out / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False,
                                                    default=float) + "\n")
    write_card(out, val, summary, lev_last_ym)
    passed = bool(val["primary"]["passed"])
    fwe = max(lev_last_ym, str(mos.sheets["acquisition_date"].max())[:4] + "-"
              + str(mos.sheets["acquisition_date"].max())[4:6])
    m = manifest.build_manifest(
        out, product=PRODUCT, layer="L2",
        inputs=[*[str(p) for p in ARCHIVES], str(DTM_DIR / "schema_hdr.zip"),
                str(DTM_DIR / "SOURCE.json"), str(LEVELING), str(TWIN_GRID_NPZ),
                str(SRTM_NPZ), str(STATIONS), str(prereg[0])],
        code=[str(p) for p in CODE],
        parameters={**DESIGN, "epochs": EPOCHS, "local_window_points": 2 * LOCAL_HALF + 1,
                    "station_columns_read": STATION_COLS,
                    "epoch_error_a_m_per_yr": a_epoch,
                    "point_sd_mad_m": sd_pt_mad,
                    "preregistration": {"id": PREREG_ID, "path": rel_to_repo(prereg[0]),
                                        "sha256": prereg[1]},
                    "grid_sha256": g["grid_sha256"], "mask_sha256": g["mask_sha256"]},
        fit_window_end=fwe, validation_report="validation.json",
        gate={"passed": passed,
              "criteria": "plan 3 WP W3: |median bias| <= 0.30 m of the epoch-corrected 1 km "
                          "cell mean against leveling benchmarks held out in 10 km blocks "
                          f"(pre-registration {PREREG_ID}); measured "
                          f"{val['primary']['median_m']:+.3f} m"},
        allowed_uses=["forcing_frozen", "forcing_recorded_arm", "model_design", "validator",
                      "after_the_fact_analysis", "design_time_evidence"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit", "prospective_scorer",
                        "calibration_target", "selection"],
        personal_data="none (public monitoring-well and benchmark sites; station contact "
                      "and address columns are never read)",
        extra={"product_card": "PRODUCT_CARD.md", "build_note": a.note,
               "flags_in_use": {"0": "cell aggregate of observed DTM points at the sheet "
                                     "epochs (value_sheet_*)",
                                "3": "epoch-corrected with leveling displacement "
                                     "interpolated from other benchmarks (value, value_ref)",
                                "5": "well-point ground elevation inferred from the DTM "
                                     "(replaces GroundHeight, H1b)",
                                "6": "epoch after the last leveling campaign (2022-01-01, "
                                     "2023-01-01): extrapolated, never for calibration",
                                "8": "station without usable coordinates"}})
    manifest.write_manifest(out, m)
    log(f"wrote {out}/MANIFEST.json; gate passed={passed}")
    return 0


def product_summary(cells, cdf, st, mos, g, dm, corr, z_t, srtm, flag_t, a_epoch, sd_pt_mad,
                    T_last) -> dict:
    zr = cdf["z_ref"].to_numpy()
    mw = st[st["model_well"]]
    zero = mw[mw["ground_height_is_zero"]]
    nz = st[(~st["ground_height_is_zero"]) & np.isfinite(st["ground_height_meta"])
            & np.isfinite(st["ground_elev_dtm_ref_m"])]
    we = st[np.isfinite(st["well_elevation_meta"]) & np.isfinite(st["ground_elev_dtm_ref_m"])]
    return {
        "n_cells": int(len(cdf)),
        "cells_full_coverage": int((cells["coverage"] >= 1).sum()),
        "cells_partial_coverage": int((cells["coverage"] < 1).sum()),
        "min_coverage": float(cells["coverage"].min()),
        "sheets_used": int(len(mos.sheets)),
        "sheets_excluded_late": int(len(mos.excluded_late)) if mos.excluded_late is not None
        else 0,
        "mosaic_sheet_acquisition_range": [str(mos.sheets["acquisition_date"].min()),
                                           str(mos.sheets["acquisition_date"].max())],
        "active_cell_sheet_acquisition_range": [mos.active_acq[0], mos.active_acq[1]],
        "sheets_feeding_active_cells": mos.active_acq[2],
        "cell_epoch_range": [float(cells["epoch_mean"].min()),
                             float(cells["epoch_mean"].max())],
        "cells_mixed_epochs": int((cells["epoch_min"] != cells["epoch_max"]).sum()),
        "duplicate_sheets_identical": int(sum(d["identical"] for d in mos.duplicates)),
        "duplicate_sheets_different": int(sum(not d["identical"] for d in mos.duplicates)),
        "overlap": mos.overlap,
        "z_ref_median_m": float(np.median(zr)), "z_ref_min_m": float(zr.min()),
        "cells_z_ref_below_0": int((zr < 0).sum()), "cells_z_ref_below_2": int((zr < 2).sum()),
        "srtm_cells_below_0": int((srtm < 0).sum()), "srtm_cells_below_2": int((srtm < 2).sum()),
        "srtm_minus_z_ref_median_m": float(np.median(srtm - zr)),
        "srtm_minus_z_ref_iqr_m": [float(x) for x in np.percentile(srtm - zr, [25, 75])],
        "sd_within_median_m": float(np.median(cells["sd"])),
        "sd_within_p95_m": float(np.percentile(cells["sd"], 95)),
        "corr_ref_abs_max_m": float(np.abs(cdf["corr_ref"]).max()),
        "corr_ref_abs_p95_m": float(np.percentile(np.abs(cdf["corr_ref"]), 95)),
        "corr_2012_range_m": [float(corr[:, 0].min()), float(corr[:, 0].max())],
        "corr_2023_range_m": [float(corr[:, -1].min()), float(corr[:, -1].max())],
        "rate_2012_2021_range_m_per_yr": [float(cdf["rate_2012_2021_m_per_yr"].min()),
                                          float(cdf["rate_2012_2021_m_per_yr"].max())],
        "leveling_campaign_epochs": [float(x) for x in dm.T],
        "leveling_increments_per_pair": dm.n_increments,
        "leveling_increments_dropped_as_resets": dm.n_dropped,
        "epochs_flag6": [int(t) for t, f in zip(EPOCHS, flag_t[0], strict=True) if f == 6],
        "epoch_error_a_m_per_yr": a_epoch, "point_sd_mad_m": sd_pt_mad,
        "stations": int(len(st)), "stations_with_dtm": int(np.isfinite(
            st["ground_elev_dtm_ref_m"]).sum()),
        "stations_xy_swapped_repaired_locally": int((st["xy_status"]
                                                     == "swapped_repaired_locally").sum()),
        "stations_xy_missing": int((st["xy_status"] == "missing").sum()),
        "model_wells": int(len(mw)), "model_wells_groundheight_zero": int(len(zero)),
        "model_wells_groundheight_zero_with_dtm": int(np.isfinite(
            zero["ground_elev_dtm_ref_m"]).sum()),
        "groundheight_nonzero_minus_dtm_median_m": float(np.median(
            nz["ground_height_meta"] - nz["ground_elev_dtm_ref_m"])) if len(nz) else None,
        "groundheight_nonzero_minus_dtm_iqr_m": [float(x) for x in np.percentile(
            nz["ground_height_meta"] - nz["ground_elev_dtm_ref_m"], [25, 75])]
        if len(nz) else None,
        "groundheight_nonzero_n": int(len(nz)),
        "wellelevation_minus_dtm_median_m": float(np.median(
            we["well_elevation_meta"] - we["ground_elev_dtm_ref_m"])) if len(we) else None,
        "wellelevation_minus_dtm_iqr_m": [float(x) for x in np.percentile(
            we["well_elevation_meta"] - we["ground_elev_dtm_ref_m"], [25, 75])]
        if len(we) else None,
        "wellelevation_n": int(len(we)),
    }


def summarize_validation(res, res_buf, ee, cells, dm, g, prereg) -> dict:
    rb = validation.per_benchmark(res, "r_mean")
    med = float(rb.median())
    passed = abs(med) <= validation.THRESHOLD_M
    sec = {}
    for name, col in (("cell_median", "r_median"), ("uncorrected_cell_mean",
                                                    "r_mean_uncorrected"),
                      ("point_dtm_bilinear", "r_point"), ("srtm_cell", "r_srtm")):
        sec[name] = {"per_benchmark": validation.summarize(validation.per_benchmark(res, col)),
                     "per_observation": validation.summarize(res[col])}
    sec["buffer_10km_cell_mean"] = {
        "per_benchmark": validation.summarize(validation.per_benchmark(res_buf, "r_mean")),
        "n_train_benchmarks_median": float(res_buf.groupby("block")["n_train_benchmarks"]
                                           .first().median())}
    yrs = {}
    for y in (2012, 2016, 2020):
        s = res[res["year"] == y]
        yrs[str(y)] = {"cell_mean": validation.summarize(s["r_mean"]),
                       "srtm_cell": validation.summarize(s["r_srtm"])}
    sec["per_year"] = yrs
    # coverage of value +- 1.645 sd_point (point in cell), block clusters
    sdp = np.sqrt(res["sd_within"] ** 2 + (ee["a_m_per_yr"] * res["dt"].abs()) ** 2)
    hit = (res["r_mean"].abs() <= validation.Z90 * sdp).to_numpy()
    sec["coverage_point_in_cell_90"] = validation.cluster_coverage(hit, res["block"].to_numpy())
    # standardized bias
    sec["median_bias_over_median_sd_within"] = float(med / np.median(res["sd_within"]))
    # strata
    strata = {}
    bm = res.groupby("sid").agg(r=("r_mean", "median"), z=("cell_mean_sheet", "first"),
                                y=("y", "first"), x=("x", "first"))
    rates = dm.rate(bm[["x", "y"]].to_numpy(), 2012.0, 2021.0)
    bm["rate"] = rates
    for lab, sel in (("north_of_y2640km_approx_changhua", bm["y"] >= 2640000),
                     ("south_of_y2640km_approx_yunlin", bm["y"] < 2640000),
                     ("cell_mean_lt_5m", bm["z"] < 5), ("cell_mean_5_20m",
                                                        (bm["z"] >= 5) & (bm["z"] < 20)),
                     ("cell_mean_ge_20m", bm["z"] >= 20),
                     ("rate_slower_than_1cm_yr", bm["rate"] > -0.01),
                     ("rate_1_3cm_yr", (bm["rate"] <= -0.01) & (bm["rate"] > -0.03)),
                     ("rate_faster_than_3cm_yr", bm["rate"] <= -0.03)):
        strata[lab] = validation.summarize(bm.loc[sel, "r"])
    sec["strata_per_benchmark"] = strata
    blk = res.groupby("sid")["block"].first()
    by_block = rb.groupby(blk).median()
    sec["per_block_median_range_m"] = [float(by_block.min()), float(by_block.max())]
    sec["blocks_with_abs_median_gt_0.30"] = int((by_block.abs() > 0.30).sum())
    return {
        "wp": "W3", "product": PRODUCT,
        "preregistration": {"id": PREREG_ID, "path": rel_to_repo(prereg[0]),
                            "sha256": prereg[1], "created": prereg[2].get("created")},
        "primary": {"statistic": "median over held-out benchmarks of the per-benchmark median "
                                 "residual (epoch-corrected 1 km cell mean minus benchmark)",
                    "median_m": med, "threshold_m": validation.THRESHOLD_M,
                    "passed": bool(passed), "n_benchmarks": int(len(rb)),
                    "n_observations": int(len(res)), "n_blocks": int(res["block"].nunique()),
                    "block_bootstrap_90ci_m": validation.block_bootstrap_median(res, "r_mean"),
                    "per_benchmark": validation.summarize(rb)},
        "secondary": sec,
        "epoch_correction_error": ee,
        "data_windows": {"leveling_used": "main campaigns (Apr-Jul) 2011-2021; validation "
                                          "observations 2012-2021",
                         "dtm_sheet_acquisition": "2016-11 .. 2018-07 inside the grid",
                         "post_origin_data": "none"},
        "metrics_added_after_registration": [
            "epoch_correction_error.coverage_leave_one_block_out (reported only)"],
        "design_time_evidence_note": "the SRTM-vs-leveling pilot (plan W3 facts) and a "
                                     "development comparison of the DTM cell means with SRTM "
                                     "(no leveling) preceded the pre-registration",
    }


def write_card(out: Path, val: dict, s: dict, lev_last_ym: str) -> None:
    p = val["primary"]
    sec = val["secondary"]
    pt = sec["point_dtm_bilinear"]["per_benchmark"]
    sr = sec["srtm_cell"]["per_benchmark"]
    cov = sec["coverage_point_in_cell_90"]
    ee = val["epoch_correction_error"]
    cvc = ee["coverage_leave_one_block_out"]
    a0, a1 = s["active_cell_sheet_acquisition_range"]
    txt = f"""# ground_elevation ({out.name}) - product card

WP W3 of the October 2026 data-remediation plan. MOI 2025-edition 20 m DTM (data.gov.tw
176927, Government Open Data License v1; airborne LiDAR bare-earth model). The
{s['sheets_feeding_active_cells']} sheets that feed the twin's 2148 active 1 km cells were
acquired {a0[:4]}-{a0[4:6]}..{a1[:4]}-{a1[4:6]} (the mosaic around the grid also holds
sheets from {s['mosaic_sheet_acquisition_range'][0][:4]}-{s['mosaic_sheet_acquisition_range'][0][4:6]},
outside every active cell). The cell mean of the 20 m points is epoch-corrected with WRA
leveling (main campaigns 2011-2021, last month used {lev_last_ym}).

**Gate: {'PASSED' if p['passed'] else 'FAILED'}.** Pre-registered test
({val['preregistration']['id']}): median of (epoch-corrected cell mean minus benchmark) over
{p['n_benchmarks']} benchmarks held out in {p['n_blocks']} 10 km blocks =
{p['median_m']:+.3f} m (block-bootstrap 90 % interval {p['block_bootstrap_90ci_m'][0]:+.3f}..
{p['block_bootstrap_90ci_m'][1]:+.3f}); threshold |bias| <= 0.30 m. The loader therefore
refuses every gated use (forcing, model design, validator) until a decision is recorded;
design-time evidence and after-the-fact analysis stay open.

What the gap is made of (pre-registered secondary metrics): the DTM at the benchmark point
(bilinear) is {pt['median']:+.3f} m from the benchmark (MAD-sd {pt['mad_sd']:.2f} m); the rest
is the cell mean lying below the point, because benchmarks sit on locally higher ground
(roads, embankments, bridges). The field it replaces, SRTM, scores {sr['median']:+.3f} m
(MAD-sd {sr['mad_sd']:.2f} m) on the same benchmarks.

- **Coverage.** {s['cells_full_coverage']} cells fully covered, {s['cells_partial_coverage']}
  coastal cells partially (minimum {s['min_coverage']:.2f} of the 2500 lattice points).
- **Arrays** (`ground_elevation_grid.npz`, active-cell order = `np.nonzero(mask)`):
  `value` (A, 12) at 1 January 2012..2023, `value_ref` at {T_REF} (static field),
  `value_sheet_mean`/`median` (uncorrected), `sd_within` (representativeness of a point in
  the cell), `sd_epoch` (epoch-correction error, a = {ee['a_m_per_yr']:.4f} m/yr x |dt|),
  `sd` (error of the cell mean, excluding any systematic DTM error), `sd_point`, `flag`,
  mask and grid hashes.
- **Calibration of the sd.** `sd_epoch`: the scale a was fitted on the held-out benchmarks'
  own displacement errors (in-sample 90 % coverage {ee['coverage_of_1.645_a_dt']:.3f}); with a
  refitted leave-one-block-out the coverage is {cvc['coverage']:.3f} (Wilson 90 %
  {cvc['wilson90'][0]:.3f}..{cvc['wilson90'][1]:.3f}, plan 4.4 rule
  {'passed' if cvc['passes_rule_4.4'] else 'failed'}). `sd_point` (sd_within with sd_epoch)
  covers point-in-cell residuals at {cov['coverage']:.3f}, Wilson 90 %
  {cov['wilson90'][0]:.3f}..{cov['wilson90'][1]:.3f}: **uncalibrated** (it ignores the siting
  offset). `sd` of the cell mean and the well-point `sd_m` (local 100 m sd, the MAD-sd of the
  held-out point residuals and sd_epoch) have no independent check: **uncalibrated**.
- **Flags.** 0 uncorrected aggregate; 3 epoch-corrected (displacement interpolated from
  other benchmarks); 6 epochs after the last leveling campaign (2022-01-01, 2023-01-01);
  5 well-point replacement of `GroundHeight` (`well_ground_elevation.parquet`, 8 = no
  coordinates).
- **Known limits.** The epoch correction interpolates leveling by inverse distance and
  extrapolates linearly after May 2021. Station coordinates stored as "y x" are swapped
  locally (H1a owns the repair).
- **Uses.** Allowed (gated): forcing_frozen, forcing_recorded_arm, model_design, validator;
  (gate-exempt): after_the_fact_analysis, design_time_evidence. Forbidden: scoring and
  calibration targets, baselines, datum fits, selection.
"""
    (out / "PRODUCT_CARD.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
