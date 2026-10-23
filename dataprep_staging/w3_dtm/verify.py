"""Independent re-validation of ``ground_elevation/v3`` (WP W3, continuation round).

The product was built and scored in the first W3 round (v1-v3). This script re-checks it
before the WP report is written. It never writes into the product and never changes its gate:

1. loader-guard probes: every use in the vocabulary is tried on v3 and the outcome recorded;
2. the primary statistic recomputed from the v3 residual table (plain pandas);
3. a full rebuild of the mosaic, cell aggregates and both held-out validations with the v3
   code, compared value by value with the v3 outputs (reproducibility);
4. re-implementations that do not use the builder's helpers: cell means by assigning each
   20 m lattice point to its model cell with floor(), a raw parse of randomly drawn DTM
   sheets against the mosaic, and the leveling campaign elevations from the panel;
5. a leakage test of the 10 km block hold-out: for every block, the benchmarks inside it get a
   synthetic trend (+0.10 m per year, below the 0.30 m reset threshold) and the validation is
   rerun; the epoch corrections of that block's rows must not move (with the buffer variant,
   the trend goes to every benchmark within 10 km of the block);
6. an after-the-fact siting diagnostic (never fed back): where the DTM value at a benchmark
   (and at a monitoring well) ranks among the 20 m points of its own 1 km cell.

Usage (from anywhere; through the CPU-slot wrapper):
    cpu_slot.sh python -I verify.py <out_json> [--leak-blocks N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))           # dataprep_staging
sys.path.insert(0, str(HERE.parents[2]))           # repository (read-only imports)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard  # noqa: E402
from common.paths import DATAPREP  # noqa: E402
from common.uses import USES  # noqa: E402
from w3_dtm import build, leveling, mosaic, validation  # noqa: E402

V3 = DATAPREP / "ground_elevation" / "v3"
TREND = 0.10                 # m per year added to the held-out benchmarks in the leak test
N_RAW_SHEETS = 25
SEED = 20261012


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def guard_probes() -> dict:
    out = {}
    for use in sorted(USES):
        try:
            guard.open_product(V3, use)
            out[use] = "granted"
        except Exception as exc:  # noqa: BLE001 - the outcome class is what is recorded
            out[use] = type(exc).__name__
    return out


def maxdiff(a, b) -> float:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape != b.shape:
        return float("inf")
    both_nan = np.isnan(a) & np.isnan(b)
    if np.any(np.isnan(a) != np.isnan(b)):
        return float("inf")
    d = np.abs(a - b)[~both_nan]
    return float(d.max()) if d.size else 0.0


def compare_residuals(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    key = ["sid", "year"]
    a = new.sort_values(key).reset_index(drop=True)
    b = old.sort_values(key).reset_index(drop=True)
    same_keys = bool((a[key].astype(str).to_numpy() == b[key].astype(str).to_numpy()).all()) \
        if len(a) == len(b) else False
    num = [c for c in b.columns if c in a.columns and pd.api.types.is_numeric_dtype(b[c])]
    diffs = {c: maxdiff(a[c], b[c]) for c in num} if same_keys else {}
    return {"rows_new": int(len(a)), "rows_v3": int(len(b)), "same_keys": same_keys,
            "max_abs_diff_by_column": diffs,
            "max_abs_diff": max(diffs.values()) if diffs else float("inf"),
            "block_equal": bool(same_keys and (a["block"].astype(str).to_numpy()
                                               == b["block"].astype(str).to_numpy()).all())}


def lattice_cell_index(mos_extent, step, shape, g) -> np.ndarray:
    """Active-cell index of every lattice point by floor() (independent of cell_blocks)."""
    e0, _, n0, _ = mos_extent
    nrow, ncol = shape
    E = e0 + step * np.arange(ncol)
    N = n0 + step * np.arange(nrow)
    col = np.floor((E - g["x0"]) / g["dx"]).astype(int)
    row = np.floor((N - g["y0"]) / g["dx"]).astype(int)
    idx = np.full(g["mask"].shape, -1, dtype=int)
    idx[g["rows"], g["cols"]] = np.arange(len(g["rows"]))
    ny, nx = g["mask"].shape
    okc = (col >= 0) & (col < nx)
    okr = (row >= 0) & (row < ny)
    out = np.full(shape, -1, dtype=int)
    rr, cc = np.meshgrid(np.where(okr)[0], np.where(okc)[0], indexing="ij")
    out[rr, cc] = idx[row[rr], col[cc]]
    return out


def independent_cell_means(H, cidx, n_cells) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(H) & (cidx >= 0)
    c = cidx[ok]
    h = H[ok].astype("float64")
    n = np.bincount(c, minlength=n_cells)
    s = np.bincount(c, weights=h, minlength=n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n, n


def raw_sheet_check(sheets: pd.DataFrame, mos_npz, archives_dir: Path, active_k: np.ndarray,
                    rng) -> dict:
    """Parse randomly drawn .grd members with a plain text split and compare with the mosaic
    at the lattice points that the mosaic attributes to that sheet."""
    H, SH = mos_npz["H"], mos_npz["sheet"]
    e0, _, n0, _ = (float(x) for x in mos_npz["extent"])
    step = float(mos_npz["step"])
    pick = rng.choice(active_k, size=min(N_RAW_SHEETS, len(active_k)), replace=False)
    rec = []
    for k in sorted(int(x) for x in pick):
        r = sheets.iloc[k]
        with zipfile.ZipFile(archives_dir / r["archive"]) as z:
            raw = z.read(r["member_grd"]).decode("ascii")
        a = np.array(raw.split(), dtype="float64").reshape(-1, 3)
        j = np.rint((a[:, 0] - e0) / step).astype(int)
        i = np.rint((a[:, 1] - n0) / step).astype(int)
        inside = (i >= 0) & (i < H.shape[0]) & (j >= 0) & (j < H.shape[1])
        valid = inside & (a[:, 2] >= mosaic.NODATA_LO) & (a[:, 2] <= mosaic.NODATA_HI)
        own = valid.copy()
        own[valid] = SH[i[valid], j[valid]] == k
        d = np.abs(H[i[own], j[own]].astype("float64")
                   - a[own, 2].astype("float32").astype("float64"))
        rec.append({"k": k, "sheet": str(r["file_sheet_id"]), "points": int(len(a)),
                    "points_owned_in_mosaic": int(own.sum()),
                    "points_valid_not_owned": int((valid & ~own).sum()),
                    "max_abs_diff_m": float(d.max()) if d.size else None})
    return {"sheets_checked": len(rec),
            "points_compared": int(sum(x["points_owned_in_mosaic"] for x in rec)),
            "max_abs_diff_m": max((x["max_abs_diff_m"] or 0.0) for x in rec),
            "points_overwritten_by_later_sheet": int(sum(x["points_valid_not_owned"]
                                                         for x in rec)),
            "detail": rec}


def independent_campaigns(path: Path) -> pd.DataFrame:
    lev = pd.read_parquet(path, columns=["datetime", "elev_m", "sid"])
    lev = lev[np.isfinite(lev["elev_m"])]
    m = lev[lev["datetime"].dt.month.between(4, 7) & lev["datetime"].dt.year.between(2011, 2021)]
    return (m.assign(year=m["datetime"].dt.year, sid=m["sid"].astype(str))
            .groupby(["sid", "year"])["elev_m"].median().rename("elev_indep").reset_index())


def leak_test(camp, cells, g, mos, srtm, res_ref, buffer: bool, blocks: list[str]) -> dict:
    allxy = camp.groupby("sid")[["x", "y"]].first()
    cols = ["corr_cell", "corr_at_bm", "corr_point"]
    ref = res_ref.set_index(["sid", "year"])
    per_block = []
    for blk in blocks:
        bx, by = (int(v) for v in blk.split("_"))
        xa, ya = g["x0"] + bx * validation.BLOCK_M, g["y0"] + by * validation.BLOCK_M
        grow = validation.BUFFER_M if buffer else 0.0
        sel = ((allxy["x"] >= xa - grow) & (allxy["x"] < xa + validation.BLOCK_M + grow)
               & (allxy["y"] >= ya - grow) & (allxy["y"] < ya + validation.BLOCK_M + grow))
        sids = set(allxy.index[sel])
        cp = camp.copy()
        hit = cp["sid"].isin(sids)
        cp.loc[hit, "elev"] = cp.loc[hit, "elev"] + TREND * (cp.loc[hit, "year"] - 2011)
        r = validation.run(camp=cp, cells=cells, centroids=g["centroids"], mask=g["mask"],
                           x0=g["x0"], y0=g["y0"], dx=g["dx"], mos=mos, srtm=srtm,
                           buffer=buffer, log=None).set_index(["sid", "year"])
        r = r.loc[ref.index]
        inb = (ref["block"] == blk).to_numpy()
        d_in = max(maxdiff(r.loc[inb, c], ref.loc[inb, c]) for c in cols)
        d_out = max(float(np.nanmax(np.abs(r.loc[~inb, c].to_numpy()
                                           - ref.loc[~inb, c].to_numpy()))) for c in cols)
        per_block.append({"block": blk, "perturbed_benchmarks": len(sids),
                          "rows_in_block": int(inb.sum()),
                          "max_abs_change_in_block_m": d_in,
                          "max_abs_change_elsewhere_m": d_out})
    return {"buffer": buffer, "trend_m_per_yr": TREND, "blocks_tested": len(per_block),
            "max_abs_change_in_block_m": max(x["max_abs_change_in_block_m"] for x in per_block),
            "blocks_where_perturbation_moved_other_blocks": int(sum(
                x["max_abs_change_elsewhere_m"] > 1e-6 for x in per_block)),
            "per_block": per_block}


def siting_ranks(H, cidx, res3, wells) -> dict:
    """Percentile rank of a point's DTM value among its cell's 20 m points (0.5 expected for
    a randomly sited point)."""
    order = {}
    ok = np.isfinite(H) & (cidx >= 0)
    c = cidx[ok]
    h = H[ok].astype("float64")
    srt = np.argsort(c, kind="stable")
    c, h = c[srt], h[srt]
    starts = np.searchsorted(c, np.arange(cidx.max() + 2))
    for cell in np.unique(np.concatenate([res3["cell"].unique(),
                                          wells["cell"][wells["cell"] >= 0].unique()])):
        order[int(cell)] = np.sort(h[starts[cell]:starts[cell + 1]])

    def rank(cell, v):
        a = order.get(int(cell))
        if a is None or not len(a) or not np.isfinite(v):
            return np.nan
        lo = np.searchsorted(a, v, side="left")
        hi = np.searchsorted(a, v, side="right")
        return (lo + hi) / 2 / len(a)

    bm = res3.groupby("sid").agg(cell=("cell", "first"), p=("dtm_point_sheet", "first"),
                                 e=("elev", "median"), cp=("corr_point", "median"))
    bm["rank_dtm_point"] = [rank(c_, v) for c_, v in zip(bm["cell"], bm["p"], strict=True)]
    bm["rank_benchmark"] = [rank(c_, e - k) for c_, e, k in zip(bm["cell"], bm["e"], bm["cp"],
                                                                strict=True)]
    w = wells[(wells["cell"] >= 0) & np.isfinite(wells["dtm_point_sheet_m"])]
    wr = np.array([rank(c_, v) for c_, v in zip(w["cell"], w["dtm_point_sheet_m"],
                                                strict=True)])
    wm = w["model_well"].to_numpy(dtype=bool)

    def q(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return {"n": int(len(v)), "median": float(np.median(v)),
                "iqr": [float(x) for x in np.percentile(v, [25, 75])],
                "frac_above_cell_median": float(np.mean(v > 0.5))}

    return {"label": "after-the-fact analysis (never fed back into the product, gate or "
                     "threshold)",
            "benchmark_dtm_point_rank_in_cell": q(bm["rank_dtm_point"]),
            "benchmark_elevation_rank_in_cell_at_sheet_epoch": q(bm["rank_benchmark"]),
            "station_dtm_point_rank_in_cell": q(wr),
            "model_well_dtm_point_rank_in_cell": q(wr[wm])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("--leak-blocks", type=int, default=0, help="0 = every block")
    a = ap.parse_args(argv)
    t_start = time.time()
    rep: dict = {"product": "ground_elevation/v3", "script": "dataprep_staging/w3_dtm/verify.py"}

    log("guard probes")
    rep["guard_probes"] = guard_probes()
    p = guard.open_product(V3, "after_the_fact_analysis")
    res3 = pd.read_parquet(p.path("benchmark_residuals_heldout.parquet"))
    resb3 = pd.read_parquet(p.path("benchmark_residuals_heldout_buffer10km.parquet"))
    cells3 = pd.read_parquet(p.path("ground_elevation_cells.parquet"))
    wells3 = pd.read_parquet(p.path("well_ground_elevation.parquet"))
    sheets3 = pd.read_csv(p.path("sheets.csv"), dtype={"file_sheet_id": str, "sheet_id": str,
                                                      "acquisition_date": str})
    val3 = json.loads(p.path("validation.json").read_text(encoding="utf-8"))
    mos3 = np.load(p.path("dtm20m_mosaic.npz"))
    grid3 = np.load(p.path("ground_elevation_grid.npz"))

    # 2. primary from the table
    rb = res3.groupby("sid")["r_mean"].median()
    rep["primary_from_table"] = {"median_m": float(rb.median()), "n_benchmarks": int(len(rb)),
                                 "n_blocks": int(res3["block"].nunique()),
                                 "v3_reported_m": val3["primary"]["median_m"],
                                 "abs_diff": abs(float(rb.median())
                                                 - val3["primary"]["median_m"])}

    # 3. rebuild
    log("rebuild mosaic")
    t0 = time.time()
    g = build.load_grid()
    mos = mosaic.build_mosaic([str(x) for x in build.ARCHIVES], log=None)
    blocks = mosaic.cell_blocks(mos, g["x0"], g["y0"], g["dx"], g["rows"], g["cols"])
    cells = mosaic.aggregate_cells(mos, blocks)
    with np.load(build.SRTM_NPZ) as z:
        srtm = np.asarray(z["dem"], dtype="float64")
    t_mos = time.time() - t0
    rep["rebuild"] = {
        "mosaic_seconds": round(t_mos, 1),
        "H_max_abs_diff": maxdiff(mos.H, mos3["H"]),
        "epoch_max_abs_diff": maxdiff(mos.epoch, mos3["epoch"]),
        "sheet_index_equal": bool(np.array_equal(mos.sheet, mos3["sheet"])),
        "cell_mean_max_abs_diff": maxdiff(cells["mean"], cells3["z_sheet_mean"]),
        "cell_median_max_abs_diff": maxdiff(cells["median"], cells3["z_sheet_median"]),
        "cell_sd_max_abs_diff": maxdiff(cells["sd"], cells3["sd_within"]),
    }
    obs = leveling.load_panel(str(build.LEVELING))
    camp = leveling.campaigns(obs, validation.DISP_YEARS)
    log("rerun validation")
    t0 = time.time()
    res = validation.run(camp=camp, cells=cells, centroids=g["centroids"], mask=g["mask"],
                         x0=g["x0"], y0=g["y0"], dx=g["dx"], mos=mos, srtm=srtm, log=None)
    t_val = time.time() - t0
    res_b = validation.run(camp=camp, cells=cells, centroids=g["centroids"], mask=g["mask"],
                           x0=g["x0"], y0=g["y0"], dx=g["dx"], mos=mos, srtm=srtm,
                           buffer=True, log=None)
    rep["rebuild"]["validation_seconds"] = round(t_val, 1)
    rep["rebuild"]["residuals"] = compare_residuals(res, res3)
    rep["rebuild"]["residuals_buffer"] = compare_residuals(res_b, resb3)
    boot = validation.block_bootstrap_median(res, "r_mean")
    rep["rebuild"]["bootstrap_90ci"] = boot
    rep["rebuild"]["bootstrap_equal_v3"] = bool(np.allclose(
        boot, val3["primary"]["block_bootstrap_90ci_m"], rtol=0, atol=1e-12))
    ee = validation.epoch_error_slope(res)
    rep["rebuild"]["epoch_a_abs_diff"] = abs(ee["a_m_per_yr"]
                                             - val3["epoch_correction_error"]["a_m_per_yr"])
    rep["rebuild"]["grid_value_ref_vs_cells_z_ref"] = maxdiff(grid3["value_ref"],
                                                              cells3["z_ref"])

    # 4. independent re-implementations
    log("independent checks")
    cidx = lattice_cell_index(mos3["extent"], float(mos3["step"]), mos3["H"].shape, g)
    m_ind, n_ind = independent_cell_means(mos3["H"], cidx, len(cells3))
    rep["independent_cell_means"] = {
        "max_abs_diff_vs_v3_z_sheet_mean": maxdiff(m_ind, cells3["z_sheet_mean"]),
        "n_points_equal": bool(np.array_equal(n_ind, cells3["n_points"].to_numpy())),
        "cells": int(len(cells3)), "cells_without_points": int((n_ind == 0).sum())}
    active_k = np.unique(mos3["sheet"][cidx >= 0])
    active_k = active_k[active_k >= 0]
    rng = np.random.default_rng(SEED)
    rep["raw_sheet_check"] = raw_sheet_check(sheets3, mos3, build.DTM_DIR, active_k, rng)
    ic = independent_campaigns(build.LEVELING)
    mm = res3[["sid", "year", "elev"]].assign(sid=res3["sid"].astype(str)).merge(
        ic, on=["sid", "year"], how="left")
    rep["independent_leveling"] = {
        "rows": int(len(mm)), "rows_unmatched": int(mm["elev_indep"].isna().sum()),
        "max_abs_diff_m": float(np.nanmax(np.abs(mm["elev"] - mm["elev_indep"]))),
        "years": [int(res3["year"].min()), int(res3["year"].max())]}

    # 5. leakage test
    blks = sorted(res3["block"].unique())
    if a.leak_blocks:
        blks = list(pd.Series(blks).sample(n=min(a.leak_blocks, len(blks)),
                                           random_state=SEED))
    log(f"leak test on {len(blks)} blocks (no buffer)")
    rep["leak_test"] = leak_test(camp, cells, g, mos, srtm, res3, False, blks)
    log(f"leak test on {len(blks)} blocks (buffer)")
    rep["leak_test_buffer"] = leak_test(camp, cells, g, mos, srtm, resb3, True, blks)

    # 6. siting diagnostic
    log("siting ranks")
    rep["siting_ranks"] = siting_ranks(mos3["H"], cidx, res3, wells3)
    rep["seconds_total"] = round(time.time() - t_start, 1)
    Path(a.out_json).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)
                                + "\n", encoding="utf-8")
    log(f"wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
