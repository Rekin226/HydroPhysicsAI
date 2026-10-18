"""After-the-fact diagnostics of the W3 held-out validation (never fed back into the product,
its gate or its threshold).

Opens ``ground_elevation/v1`` through the loader guard as ``after_the_fact_analysis`` and asks
why the epoch-corrected cell mean sits below the leveling benchmarks:

* siting: how much of the gap is the cell mean versus the DTM at the benchmark point, and is
  the benchmark at a local high of the 20 m DTM (3 x 3 and 5 x 5 windows)?
* site type from the benchmark name (bridge 橋, school 國小/國中, temple 宮/廟);
* DTM provenance: residuals by sheet agency, sensor, acquisition date and by the header's
  height-datum label;
* spatial structure: per-block medians of the point residual, neighbour correlation;
* agency metadata: station GroundHeight and WellElevation against the DTM point.

Usage: python -I diagnostics.py <product_version_dir> <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard  # noqa: E402
from common.paths import REPO  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

LEVELING = REPO / "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"


def summ(v) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)),
            "iqr": [float(x) for x in np.percentile(v, [25, 75])]}


def main(prod: str, out_json: str) -> int:
    p = guard.open_product(prod, "after_the_fact_analysis")
    res = p.read_table("benchmark_residuals_heldout.parquet")
    with np.load(p.path("dtm20m_mosaic.npz")) as z:
        H, SH = z["H"], z["sheet"]
        ext, step = z["extent"], float(z["step"])
    sheets = pd.read_csv(p.path("sheets.csv"), dtype={"acquisition_date": str,
                                                      "dtm_date": str})
    wells = p.read_table("well_ground_elevation.parquet")
    bm = res.groupby("sid").agg(x=("x", "first"), y=("y", "first"), r_mean=("r_mean", "median"),
                                r_point=("r_point", "median"), elev=("elev", "median"),
                                corr_point=("corr_point", "median"),
                                block=("block", "first"), sd_within=("sd_within", "first"))
    j = np.rint((bm["x"].to_numpy() - ext[0]) / step).astype(int)
    i = np.rint((bm["y"].to_numpy() - ext[2]) / step).astype(int)
    loc = {}
    for half in (1, 2):
        mx, mn = [], []
        for ii, jj in zip(i, j, strict=True):
            w = H[ii - half:ii + half + 1, jj - half:jj + half + 1].astype(float)
            mx.append(np.nanmax(w) if np.isfinite(w).any() else np.nan)
            mn.append(np.nanmin(w) if np.isfinite(w).any() else np.nan)
        mx, mn = np.array(mx), np.array(mn)
        # epoch-correct the window extremes like the point value
        r_max = mx + bm["corr_point"].to_numpy() - bm["elev"].to_numpy()
        r_min = mn + bm["corr_point"].to_numpy() - bm["elev"].to_numpy()
        loc[f"window_{2 * half + 1}x{2 * half + 1}"] = {
            "r_local_max": summ(r_max), "r_local_min": summ(r_min),
            "frac_benchmark_within_min_max": float(np.mean((r_min <= 0) & (r_max >= 0))),
            "frac_benchmark_above_local_max": float(np.mean(r_max < 0))}
    names = pd.read_parquet(LEVELING, columns=["sid", "name"]).drop_duplicates("sid") \
        .set_index("sid")["name"].astype(str)
    bm["name"] = names.reindex(bm.index).to_numpy()
    site = {}
    for lab, pat in (("bridge_橋", "橋"), ("school_國小_國中", "國小|國中|小學"),
                     ("temple_宮_廟", "宮|廟")):
        sel = bm["name"].str.contains(pat, regex=True, na=False)
        site[lab] = {"r_point": summ(bm.loc[sel, "r_point"]),
                     "r_mean": summ(bm.loc[sel, "r_mean"])}
    sel = ~bm["name"].str.contains("橋|國小|國中|小學|宮|廟", regex=True, na=False)
    site["other"] = {"r_point": summ(bm.loc[sel, "r_point"]),
                     "r_mean": summ(bm.loc[sel, "r_mean"])}
    # provenance of the DTM at the benchmark
    k = SH[np.clip(i, 0, SH.shape[0] - 1), np.clip(j, 0, SH.shape[1] - 1)]
    bm["sheet_k"] = k
    sh = sheets.reset_index().rename(columns={"index": "k"})
    bm = bm.merge(sh[["k", "acquisition_agency", "acquisition_equipment",
                      "acquisition_date", "height_datum", "dtm_producer"]],
                  left_on="sheet_k", right_on="k", how="left").set_index(bm.index)
    prov = {}
    for col in ("acquisition_agency", "acquisition_equipment", "acquisition_date",
                "height_datum", "dtm_producer"):
        prov[col] = {str(kk): summ(g["r_point"]) for kk, g in bm.groupby(col)}
    # spatial structure of the point residual
    by_block = bm.groupby("block")["r_point"].agg(["median", "size"])
    xy = bm[["x", "y"]].to_numpy()
    tr = cKDTree(xy)
    d, nn = tr.query(xy, k=2)
    rp = bm["r_point"].to_numpy()
    dev = rp - np.nanmedian(rp)
    ok = np.isfinite(dev) & np.isfinite(dev[nn[:, 1]]) & (d[:, 1] < 2000)
    spat = {"per_block_point_median_range": [float(by_block["median"].min()),
                                             float(by_block["median"].max())],
            "per_block_point_median_iqr": [float(x) for x in
                                           np.percentile(by_block["median"], [25, 75])],
            "blocks": int(len(by_block)),
            "nearest_neighbour_corr_within_2km": float(np.corrcoef(dev[ok],
                                                                   dev[nn[ok, 1]])[0, 1]),
            "n_pairs": int(ok.sum()),
            "nearest_neighbour_median_distance_m": float(np.median(d[ok, 1]))}
    # agency metadata at the wells against the DTM point (sheet epoch, no correction needed
    # for a median)
    w = wells[np.isfinite(wells["dtm_point_sheet_m"])]
    gh = w[(w["ground_height_meta"] != 0) & np.isfinite(w["ground_height_meta"])]
    meta = {"groundheight_nonzero_minus_dtm_point": summ(gh["ground_height_meta"]
                                                         - gh["dtm_point_sheet_m"]),
            "wellelevation_minus_dtm_point": summ(w["well_elevation_meta"]
                                                  - w["dtm_point_sheet_m"]),
            "baseelevation_minus_dtm_point": summ(w["base_elevation_meta"]
                                                  - w["dtm_point_sheet_m"])}
    # cell-versus-point gap at the benchmarks (pure DTM, no leveling)
    gap = summ(bm["r_mean"] - bm["r_point"])
    out = {
        "label": f"after-the-fact analysis of {Path(prod).parent.name}/{Path(prod).name} "
                 f"(gate failed); never fed back into the product, its gate or its threshold",
        "guard_notes": p.notes,
        "decomposition": {"r_mean_cell_minus_benchmark": summ(bm["r_mean"]),
                          "r_point_dtm_minus_benchmark": summ(bm["r_point"]),
                          "cell_mean_minus_dtm_point_at_benchmarks": gap},
        "local_window": loc, "site_type_from_name": site, "dtm_provenance": prov,
        "spatial": spat, "agency_metadata_at_wells": meta,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
