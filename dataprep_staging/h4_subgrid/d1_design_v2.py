"""WP H4 repair round, design-time check (run BEFORE the v2 pre-registration).

Reads NO head value. Parts:

1. the verifier's diagnostic for the v1 kernel (Yunlin pumps, h4lib.PRIMARY): the v1
   seasonal / slow unit regressors split into the part a cell-mean head carries (the kernel
   averaged over the well's own cell) and the sub-grid remainder (point minus cell mean);
2. the repaired sub-grid regressors (h4v2lib.PRIMARY and the members) at the 158 calibration
   wells: support, unit-multiplier sd, correlation with the v1 regressor;
3. the held-out wells (non-calibration stations in active cells; station metadata only): the
   mapping rule checked on the 158 wells, support and regressor sd;
4. ``--availability``: month COUNTS of flag 0-1 rows per held-out well in the fit window
   (2012-02..2019-12) and the test window (2020-01..2022-12), read from the columns sid,
   month and flag only (no value column);
5. the cost of one placebo draw.

Writes results/dataprep/reports/H4_evidence/repair/design_time/d1.json.
Usage: python d1_design_v2.py [--availability]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
import h4v2lib as V  # noqa: E402
from common import guard  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

OUT = DATAPREP / "reports/H4_evidence/repair/design_time"


def pct(x, p=(0, 10, 25, 50, 75, 90, 100)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {str(k): round(float(v), 6) for k, v in zip(p, np.percentile(x, p), strict=True)} \
        if x.size else {}


def rowsd(X):
    return X.std(axis=1, ddof=1)


def corr_rows(A, B):
    out = []
    for a, b in zip(A, B, strict=True):
        if a.std() > 0 and b.std() > 0:
            out.append(float(np.corrcoef(a, b)[0, 1]))
        else:
            out.append(np.nan)
    return np.array(out)


def availability(sids) -> pd.DataFrame:
    p = guard.open_product(L.HEADS, "selection", window_end="2022-12")
    df = p.read_table("heads_monthly.parquet", columns=["sid", "month", "flag"])
    if "value" in df.columns:
        raise RuntimeError("value column read")
    df["sid"] = df["sid"].astype(str)
    df = df[df["flag"].isin([0, 1]) & df["sid"].isin(set(sids))]
    fit = df[(df.month >= "2012-02") & (df.month <= "2019-12")].groupby("sid").size()
    te = df[(df.month >= "2020-01") & (df.month <= "2022-12")].groupby("sid").size()
    return pd.DataFrame({"n_fit": fit, "n_test": te}).reindex(sids).fillna(0).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--availability", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    grid = V.load_grid()
    res = {"reads_head_values": False, "availability_counts_read": bool(a.availability),
           "grid": {"x0": grid.x0, "y0": grid.y0, "dx": grid.dx, "nx": grid.nx, "ny": grid.ny,
                    "n_active": int(grid.mask.sum())}}

    # ---- 1. verifier diagnostic on the v1 operator (Yunlin pumps, v1 config) ----------------
    d1 = L.build_data(L.PRIMARY, with_heads=False)
    res["mapping_check_158"] = V.check_cal_mapping(grid, d1)
    Gs1, Gl1, n1 = L.operator_matrices(d1)
    S1, L1 = L.regressors(d1, Gs1, Gl1)
    cfg_sub_v1 = V.with_config(V.PRIMARY, counties=("yunlin",))
    tabs = V.make_tables(cfg_sub_v1, grid)
    d1s = replace_cfg(d1, cfg_sub_v1)
    Gss, Gls, _ = V.operator_matrices(d1s, grid, tabs)
    Ss, Ls = L.regressors(d1s, Gss, Gls)
    sup = n1 > 0
    S_cell, L_cell = S1 - Ss, L1 - Ls                  # cell-mean part of the v1 kernel
    sd_pt, sd_cell, sd_sub = rowsd(S1), rowsd(S_cell), rowsd(Ss)
    big = sup & (sd_pt > 0.02)
    res["v1_kernel_split"] = {
        "n_supported": int(sup.sum()),
        "seasonal_corr_cellmean_vs_point": pct(corr_rows(S_cell[sup], S1[sup])),
        "seasonal_sd_ratio_cellmean_over_point": pct(sd_cell[sup] / sd_pt[sup]),
        "seasonal_sd_ratio_subgrid_over_point": pct(sd_sub[sup] / sd_pt[sup]),
        "n_point_sd_gt_0.02m": int(big.sum()),
        "seasonal_sd_ratio_subgrid_over_point_sd_gt_0.02m": pct(sd_sub[big] / sd_pt[big]),
        "slow_sd_ratio_subgrid_over_point": pct(rowsd(Ls)[sup] / np.maximum(rowsd(L1)[sup],
                                                                             1e-12)),
        "seasonal_sd_point_m": pct(sd_pt[sup]), "seasonal_sd_subgrid_m": pct(sd_sub[sup]),
        "note": "the cell-mean part is computed with the untruncated kernel minus the v1 "
                "offset, on the same pump set (pumps within 2 km of the well)"}
    del d1, Gs1, Gl1, Gss, Gls

    # ---- 2. repaired operator at the 158 calibration wells --------------------------------
    members = {"primary": {}, "yunlin_only": {"counties": ("yunlin",)},
               "r_max_3km": {"r_max_m": 3000.0}, "centroid_reference": {"kernel_mode":
                                                                        "centroid"}}
    cal_cache = {}
    res["calibration_wells"] = {}
    for name, kw in members.items():
        cfg = V.with_config(V.PRIMARY, **kw)
        key = (cfg.counties,)
        if key not in cal_cache:
            cal_cache.clear()
            cal_cache[key] = L.build_data(cfg, with_heads=False)
        d = replace_cfg(cal_cache[key], cfg)
        tb = V.make_tables(cfg, grid)
        t1 = time.perf_counter()
        Gs, Gl, n = V.operator_matrices(d, grid, tb)
        t_op = time.perf_counter() - t1
        S, Lx = L.regressors(d, Gs, Gl)
        s = n > 0
        res["calibration_wells"][name] = {
            "config_change": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
            "n_pumps": int(len(d.pump_xy)),
            "n_supported": int(s.sum()),
            "supported_by_county_layer": {f"{c}_L{lay}": int((s & (d.county == c)
                                                             & (d.layer == lay)).sum())
                                          for c in ("YL", "CH") for lay in (0, 1)},
            "seasonal_unit_sd_m": pct(rowsd(S)[s]), "slow_unit_sd_m": pct(rowsd(Lx)[s]),
            "seasonal_unit_mean_abs_m": pct(np.abs(S[s]).mean(axis=1)),
            "operator_build_s": round(t_op, 1)}
        if name == "primary":
            res["calibration_wells"][name]["corr_with_v1_seasonal_regressor_yunlin"] = pct(
                corr_rows(S[sup], S1[sup]))
            prim_d, prim_tabs, prim_n = d, tb, n

    # ---- 3. held-out wells -----------------------------------------------------------------
    ft, notes = V.fresh_well_table(grid, prim_d.sids)
    res["heldout_selection"] = notes[0]
    fd = V.fresh_data(prim_d, ft, with_heads=False)
    Gs, Gl, nf = V.operator_matrices(fd, grid, prim_tabs)
    S, Lx = L.regressors(fd, Gs, Gl)
    tab = ft[["sid", "county", "layer", "cell"]].copy()
    tab["w"] = fd.w
    tab["n_pumps_2km"] = nf
    tab["seasonal_unit_sd_m"] = rowsd(S)
    tab["T"] = fd.T
    if a.availability:
        av = availability(ft["sid"].tolist())
        tab = tab.join(av, on="sid")
        tab["eligible_months"] = (tab.n_fit >= V.PRIMARY.min_train) & (
            tab.n_test >= V.PRIMARY.min_test)
    res["heldout_wells"] = tab.to_dict(orient="records")
    res["heldout_counts"] = {
        "candidates": int(len(tab)),
        "supported": int((tab.n_pumps_2km > 0).sum()),
        "by_county_layer": tab.groupby(["county", "layer"]).size().rename(
            "n").reset_index().to_dict(orient="records")}
    if a.availability:
        el = tab[tab.eligible_months]
        res["heldout_counts"]["eligible"] = int(len(el))
        res["heldout_counts"]["eligible_supported"] = int((el.n_pumps_2km > 0).sum())
        res["heldout_counts"]["eligible_supported_by_county"] = el[el.n_pumps_2km > 0].groupby(
            "county").size().to_dict()

    # ---- 5. placebo cost -------------------------------------------------------------------
    t1 = time.perf_counter()
    rng = np.random.default_rng(0)
    V.operator_matrices(prim_d, grid, prim_tabs, pump_xy=L.shuffle_xy(prim_d, rng))
    res["one_shuffle_operator_s"] = round(time.perf_counter() - t1, 1)
    res["runtime_s"] = round(time.perf_counter() - t0, 1)
    name = "d1_availability.json" if a.availability else "d1.json"
    (OUT / name).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("v1_kernel_split", "heldout_counts",
                                          "mapping_check_158")}, indent=1, default=float))
    for k, v in res["calibration_wells"].items():
        print(k, v["n_supported"], v["supported_by_county_layer"], v["seasonal_unit_sd_m"]["50"],
              v["slow_unit_sd_m"]["50"])


def replace_cfg(d: L.Data, cfg) -> L.Data:
    from dataclasses import replace
    return replace(d, cfg=cfg)


if __name__ == "__main__":
    main()
