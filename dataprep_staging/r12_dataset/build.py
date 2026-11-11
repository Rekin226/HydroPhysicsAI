"""Round-12 analysis-ready dataset for the statistical subsidence layer (local output only).

Writes parquet tables and a QC report to ``--out`` (default ``$HYDRO_R12_OUT/dataset``).
Nothing written here is committed: the tables derive from agency data.

Tables
  benchmarks                one row per leveling benchmark (coordinates, county, grid cell)
  leveling_increments       benchmark x year May-to-May drop (m), survey months, QC flag
  wells_year                well x hydrological-year head statistics (anomalies, m)
  heads_at_benchmarks       benchmark x year head statistics (IDW by aquifer group) and the
                            10 km smoothed driver used by the layer
  rain_at_benchmarks        benchmark x hydrological-year rainfall total (mm)
  county_year               county x year: official sinking area, mean head change, electricity
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataprep_staging.r12_leveling.increments import (
    DDIR,
    hydro_year,
    site_table,
    to_sites,
    well_year_stats,
)
from dataprep_staging.r12_leveling.nowcast import may_change
from dataprep_staging.r12_leveling.paths import DATAPREP as SHARED
from dataprep_staging.r12_leveling.paths import OUT, POLY
from hydrophysics.twin.leveling import load_panel

FEAT_YEARS = list(range(2014, 2027))
GROUPS = (("s", (1,)), ("d", (2, 3, 4)), ("a", (1, 2, 3, 4)))
NAMES = ("mean", "min", "dmean", "dmay", "exceed", "amp")


def smooth(v, xy, L=10_000.0, k=200):
    ok = np.isfinite(v)
    if ok.sum() < 5:
        return np.full(len(v), np.nan)
    d, ix = cKDTree(xy[ok]).query(xy, k=min(k, int(ok.sum())))
    w = np.exp(-0.5 * (d / L) ** 2)
    return (w * v[ok][ix]).sum(1) / w.sum(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "dataset"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    qc: dict = {}

    # --- leveling -------------------------------------------------------------------
    panel = load_panel(DDIR)
    d, xy, sids, elev = site_table()
    stats, meta = well_year_stats()
    stats["dmay"] = may_change().reindex(meta.index)
    wxy = meta[["x_twd97", "y_twd97"]].to_numpy("float64")
    dist_w, iw = cKDTree(wxy).query(xy)
    county = meta.CountyName.to_numpy()[iw]
    from hydrophysics.twin.grid import build_grid
    g = build_grid(POLY)
    dc, cell = cKDTree(g.centroids()).query(xy)
    surveys = panel.groupby("sid").datetime.agg(["count", "min", "max"]).reindex(sids)
    bm = pd.DataFrame({"sid": sids, "x": xy[:, 0], "y": xy[:, 1], "county": county,
                       "nearest_well_km": dist_w / 1000, "cell": np.where(dc <= g.dx, cell, -1),
                       "n_surveys": surveys["count"].to_numpy(),
                       "first_survey": surveys["min"].to_numpy(),
                       "last_survey": surveys["max"].to_numpy()})
    bm.to_parquet(os.path.join(a.out, "benchmarks.parquet"))

    p = panel[(panel.datetime.dt.month >= 4) & (panel.datetime.dt.month <= 8)].copy()
    p["year"] = p.datetime.dt.year
    month = p.groupby(["sid", "year"]).datetime.first().dt.month.unstack("year")
    raw = -(elev.diff(axis=1))
    inc = raw.stack().rename("d_m").reset_index()
    inc.columns = ["sid", "year", "d_m"]
    inc["survey_month"] = [month.at[s, y] if y in month.columns else np.nan
                           for s, y in zip(inc.sid, inc.year, strict=True)]
    inc["prev_survey_month"] = [month.at[s, y - 1] if (y - 1) in month.columns else np.nan
                                for s, y in zip(inc.sid, inc.year, strict=True)]
    inc["flag"] = 0
    inc.loc[inc.d_m.abs() > 0.5, "flag"] = 2                      # step artefact
    inc.loc[(inc.flag == 0) & (inc.survey_month != inc.prev_survey_month), "flag"] = 1
    inc.to_parquet(os.path.join(a.out, "leveling_increments.parquet"))
    qc["leveling"] = {
        "benchmarks": int(len(bm)), "increments": int(len(inc)),
        "flag0_same_month": int((inc.flag == 0).sum()),
        "flag1_month_shift": int((inc.flag == 1).sum()),
        "flag2_step_gt_50cm": int((inc.flag == 2).sum()),
        "increments_per_year": inc[inc.flag < 2].groupby("year").size().to_dict(),
        "last_survey_year": int(inc.year.max()),
        "by_county": bm.county.value_counts().to_dict(),
        "note": "No spring survey in 2006 (October campaign); increments 2006 and 2007 are "
                "absent. Per-benchmark leveling ends 2021; later years exist only as county "
                "and township summaries.",
    }

    # --- heads ----------------------------------------------------------------------
    wy = pd.concat({k: v.stack(future_stack=True) for k, v in stats.items()}, axis=1)
    wy.index.names = ["sid", "year"]
    wy = wy.reset_index().merge(meta.reset_index().rename(columns={"index": "sid"}), on="sid")
    wy.to_parquet(os.path.join(a.out, "wells_year.parquet"))
    cols = {}
    for grp, layers in GROUPS:
        for n in NAMES:
            cols[f"{n}_{grp}"] = to_sites(stats[n].reindex(columns=FEAT_YEARS), meta, xy, layers)
    for n in ("dmean", "exceed"):
        cols[f"{n}_a_sm10"] = np.stack([smooth(cols[f"{n}_a"][:, j], xy)
                                        for j in range(len(FEAT_YEARS))], 1)
    hb = pd.DataFrame({"sid": np.repeat(sids, len(FEAT_YEARS)),
                       "year": np.tile(FEAT_YEARS, len(sids)),
                       **{k: v.ravel() for k, v in cols.items()}})
    hb.to_parquet(os.path.join(a.out, "heads_at_benchmarks.parquet"))
    lay = pd.to_numeric(meta.layer_code, errors="coerce")
    qc["heads"] = {
        "wells_with_long_record": int(len(meta)),
        "by_county": meta.CountyName.value_counts().to_dict(),
        "by_layer": lay.value_counts().sort_index().to_dict(),
        "years": [int(FEAT_YEARS[0]), int(FEAT_YEARS[-1])],
        "wells_with_year_stat": stats["mean"].notna().sum().to_dict(),
        "benchmarks_farther_than_5km_from_a_well": int((dist_w > 5000).sum()),
        "note": "Monthly heads start 2012-01, so head statistics start with hydrological "
                "year 2013 and year-on-year changes with 2014. Only 13 wells have 2010-11 "
                "data. This limits every head-to-leveling fit to 2014-2021 (8 increments).",
    }

    # --- rain -----------------------------------------------------------------------
    rf = pd.read_csv(os.path.join(DDIR, "rf_timeseries.csv"), index_col=0, parse_dates=True)
    st = pd.read_csv(os.path.join(DDIR, "rf_stations.csv"))
    qc["rain"] = {"gauges": int(rf.shape[1]), "nan_share": float(rf.isna().mean().mean()),
                  "span": [str(rf.index.min().date()), str(rf.index.max().date())],
                  "station_columns": st.columns.tolist()}
    hy = hydro_year(rf.index)
    ok_days = rf.notna().groupby(hy).sum()
    tot = (rf.groupby(hy).sum(min_count=1) / ok_days * 365.25).where(ok_days >= 300)
    xcol = next((c for c in st.columns if c.lower() in ("x", "x_3826", "x_twd97", "tm_x97")), None)
    ycol = next((c for c in st.columns if c.lower() in ("y", "y_3826", "y_twd97", "tm_y97")), None)
    idc = next((c for c in st.columns if st[c].astype(str).isin(rf.columns).mean() > 0.8), None)
    if xcol and ycol and idc:
        st = st.set_index(st[idc].astype(str)).reindex(tot.columns)
        gxy = st[[xcol, ycol]].to_numpy("float64")
        yrs = [y for y in tot.index if 2013 <= y <= 2026]
        out = np.full((len(sids), len(yrs)), np.nan)
        for j, y in enumerate(yrs):
            v = tot.loc[y].to_numpy("float64")
            k = np.isfinite(v) & np.isfinite(gxy).all(1)
            if k.sum() >= 3:
                dd, ix = cKDTree(gxy[k]).query(xy, k=min(4, int(k.sum())))
                w = 1 / np.maximum(dd, 500.0) ** 2
                out[:, j] = (w * v[k][ix]).sum(1) / w.sum(1)
        pd.DataFrame({"sid": np.repeat(sids, len(yrs)), "year": np.tile(yrs, len(sids)),
                      "rain_mm": out.ravel()}).to_parquet(
            os.path.join(a.out, "rain_at_benchmarks.parquet"))
        qc["rain"]["gauges_with_full_year"] = (ok_days >= 300).sum(1).loc[yrs].to_dict()
    else:
        qc["rain"]["skipped"] = "gauge coordinates not resolved"

    # --- county-year ----------------------------------------------------------------
    rep = SHARED + "subsidence_public_reports/"
    A = pd.concat([pd.read_csv(rep + "preorigin/v1/county_area_by_year.csv"),
                   pd.read_csv(rep + "postorigin/v1/county_area_by_year.csv")])
    A = A[A.county.isin(["彰化", "雲林"])].sort_values("edition_roc")
    A = A.groupby(["county", "year_label"]).area_km2.last().reset_index()
    A["year"] = pd.to_numeric(A.year_label.astype(str).str[:3], errors="coerce") + 1911
    A = A.dropna(subset=["year"]).astype({"year": int})
    rows = []
    for cty, full in (("彰化", "彰化縣"), ("雲林", "雲林縣")):
        w = meta.index[meta.CountyName == full]
        for n in ("dmean", "exceed", "mean", "min"):
            s = stats[n].loc[w].mean()
            rows.append(pd.DataFrame({"county": cty, "year": s.index, "stat": n, "v": s.values}))
    hc = pd.concat(rows).pivot_table(index=["county", "year"], columns="stat", values="v")
    cy = hc.reset_index().merge(A[["county", "year", "area_km2"]], on=["county", "year"],
                                how="outer").sort_values(["county", "year"])
    cy.to_parquet(os.path.join(a.out, "county_year.parquet"))
    qc["county_year"] = {"rows": int(len(cy)),
                         "area_years": sorted(A.year.unique().tolist()),
                         "note": "Official area is a validator (after-the-fact use only)."}

    with open(os.path.join(a.out, "QC_REPORT.json"), "w") as f:
        json.dump(qc, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(qc, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
