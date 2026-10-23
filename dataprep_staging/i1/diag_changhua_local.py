"""After-the-fact diagnosis, part 2 (2026-10-19, user request): what drives the error at the worst
Changhua wells, and what makes Changhua different from Yunlin.

Part A, local checks. For every scored well (out-of-fold, R0's design-time folds) of R0 and R1c:
the gate metric (own-mean anomaly RMSE) and its change with the census (R1c - R0). Against:

* local census electricity within 2 and 5 km (R1c's gridded forcing, GWh/yr) and the distance
  to the nearest kept in-grid census pump;
* doubtful pump coordinates within 2 km: points shared by 5 or more pumps, coordinates on a
  100 m lattice (both ends 00), and pumps whose cell is not in the grid but lie within 2 km of
  the well;
* the well's layer and depth (the census carries no usable pump depth: well_depth_m is empty
  for every kept pump, so pump depth cannot be compared);
* the record's QC (L1_clean v3 well_qc_summary, pre-origin period).

Spearman correlations within each county, and the medians of the worst quartile against the rest.

Part B, Changhua against Yunlin (wells and cells of each county):

* observed heads: seasonal amplitude, interannual sd, depth to water (ground = WellElevation,
  else the DEM cell; GroundHeight is 0 or blank at many stations), trend, and where the wells sit
  (easting bands west < 185 km, mid 185-205 km, east >= 205 km), with the data response per band;
* the data's own response: per well, the regression of the monthly head anomaly on the anomaly
  of local census electricity (3 km disc, lag 0-2 months, best lag): slope in m per GWh/month and
  correlation, i.e. what the heads say about pumping without any model;
* pumping: energy per km2, seasonal swing of the electricity, pump depth distribution, purpose
  and billing mix, horsepower;
* water supply: canal deliveries and rain recharge (mm/yr);
* the model: predicted/observed amplitude and correlation, and R1c's calibrated mid-zone
  storage for each county.

Report only (U7): never a gate, a threshold or a rung choice. Logs one look (after_the_fact).

Run from the repository root:
    dataprep_staging/tools/cpu_slot.sh python dataprep_staging/i1/diag_changhua_local.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(REPO))

from harness import prereg3  # noqa: E402

OUT = REPO / "results/dataprep/reports/I1_changhua_local"
RUN = {"R0": "cwa", "R1c": "ccz"}
STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"
CENSUS_L1 = REPO / "results/dataprep/pump_electricity/census_L1/v3/census_pumps.parquet"
QC = REPO / "results/dataprep/heads/L1_clean/v3/well_qc_summary.csv"
COUNTY_MAP = REPO / "results/dataprep/zone_county_map/v2/zone_county_map.npz"
SW = REPO / "results/dataprep/surface_water_v4/model_npz/v1/surface_water_v4_model.npz"
COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}
CENSUS_COLS = ["county", "x_twd97", "y_twd97", "in_grid", "kept", "pump_hp", "purpose_class",
               "billing_code", "well_depth_m", "install_month"]


def run_dir(s: str) -> Path:
    return REPO / f"results/twin_runs/stage3_datum_gate_{s}"


def per_entry(suffix: str, st: pd.DataFrame) -> pd.DataFrame:
    z = np.load(run_dir(suffix) / "stage3_per_entry.npz", allow_pickle=True)
    w = pd.read_csv(run_dir(suffix) / "stage3_kfold_wells.csv", dtype={"sid": str}
                    ).set_index("entry")
    sids = w.loc[z["entry"], "sid"].to_numpy()
    rows = []
    for i, sid in enumerate(sids):
        o, p = z["obs"][i].astype(float), z["pred"][i].astype(float)
        m = np.isfinite(o) & np.isfinite(p)
        if m.sum() < 12:
            continue
        oa, pa = o[m] - o[m].mean(), p[m] - p[m].mean()
        rows.append({"entry": int(z["entry"][i]), "sid": sid, "x": float(z["x"][i]),
                     "y": float(z["y"][i]), "layer": int(z["layer"][i]),
                     "county": st.loc[sid, "CountyName"],
                     "rmse": float(np.sqrt(np.mean((pa - oa) ** 2))),
                     "amp": float(pa.std() / oa.std()), "r": float(np.corrcoef(pa, oa)[0, 1]),
                     "obs": o, "pred": p})
    return pd.DataFrame(rows).set_index("entry")


def cell_xy(grid) -> np.ndarray:
    return np.asarray(grid.centroids(), dtype=float)


def climatology_parts(x: np.ndarray, cal: np.ndarray) -> tuple[float, float, float]:
    """(seasonal amplitude = sd of the month-of-year means, interannual sd of the 12-month
    smoothed deseasonalised series, linear trend m/yr) of one monthly series."""
    a = x - np.nanmean(x)
    clim = np.array([np.nanmean(a[cal == k]) if np.isfinite(a[cal == k]).any() else np.nan
                     for k in range(12)])
    d = a - clim[cal]
    sm = np.array([np.nanmean(d[max(0, t - 6):t + 6]) if np.isfinite(d[max(0, t - 6):t + 6]
                                                                     ).sum() >= 8 else np.nan
                   for t in range(len(d))])
    t = np.arange(len(x))[np.isfinite(x)] / 12.0
    tr = float(np.polyfit(t, x[np.isfinite(x)], 1)[0]) if len(t) > 24 else np.nan
    return float(np.nanstd(clim)), float(np.nanstd(sm)), tr


def response(head: np.ndarray, e_local: np.ndarray, max_lag: int = 2) -> tuple[float, float, int]:
    """Best-lag regression of the head anomaly on the local electricity anomaly (both own-mean
    removed): (slope m per GWh/month, correlation, lag)."""
    best = (np.nan, np.nan, -1)
    for lag in range(max_lag + 1):
        e = np.r_[np.full(lag, np.nan), e_local[:len(e_local) - lag]] if lag else e_local
        m = np.isfinite(head) & np.isfinite(e)
        if m.sum() < 36 or np.std(e[m]) == 0:
            continue
        h, x = head[m] - head[m].mean(), (e[m] - e[m].mean()) / 1e6
        r = float(np.corrcoef(h, x)[0, 1])
        if not np.isfinite(best[1]) or r < best[1]:        # pumping lowers heads: most negative
            best = (float((x @ h) / (x @ x)), r, lag)
    return best


def main() -> None:
    st = pd.read_parquet(STATIONS, columns=["sid", "CountyName", "WellDepth", "FinishDepth",
                                            "WellElevation", "GroundwaterLayerCode"]
                         ).set_index("sid")
    r0, r1 = per_entry(RUN["R0"], st), per_entry(RUN["R1c"], st)
    wells = r1[["sid", "x", "y", "layer", "county", "rmse", "amp", "r"]].rename(
        columns={"rmse": "rmse_r1c", "amp": "amp_r1c", "r": "r_r1c"})
    wells = wells.join(r0[["rmse", "amp", "r"]].rename(
        columns={"rmse": "rmse_r0", "amp": "amp_r0", "r": "r_r0"}))
    wells["d_rmse"] = wells["rmse_r1c"] - wells["rmse_r0"]
    wells["well_depth_m"] = pd.to_numeric(st.loc[wells["sid"], "WellDepth"], errors="coerce"
                                          ).to_numpy()
    with open(run_dir(RUN["R1c"]) / "inputs.local.pkl", "rb") as fh:
        inp = pickle.load(fh)
    cxy = cell_xy(inp.grid)
    E = np.asarray(sum(inp.E_by_class.values()), dtype=float)          # (A, 132) kWh/month
    R = np.asarray(inp.recharge_field, dtype=float)
    dates = pd.DatetimeIndex(inp.dates)
    cal = np.asarray(dates[1:].month - 1)
    years = len(dates) / 12.0
    # ---- census pumps (kept, in grid) and doubtful coordinates
    cp = pd.read_parquet(CENSUS_L1, columns=CENSUS_COLS)
    cp = cp[cp["county"].isin(["changhua", "yunlin"])].copy()
    pts = cp[["x_twd97", "y_twd97"]].to_numpy(dtype=float)
    key = pd.Series(list(zip(cp["x_twd97"].round(1), cp["y_twd97"].round(1), strict=True)), index=cp.index)
    cp["shared_point"] = key.map(key.value_counts()).to_numpy() >= 5
    cp["lattice100"] = ((np.mod(pts[:, 0], 100) == 0) & (np.mod(pts[:, 1], 100) == 0))
    kept = cp[cp["kept"].astype(bool) & cp["in_grid"].astype(bool)]
    kxy = kept[["x_twd97", "y_twd97"]].to_numpy(dtype=float)
    feats = []
    for e, w in wells.iterrows():
        d_cell = np.hypot(cxy[:, 0] - w.x, cxy[:, 1] - w.y)
        d_pump = np.hypot(kxy[:, 0] - w.x, kxy[:, 1] - w.y)
        d_all = np.hypot(pts[:, 0] - w.x, pts[:, 1] - w.y)
        near2 = d_all <= 2000.0
        kn2 = d_pump <= 2000.0
        feats.append({
            "entry": e,
            "E_2km_GWh_yr": E[d_cell <= 2000.0].sum() / years / 1e6,
            "E_5km_GWh_yr": E[d_cell <= 5000.0].sum() / years / 1e6,
            "nearest_pump_m": float(d_pump.min()) if len(d_pump) else np.nan,
            "pumps_2km": int(kn2.sum()),
            "pump_depth_2km_m": float(np.nanmedian(kept["well_depth_m"].to_numpy()[kn2]))
            if kn2.any() else np.nan,
            "dub_shared_2km": float(cp["shared_point"].to_numpy()[near2].mean())
            if near2.any() else np.nan,
            "dub_lattice_2km": float(cp["lattice100"].to_numpy()[near2].mean())
            if near2.any() else np.nan,
            "dub_offgrid_2km": float((~cp["in_grid"].astype(bool).to_numpy()[near2]).mean())
            if near2.any() else np.nan})
    wells = wells.join(pd.DataFrame(feats).set_index("entry"))
    qc = pd.read_csv(QC, dtype={"sid": str})
    qc = qc[qc["period"] == "pre"].set_index("sid")
    for col in ("share_days_flag9", "persistent_jumps_per_year", "offset_episodes",
                "steps_corrected", "dropouts"):
        wells["qc_" + col] = pd.to_numeric(qc[col], errors="coerce").reindex(wells["sid"]
                                                                              ).to_numpy()
    # ---- Part A statistics
    covs = ["E_2km_GWh_yr", "E_5km_GWh_yr", "nearest_pump_m", "pumps_2km", "pump_depth_2km_m",
            "dub_shared_2km", "dub_lattice_2km", "dub_offgrid_2km", "layer", "well_depth_m",
            "qc_share_days_flag9", "qc_persistent_jumps_per_year", "qc_offset_episodes",
            "qc_steps_corrected", "amp_r1c", "r_r1c"]
    part_a = {}
    for cname, cc in COUNTIES.items():
        d = wells[wells["county"] == cc]
        worst = d["rmse_r1c"] >= d["rmse_r1c"].quantile(0.75)
        out = {"n_wells": int(len(d)), "rmse_r1c_mean": float(d["rmse_r1c"].mean()),
               "rmse_r0_mean": float(d["rmse_r0"].mean()),
               "wells_worse_with_census": int((d["d_rmse"] > 0).sum()),
               "spearman": {}, "worst_quartile_vs_rest_median": {}}
        for t in ("rmse_r1c", "d_rmse"):
            out["spearman"][t] = {}
            for c in covs:
                m = d[[t, c]].dropna()
                if len(m) >= 10 and m[c].nunique() > 1:
                    rho, p = spearmanr(m[t], m[c])
                    out["spearman"][t][c] = {"rho": float(rho), "p": float(p), "n": int(len(m))}
        for c in covs + ["rmse_r0", "d_rmse"]:
            out["worst_quartile_vs_rest_median"][c] = [float(d.loc[worst, c].median()),
                                                       float(d.loc[~worst, c].median())]
        out["worst_wells"] = (d.loc[worst].sort_values("rmse_r1c", ascending=False)
                              [["sid", "layer", "rmse_r0", "rmse_r1c", "amp_r1c", "r_r1c",
                                "E_2km_GWh_yr", "nearest_pump_m", "pump_depth_2km_m",
                                "well_depth_m", "dub_shared_2km"]]
                              .round(3).to_dict(orient="records"))
        part_a[cname] = out
    # ---- Part B: county comparison
    with np.load(COUNTY_MAP) as z:
        ch_cells = np.asarray(z["changhua"], dtype=bool)
    with np.load(SW) as z:
        swd = pd.DatetimeIndex(pd.to_datetime([str(t) for t in z["dates"]]))
        sw = np.asarray(z["sw_m_per_day"], dtype=float)[:, swd.get_indexer(dates)]
    th = json.loads((run_dir(RUN["R1c"]) / "stage3_theta.json").read_text())["theta"]
    part_b = {}
    for cname, cc in COUNTIES.items():
        cells = ch_cells if cname == "changhua" else ~ch_cells
        area_km2 = cells.sum() * (inp.grid.dx / 1000.0) ** 2
        e_mon = E[cells].sum(axis=0)
        e_clim = np.array([e_mon[1:][cal == k].mean() for k in range(12)])
        d = wells[wells["county"] == cc]
        seas, inter, trend, slope, rr, lag, dtw, band = [], [], [], [], [], [], [], []
        ge = np.asarray(inp.ground_elev, dtype=float)
        for e, w in d.iterrows():
            o = r1.loc[e, "obs"]
            s_, i_, t_ = climatology_parts(o, cal)
            seas.append(s_)
            inter.append(i_)
            trend.append(t_)
            dc = np.hypot(cxy[:, 0] - w.x, cxy[:, 1] - w.y) <= 3000.0
            sl, r_, lg = response(o, E[dc].sum(axis=0)[1:])
            slope.append(sl)
            rr.append(r_)
            lag.append(lg)
            # GroundHeight is 0 or blank for many stations (a missing-value code), so the
            # ground is the surveyed WellElevation, or the model's DEM cell when that is missing
            we = pd.to_numeric(st.loc[w.sid, "WellElevation"], errors="coerce")
            g = we if np.isfinite(we) and we > 0 else ge[int(np.argmin(
                np.hypot(cxy[:, 0] - w.x, cxy[:, 1] - w.y)))]
            dtw.append(float(g - np.nanmean(o)))
            band.append("west_lt185km" if w.x < 185000 else
                        "mid_185_205km" if w.x < 205000 else "east_ge205km")
        kc = kept[kept["county"] == cname]
        dep = pd.to_numeric(kc["well_depth_m"], errors="coerce")
        part_b[cname] = {
            "grid_area_km2": float(area_km2), "n_scored_wells": int(len(d)),
            "wells_per_100km2": float(len(d) / area_km2 * 100),
            "well_layer_counts": {str(k): int(v) for k, v in d["layer"].value_counts()
                                  .sort_index().items()},
            "obs_seasonal_amp_m_median": float(np.nanmedian(seas)),
            "obs_interannual_sd_m_median": float(np.nanmedian(inter)),
            "obs_trend_m_per_yr_median": float(np.nanmedian(trend)),
            "depth_to_water_m_median": float(np.nanmedian(dtw)),
            "depth_to_water_by_layer_m": {str(k): float(np.nanmedian(np.asarray(dtw)[
                d["layer"].to_numpy() == k])) for k in sorted(d["layer"].unique())},
            "wells_by_easting_band": {b: int(n) for b, n in pd.Series(band).value_counts()
                                      .items()},
            "data_response_by_band": {b: {
                "slope_median": float(np.nanmedian(np.asarray(slope)[np.asarray(band) == b])),
                "r_median": float(np.nanmedian(np.asarray(rr)[np.asarray(band) == b])),
                "seasonal_amp_median": float(np.nanmedian(np.asarray(seas)[np.asarray(band)
                                                                           == b])),
                "n": int((np.asarray(band) == b).sum())} for b in sorted(set(band))},
            "data_response_slope_m_per_GWh_month_median": float(np.nanmedian(slope)),
            "data_response_r_median": float(np.nanmedian(rr)),
            "data_response_best_lag_counts": {str(k): int(v) for k, v in
                                              pd.Series(lag).value_counts().sort_index().items()},
            "energy_GWh_per_yr": float(e_mon.sum() / years / 1e6),
            "energy_MWh_per_km2_yr": float(e_mon.sum() / years / 1e3 / area_km2),
            "energy_seasonal_max_over_min": float(e_clim.max() / e_clim.min()),
            "energy_peak_month": int(np.argmax(e_clim) + 1),
            "kept_pumps_in_grid": int(len(kc)),
            "pumps_per_km2": float(len(kc) / area_km2),
            "pump_hp_median": float(pd.to_numeric(kc["pump_hp"], errors="coerce").median()),
            # the census carries no usable pump depth (well_depth_m is empty for kept pumps)
            "pump_depth_known_share": float(dep.notna().mean()),
            "purpose_share": {k: float(v) for k, v in
                              kc["purpose_class"].value_counts(normalize=True).items()},
            "billing_share": {str(k): float(v) for k, v in
                              kc["billing_code"].value_counts(normalize=True).head(5).items()},
            "canal_mm_per_yr": float(sw[cells].mean() * 365.25 * 1000),
            "rain_recharge_mm_per_yr": float(R[cells].mean() * 365.25 * 1000),
            "model_r1c_amp_median": float(d["amp_r1c"].median()),
            "model_r1c_r_median": float(d["r_r1c"].median()),
            "model_r0_amp_median": float(d["amp_r0"].median()),
            "dub_shared_point_share": float(cp.loc[cp["county"] == cname, "shared_point"].mean()),
            "dub_lattice100_share": float(cp.loc[cp["county"] == cname, "lattice100"].mean())}
    part_b["r1c_mid_storage_log_S"] = {"rest": th.get("log_S_mid"),
                                       "changhua": th.get("log_S_mid_changhua")}
    rec = {"schema": "i1-diag-changhua-local/1", "runs": RUN,
           "role": "after_the_fact_analysis (U7: never a gate, threshold or choice)",
           "part_a_local": part_a, "part_b_counties": part_b}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    wells.drop(columns=[c for c in wells.columns if c in ("obs", "pred")]).to_csv(
        OUT / "wells.csv")
    look = prereg3.record_look(
        ["I1:design_oof_heads_R0_R1e"], "heads:2012-01..2022-12", "I1", "after_the_fact",
        note="Local checks on the worst Changhua wells and the Changhua/Yunlin comparison "
             "(user request 2026-10-19); report only. "
             "results/dataprep/reports/I1_changhua_local/report.json")
    print("look logged:", look["time"])


if __name__ == "__main__":
    main()
