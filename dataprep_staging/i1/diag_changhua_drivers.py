"""After-the-fact diagnosis, part 3 (2026-10-19, user request): what drives Changhua's heads, if
not the census electricity? A model-free test of every driver already in hand.

For every scored well (out-of-fold observations of R0's design-time folds; the predictions are
not used), the monthly own-mean head anomaly is regressed on each candidate driver's anomaly
passed through causal exponential filters with response times of 0 (same month), 2, 6 and 18
months: the transfer-function form of a well hydrograph (a driver's effect builds up and decays
over months). Ridge regression on standardised features, alpha = 1, scored out of sample by
leave-one-year-out cross-validation (11 folds, 2012-2022). Reported per county (Changhua, and
Yunlin as the reference): the median out-of-sample R2 of

* month-of-year climatology alone (the no-driver baseline);
* each driver alone, and every driver together, with and without the climatology;
* the drop in R2 when one driver is left out of the full set;

for the raw anomaly and for the deseasonalised anomaly (each well's month-of-year means removed;
drivers deseasonalised the same way), which isolates the year-to-year part.

Drivers (all 2012-02..2022-12, as the scored months):
* census electricity within 3 km of the well, and over the well's county (R1c inputs);
* rain recharge at the well's cell (rain - ET0, clamped at 0; R1c inputs);
* W1 canal deliveries at the well's cell (surface_water_v4 grid) and W1's district monthly
  deliveries of the well's own irrigation district (changhua / yunlin), pre-origin v2;
* Zhuoshui discharge at 彰雲橋 (apex, H057) and 溪州大橋 (mid-fan, H071), monthly means of the
  yearbook daily series, pre-origin v2;
* the P6 public-well abstraction over the well's county.

Report only (U7): never a gate, a threshold or a rung choice. Logs one look (after_the_fact).

Run from the repository root:
    dataprep_staging/tools/cpu_slot.sh python dataprep_staging/i1/diag_changhua_drivers.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(REPO))

from harness import prereg3  # noqa: E402

OUT = REPO / "results/dataprep/reports/I1_changhua_drivers"
RUN = REPO / "results/twin_runs/stage3_datum_gate_ccz"
STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"
COUNTY_MAP = REPO / "results/dataprep/zone_county_map/v2/zone_county_map.npz"
W1 = REPO / "results/dataprep/surface_water_v4/preorigin/v2"
SW = REPO / "results/dataprep/surface_water_v4/model_npz/v1/surface_water_v4_model.npz"
PW = REPO / "results/dataprep/public_wells/model_npz/v1/public_wells_model.npz"
COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}
TAUS = (0, 2, 6, 18)
ALPHA = 1.0
MIN_MONTHS = 60
GAUGES = {"river_apex_H057": "H057", "river_midfan_H071": "H071"}


def efilter(x: np.ndarray, tau: float) -> np.ndarray:
    """Causal exponential filter (unit gain); tau = 0 returns x."""
    if tau == 0:
        return x.copy()
    a = np.exp(-1.0 / tau)
    out = np.empty_like(x)
    acc = 0.0
    for t, v in enumerate(x):
        acc = a * acc + (1 - a) * v
        out[t] = acc
    return out


def deseason(x: np.ndarray, cal: np.ndarray) -> np.ndarray:
    clim = np.array([np.nanmean(x[cal == k]) for k in range(12)])
    return x - clim[cal]


def features(series: dict, names: list[str]) -> np.ndarray:
    cols = []
    for n in names:
        s = series[n]
        s = s - np.nanmean(s)
        for tau in TAUS:
            cols.append(efilter(s, tau))
    return np.column_stack(cols) if cols else np.zeros((len(next(iter(series.values()))), 0))


def cv_r2(y: np.ndarray, X: np.ndarray, year: np.ndarray) -> float:
    """Leave-one-year-out out-of-sample R2 of ridge(y ~ X) (standardised X, intercept)."""
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    pred = np.full_like(y, np.nan)
    for yr in np.unique(year[m]):
        tr, te = m & (year != yr), m & (year == yr)
        if te.sum() == 0 or tr.sum() < 24:
            continue
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd[sd == 0] = 1.0
        Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
        yc = y[tr].mean()
        if Xt.shape[1]:
            beta = np.linalg.solve(Xt.T @ Xt + ALPHA * np.eye(Xt.shape[1]), Xt.T @ (y[tr] - yc))
            pred[te] = yc + Xe @ beta
        else:
            pred[te] = yc
    ok = m & np.isfinite(pred)
    if ok.sum() < 24:
        return float("nan")
    resid = ((y[ok] - pred[ok]) ** 2).sum()
    tot = ((y[ok] - y[ok].mean()) ** 2).sum()
    return float(1.0 - resid / tot)


def main() -> None:
    st = pd.read_parquet(STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    z = np.load(RUN / "stage3_per_entry.npz", allow_pickle=True)
    w = pd.read_csv(RUN / "stage3_kfold_wells.csv", dtype={"sid": str}).set_index("entry")
    sids = w.loc[z["entry"], "sid"].to_numpy()
    with open(RUN / "inputs.local.pkl", "rb") as fh:
        inp = pickle.load(fh)
    dates = pd.DatetimeIndex(inp.dates)
    scored = dates[1:]
    cal = np.asarray(scored.month - 1)
    year = np.asarray(scored.year)
    cxy = np.asarray(inp.grid.centroids(), dtype=float)
    E = np.asarray(sum(inp.E_by_class.values()), dtype=float)[:, 1:]
    R = np.asarray(inp.recharge_field, dtype=float)[:, 1:]
    with np.load(COUNTY_MAP) as zz:
        ch_cells = np.asarray(zz["changhua"], dtype=bool)
    with np.load(SW) as zz:
        swd = pd.DatetimeIndex(pd.to_datetime([str(t) for t in zz["dates"]]))
        sw = np.asarray(zz["sw_m_per_day"], dtype=float)[:, swd.get_indexer(scored)]
    with np.load(PW) as zz:
        pwd = pd.PeriodIndex([str(t) for t in zz["dates"]], freq="M")
        pw = np.asarray(zz["q_m3d"], dtype=float).sum(axis=0)[
            :, pwd.get_indexer(scored.to_period("M"))]
    dm = pd.read_csv(W1 / "district_monthly.csv", parse_dates=["date"])
    district = {d: g.set_index("date")["value"].reindex(scored).to_numpy(dtype=float)
                for d, g in dm.groupby("district")}
    gd = pd.read_parquet(W1 / "gauges_daily_L1.parquet")
    gd["date"] = pd.to_datetime(gd["date"])
    river = {}
    for name, code in GAUGES.items():
        s = gd[gd["station_no"] == code].set_index("date")["value"].astype(float)
        river[name] = s.resample("MS").mean().reindex(scored).to_numpy()
    county_cells = {"changhua": ch_cells, "yunlin": ~ch_cells}
    drivers_all = ["electricity_local_3km", "electricity_county", "rain_recharge_cell",
                   "canal_w1_cell", "canal_district", *GAUGES, "public_wells_county"]
    rows = []
    for i, sid in enumerate(sids):
        county = next((k for k, v in COUNTIES.items() if st.loc[sid, "CountyName"] == v), None)
        if county is None:
            continue
        y = z["obs"][i].astype(float)
        if np.isfinite(y).sum() < MIN_MONTHS:
            continue
        y = y - np.nanmean(y)
        x, yy = float(z["x"][i]), float(z["y"][i])
        dcell = np.hypot(cxy[:, 0] - x, cxy[:, 1] - yy)
        c0 = int(np.argmin(dcell))
        cc = county_cells[county]
        series = {"electricity_local_3km": E[dcell <= 3000.0].sum(axis=0),
                  "electricity_county": E[cc].sum(axis=0),
                  "rain_recharge_cell": R[c0],
                  "canal_w1_cell": sw[c0],
                  "canal_district": district[county],
                  **river,
                  "public_wells_county": pw[cc].sum(axis=0)}
        clim = np.column_stack([(cal == k).astype(float) for k in range(11)])
        rec = {"sid": sid, "county": county, "layer": int(z["layer"][i])}
        for target in ("raw", "deseasonalised"):
            if target == "raw":
                yt, sr = y, series
            else:
                yt = deseason(y, cal)
                sr = {k: deseason(v, cal) for k, v in series.items()}
            pre = f"{target}:"
            if target == "raw":
                rec[pre + "climatology"] = cv_r2(yt, clim, year)
            for d in drivers_all:
                rec[pre + d] = cv_r2(yt, features(sr, [d]), year)
            full = features(sr, drivers_all)
            rec[pre + "all_drivers"] = cv_r2(yt, full, year)
            if target == "raw":
                rec[pre + "all_drivers+climatology"] = cv_r2(yt, np.c_[full, clim], year)
            for d in drivers_all:
                rec[pre + "all_minus_" + d] = rec[pre + "all_drivers"] - cv_r2(
                    yt, features(sr, [k for k in drivers_all if k != d]), year)
        rows.append(rec)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "wells.csv", index=False)
    summary = {}
    for county in COUNTIES:
        d = df[df["county"] == county]
        summary[county] = {"n_wells": int(len(d)),
                           "median_cv_r2": {c: float(d[c].median()) for c in df.columns
                                            if ":" in c},
                           "share_wells_cv_r2_gt_0p3": {c: float((d[c] > 0.3).mean())
                                                        for c in df.columns
                                                        if ":" in c and "minus" not in c}}
    rec = {"schema": "i1-diag-changhua-drivers/1",
           "role": "after_the_fact_analysis (U7: never a gate, threshold or choice)",
           "method": {"taus_months": list(TAUS), "ridge_alpha": ALPHA,
                      "cv": "leave-one-year-out, 2012-2022", "min_months": MIN_MONTHS,
                      "gauges": GAUGES},
           "summary": summary}
    (OUT / "report.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    look = prereg3.record_look(
        ["I1:design_oof_heads_R0_R1e"], "heads:2012-01..2022-12", "I1", "after_the_fact",
        note="Model-free driver regression of Changhua and Yunlin head anomalies "
             "(user request 2026-10-19); report only. "
             "results/dataprep/reports/I1_changhua_drivers/report.json")
    print("look logged:", look["time"])


if __name__ == "__main__":
    main()
