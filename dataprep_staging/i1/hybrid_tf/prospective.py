"""Frozen model code of the DRAFT test ``I1.hybrid_tf.prospective_2026-11`` (unregistered).

This module is what the draft declaration hashes. It is never run on head months after
2022-12 by the design study; at registration the registrar runs :func:`fit` once with
``variant="full"`` and ``fit_end="2026-09"``, records the sha256 of ``parameters.json`` in the
registration, and changes nothing afterwards.

Model (primary, test T1, one month ahead): for each well,

    h(t) = clim(m(t)) + rho * (h(t-1) - clim(m(t-1))) + TF(t),

the existing AR(1) statistical layer (``twin.hybrid`` ``stat``, one month ahead) plus a ridge
transfer-function model ``TF`` of its one-month innovations on the drivers' filtered
anomalies and month-of-year dummies ("ARX"; ``evaluate`` ``arx*``). ``clim`` and ``rho`` come
from the fit months (flags 0-1, U1); the ridge has the frozen hyperparameters of ``config``.
Secondary (test T2, reported only): the pure TF simulation ``h(t) = TF_h(t)`` with recorded
drivers and no head after the origin (``evaluate`` ``tf``).

Driver months: a month counts as recorded under the rules of ``config`` (grid share of flags
0-5 >= 0.99; canal flag 6 is not a record; rain >= 5 gauges with >= 90 % of days; river
>= 20 daily values; public wells to 2022-12). Every other month takes the driver's
month-of-year climatology over the recorded fit months, except the public wells, which hold
their last recorded value (rules fixed before any score was read; ``config.HOLD_LAST``).

Sources by variant (each opened through ``common.guard.open_product``):

* ``full`` (registration and scoring): release full components with use
  ``statistical_layer_history`` (heads, electricity, canal, ET0; river from
  surface_water_v4/full/v4), the P6 member with use ``forcing_frozen``, the county map with
  ``model_design``. Heads rows are filtered to flags 0-1.
* ``preorigin`` (code checks on 2012-2022 only): the pre-origin components of ``data``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .model import Ridge, features, lag1, month_dummies

sys.path.insert(0, str(C.REPO / "dataprep_staging"))
sys.path.insert(0, str(C.REPO))

from common.guard import filter_flags, open_product  # noqa: E402

SOURCES = {
    "full": {
        "heads": (C.REL / "full/heads_observed/v1", "statistical_layer_history"),
        "elec": (C.REL / "full/electricity_census/v1", "statistical_layer_history"),
        "sw": (C.REL / "full/surface_water/v1", "statistical_layer_history"),
        "et0": (C.REL / "full/et0/v1", "statistical_layer_history"),
        "river": (C.REPO / "results/dataprep/surface_water_v4/full/v4",
                  "statistical_layer_history"),
        "pw": (C.PW_PRE, "forcing_frozen"),
        "county": (C.COUNTY_MAP, "model_design"),
    },
    "preorigin": {
        "heads": (C.HEADS_PRE, "design_time_evidence"),
        "elec": (C.ELEC_PRE, "design_time_evidence"),
        "sw": (C.SW_PRE, "design_time_evidence"),
        "et0": (C.ET0_PRE, "statistical_layer_history"),
        "river": (C.W1_PRE, "design_time_evidence"),
        "pw": (C.PW_PRE, "design_time_evidence"),
        "county": (C.COUNTY_MAP, "model_design"),
    },
}
REF_END = "2022-12"     # driver anomalies are centred on their 2012-01..2022-12 mean (design)


def months(a: str, b: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{a}-01", f"{b}-01", freq="MS")


def _open(src: dict, key: str):
    path, use = src[key]
    return open_product(path, use)


def _npz_output(pr) -> str:
    return next(o for o in pr.outputs if o.endswith(".npz"))


def wells_frozen() -> pd.DataFrame:
    """R0's 158 scored wells (``stage3_datum_gate_cwa``), the well set of the draft test."""
    z = np.load(C.R0_RUN / "stage3_per_entry.npz", allow_pickle=True)
    kw = pd.read_csv(C.R0_RUN / "stage3_kfold_wells.csv", dtype={"sid": str}).set_index("entry")
    return pd.DataFrame({"sid": kw.loc[z["entry"], "sid"].to_numpy(),
                         "layer": np.asarray(z["layer"], int)})


def heads_matrix(src: dict, sids, hd: pd.DatetimeIndex, use_override: str | None = None
                 ) -> np.ndarray:
    """(W, T) monthly heads, flags 0-1 only (U1), on the months ``hd``."""
    path, use = src["heads"]
    pr = open_product(path, use_override or use)
    h = filter_flags(pr.read_table("heads_monthly.parquet",
                                   columns=["sid", "date", "value", "flag"]), "baseline")
    h["date"] = pd.to_datetime(h["date"])
    h = h[(h["date"] >= hd[0]) & (h["date"] <= hd[-1])]
    piv = h.pivot_table(index="sid", columns="date", values="value", aggfunc="mean")
    return piv.reindex(index=list(sids), columns=hd).to_numpy(float)


def driver_panel(src: dict, sids, dd: pd.DatetimeIndex) -> tuple[dict, dict, dict]:
    """Raw driver series per well on months ``dd`` and each driver's recorded-month mask.

    Returns (values name -> (W, T) with NaN where not recorded, recorded name -> (T,) bool,
    station attributes)."""
    T = len(dd)
    ym = [d.strftime("%Y-%m") for d in dd]
    hp = _open(src, "heads")
    st = hp.read_table("stations.parquet", columns=["sid", "CountyName", "x_twd97", "y_twd97",
                                                     "cell_index"]).set_index("sid")
    inv = {v: k for k, v in C.COUNTIES.items()}
    county = np.array([inv.get(st.loc[s, "CountyName"], "") for s in sids])
    xy = st.loc[list(sids), ["x_twd97", "y_twd97"]].to_numpy(float)
    cell = st.loc[list(sids), "cell_index"].to_numpy(int)
    cm = _open(src, "county")
    with np.load(cm.path("zone_county_map.npz")) as zz:
        ch = np.asarray(zz["changhua"], bool)
        cxy = np.asarray(zz["centroids"], float)
    cc = {"changhua": ch, "yunlin": ~ch}

    def on_months(dates, arr):
        idx = pd.Index([str(x)[:7] for x in dates]).get_indexer(ym)
        out = np.full(arr.shape[:-1] + (T,), np.nan)
        ok = idx >= 0
        out[..., ok] = arr[..., idx[ok]]
        return out

    ep = _open(src, "elec")
    with np.load(ep.path("cells_total.npz")) as z:
        ev = on_months(z["dates"], np.asarray(z["value_all"], float))
        ef = on_months(z["dates"], np.asarray(z["flag_all"], float))
    active = np.nanmax(np.where(np.isfinite(ev), ev, 0), axis=1) > 0
    ok = np.isin(ef, range(6)) & np.isfinite(ev)
    rec_e = ok[active].mean(axis=0) >= C.GRID_MIN_SHARE
    E = np.where(ok, ev, 0.0)
    sp = _open(src, "sw")
    with np.load(sp.path("grid_sw_v4.npz")) as z:
        sv = on_months(z["dates"], np.asarray(z["value"], float))
        sf = on_months(z["dates"], np.asarray(z["flag"], float))
    rec_s = (np.isin(sf, range(6)) & np.isfinite(sv)).mean(axis=0) >= C.GRID_MIN_SHARE
    dm = sp.read_table("district_monthly.csv", columns=["date", "district", "value", "flag"])
    dm["ym"] = dm["date"].astype(str).str[:7]
    dist, rec_dist = {}, {}
    for d, g in dm.groupby("district"):
        g = g.set_index("ym").reindex(ym)
        dist[d] = g["value"].to_numpy(float)
        rec_dist[d] = np.isin(g["flag"].to_numpy(), range(6)) & np.isfinite(dist[d])
    pp = _open(src, "pw")
    with np.load(pp.path("sensitivity_member.npz")) as z:
        pw = on_months(z["dates"], np.asarray(z["value"], float).sum(axis=0))
    rec_p = np.isfinite(pw).all(axis=0)
    rp = _open(src, "river")
    gd = rp.read_table("gauges_daily_L1.parquet", columns=["date", "station_no", "value"])
    gd["date"] = pd.to_datetime(gd["date"])
    river, rec_r = {}, {}
    for name, code in C.GAUGES.items():
        s = gd[gd["station_no"] == code].set_index("date")["value"].astype(float)
        mm = s.resample("MS").agg(["mean", "count"]).reindex(dd)
        river[name] = mm["mean"].to_numpy()
        rec_r[name] = (mm["count"].fillna(0).to_numpy() >= C.RIVER_MIN_DAYS)
    from hydrophysics.subsidence import idw_interp
    tp = _open(src, "et0")
    with np.load(tp.path(_npz_output(tp))) as z:
        et = pd.DataFrame(np.asarray(z["et0"], float).T, index=pd.to_datetime(z["dates"]))
        et_ids = [str(x) for x in z["well_ids"]]
    gws = pd.read_csv(C.GW_STATIONS).set_index("st_id")
    et_xy = gws.loc[et_ids, ["TM_X97", "TM_Y97"]].to_numpy(float)
    etm = et.resample("MS").mean().reindex(dd)
    rec_et = etm.notna().all(axis=1).to_numpy()
    rf = pd.read_csv(C.RAIN, parse_dates=["date time"]).set_index("date time")
    rf = rf[(rf.index >= dd[0]) & (rf.index < dd[-1] + pd.offsets.MonthBegin(1))]
    rst = pd.read_csv(C.RAIN_STATIONS).set_index("rf_id")
    grp = rf.index.to_period("M")
    full_month = rf.groupby(grp).size().reindex(dd.to_period("M")).fillna(0).to_numpy() \
        >= np.asarray(dd.days_in_month)
    cover = (rf.notna().groupby(grp).mean() >= 0.9).sum(axis=1).reindex(
        dd.to_period("M")).fillna(0).to_numpy()
    rec_rain = full_month & (cover >= C.RAIN_MIN_GAUGES) & rec_et
    rfm = rf.groupby(grp).mean().reindex(dd.to_period("M"))
    rf_xy = rst.loc[rfm.columns, ["TM_X97", "TM_Y97"]].to_numpy(float)
    recharge = np.clip(idw_interp(xy, rf_xy, rfm.to_numpy().T)
                       - idw_interp(xy, et_xy, etm.to_numpy().T), 0.0, None) / 1000.0

    W = len(sids)
    vals = {k: np.full((W, T), np.nan) for k in C.DRIVERS}
    recm = {k: np.zeros((W, T), bool) for k in C.DRIVERS}
    for i in range(W):
        dcell = np.hypot(cxy[:, 0] - xy[i, 0], cxy[:, 1] - xy[i, 1])
        c = cc[county[i]]
        vals["electricity_local_3km"][i] = E[dcell <= C.LOCAL_RADIUS_M].sum(axis=0)
        vals["electricity_county"][i] = E[c].sum(axis=0)
        recm["electricity_local_3km"][i] = recm["electricity_county"][i] = rec_e
        vals["rain_recharge"][i] = recharge[i]
        recm["rain_recharge"][i] = rec_rain
        vals["canal_w1_cell"][i] = sv[cell[i]]
        recm["canal_w1_cell"][i] = rec_s
        vals["canal_district"][i] = dist[county[i]]
        recm["canal_district"][i] = rec_dist[county[i]]
        for name in C.GAUGES:
            vals[name][i] = river[name]
            recm[name][i] = rec_r[name]
        vals["public_wells_county"][i] = pw[c].sum(axis=0)
        recm["public_wells_county"][i] = rec_p
    for k in C.DRIVERS:
        vals[k] = np.where(recm[k], vals[k], np.nan)
    attrs = {"county": county, "xy": xy, "cell": cell}
    return vals, recm, attrs


def fill(vals: np.ndarray, rec: np.ndarray, cal: np.ndarray, fit: np.ndarray, hold: bool
         ) -> tuple[np.ndarray, np.ndarray]:
    """Fill the months not recorded: month-of-year mean over the recorded fit months, or the
    last recorded value (``hold``). Returns (filled, the 12 climatology values)."""
    v = vals.copy()
    sel = rec & fit
    allm = float(np.nanmean(v[sel])) if sel.any() else 0.0
    clim = np.array([np.nanmean(v[sel & (cal == m)]) if (sel & (cal == m)).any() else allm
                     for m in range(12)])
    if hold:
        last = np.nan
        for t in range(len(v)):
            if rec[t] and np.isfinite(v[t]):
                last = v[t]
            elif np.isfinite(last):
                v[t] = last
            else:
                v[t] = clim[cal[t]]
    else:
        v = np.where(rec & np.isfinite(v), v, clim[cal])
    return v, clim


def fit(variant: str, fit_end: str, out_dir: Path | None = None, sids=None) -> dict:
    """Fit the frozen model on heads 2012-02..``fit_end`` and drivers 2012-01..``fit_end``."""
    src = SOURCES[variant]
    wl = wells_frozen() if sids is None else pd.DataFrame({"sid": list(sids)})
    sids = wl["sid"].tolist()
    dd = months(C.DRIVER_START, fit_end)
    hd = dd[1:]
    y = heads_matrix(src, sids, hd)
    vals, recm, attrs = driver_panel(src, sids, dd)
    cal_d, cal_h = np.asarray(dd.month - 1), np.asarray(hd.month - 1)
    ref_sel = dd <= pd.Timestamp(f"{REF_END}-01")
    dum = month_dummies(cal_h)
    fit_d = np.ones(len(dd), bool)
    wells = {}
    rhos = []
    for i, s in enumerate(sids):
        series, dclim, ref = {}, {}, {}
        for k in C.DRIVERS:
            f, cl = fill(vals[k][i], recm[k][i], cal_d, fit_d, k in C.HOLD_LAST)
            series[k], dclim[k], ref[k] = f, cl, float(f[ref_sel].mean())
        X = np.c_[features(series, C.DRIVERS, ref)[1:], dum]
        ok = np.isfinite(y[i])
        rec = {"county": str(attrs["county"][i]), "n_fit": int(ok.sum())}
        if ok.sum() < C.MIN_TRAIN:
            wells[s] = {**rec, "status": "too few fit months"}
            continue
        m_all = float(y[i, ok].mean())
        clim = np.array([y[i, ok & (cal_h == m)].mean() if (ok & (cal_h == m)).any() else m_all
                         for m in range(12)])
        a = y[i] - clim[cal_h]
        rho, npair = lag1(a, ok)
        rhos.append(rho)
        rec.update({"clim_m": clim.tolist(), "rho": rho, "rho_pairs": npair,
                    "driver_ref_mean": ref, "driver_clim": {k: v.tolist()
                                                            for k, v in dclim.items()},
                    "last_recorded": {k: (str(dd[np.flatnonzero(recm[k][i])[-1]])[:7]
                                          if recm[k][i].any() else None) for k in C.DRIVERS},
                    "_X": X, "_a": a, "_ok": ok,
                    "_tf": Ridge().fit(X[ok], y[i, ok])})
        wells[s] = rec
    pooled = float(np.nanmedian(rhos)) if rhos else 0.0
    for rec in wells.values():
        if "_X" not in rec:
            continue
        r = rec["rho"] if np.isfinite(rec["rho"]) else pooled
        rec["rho_used"] = r
        ok, a, X = rec.pop("_ok"), rec.pop("_a"), rec.pop("_X")
        tr = np.flatnonzero(ok[1:] & ok[:-1]) + 1
        rec["arx"] = _ridge_params(Ridge().fit(X[tr], a[tr] - r * a[tr - 1]))
        rec["tf"] = _ridge_params(rec.pop("_tf"))
        rec["status"] = "fitted"
    params = {"schema": "i1-hybrid-tf-frozen/1", "variant": variant,
              "fit_heads": f"{C.HEAD_START}..{fit_end}", "fit_drivers":
              f"{C.DRIVER_START}..{fit_end}", "ref_centring": f"{C.DRIVER_START}..{REF_END}",
              "rho_pooled": pooled, "hyperparameters": hyperparameters(),
              "code_sha256": code_hashes(), "wells": wells}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        data = json.dumps(params, indent=1, ensure_ascii=False, default=float).encode()
        (out_dir / "parameters.json").write_bytes(data)
        params["parameters_sha256"] = hashlib.sha256(data).hexdigest()
    return params


def _ridge_params(m: Ridge) -> dict:
    return {"keep": m.keep.tolist(), "mu": m.mu.tolist(), "sd": m.sd.tolist(),
            "lo": m.lo.tolist(), "hi": m.hi.tolist(), "c": m.c, "beta": m.beta.tolist()}


def _ridge_from(p: dict) -> Ridge:
    m = Ridge()
    m.keep = np.asarray(p["keep"], bool)
    for k in ("mu", "sd", "lo", "hi", "beta"):
        setattr(m, k, np.asarray(p[k], float))
    m.c = float(p["c"])
    return m


def predict(params: dict, variant: str, end: str, y_prev: np.ndarray | None,
            sids=None) -> dict:
    """Predictions on every month 2012-02..``end`` from the frozen parameters.

    ``y_prev`` (W, T) holds the heads used as the previous-month input of the one-month-ahead
    forecast (flags 0-1; NaN where not observed); returns ``{"dates", "sids", "arx1",
    "stat_ar1_1", "persist1", "clim", "tf"}`` arrays (W, T). Unrecorded driver months use the
    frozen climatology (or hold), as in :func:`fit`."""
    src = SOURCES[variant]
    sids = list(params["wells"]) if sids is None else list(sids)
    dd = months(C.DRIVER_START, end)
    hd = dd[1:]
    vals, recm, _ = driver_panel(src, sids, dd)
    cal_d, cal_h = np.asarray(dd.month - 1), np.asarray(hd.month - 1)
    dum = month_dummies(cal_h)
    W, T = len(sids), len(hd)
    out = {k: np.full((W, T), np.nan) for k in ("arx1", "stat_ar1_1", "persist1", "clim",
                                                  "tf")}
    for i, s in enumerate(sids):
        p = params["wells"].get(s, {})
        if p.get("status") != "fitted":
            continue
        series = {}
        for k in C.DRIVERS:
            cl = np.asarray(p["driver_clim"][k], float)
            v, r = vals[k][i], recm[k][i]
            if k in C.HOLD_LAST:
                series[k], _ = fill(v, r, cal_d, np.ones(len(dd), bool), True)
            else:
                series[k] = np.where(r & np.isfinite(v), v, cl[cal_d])
        X = np.c_[features(series, C.DRIVERS, p["driver_ref_mean"])[1:], dum]
        clim = np.asarray(p["clim_m"], float)[cal_h]
        out["clim"][i] = clim
        out["tf"][i] = _ridge_from(p["tf"]).predict(X)
        if y_prev is None:
            continue
        yp = np.r_[np.nan, y_prev[i, :-1]]
        cp = np.r_[np.nan, clim[:-1]]
        base = clim + p["rho_used"] * (yp - cp)
        out["persist1"][i] = yp
        out["stat_ar1_1"][i] = base
        out["arx1"][i] = base + _ridge_from(p["arx"]).predict(X)
    out["dates"] = np.array([d.strftime("%Y-%m") for d in hd])
    out["sids"] = np.array(sids)
    return out


def hyperparameters() -> dict:
    return {"taus_months": list(C.TAUS), "ridge_alpha": C.ALPHA,
            "local_radius_m": C.LOCAL_RADIUS_M, "min_train": C.MIN_TRAIN,
            "rho_min_pairs": C.RHO_MIN_PAIRS, "drivers": list(C.DRIVERS),
            "hold_last": sorted(C.HOLD_LAST), "rain_min_gauges": C.RAIN_MIN_GAUGES,
            "river_min_days": C.RIVER_MIN_DAYS, "grid_min_share": C.GRID_MIN_SHARE,
            "ridge_guards": "constant fit columns dropped; standardised features clipped to "
                            "their fit range", "month_dummies": 11,
            "driver_centring": f"{C.DRIVER_START}..{REF_END} mean of the filled series"}


def code_hashes() -> dict:
    d = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.glob("*.py"))}
