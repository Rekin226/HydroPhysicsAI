"""Guarded loading of the 2012-2022 design data of the hybrid TF layer.

Every product with a manifest is opened through ``common.guard.open_product`` with the use it
serves here (rule 4 of the task; U5):

=====================================  ==========================  ==========================
product                                use                         role
=====================================  ==========================  ==========================
preorigin/heads_observed (H1a L1 v3)   design_time_evidence        targets; rows filtered to
                                                                   flags 0-1 as for a baseline
                                                                   (U1)
preorigin/electricity_census (C0)      design_time_evidence        driver (flags 0-5 kept)
preorigin/surface_water (W1)           design_time_evidence        driver
preorigin/public_wells_member (P6)     design_time_evidence        driver
preorigin/et0 (W2)                     statistical_layer_history   driver (its manifest grants
                                                                   no design use; this layer
                                                                   is a statistical layer)
surface_water_v4/preorigin/v2 gauges   design_time_evidence        driver (river)
zone_county_map/v2                     model_design                county of each cell
=====================================  ==========================  ==========================

Inputs without a manifest: R0's out-of-fold heads (``stage3_per_entry.npz`` of
``stage3_datum_gate_cwa``) and the twin's rain cache, read only up to 2022-12. No head month
after 2022-12 is read anywhere in this module.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

sys.path.insert(0, str(C.REPO / "dataprep_staging"))
sys.path.insert(0, str(C.REPO))

from common.guard import filter_flags, open_product  # noqa: E402

FORCING_FLAGS = (0, 1, 2, 3, 4, 5)


@dataclass
class DesignData:
    """Wells, heads and drivers on the design window."""
    sids: np.ndarray                 # (W,) station ids, R0's entry order
    county: np.ndarray               # (W,) 'changhua' / 'yunlin'
    layer: np.ndarray                # (W,)
    head_dates: pd.DatetimeIndex     # 131 months 2012-02..2022-12
    drv_dates: pd.DatetimeIndex      # 132 months 2012-01..2022-12
    y: np.ndarray                    # (W, 131) observed heads, flags 0-1, m
    p: np.ndarray                    # (W, 131) R0 out-of-fold heads, m
    drivers: dict[str, np.ndarray]   # name -> (W, 132)
    notes: list[str] = field(default_factory=list)
    reads: list[dict] = field(default_factory=list)


def _months(a: str, b: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{a}-01", f"{b}-01", freq="MS")


def _ym_index(dates, want: pd.DatetimeIndex) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime([str(x)[:7] + "-01" for x in dates]))
    idx = d.get_indexer(want)
    if (idx < 0).any():
        raise ValueError(f"missing months {want[idx < 0][:3].tolist()}")
    return idx


def _fill_cell_climatology(v: np.ndarray, ok: np.ndarray, cal: np.ndarray) -> tuple:
    """Replace entries not ``ok`` by the cell's month-of-year mean of its ok entries."""
    out = np.where(ok, v, np.nan)
    n_bad = int((~ok).sum())
    for m in range(12):
        sel = cal == m
        blk = out[:, sel]
        cnt = np.isfinite(blk).sum(axis=1)
        mu = np.where(cnt > 0, np.nansum(blk, axis=1) / np.maximum(cnt, 1), 0.0)
        out[:, sel] = np.where(np.isfinite(blk), blk, mu[:, None])
    return out, n_bad


def load_design(verbose: bool = True) -> DesignData:
    notes, reads = [], []
    hd = _months(C.HEAD_START, C.DESIGN_END)
    dd = _months(C.DRIVER_START, C.DESIGN_END)
    cal = np.asarray(dd.month - 1)

    # ---- wells (R0's 158 scored entries) and their station attributes
    z = np.load(C.R0_RUN / "stage3_per_entry.npz", allow_pickle=True)
    kw = pd.read_csv(C.R0_RUN / "stage3_kfold_wells.csv", dtype={"sid": str}).set_index("entry")
    sids = kw.loc[z["entry"], "sid"].to_numpy()
    if z["pred"].shape[1] != len(hd):
        raise ValueError("R0 per-entry arrays do not span 2012-02..2022-12")
    hp = open_product(C.HEADS_PRE, "design_time_evidence")
    reads.append({"product": str(C.HEADS_PRE.relative_to(C.REPO)), "use": hp.use,
                  "notes": hp.notes})
    st = hp.read_table("stations.parquet", columns=["sid", "CountyName", "x_twd97", "y_twd97",
                                                     "cell_index"]).set_index("sid")
    inv = {v: k for k, v in C.COUNTIES.items()}
    county = np.array([inv.get(st.loc[s, "CountyName"], "") for s in sids])
    xy = st.loc[sids, ["x_twd97", "y_twd97"]].to_numpy(float)
    cell = st.loc[sids, "cell_index"].to_numpy(int)
    if (cell < 0).any() or (county == "").any():
        raise ValueError("an R0 well lies off the grid or outside the two counties")

    # ---- heads 2012-02..2022-12, flags 0-1 (U1: the same rows a baseline may use)
    h = filter_flags(hp.read_table("heads_monthly.parquet",
                                   columns=["sid", "date", "value", "flag"]), "baseline")
    h["date"] = pd.to_datetime(h["date"])
    if h["date"].max() > pd.Timestamp(f"{C.DESIGN_END}-01"):
        raise ValueError("pre-origin heads carry a month after 2022-12")
    piv = h.pivot_table(index="sid", columns="date", values="value", aggfunc="mean")
    y = piv.reindex(index=sids, columns=hd).to_numpy(float)
    p = np.asarray(z["pred"], dtype=float)
    # consistency check against the observations R0 was scored on (same months)
    o = np.asarray(z["obs"], dtype=float)
    both = np.isfinite(o) & np.isfinite(y)
    diff = np.abs(o - y)[both]
    notes.append(f"heads: release H1a L1 v3 vs R0's own observations on {both.sum()} shared "
                 f"well-months: median |diff| {np.median(diff):.3f} m, share within 0.05 m "
                 f"{(diff < 0.05).mean():.3f}; release-only months "
                 f"{int((np.isfinite(y) & ~np.isfinite(o)).sum())}, R0-only months "
                 f"{int((np.isfinite(o) & ~np.isfinite(y)).sum())}")

    # ---- county of each cell
    cm = open_product(C.COUNTY_MAP, "model_design")
    reads.append({"product": str(C.COUNTY_MAP.relative_to(C.REPO)), "use": cm.use,
                  "notes": cm.notes})
    with np.load(cm.path("zone_county_map.npz")) as zz:
        ch_cells = np.asarray(zz["changhua"], dtype=bool)
        cxy = np.asarray(zz["centroids"], dtype=float)
    county_cells = {"changhua": ch_cells, "yunlin": ~ch_cells}

    # ---- electricity (C0 pre-origin, value_all, forcing flags 0-5)
    ep = open_product(C.ELEC_PRE, "design_time_evidence")
    reads.append({"product": str(C.ELEC_PRE.relative_to(C.REPO)), "use": ep.use,
                  "notes": ep.notes})
    with np.load(ep.path("cells_total.npz")) as zz:
        ix = _ym_index(zz["dates"], dd)
        ev = np.asarray(zz["value_all"], dtype=float)[:, ix]
        ef = np.asarray(zz["flag_all"])[:, ix]
    E, n_bad = _fill_cell_climatology(ev, np.isin(ef, FORCING_FLAGS) & np.isfinite(ev), cal)
    notes.append(f"electricity: {n_bad} of {ef.size} cell-months had flags outside 0-5 "
                 f"({100 * n_bad / ef.size:.2f} %) and were replaced by the cell's "
                 f"month-of-year mean")

    # ---- canal (W1 pre-origin): grid cell and district monthly
    sp = open_product(C.SW_PRE, "design_time_evidence")
    reads.append({"product": str(C.SW_PRE.relative_to(C.REPO)), "use": sp.use,
                  "notes": sp.notes})
    with np.load(sp.path("grid_sw_v4.npz")) as zz:
        sw = np.asarray(zz["value"], dtype=float)[:, _ym_index(zz["dates"], dd)]
    dm = sp.read_table("district_monthly.csv", columns=["date", "district", "value"])
    dm["date"] = pd.to_datetime(dm["date"])
    district = {d: g.set_index("date")["value"].reindex(dd).to_numpy(float)
                for d, g in dm.groupby("district")}

    # ---- public wells (P6 sensitivity member, primary), summed over layers
    pp = open_product(C.PW_PRE, "design_time_evidence")
    reads.append({"product": str(C.PW_PRE.relative_to(C.REPO)), "use": pp.use,
                  "notes": pp.notes})
    with np.load(pp.path("sensitivity_member.npz")) as zz:
        pw = np.asarray(zz["value"], dtype=float).sum(axis=0)[:, _ym_index(zz["dates"], dd)]

    # ---- river (W1 L1 daily gauges, pre-origin v2), monthly means
    wp = open_product(C.W1_PRE, "design_time_evidence")
    reads.append({"product": str(C.W1_PRE.relative_to(C.REPO)), "use": wp.use,
                  "notes": wp.notes})
    gd = wp.read_table("gauges_daily_L1.parquet", columns=["date", "station_no", "value"])
    gd["date"] = pd.to_datetime(gd["date"])
    river = {}
    for name, code in C.GAUGES.items():
        s = gd[gd["station_no"] == code].set_index("date")["value"].astype(float)
        river[name] = s.resample("MS").mean().reindex(dd).to_numpy()

    # ---- rain recharge at the well: IDW rain (26 gauges) - IDW ET0 (61 points), >= 0, m/day
    from hydrophysics.subsidence import idw_interp
    tp = open_product(C.ET0_PRE, "statistical_layer_history")
    reads.append({"product": str(C.ET0_PRE.relative_to(C.REPO)), "use": tp.use,
                  "notes": tp.notes})
    with np.load(tp.path("et0_daily_2012_2022_preorigin.npz")) as zz:
        et = pd.DataFrame(np.asarray(zz["et0"], dtype=float).T,
                          index=pd.to_datetime(zz["dates"]))
        et_ids = [str(x) for x in zz["well_ids"]]
    # the twin's convention: each ET0 series sits at its curated well (gw_stations.csv st_id)
    gws = pd.read_csv(C.GW_STATIONS).set_index("st_id")
    et_xy = gws.loc[et_ids, ["TM_X97", "TM_Y97"]].to_numpy(float)
    et_m = et.resample("MS").mean().reindex(dd).to_numpy().T             # (61, T) mm/day
    rf = pd.read_csv(C.RAIN, parse_dates=["date time"]).set_index("date time")
    rf = rf[rf.index <= pd.Timestamp(f"{C.DESIGN_END}-01") + pd.offsets.MonthEnd(0)]
    rst = pd.read_csv(C.RAIN_STATIONS).set_index("rf_id")
    rf_m = rf.resample("MS").mean().reindex(dd)
    rf_xy = rst.loc[rf_m.columns, ["TM_X97", "TM_Y97"]].to_numpy(float)
    rain_w = idw_interp(xy, rf_xy, rf_m.to_numpy().T)
    et_w = idw_interp(xy, et_xy, et_m)
    recharge = np.clip(rain_w - et_w, 0.0, None) / 1000.0
    notes.append("rain: the twin's cache chou-shui-data/data/rf_timeseries.csv (CODIS rebuild, "
                 "no manifest) read up to 2022-12-31 only")

    drivers = {k: np.zeros((len(sids), len(dd))) for k in C.DRIVERS}
    for i in range(len(sids)):
        dcell = np.hypot(cxy[:, 0] - xy[i, 0], cxy[:, 1] - xy[i, 1])
        cc = county_cells[county[i]]
        drivers["electricity_local_3km"][i] = E[dcell <= C.LOCAL_RADIUS_M].sum(axis=0)
        drivers["electricity_county"][i] = E[cc].sum(axis=0)
        drivers["rain_recharge"][i] = recharge[i]
        drivers["canal_w1_cell"][i] = sw[cell[i]]
        drivers["canal_district"][i] = district[county[i]]
        for name in C.GAUGES:
            drivers[name][i] = river[name]
        drivers["public_wells_county"][i] = pw[cc].sum(axis=0)
    for k, v in drivers.items():
        nn = int((~np.isfinite(v)).sum())
        if nn:
            notes.append(f"driver {k}: {nn} missing well-months, filled by the driver's "
                         f"month-of-year mean (2012-2022)")
            for m in range(12):
                sel = cal == m
                mu = np.nanmean(v[:, sel], axis=1)
                v[:, sel] = np.where(np.isfinite(v[:, sel]), v[:, sel], mu[:, None])
    if verbose:
        for n in notes:
            print(n, flush=True)
    return DesignData(sids=sids, county=county, layer=np.asarray(z["layer"], int),
                      head_dates=hd, drv_dates=dd, y=y, p=p, drivers=drivers, notes=notes,
                      reads=reads)
