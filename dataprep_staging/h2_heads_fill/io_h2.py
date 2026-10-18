"""Loading the H2 inputs through the loader guard (U3, U5) and building the monthly matrix.

Pre-origin variant: ``heads/L1_clean/v3`` opened for ``initial_conditions`` with
``window_end = 2022-12`` (time slice on ``date``) for the head tables, and through the
per-output route for ``station_meta_preorigin.parquet`` (its own fit_window_end 2022-12).
Full variant: the same product opened for ``statistical_layer_history`` (no window).
The model wells come from ``validation_units/v3`` (opened for ``selection``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from h2_heads_fill import config as C

sys.path.insert(0, str(C.REPO / "dataprep_staging"))
from common import guard  # noqa: E402

MONTHLY_COLS = ["sid", "date", "month", "value", "flag", "c20", "n_clean_days",
                "sigma_obs", "n_days_flag9", "n_days_flag3", "days_in_month"]
META_COLS = ["sid", "x_twd97", "y_twd97", "layer_code", "CountyName", "WellElevation",
             "WellDepth"]


@dataclass
class HeadData:
    variant: str                       # 'pre' or 'full'
    sids: list[str]
    dates: pd.DatetimeIndex            # month starts
    Y: np.ndarray                      # (W, T) observed C20 monthly means (L1 flag 0/1)
    l1_flag: np.ndarray                # (W, T) L1 flag of the row read (255 = no row read)
    kclean: np.ndarray                 # (W, T) clean days of the month (0 = no row)
    ndays: np.ndarray                  # (T,) days in month
    flag9days: np.ndarray              # (W, T) flag-9 days inside an observed/partial month
    partial: np.ndarray                # (W, T) L1 flag 0/1 month with < 20 clean days
    partial_mean: np.ndarray           # (W, T) its mean of clean days (NaN otherwise)
    meta: pd.DataFrame                 # per well: sid, x, y, layer, county, in_model, dem, cell
    daily: pd.DataFrame                # clean days: sid, date, value
    notes: list[str] = field(default_factory=list)

    @property
    def months(self) -> list[str]:
        return [d.strftime("%Y-%m") for d in self.dates]

    def xy(self) -> np.ndarray:
        return self.meta[["x", "y"]].to_numpy(float)


def _grid():
    from hydrophysics.twin.grid import FanGrid
    z = np.load(C.GRID_NPZ, allow_pickle=False)
    return FanGrid(int(z["nx"]), int(z["ny"]), float(z["dx"]), float(z["x0"]),
                   float(z["y0"]), z["mask"])


def grid_and_dem():
    """(FanGrid, SRTM elevation on the active cells) of the model grid."""
    g = _grid()
    with np.load(C.BASEMAP_NPZ, allow_pickle=False) as z:
        dem = np.asarray(z["dem"], dtype=float)
    if dem.shape[0] != g.n_active:
        raise ValueError(f"basemap dem has {dem.shape[0]} cells, grid {g.n_active}")
    return g, dem


def model_wells() -> pd.DataFrame:
    p = guard.open_product(C.UNITS_DIR, "selection")
    w = p.read_table("wells.csv")
    w["sid"] = w["sid"].astype(str).str.zfill(8)
    return w


def open_l1(variant: str):
    """(product for head tables, product for station metadata, metadata file name)."""
    if variant == "pre":
        heads = guard.open_product(C.L1_DIR, "initial_conditions", window_end=C.PRE_END)
        meta = guard.open_product(C.L1_DIR, "initial_conditions")   # per-output route
        return heads, meta, "station_meta_preorigin.parquet"
    if variant == "full":
        p = guard.open_product(C.L1_DIR, "statistical_layer_history")
        return p, p, "station_meta_clean.parquet"
    raise ValueError(variant)


def load(variant: str, with_daily: bool = True) -> HeadData:
    hp, mp, meta_name = open_l1(variant)
    end = C.PRE_END if variant == "pre" else C.FULL_END
    dates = pd.date_range(f"{C.START}-01", f"{end}-01", freq="MS")
    mon = hp.read_table("heads_monthly.parquet", columns=MONTHLY_COLS)
    mon = mon[(mon.date >= dates[0]) & (mon.date <= dates[-1])]
    mon["sid"] = mon["sid"].astype(str)
    have = mon[mon.flag.isin(C.OBS_FLAGS)]
    sids = sorted(have.sid.unique())
    W, T = len(sids), len(dates)
    si = {s: i for i, s in enumerate(sids)}
    ti = {d: j for j, d in enumerate(dates)}
    Y = np.full((W, T), np.nan)
    l1f = np.full((W, T), 255, dtype=np.uint8)
    kc = np.zeros((W, T), dtype=int)
    f9 = np.zeros((W, T), dtype=int)
    part = np.zeros((W, T), dtype=bool)
    pmean = np.full((W, T), np.nan)
    m = mon[mon.sid.isin(si)]
    ii = m.sid.map(si).to_numpy()
    jj = m.date.map(ti).to_numpy()
    l1f[ii, jj] = m.flag.to_numpy().astype(np.uint8)
    obs01 = m.flag.isin(C.OBS_FLAGS).to_numpy()
    kc[ii[obs01], jj[obs01]] = m.n_clean_days.to_numpy()[obs01]
    f9[ii, jj] = m.n_days_flag9.to_numpy()
    c20 = obs01 & m.c20.to_numpy().astype(bool)
    Y[ii[c20], jj[c20]] = m.value.to_numpy()[c20]
    pp = obs01 & ~m.c20.to_numpy().astype(bool)
    part[ii[pp], jj[pp]] = True
    pmean[ii[pp], jj[pp]] = m.value.to_numpy()[pp]
    ndays = np.array([d.days_in_month for d in dates])

    md = mp.read_table(meta_name, columns=META_COLS)
    md["sid"] = md["sid"].astype(str)
    md = md.set_index("sid").reindex(sids)
    mw = model_wells()
    g, dem = grid_and_dem()
    cells, dems = [], []
    for x, y in zip(md.x_twd97, md.y_twd97, strict=True):
        c = g.active_index(x, y) if np.isfinite(x) and np.isfinite(y) else None
        cells.append(-1 if c is None else int(c))
        dems.append(np.nan if c is None else float(dem[c]))
    county = md.CountyName.map({"彰化縣": "CH", "雲林縣": "YL"}).fillna("other")
    meta = pd.DataFrame({
        "sid": sids, "x": md.x_twd97.to_numpy(float), "y": md.y_twd97.to_numpy(float),
        "layer": md.layer_code.astype(str).to_numpy(), "county": county.to_numpy(),
        "in_model": np.isin(sids, mw.sid.to_numpy()), "cell": cells, "dem": dems,
        "well_elev": md.WellElevation.to_numpy(float)})
    daily = pd.DataFrame(columns=["sid", "date", "value"])
    if with_daily:
        dd = hp.read_table("heads_daily.parquet",
                           columns=["sid", "date", "value", "flag", "day_valid"])
        dd["sid"] = dd["sid"].astype(str)
        dd = dd[dd.flag.isin(C.OBS_FLAGS) & dd.day_valid.astype(bool)
                & (dd.date >= dates[0]) & (dd.date < dates[-1] + pd.offsets.MonthBegin(1))
                & dd.sid.isin(si)]
        daily = dd[["sid", "date", "value"]].sort_values(["sid", "date"]).reset_index(drop=True)
    notes = list(hp.notes) + list(mp.notes)
    return HeadData(variant, sids, dates, Y, l1f, kc, ndays, f9, part, pmean, meta, daily,
                    notes)


def clean_truth(h: HeadData) -> np.ndarray:
    """Observed months whose truth is 'clean': no flag-9 or flag-3 day in the month and none
    in an adjacent observed month, and the well is not one of the three pumping-cycle wells
    (H1a: their flagged episodes may be pump regimes). The plan's pilot used the audit's raw
    offset episodes; L1 v3 already removes flagged days, so this proxy marks the months next
    to them (flag-9 objects are invisible to a pre-origin read except through the counts)."""
    obs = np.isfinite(h.Y)
    bad = h.flag9days > 0
    near = bad.copy()
    near[:, 1:] |= bad[:, :-1]
    near[:, :-1] |= bad[:, 1:]
    pump = np.isin(np.array(h.sids), C.PUMP_CYCLE_WELLS)[:, None]
    return obs & ~near & ~pump


def month_index(dates: pd.DatetimeIndex, ym: str) -> int:
    return int(dates.get_loc(pd.Timestamp(f"{ym}-01")))
