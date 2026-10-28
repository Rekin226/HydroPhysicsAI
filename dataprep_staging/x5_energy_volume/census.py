"""Census electricity on three bases, per pump and per county-class-month (X5 shared library).

Bases (every number is kWh of the TPC census L0 copy; NaN = pre-install = 0):

* ``raw``   - the sum of every pump's series (shared meters counted once per pump);
* ``dedup`` - each meter's series counted once (its first pump's copy, as
  ``pumping.clean_census`` does) and shared over the meter's pumps by rated horsepower
  (even split when the HP sum is 0), i.e. ``clean_census(cap_duty=None)``;
* ``kept``  - ``dedup`` restricted to the twin's kept meters (census_L1 v3 ``kept``:
  ``clean_census`` cap 1.0, duty on 2012-01..2022-12).

Classes are the pump's own ``purpose_class`` (``scenario.purpose_class``), so kept is a
subset of dedup pump by pump. Pump attributes come from census_L1 v3 (no personal column;
explicit column list), kWh from the L0 long tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_energy_volume import x5const as K  # noqa: E402

DATES = pd.date_range("2007-01-01", "2026-07-01", freq="MS")
T_ALL = len(DATES)
L1_COLS = ["county", "row_idx", "sid", "town", "work_station", "irrigation_district",
           "x_twd97", "y_twd97", "cell", "in_grid", "pump_hp", "purpose_class",
           "billing_code", "tube_dia_raw", "meter_idx", "meter_n_pumps", "meter_first_pump",
           "meter_hp_sum", "meter_duty_2012_2022", "kept"]


def tidx(month: str) -> int:
    t = pd.Timestamp(month)
    return (t.year - 2007) * 12 + t.month - 1


def load_pumps(county: str) -> pd.DataFrame:
    assert not set(L1_COLS) & set(K.FORBIDDEN_COLS)
    p = pd.read_parquet(K.CENSUS_L1, columns=L1_COLS)
    p = p[p["county"] == county].sort_values("row_idx").reset_index(drop=True)
    if not (p["row_idx"].to_numpy() == np.arange(len(p))).all():
        raise SystemExit(f"{county}: census_L1 rows are not L0 rows 0..n-1")
    hp = pd.to_numeric(p["pump_hp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tot = hp.groupby(p["meter_idx"]).transform("sum")
    n = p.groupby("meter_idx")["sid"].transform("size")
    p["hp_share_all"] = np.where(tot > 0, hp / tot.where(tot > 0, 1.0), 1.0 / n)
    p["tube_in"] = pd.to_numeric(p["tube_dia_raw"], errors="coerce")
    return p


def load_volt(county: str) -> np.ndarray:
    """The L0 'volt' column (supply voltage class; not personal), census row order."""
    v = pd.read_parquet(K.L0_PARQ / f"{county}_census.parquet", columns=["row_idx", "volt"])
    v = v.sort_values("row_idx")
    return v["volt"].astype("string").to_numpy()


def load_kwh(county: str, n: int) -> np.ndarray:
    """Dense (n, T_ALL) L0 kWh, NaN where L0 is NaN."""
    k = pd.read_parquet(K.L0_PARQ / f"{county}_kwh_long.parquet",
                        columns=["row_idx", "datetime", "electricity_kwh"])
    dt = pd.to_datetime(k["datetime"])
    ti = ((dt.dt.year - 2007) * 12 + dt.dt.month - 1).to_numpy()
    ri = k["row_idx"].to_numpy()
    L = np.full((n, T_ALL), np.nan)
    L[ri, ti] = k["electricity_kwh"].to_numpy(dtype="float64")
    return L


def bases(p: pd.DataFrame, L: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pump (n, T_ALL) kWh on the raw, dedup and kept bases (NaN -> 0)."""
    X = np.nan_to_num(L, nan=0.0)
    first = p.index[p["meter_first_pump"].to_numpy()].to_numpy()
    mser = np.zeros((int(p["meter_idx"].max()) + 1, T_ALL))
    mser[p.loc[first, "meter_idx"].to_numpy()] = X[first]
    dedup = mser[p["meter_idx"].to_numpy()] * p["hp_share_all"].to_numpy()[:, None]
    kept = dedup * p["kept"].to_numpy(dtype=bool)[:, None]
    return {"raw": X, "dedup": dedup, "kept": kept}


def class_month(p: pd.DataFrame, E: np.ndarray) -> pd.DataFrame:
    """(class x month) totals of a per-pump matrix."""
    g = pd.DataFrame(E, columns=DATES).groupby(p["purpose_class"].to_numpy()).sum()
    return g.reindex(K.CLASSES).fillna(0.0)


def window_mean(cm: pd.DataFrame, y0: int, y1: int, months=None) -> pd.Series:
    """Mean per year over y0..y1 of the (months-restricted) class totals."""
    cols = [d for d in cm.columns if y0 <= d.year <= y1 and (months is None or d.month in months)]
    return cm[cols].sum(axis=1) / (y1 - y0 + 1)


def per_pump_window(E: np.ndarray, y0: int, y1: int, months=None) -> np.ndarray:
    sel = np.array([y0 <= d.year <= y1 and (months is None or d.month in months)
                    for d in DATES])
    return E[:, sel].sum(axis=1) / (y1 - y0 + 1)
