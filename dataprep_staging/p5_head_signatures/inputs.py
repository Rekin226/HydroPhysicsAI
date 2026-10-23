"""Guarded access to every P5 input (plan 2.5 U5: the loader guard checks allowed uses).

Declared uses (one per product, plan 4.1):

* ``heads/L1_clean/v3`` (H1a): ``validator``. P5's indices are validators and benchmarks;
  every test statistic uses rows dated <= 2022-12 only (U7), selected explicitly.
* ``pump_electricity/{census_L1,preorigin}/v3`` (C0): ``validator``. The census is the truth
  against which head-derived claims are checked (pre-origin variant, 2012-01..2022-12).
* ``et0/v1`` (W2) and ``surface_water_v4/preorigin/v2`` (W1): ``after_the_fact_analysis``
  (their manifests do not allow ``validator``): forcing of P5's per-well transfer-function
  benchmarks, whose gains are never fed back into a scored model.
* ``validation_units/v3`` (A1): ``validator`` (well list of the 158 model wells).
* Rain: L0 ``chou-shui-data/data/rf_timeseries.csv`` (read-only, hashed in the manifest).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from p5_head_signatures import config as C

STAGING = C.REPO / "dataprep_staging"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from common import guard  # noqa: E402

USE_L1 = "validator"
USE_C0 = "validator"
USE_FORCING = "after_the_fact_analysis"


def l1() -> guard.Product:
    return guard.open_product(C.L1_DIR, USE_L1)


def station_meta(p: guard.Product | None = None) -> pd.DataFrame:
    """Non-personal station columns of L1 v3 (repaired coordinates, labels, layer codes)."""
    p = p or l1()
    cols = ["sid", "CountyName", "TownName", "x_twd97", "y_twd97", "xy_flag", "layer_code",
            "layer_flag", "WellDepth", "FinishDepth", "WellElevation", "era5_cell",
            "in_hist_cache", "in_new_network"]
    m = pd.read_parquet(p.path("station_meta_clean.parquet"), columns=cols)
    m["sid"] = m["sid"].astype(str)
    m["county"] = m["CountyName"].map(C.COUNTIES).fillna("other")
    m["layer"] = m["layer_code"].astype(str).replace({"nan": "", "None": ""})
    return m


def be_table(p: guard.Product | None = None) -> pd.DataFrame:
    p = p or l1()
    b = pd.read_csv(p.path("barometric_efficiency.csv"), dtype={"sid": str})
    return b


def pressure_wide(p: guard.Product | None = None) -> pd.DataFrame:
    """Hourly ERA5 surface pressure (hPa) per cell, local time, full hourly index; the same
    frame ``h1_heads_qc.baro.load_pressure`` builds from L0 (checked in s1_extract)."""
    p = p or l1()
    x = pd.read_parquet(p.path("pressure_hourly.parquet"))
    P = x.pivot(index="date", columns="era5_cell", values="surface_pressure_hpa").sort_index()
    P = P.where((P > 850) & (P < 1100))
    full = pd.date_range(P.index.min(), P.index.max(), freq="h")
    P = P.reindex(full)
    P.columns.name = None
    return P


def daily_flags(p: guard.Product | None = None, columns=None) -> pd.DataFrame:
    """L1 v3 daily table (validator use: the guard drops flags 8 and 9)."""
    p = p or l1()
    cols = columns or ["sid", "date", "flag", "value", "baro_m", "corr_m", "n_hours"]
    d = p.read_table("heads_daily.parquet", columns=cols)
    d["sid"] = d["sid"].astype(str)
    return d


def monthly_heads(p: guard.Product | None = None) -> pd.DataFrame:
    p = p or l1()
    d = p.read_table("heads_monthly.parquet",
                     columns=["sid", "month", "value", "flag", "c20", "n_clean_days",
                              "sigma_obs"])
    d["sid"] = d["sid"].astype(str)
    return d


def c0_census() -> guard.Product:
    return guard.open_product(C.C0_CENSUS_DIR, USE_C0)


def c0_pre() -> guard.Product:
    return guard.open_product(C.C0_PRE_DIR, USE_C0)


def et0() -> guard.Product:
    return guard.open_product(C.ET0_DIR, USE_FORCING)


def surface_water() -> guard.Product:
    return guard.open_product(C.SW_PRE_DIR, USE_FORCING)


def units() -> guard.Product:
    return guard.open_product(C.UNITS_DIR, "validator")


def census_pumps(cols=None) -> pd.DataFrame:
    """Pump table (C0 census_L1 v3): only non-personal columns plus coordinates (kept under
    results/). The census names/addresses were never part of this product."""
    p = c0_census()
    cols = cols or ["county", "uid", "x_twd97", "y_twd97", "cell", "in_grid", "pump_hp",
                    "purpose_class", "billing_code", "kept", "hp_share"]
    return pd.read_parquet(p.path("census_pumps.parquet"), columns=cols)


def pump_kwh(county: str) -> dict:
    """Per-pump monthly kWh (C0 preorigin v3, 2007-01..2022-12): value, flag, uid, dates."""
    p = c0_pre()
    name = {"yunlin": "pumps_yunlin.npz", "changhua": "pumps_changhua.npz",
            "chiayi": "pumps_chiayi_ingrid.npz"}[county]
    z = np.load(p.path(name), allow_pickle=True)
    return {k: z[k] for k in ("value", "flag", "uid", "dates", "purpose_class", "hp_share")}
