"""Raw head-record access for WP H1a (read-only on L0).

L0 sources:
  historical cache  AMP_V2/data/wells/<sid>.parquet                   (2012-01 .. 2022-12)
  new network       data_fetch_api/new_measurements/heads_202210_202610/<sid>.parquet
                                                                       (2022-10 .. 2026-09)
Each file holds one column ``value`` (m) indexed by ``datetime`` (local time, Asia/Taipei),
hourly in the early years and every 10 minutes later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from common.paths import REPO

HIST_DIR = REPO / "AMP_V2" / "data" / "wells"
NEW_DIR = REPO / "data_fetch_api" / "new_measurements" / "heads_202210_202610"
STATIONS = REPO / "AMP_V2" / "data" / "fan_stations.parquet"

#: The merge rule: historical cache up to and including this day, new network after it.
#: Their 2022-10..12 overlap is identical (p0_profile.py checks it).
SPLIT = pd.Timestamp("2023-01-01")
T0 = pd.Timestamp("2012-01-01")
T1 = pd.Timestamp("2026-10-01")      # exclusive

#: Non-personal station columns that may be read (plan 2.6: contact fields never copied;
#: fan_stations.parquet has none, LocationAddress is left out as unnecessary).
STATION_COLUMNS = [
    "sid", "AreaCode", "BaseElevation", "CountyCode", "CountyName", "DisuseDate",
    "EstablishDate", "FinishDate", "FinishDepth", "GroundHeight", "GroundwaterLayerCode",
    "GroundwaterZone", "GroundwaterZoneIdentifier", "LayerAttribute", "LocationByTWD67",
    "LocationByTWD97", "ModificationDate", "ModificationReason", "ObservationtDate",
    "OrganizationIdentifier", "PipeMaterial", "Remarks", "TownName", "WaterLevel",
    "WaterResourcesDistrictName", "WellDepth", "WellElevation", "WellIdentifier", "WellName",
    "WellNumber", "WellRadius", "WellType", "start_datetime_from_dataset",
    "end_datetime_from_dataset",
]


def load_raw(path: str | Path, keep_sentinels: bool = False) -> pd.Series:
    """Sorted, de-duplicated raw series. Sentinels (|v| >= 1000, e.g. -999998) are dropped
    unless ``keep_sentinels``."""
    f = pd.read_parquet(path)
    if "datetime" in f:
        f = f.set_index("datetime")
    s = pd.to_numeric(f["value"], errors="coerce")
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()
    if not keep_sentinels:
        s = s[s.abs() < 1000]
    return s


def load_well(sid: str, keep_sentinels: bool = True, pre_only: bool = False
              ) -> tuple[pd.Series, pd.Series]:
    """(pre-origin part < SPLIT, post-origin part >= SPLIT) for one station; empty if absent.

    The pre-origin part comes from the historical cache; the new network's 2022-10..12
    records are dropped (identical to the cache, ``p0_profile.py``), except for a well whose
    historical record is absent or empty, whose new-network records before SPLIT form its
    pre-origin part (none in the 2026-10 data). ``pre_only``: the new-network file is not
    opened unless the historical record is absent or empty (leakage probe)."""
    h = HIST_DIR / f"{sid}.parquet"
    n = NEW_DIR / f"{sid}.parquet"
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    sh = load_raw(h, keep_sentinels) if h.exists() else empty
    need_new = (not pre_only) or sh.empty
    sn = load_raw(n, keep_sentinels) if (n.exists() and need_new) else empty.copy()
    if sh.empty:
        sh = empty
    if sn.empty:
        sn = empty.copy()
    sh = sh[(sh.index >= T0) & (sh.index < SPLIT)]
    if sh.empty:
        sh = sn[(sn.index >= T0) & (sn.index < SPLIT)]
    sn = sn[(sn.index >= SPLIT) & (sn.index < T1)]
    return sh, sn


def all_sids() -> list[str]:
    s = {p.stem for p in HIST_DIR.glob("*.parquet")} | {p.stem for p in NEW_DIR.glob("*.parquet")}
    return sorted(s)


def stations() -> pd.DataFrame:
    return pd.read_parquet(STATIONS, columns=STATION_COLUMNS)


def sentinel_mask(v: np.ndarray) -> np.ndarray:
    return ~np.isfinite(v) | (np.abs(v) >= 1000)
