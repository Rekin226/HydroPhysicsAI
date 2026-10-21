"""Monthly panel of WP X1: TPC county sales, the census series the hybrid uses, and weather.

Every product with a manifest is opened through ``common.guard.open_product`` with a declared
use (U5):

==============================================  ===========================  ================
product                                         use                          role
==============================================  ===========================  ================
pump_electricity/full/v3 (county_class_monthly) after_the_fact_analysis      task 2 validation
release full/electricity_census (cells_total)   statistical_layer_history    task 3 target
release preorigin/heads_observed (stations)     design_time_evidence         well x, y only
zone_county_map/v2                              model_design                 county of a cell
release full/et0                                statistical_layer_history    weather feature
==============================================  ===========================  ================

``stations.parquet`` is the only table read from the heads product: well coordinates and
county. No head value is read anywhere in this package. Inputs without a manifest: the TPC
and MOEAEA L0 downloads (untrusted; parsed as text tables) and the twin's rain cache
(``chou-shui-data/data/rf_timeseries.csv``, CODIS rebuild; read as the hybrid layer reads it).

The census series the hybrid uses (``i1/hybrid_tf/prospective.driver_panel``): cells_total
``value_all`` with flags 0-5 (others set to 0), summed over the in-grid cells of each county
(``changhua`` mask, the rest is Yunlin) and over the cells within 3 km of each well. A month
counts as recorded when >= 99 % of active cells carry flags 0-5 (the last is 2026-06).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.guard import open_product  # noqa: E402
from common.paths import REPO  # noqa: E402

L0 = REPO / "data_fetch_api/new_measurements"
REL = REPO / "results/dataprep/model_ready/release_20261012b2"
CENSUS_C0 = REPO / "results/dataprep/pump_electricity/full/v3"
ELEC_FULL = REL / "full/electricity_census/v1"
HEADS_PRE = REL / "preorigin/heads_observed/v1"
ET0_FULL = REL / "full/et0/v1"
COUNTY_MAP = REPO / "results/dataprep/zone_county_map/v2"
RAIN = REPO / "chou-shui-data/data/rf_timeseries.csv"
WELLS_CSV = REPO / "results/dataprep/reports/I1_hybrid_tf/draft_declaration/wells.csv"

#: TPC d007012 vintages in hand (newest last). Later fetches are appended by the caller.
TPC_FILES = [L0 / "tpc_d007012/2026-10-12/d007012.csv"]
NAT_FILES = (L0 / "moeaea_agri_electricity/2026-10-12/set_256.csv",
             L0 / "moeaea_agri_electricity/2026-10-12/set_236.csv")

CH, YL = "彰化縣", "雲林縣"
COUNTY_KEYS = {"changhua": CH, "yunlin": YL}
SECTORS = {"農林漁牧": "ag", "住宅": "res", "服務業": "srv", "工業": "ind"}
FORCING_FLAGS = (0, 1, 2, 3, 4, 5)
GRID_MIN_SHARE = 0.99
LOCAL_RADIUS_M = 3000.0
AGRI_CLASSES = ("irrigation", "aquaculture", "livestock")


def read_tpc(path: Path) -> pd.DataFrame:
    """d007012 long table: date, county, sector, kwh (all counties)."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").iloc[:, :5]
    df.columns = ["ym", "county", "sector", "kwh", "pct"]
    df = df.dropna(subset=["ym"])
    s = df["ym"].str.strip()
    df["date"] = pd.to_datetime(dict(year=s.str[:-2].astype(int) + 1911,
                                     month=s.str[-2:].astype(int), day=1))
    df["county"] = df["county"].str.strip().str.replace("臺", "台")
    df["sector"] = df["sector"].str.strip()
    df["kwh"] = pd.to_numeric(df["kwh"].str.replace(",", "").str.strip(), errors="coerce")
    return df[["date", "county", "sector", "kwh"]]


def tpc_wide(files=None) -> pd.DataFrame:
    """GWh/month columns ``<sector>_<county>`` for Changhua and Yunlin (index date). Later
    vintages override earlier ones month by month (no revision was ever seen)."""
    frames = [read_tpc(Path(f)) for f in (files or TPC_FILES)]
    df = pd.concat(frames).drop_duplicates(["date", "county", "sector"], keep="last")
    out = {}
    for key, cty in COUNTY_KEYS.items():
        g = df[df.county == cty]
        for zh, short in SECTORS.items():
            s = g[g.sector == zh].set_index("date")["kwh"].sort_index() / 1e6
            out[f"{short}_{key}"] = s
    w = pd.DataFrame(out)
    w.index = pd.DatetimeIndex(w.index, name="date")
    return w.sort_index()


def national_ag(files=NAT_FILES) -> pd.Series:
    parts = []
    for p in files:
        d = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
        parts.append(pd.Series(pd.to_numeric(d["電力消費_農業部門(數值)"], errors="coerce").values,
                               index=pd.to_datetime(d.iloc[:, 0].str.strip(), format="%Y%m")))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index, name="date")
    return s.rename("nat_ag")


@dataclass
class Census:
    dates: pd.DatetimeIndex
    county: pd.DataFrame            # in-grid totals GWh, columns changhua, yunlin
    recorded: pd.Series             # bool per month
    discs: pd.DataFrame             # (T, W) GWh, columns = well sid
    well_county: pd.Series          # sid -> changhua / yunlin
    notes: list[str] = field(default_factory=list)
    reads: list[dict] = field(default_factory=list)


def wells() -> pd.DataFrame:
    w = pd.read_csv(WELLS_CSV, dtype={"sid": str})
    return w


def census_hybrid() -> Census:
    """The electricity series of the hybrid layer, from the release full component."""
    reads, notes = [], []
    ep = open_product(ELEC_FULL, "statistical_layer_history")
    reads.append({"product": str(ELEC_FULL.relative_to(REPO)), "use": ep.use})
    with np.load(ep.path("cells_total.npz")) as z:
        dates = pd.DatetimeIndex(pd.to_datetime([f"{d}-01" for d in z["dates"]]), name="date")
        ev = np.asarray(z["value_all"], float)
        ef = np.asarray(z["flag_all"])
    active = np.nanmax(np.where(np.isfinite(ev), ev, 0), axis=1) > 0
    ok = np.isin(ef, FORCING_FLAGS) & np.isfinite(ev)
    rec = ok[active].mean(axis=0) >= GRID_MIN_SHARE
    E = np.where(ok, ev, 0.0) / 1e6                       # GWh
    cm = open_product(COUNTY_MAP, "model_design")
    reads.append({"product": str(COUNTY_MAP.relative_to(REPO)), "use": cm.use})
    with np.load(cm.path("zone_county_map.npz")) as zz:
        ch = np.asarray(zz["changhua"], bool)
        cxy = np.asarray(zz["centroids"], float)
    county = pd.DataFrame({"changhua": E[ch].sum(0), "yunlin": E[~ch].sum(0)}, index=dates)
    hp = open_product(HEADS_PRE, "design_time_evidence")
    reads.append({"product": str(HEADS_PRE.relative_to(REPO)), "use": hp.use,
                  "table": "stations.parquet only (coordinates, county)"})
    st = hp.read_table("stations.parquet", columns=["sid", "CountyName", "x_twd97",
                                                     "y_twd97"]).set_index("sid")
    w = wells()
    inv = {v: k for k, v in COUNTY_KEYS.items()}
    wc = pd.Series([inv[st.loc[s, "CountyName"]] for s in w.sid], index=w.sid)
    xy = st.loc[w.sid, ["x_twd97", "y_twd97"]].to_numpy(float)
    discs = {}
    for s, (x, y) in zip(w.sid, xy, strict=True):
        d = np.hypot(cxy[:, 0] - x, cxy[:, 1] - y)
        discs[s] = E[d <= LOCAL_RADIUS_M].sum(0)
    discs = pd.DataFrame(discs, index=dates)
    notes.append(f"census: {int(rec.sum())} recorded months, last "
                 f"{dates[rec][-1].strftime('%Y-%m')}; {len(w)} wells")
    return Census(dates, county, pd.Series(rec, index=dates), discs, wc, notes, reads)


def census_c0_county() -> tuple[pd.DataFrame, list[dict]]:
    """Census county-scope (whole county) monthly GWh by class group, flags 0 only."""
    p = open_product(CENSUS_C0, "after_the_fact_analysis")
    t = p.read_table("county_class_monthly.parquet",
                     columns=["county", "scope", "class", "date", "value", "flag"])
    t = t[(t.scope == "county") & t.county.isin(["changhua", "yunlin"])].copy()
    t["date"] = pd.to_datetime(t["date"].astype(str).str[:7] + "-01")
    t["value"] = np.where(t["flag"] == 0, t["value"], np.nan) / 1e6
    piv = t.pivot_table(index="date", columns=["county", "class"], values="value",
                        aggfunc="sum", dropna=False)
    out = {}
    for c in ("changhua", "yunlin"):
        out[f"all_{c}"] = piv[(c, "all")]
        out[f"agri_{c}"] = sum(piv[(c, k)] for k in AGRI_CLASSES)
        out[f"irr_{c}"] = piv[(c, "irrigation")]
    df = pd.DataFrame(out)
    df.index = pd.DatetimeIndex(df.index, name="date")
    return df, [{"product": str(CENSUS_C0.relative_to(REPO)), "use": p.use,
                 "table": "county_class_monthly.parquet (county scope)"}]


def weather() -> tuple[pd.DataFrame, list[dict]]:
    """Fan-mean monthly rain (mm/day, mean over the gauges reporting) and ET0 (mm/day)."""
    tp = open_product(ET0_FULL, "statistical_layer_history")
    npz = next(o for o in tp.outputs if o.endswith(".npz"))
    with np.load(tp.path(npz)) as z:
        et = pd.DataFrame(np.asarray(z["et0"], float).T, index=pd.to_datetime(z["dates"]))
    et_m = et.mean(axis=1).resample("MS").mean()
    et_n = et.notna().all(axis=1).resample("MS").sum()
    full = et_n.to_numpy() >= np.asarray(et_n.index.days_in_month)
    et_m[~full] = np.nan
    rf = pd.read_csv(RAIN, parse_dates=["date time"]).set_index("date time")
    rf_m = rf.mean(axis=1).resample("MS").mean()
    cnt = rf.groupby(rf.index.to_period("M")).size()
    ok = cnt.reindex(rf_m.index.to_period("M")).to_numpy() >= np.asarray(
        rf_m.index.days_in_month)
    rf_m[~ok] = np.nan
    w = pd.DataFrame({"et0": et_m, "rain": rf_m})
    w.index = pd.DatetimeIndex(w.index, name="date")
    return w, [{"product": str(ET0_FULL.relative_to(REPO)), "use": tp.use},
               {"input": str(RAIN.relative_to(REPO)), "note": "no manifest; complete months "
                "only"}]
