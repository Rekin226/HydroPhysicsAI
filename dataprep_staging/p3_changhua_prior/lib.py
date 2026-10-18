"""Shared paths, grid and loaders for WP P3 (out-of-sample test of the land-use / crop
transfer priors against the Changhua census; plan WP P3 as repurposed on 2026-10-12).

Conventions:
- Model grid: results/twin_forward/datum_gate_innov.npz (59 x 77, 1 km, 2148 active cells,
  row-major over the mask, row 0 south), the same grid as the C0 census arrays (checked by
  mask hash in features.py).
- Truth and training targets come from WP C0 ``pump_electricity/preorigin/v3`` (cleaned census
  kWh in model units, 2007-01..2022-12), opened through the loader guard.
- The pilots (results/dataprep/_pilots/landuse_crop_priors, design_ml_transfer) are design-time
  evidence; their code is ported here (logic unchanged, paths patched) and every product is
  rebuilt from L0.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGING = REPO / "dataprep_staging"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

L0 = REPO / "data_fetch_api" / "new_measurements"
LUIMAP_PNG = REPO / "chou-shui-data/data/water/irrigation/luimap_z14.png"
LUIMAP_BOX = REPO / "chou-shui-data/data/water/irrigation/luimap_z14.box.npy"
TW_COUNTY = REPO / "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg"
SEA_SHP = REPO / "chou-shui-data/data/water/sea_TWD97.shp"
BASEMAP = REPO / "results/twin/basemap.npz"
GRID_NPZ = REPO / "results/twin_forward/datum_gate_innov.npz"
TOWNS_TOPO = L0 / "taiwan_atlas" / "towns-10t_2021.json"
MOA038_DIR = L0 / "moa_crop_038_p3"
MOAJ51_DIR = L0 / "moa_rice_J51_p3"
PILOT_LU = REPO / "results/dataprep/_pilots/landuse_crop_priors"
PILOT_ML = REPO / "results/dataprep/_pilots/design_ml_transfer"
C0_PRE = REPO / "results/dataprep/pump_electricity/preorigin/v3"
C0_L1 = REPO / "results/dataprep/pump_electricity/census_L1/v3"
EVID = REPO / "results/dataprep/reports/P3_evidence"
WORK = EVID / "work"

YEARS_PRE = list(range(2012, 2023))            # pre-origin survey / census years
T0, T1 = "2012-01", "2022-12"                  # census window (annual means)
STEP = 25.0                                    # LUIMAP lattice step (m), as the pilot

# regression groups (sums of LUIMAP groups), as the pilot's common.REG
REG = {
    "paddy": ["paddy"], "dry": ["dry"], "orchard": ["orchard"], "aqua": ["aqua"],
    "livestock": ["livestock"], "agfac": ["agfac"], "forest": ["forest", "grass"],
    "built": ["resid", "commerce", "public", "recreation", "transport"],
    "industry": ["industry", "mining"],
    "water": ["river", "pond", "hydrofac", "sea", "canal"], "unused": ["unused", "unk"],
}
FEATS = list(REG)
FARM = ["paddy", "dry", "orchard", "aqua", "livestock", "agfac"]
GEO = ["d_coast_km", "d_river_km", "dem", "canal3_ha"]
COV = ["d_coast_km", "d_river_km", "dem", "canal3_ha", "x"]
CLASSES = ["irrigation", "aquaculture", "livestock", "domestic", "industry", "other"]
TARGETS4 = ["E_irrigation", "E_aquaculture", "E_livestock", "E_other"]


def load_grid() -> SimpleNamespace:
    z = np.load(GRID_NPZ, allow_pickle=True)
    mask = z["mask"].astype(bool)
    return SimpleNamespace(mask=mask, nx=int(z["nx"]), ny=int(z["ny"]), dx=float(z["dx"]),
                           x0=float(z["x0"]), y0=float(z["y0"]), n_active=int(mask.sum()))


def active_rc(grid) -> tuple[np.ndarray, np.ndarray]:
    return np.nonzero(grid.mask)


def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_jdefault)


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ----------------------------------------------------------------------------- MOA tables
MUSHROOM = ("菇", "木耳", "靈芝", "蕈")
CROP_FEATS = ["rice1", "rice2", "upland", "veg", "fruit", "special", "flower", "forage"]


def _read_pages(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            rows += json.load(fh)
    return rows


def crop_table(d038: Path = MOA038_DIR, dj51: Path = MOAJ51_DIR, pilot_layout: bool = False
               ) -> pd.DataFrame:
    """Township x year crop-season areas (ha), the pilot's township_crops.crop_table logic:
    MOA 038 annual ('全年') rows grouped by crop-code prefix (mushrooms dropped), plus J51 rice
    planted area by season (rice1, rice2), filtered to Changhua and Yunlin. Index (年度 str,
    town = 縣市+鄉鎮). ``pilot_layout`` reads the pilot's concatenated files instead."""
    if pilot_layout:
        rows = _read_pages(str(d038 / "crop_*.json"))
    else:
        rows = _read_pages(str(d038 / "crop_*_p*.json"))
    d = pd.DataFrame(rows)
    d["code"] = d["作物代碼"].astype(str).str.strip()
    d = d[d["期作"] == "全年"]
    d = d[~d["作物"].str.contains("|".join(MUSHROOM))]
    pre = d["code"].str[0]
    grp = np.select([pre.isin(["0", "1", "2"]), pre == "3", pre == "4", pre == "6",
                     pre == "7", pre == "9", d["作物"] == "休閒面積", d["作物"] == "荒廢面積"],
                    ["upland", "special", "veg", "fruit", "forage", "flower", "fallow", "idle"],
                    "other")
    d = d.assign(grp=grp, town=d["縣市"] + d["鄉鎮"])
    d["年度"] = d["年度"].astype(str)
    d["種植面積(公頃)"] = pd.to_numeric(d["種植面積(公頃)"], errors="coerce").fillna(0.0)
    t = d.groupby(["年度", "town", "grp"])["種植面積(公頃)"].sum().unstack(fill_value=0.0)
    r = pd.DataFrame(_read_pages(str(dj51 / ("rice_*.json" if pilot_layout else
                                             "rice_*_p*.json"))))
    r = r[r["縣市"].isin(["彰化縣", "雲林縣"])]
    r = r.assign(town=r["縣市"] + r["鄉鎮"], crop=r["期作"].map({"一期作": "rice1",
                                                             "二期作": "rice2"}))
    r["年度"] = r["年度"].astype(str)
    r["種植面積"] = pd.to_numeric(r["種植面積"], errors="coerce").fillna(0.0)
    rr = r.dropna(subset=["crop"]).groupby(["年度", "town", "crop"])["種植面積"].sum() \
        .unstack(fill_value=0.0)
    t = t.join(rr, how="outer").fillna(0.0)
    for c in CROP_FEATS + ["fallow", "idle", "other"]:
        if c not in t:
            t[c] = 0.0
    return t


def roc(year: int) -> str:
    return str(year - 1911)


# census township names -> MOA / MOI (atlas 2021) names. The census attribute table writes
# 台西鄉 and the pre-2015 員林鎮; MOA 038/J51 and the MOI polygons write 臺西鄉 and 員林市.
# (The pilot joined the census and MOA names without this map and so dropped 臺西鄉 from its
# Yunlin crop-statistics fit; P3 keeps that pilot behaviour only in its reproduction variant.)
TOWN_ALIASES = {"台西鄉": "臺西鄉", "員林鎮": "員林市"}


def norm_town(county_zh: str, town) -> str | None:
    if town is None or (isinstance(town, float) and np.isnan(town)):
        return None
    t = str(town).strip()
    t = TOWN_ALIASES.get(t, t)
    return county_zh + t


COUNTY_ZH = {"yunlin": "雲林縣", "changhua": "彰化縣", "chiayi": "嘉義縣"}
