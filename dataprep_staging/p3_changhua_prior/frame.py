"""Per-cell modelling frame for WP P3 (port of the landuse_crop_priors pilot's common.load,
logic unchanged except the data sources):

- features: rebuilt from L0 by features.py (identical to the pilot's features.npz);
- Yunlin training target: the YUNLIN-county component of WP C0 pump_electricity/preorigin/v3
  (cells_by_class.npz value_by_county), opened through the loader guard for the pre-origin use
  'prior' (the plan's P3 role), annual means over 2012-01..2022-12;
- the cell's township id (training groups): majority census township of its kept YUNLIN
  pumps, empty cells filled from the nearest kept Yunlin pump (pilot rule), from
  pump_electricity/census_L1/v3.

No Changhua census value is read here; the Changhua truth is read only by score.py, after the
pre-registration's assert_can_score.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
from common import guard, hashing  # noqa: E402

T_FIRST, T_LAST = "2012-01", "2022-12"


def _window(dates) -> np.ndarray:
    d = np.asarray(dates).astype(str)
    return (d >= T_FIRST) & (d <= T_LAST)


def open_c0(product_dir, use: str):
    return guard.open_product(product_dir, use)


def class_cube(county: str, use: str):
    """(classes, E (C, A, Tw) kWh/month, dates (Tw,)) of one county's in-grid cells,
    2012-01..2022-12, from C0 preorigin/v3 through the guard."""
    p = open_c0(lib.C0_PRE, use)
    z = np.load(p.path("cells_by_class.npz"), allow_pickle=True)
    counties = [str(c) for c in z["counties"]]
    classes = [str(c) for c in z["classes"]]
    w = _window(z["dates"])
    v = z["value_by_county"][counties.index(county)][:, :, w]
    return classes, np.nan_to_num(v.astype(float)), np.asarray(z["dates"]).astype(str)[w], z


def census_attrs(county: str, use: str, cols: list[str]) -> pd.DataFrame:
    """Selected columns of census_L1/v3 for one county (explicit columns only; the L1 table
    holds no name or address column)."""
    p = open_c0(lib.C0_L1, use)
    want = sorted(set(["county"] + cols))
    df = pd.read_parquet(p.path("census_pumps.parquet"), columns=want)
    return df[df["county"] == county].reset_index(drop=True)


def load(use: str = "prior"):
    grid = lib.load_grid()
    f = np.load(lib.WORK / "features.npz", allow_pickle=True)
    tz = np.load(lib.WORK / "township.npz", allow_pickle=True)
    groups = list(f["groups"])
    A = f["area_m2"].shape[1]
    df = pd.DataFrame(index=np.arange(A))
    for k, members in lib.REG.items():
        df[k] = sum(f["area_m2"][groups.index(m)] for m in members) / 1e4      # ha
    for k, members in lib.REG.items():
        for ci, cn in ((0, "chg"), (1, "yun")):
            df[f"{k}_{cn}"] = sum(f["area_cty_m2"][ci, groups.index(m)] for m in members) / 1e4
    sh = f["county_share"]
    df["sh_chg"], df["sh_yun"], df["sh_nan"], df["sh_oth"] = sh
    df["x"], df["y"] = f["cxy"][:, 0], f["cxy"][:, 1]
    df["row"], df["col"] = f["rc"][:, 0], f["rc"][:, 1]
    df["d_coast_km"] = f["d_coast_m"] / 1e3
    df["d_river_km"] = f["d_river_m"] / 1e3
    df["dem"] = f["dem"]
    df["canal3_ha"] = f["canal3_m2"] / 1e4
    df["canal_ha"] = f["area_m2"][groups.index("canal")] / 1e4
    df["zone"] = np.where(df["x"] >= 205e3, 0, np.where(df["x"] >= 182e3, 1, 2))
    # Yunlin target (C0 preorigin/v3, Yunlin county component)
    classes, E, dates, z = class_cube("yunlin", use)
    assert str(z["mask_hash"]) == hashing.mask_hash(grid.mask)
    yrs = np.array([int(d[:4]) for d in dates])
    mon = np.array([int(d[5:7]) for d in dates])
    ny = len(dates) / 12
    for c in classes:
        e = E[classes.index(c)]
        df[f"E_{c}"] = e.sum(1) / ny
        df[f"E1416_{c}"] = e[:, (yrs >= 2014) & (yrs <= 2016)].sum(1) / 3
    df["E_tot"] = sum(df[f"E_{c}"] for c in classes)
    df["E1416_tot"] = sum(df[f"E1416_{c}"] for c in classes)
    df["E_other"] = df["E_domestic"] + df["E_industry"] + df["E_other"]
    etot = E.sum(0)
    df["E_crop1"] = etot[:, (mon >= 2) & (mon <= 6)].sum(1) / ny
    df["E_crop2"] = etot[:, (mon >= 7) & (mon <= 11)].sum(1) / ny
    df["E_off"] = etot[:, (mon == 12) | (mon == 1)].sum(1) / ny
    # cell township (training groups): majority township of kept Yunlin pumps in the cell
    L = census_attrs("yunlin", use, ["town", "cell", "kept", "x_twd97", "y_twd97"])
    pin = L[(L["kept"]) & (L["cell"] >= 0)].copy()
    pin["town"] = pin["town"].map(lambda t: lib.norm_town("雲林縣", t))
    # order as the pilot's ids (census spelling 台西鄉 sorts before 四湖鄉); same partition
    towns = sorted(pin["town"].dropna().unique(), key=lambda t: t.replace("臺西鄉", "台西鄉"))
    town_cell = pin.groupby("cell")["town"].agg(lambda s: s.value_counts().index[0])
    tid = np.full(A, -1)
    for c_, t_ in town_cell.items():
        tid[int(c_)] = towns.index(t_)
    from scipy.spatial import cKDTree
    ok = pin[["x_twd97", "y_twd97"]].notna().all(1)
    tree = cKDTree(pin.loc[ok, ["x_twd97", "y_twd97"]].to_numpy(float))
    ptown = pin.loc[ok, "town"].map({t: i for i, t in enumerate(towns)}).to_numpy()
    miss = tid < 0
    _, j = tree.query(df[["x", "y"]].to_numpy()[miss])
    tid[miss] = ptown[j]
    df["town"] = tid
    df["block5"] = (df["row"] // 5) * 100 + (df["col"] // 5)
    names = [str(n) for n in tz["names"]]
    shm = tz["share"]
    for cz, col in (("彰化縣", "moi_chg"), ("雲林縣", "moi_yun"), ("南投縣", "moi_nan")):
        df[col] = shm[[i for i, n in enumerate(names) if n.startswith(cz)]].sum(0)
    aux = {"towns": towns, "dates": dates, "classes": classes, "groups": groups, "f": f,
           "tz": tz, "town_names": names}
    return df, aux


def training_cells(df: pd.DataFrame) -> pd.Series:
    """The pilot's Yunlin training rule (TW_COUNTY shares)."""
    return (df.sh_yun + df.sh_oth >= 0.95) & (df.sh_yun > 0.3) & (df.sh_chg < 0.02)
