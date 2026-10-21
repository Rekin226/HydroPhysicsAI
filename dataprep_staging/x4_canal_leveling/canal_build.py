"""X4 canal product: redistribute the W1 district canal series inside each irrigation district.

    python -I canal_build.py [--out-root <dir>]   (default results/dataprep; a scratch root
                                                   for a dry run)

W1 (surface_water_v4) spreads a district's monthly canal volume over all LUIMAP-2024 farmland
of the county inside the model box. This build offers three alternative allocations, each
preserving W1's district monthly volumes exactly, and measures how much they move the water:

* ``service``  the irrigation service area of the management office (農田水利事業區域內灌溉
  制度公告圖, 2024), every regime at weight 1 (the area the office is legally bound to serve);
* ``duty``     the same map, regimes weighted by paddy seasons: 雙期作田 1, 單期作田 0.5,
  輪作田 1/3 (one paddy year in three); a sensitivity, not a measured duty;
* ``rice``     the office's in-county volume split between townships by the MOA township
  rice planted area of the year and crop season (J51), then spread inside each township over
  its service-area pixels; the part of the office's service area outside its main county
  (Nantou, Chiayi) keeps its static ``service`` share.

Outputs per variant (pre-origin 2012-2022 and full 2012-2025): cell shares, a monthly field
(A, T) in m/day (same convention as W1 ``grid_sw_v4.npz``), township tables and comparison
metrics against W1. Code only; no value is fitted.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import hashing, manifest  # noqa: E402
from common.paths import ORIGIN, REPO, rel_to_repo  # noqa: E402

from x4_canal_leveling import canal_maps as cm  # noqa: E402
from x4_canal_leveling.config import (  # noqa: E402
    DATAPREP,
    L0_CANAL,
    L0_NEW,
    L0_RICE,
    PROD_CANAL,
)

GRID_NPZ = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"
TOWNS = L0_NEW / "taiwan_atlas" / "towns-10t_2021.json"
W1_PRE = DATAPREP / "surface_water_v4" / "preorigin" / "v2"
W1_FULL = DATAPREP / "surface_water_v4" / "full" / "v4"
RICE_NEW = L0_NEW / "moa_town_crop_rice_2023_2025" / "2026-10-21"
STATIONS = REPO / "AMP_V2" / "data" / "fan_stations.parquet"
IA_STATS = {"2022": L0_CANAL / "stats111_benefit_area.ods",
            "2025": L0_CANAL / "stats114_service_area.ods"}
OFFICE = {"chc": {"county": "彰化縣", "w1_key": "share_changhua", "district": "changhua",
                  "stats_name": "彰 化"},
          "yl": {"county": "雲林縣", "w1_key": "share_yunlin", "district": "yunlin",
                 "stats_name": "雲 林"}}
DUTY = {"double": 1.0, "single": 0.5, "rotation": 1.0 / 3.0}
CROP_OF_MONTH = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2}


def load_grid() -> SimpleNamespace:
    z = np.load(GRID_NPZ, allow_pickle=True)
    return SimpleNamespace(mask=z["mask"].astype(bool), nx=int(z["nx"]), ny=int(z["ny"]),
                           dx=float(z["dx"]), x0=float(z["x0"]), y0=float(z["y0"]))


def towns_3826():
    import geopandas as gpd

    tw = gpd.read_file(TOWNS, layer="towns").set_crs(4326, allow_override=True).to_crs(3826)
    tw = tw[tw.COUNTYNAME.isin(["彰化縣", "雲林縣", "南投縣", "嘉義縣", "嘉義市", "臺中市",
                                "台中市"])]
    return tw.reset_index(drop=True)


def pixel_town(points: dict, tw) -> dict[str, np.ndarray]:
    import shapely

    out = {}
    for k, (x, y) in points.items():
        tid = np.full(x.size, -1, np.int32)
        for i, geom in enumerate(tw.geometry):
            minx, miny, maxx, maxy = geom.bounds
            sel = np.flatnonzero((x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
                                 & (tid < 0))
            if sel.size:
                tid[sel[shapely.contains_xy(geom, x[sel], y[sel])]] = i
        out[k] = tid
    return out


def cell_of(x, y, grid) -> np.ndarray:
    act = np.flatnonzero(grid.mask.ravel())
    lut = np.full(grid.mask.size, -1, np.int64)
    lut[act] = np.arange(act.size)
    c = np.floor((x - grid.x0) / grid.dx).astype(np.int64)
    r = np.floor((y - grid.y0) / grid.dx).astype(np.int64)
    ok = (c >= 0) & (c < grid.nx) & (r >= 0) & (r < grid.ny)
    return np.where(ok, lut[np.where(ok, r * grid.nx + c, 0)], -1)


def rice_table() -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(str(L0_RICE / "rice_*_p*.json")))
    files += sorted(glob.glob(str(RICE_NEW / "rice_*_p*.json")))
    for f in files:
        with open(f, encoding="utf-8-sig") as fh:
            page = json.load(fh)
        if isinstance(page, list):
            rows += page
    d = pd.DataFrame(rows)
    d = d[d["縣市"].isin([o["county"] for o in OFFICE.values()])]
    d["year"] = d["年度"].astype(int) + 1911
    d["crop"] = d["期作"].map({"一期作": 1, "二期作": 2})
    d["ha"] = pd.to_numeric(d["種植面積"], errors="coerce").fillna(0.0)
    d["town"] = d["縣市"] + d["鄉鎮"].str.replace("台西鄉", "臺西鄉").str.replace("員林鎮",
                                                                                 "員林市")
    t = d.dropna(subset=["crop"]).groupby(["year", "crop", "town"])["ha"].sum()
    return t.reset_index(), files


CROP_DIRS = (L0_NEW / "moa_crop_038_p3", RICE_NEW)


def season_table(y1: int) -> tuple[pd.DataFrame, list[str]]:
    """Township x year x season: rice planted (J51) and fallow / idle land (MOA 038 休閒面積,
    荒廢面積), ha, for Changhua and Yunlin, years 2012..y1."""
    rows, files = [], []
    for dd in CROP_DIRS:
        for f in sorted(glob.glob(str(dd / "crop_*_p*.json"))):
            if int(Path(f).stem.split("_")[2]) + 1911 > y1:
                continue
            with open(f, encoding="utf-8-sig") as fh:
                page = json.load(fh)
            if isinstance(page, list):
                rows += page
                files.append(f)
    d = pd.DataFrame(rows)
    d = d[d["作物"].isin(["休閒面積", "荒廢面積"]) & d["期作"].isin(["一期作", "二期作", "裡作"])]
    d["year"] = d["年度"].astype(int) + 1911
    d["season"] = d["期作"].map({"一期作": "crop1", "二期作": "crop2", "裡作": "winter"})
    d["town"] = d["縣市"] + d["鄉鎮"]
    d["what"] = d["作物"].map({"休閒面積": "fallow_ha", "荒廢面積": "idle_ha"})
    d["ha"] = pd.to_numeric(d["種植面積(公頃)"], errors="coerce").fillna(0.0)
    t = d.pivot_table(index=["year", "season", "town"], columns="what", values="ha",
                      aggfunc="sum", fill_value=0.0).reset_index()
    rice, _ = rice_table()
    rice = rice[rice.year <= y1].assign(season=lambda x: x.crop.map({1: "crop1", 2: "crop2"}))
    t = t.merge(rice.rename(columns={"ha": "rice_planted_ha"})[["year", "season", "town",
                                                                 "rice_planted_ha"]],
                on=["year", "season", "town"], how="outer")
    return t.sort_values(["town", "year", "season"]).reset_index(drop=True), files


def ia_stats() -> dict:
    """Published service areas (ha) per office and regime, 2022 (benefit area) and 2025."""
    from x4_canal_leveling.ods_read import read_ods

    out = {}
    for yr, p in IA_STATS.items():
        sheets = read_ods(p)
        for _name, rows in sheets.items():
            if not rows or "管理處" not in " ".join(" ".join(r) for r in rows[:6]):
                continue
            for r in rows:
                if r and r[0].strip() in [o["stats_name"] for o in OFFICE.values()]:
                    vals = [float(x.replace(",", "")) if x.replace(",", "").replace(".", "")
                            .isdigit() else np.nan for x in r[1:]]
                    key = [k for k, o in OFFICE.items() if o["stats_name"] == r[0].strip()][0]
                    if yr == "2022":       # 埤, 圳, 實際耕種, 合計, 灌溉專用 計, 兩期, 單期, 輪作, 旱田
                        rec = {"total_irrig_drain": vals[3], "double": vals[5],
                               "single": vals[6], "rotation": vals[7], "dry": vals[8]}
                    else:                  # 灌溉服務, 擴大, 合計, 灌溉專用 計, 兩期, 單期, 輪作, 旱田
                        rec = {"service_total": vals[0], "extended": vals[1],
                               "total_irrig_drain": vals[2], "double": vals[4],
                               "single": vals[5], "rotation": vals[6], "dry": vals[7]}
                    out.setdefault(key, {})[yr] = rec
    return out


def months(series: pd.DataFrame, district: str, y0: int, y1: int) -> pd.DataFrame:
    s = series[(series.district == district) & (series.date.dt.year >= y0)
               & (series.date.dt.year <= y1)].sort_values("date")
    return s[["date", "value", "sd"]].reset_index(drop=True)


def build(variant: str, years: tuple[int, int], w1_dir: Path, out_dir: Path, digi: dict,
          grid, tw, rice: pd.DataFrame, wells: pd.DataFrame) -> dict:
    A = int(grid.mask.sum())
    w1 = np.load(w1_dir / "grid_sw_v4.npz", allow_pickle=True)
    series = pd.read_csv(w1_dir / "district_monthly.csv", parse_dates=["date"])
    tnames = [f"{c}{t}" for c, t in zip(tw.COUNTYNAME, tw.TOWNNAME, strict=True)]
    dates = None
    fields = {k: None for k in ("w1", "service", "duty", "rice")}
    shares_out, town_rows, metrics = {}, [], {}
    for off, o in OFFICE.items():
        d = digi[off]
        cls = list(d["points"])
        cells = {k: cell_of(*d["points"][k], grid) for k in cls}
        towns = d["towns"]
        tot_area = {k: d["points"][k][0].size * d["px_area_m2"] for k in cls}
        all_px = sum(d["points"][k][0].size for k in cls)

        def share_static(weights: dict, d=d, cls=cls, cells=cells) -> np.ndarray:
            den = sum(weights[k] * d["points"][k][0].size for k in cls)
            s = np.zeros(A)
            for k in cls:
                good = cells[k] >= 0
                s += np.bincount(cells[k][good], minlength=A) * weights[k] / den
            return s

        s_serv = share_static({k: 1.0 for k in cls})
        s_duty = share_static({k: DUTY[k] for k in cls})
        s_w1 = np.asarray(w1[o["w1_key"]], float)
        # rice variant: per (year, crop) township weights inside the main county
        is_cty = np.array([n.startswith(o["county"]) for n in tnames])
        in_cty = {k: (towns[k] >= 0) & is_cty[np.maximum(towns[k], 0)] for k in cls}
        n_in = sum(in_cty[k].sum() for k in cls)
        f_out = 1.0 - n_in / all_px
        s_out = np.zeros(A)       # static share of the out-of-county service pixels
        for k in cls:
            sel = (~in_cty[k]) & (cells[k] >= 0)
            s_out += np.bincount(cells[k][sel], minlength=A) / all_px
        # per township: cell distribution of its in-county service pixels
        tpix, tcell = {}, {}
        for k in cls:
            for i in np.unique(towns[k][in_cty[k]]):
                sel = in_cty[k] & (towns[k] == i)
                tpix[tnames[i]] = tpix.get(tnames[i], 0) + int(sel.sum())
                c = cells[k][sel]
                v = np.bincount(c[c >= 0], minlength=A).astype(float)
                tcell[tnames[i]] = tcell.get(tnames[i], np.zeros(A)) + v
        for t in tcell:
            tcell[t] = tcell[t] / tpix[t]          # fraction of the township's pixels per cell
        ser = months(series, o["district"], *years)
        dates = pd.DatetimeIndex(ser["date"])
        T = len(dates)
        s_rice = np.zeros((A, T))
        rice_used = []
        for j, day in enumerate(dates):
            crop = CROP_OF_MONTH.get(day.month)
            if crop is None:
                s_rice[:, j] = s_serv               # zero-volume months: any share
                continue
            r = rice[(rice.year == day.year) & (rice.crop == crop)
                     & rice.town.isin(list(tcell))].set_index("town")["ha"]
            if r.sum() <= 0:
                raise ValueError(f"no rice area for {off} {day:%Y-%m}")
            w = r / r.sum()
            s_rice[:, j] = s_out + (1 - f_out) * sum(w[t] * tcell[t] for t in w.index)
            if day.month in (3, 8):
                rice_used.append({"office": off, "year": day.year, "crop": crop,
                                  "towns": int((r > 0).sum()), "rice_ha": float(r.sum())})
        days = np.array([x.days_in_month for x in dates], float)
        vol = ser["value"].to_numpy() * 1e8
        cell_area = grid.dx ** 2

        def field(sh, vol=vol, cell_area=cell_area, days=days):
            sh2 = sh if sh.ndim == 2 else sh[:, None]
            return sh2 * vol[None, :] / cell_area / days[None, :]

        for k, sh in (("w1", s_w1), ("service", s_serv), ("duty", s_duty), ("rice", s_rice)):
            f = field(sh)
            fields[k] = f if fields[k] is None else fields[k] + f
        shares_out[off] = {"w1": s_w1, "service": s_serv, "duty": s_duty, "rice": s_rice,
                           "rice_dates": dates}
        # township table (time-mean of the volume-weighted share, active months)
        act = vol > 0
        wvol = vol[act] / vol[act].sum()
        s_rice_mean = (s_rice[:, act] * wvol[None, :]).sum(1)
        cell_town = np.zeros((len(tnames), A))
        for k in cls:
            good = cells[k] >= 0
            for i in np.unique(towns[k][good & (towns[k] >= 0)]):
                sel = good & (towns[k] == i)
                cell_town[i] += np.bincount(cells[k][sel], minlength=A)
        # assign W1 cell shares to townships by the cell's township composition of service
        # pixels where present, else by the atlas majority (W1 farm cells without service)
        comp = cell_town / np.maximum(cell_town.sum(0, keepdims=True), 1)
        nos = cell_town.sum(0) == 0
        maj = d["cell_majority_town"]
        comp[:, nos] = 0.0
        for c in np.flatnonzero(nos):
            if maj[c] >= 0:
                comp[maj[c], c] = 1.0
        for i, t in enumerate(tnames):
            vals = {"w1": float(comp[i] @ s_w1), "service": float(comp[i] @ s_serv),
                    "duty": float(comp[i] @ s_duty), "rice": float(comp[i] @ s_rice_mean)}
            if max(vals.values()) < 1e-5:
                continue
            town_rows.append({"office": off, "town": t, **{f"share_{k}": v for k, v in
                                                           vals.items()},
                              "service_ha_digitised": float(cell_town[i].sum()
                                                            * d["px_area_m2"] / 1e4)})
        # comparison metrics (time-mean, volume weighted)
        m = {}
        for k, sh in (("service", s_serv), ("duty", s_duty), ("rice", s_rice_mean)):
            m[k] = {"infan_fraction": float(sh.sum()),
                    "w1_infan_fraction": float(s_w1.sum()),
                    "moved_fraction_of_district": float(0.5 * np.abs(sh - s_w1).sum()),
                    "corr_cell_share": float(np.corrcoef(sh, s_w1)[0, 1]),
                    "cells_w1_positive_now_zero": int(((s_w1 > 0) & (sh == 0)).sum()),
                    "cells_zero_now_positive": int(((s_w1 == 0) & (sh > 0)).sum())}
        # rice variant: interannual movement (each year's crop-1 March share vs the mean)
        yr_moves = []
        for j, day in enumerate(dates):
            if day.month in (3, 8):
                yr_moves.append({"year": day.year, "crop": CROP_OF_MONTH[day.month],
                                 "moved_vs_service": float(0.5 * np.abs(s_rice[:, j]
                                                                        - s_serv).sum()),
                                 "moved_vs_w1": float(0.5 * np.abs(s_rice[:, j] - s_w1).sum())})
        m["rice_by_year"] = yr_moves
        m["rice_inputs"] = rice_used
        m["out_of_county_service_fraction"] = float(f_out)
        m["digitised_area_ha"] = {k: float(v / 1e4) for k, v in tot_area.items()}
        # wells: depth at the cells holding the district's monitoring wells (no head data)
        wc = wells[wells.CountyName == o["county"]]
        cidx = cell_of(wc.x.to_numpy(), wc.y.to_numpy(), grid)
        cidx = np.unique(cidx[cidx >= 0])
        m["well_cells"] = int(cidx.size)
        m["well_cell_corr_vs_w1"] = {
            k: float(np.corrcoef(sh[cidx], s_w1[cidx])[0, 1])
            for k, sh in (("service", s_serv), ("duty", s_duty), ("rice", s_rice_mean))}
        m["well_cell_share_ratio_vs_w1"] = {
            k: float(sh[cidx].sum() / max(s_w1[cidx].sum(), 1e-12))
            for k, sh in (("service", s_serv), ("duty", s_duty), ("rice", s_rice_mean))}
        metrics[off] = m
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "canal_fields_x4.npz",
        dates=np.array([x.strftime("%Y-%m") for x in dates]),
        **{f"value_{k}": v for k, v in fields.items()},
        **{f"share_{off}_{k}": shares_out[off][k] for off in OFFICE
           for k in ("w1", "service", "duty", "rice")},
        mask_hash=hashing.mask_hash(grid.mask),
        grid_hash=hashing.grid_hash(grid.nx, grid.ny, grid.dx, grid.x0, grid.y0, grid.mask),
        nx=grid.nx, ny=grid.ny, dx=grid.dx, x0=grid.x0, y0=grid.y0,
        units="value_*: m/day of canal water delivered to the cell (monthly mean daily rate), "
              "W1 convention; share_*: fraction of the district volume per cell",
        note=f"variant {variant}; district volumes = W1 {rel_to_repo(w1_dir)} "
             "district_monthly.value, unchanged; only the within-district allocation differs")
    pd.DataFrame(town_rows).to_csv(out_dir / "township_shares.csv", index=False)
    seas, seas_files = season_table(years[1])
    seas.to_csv(out_dir / "township_season_rice_fallow.csv", index=False)
    metrics["season_table_files"] = [rel_to_repo(f) for f in seas_files]
    rebuild = np.abs(fields["w1"] - np.asarray(w1["value"])[:, :fields["w1"].shape[1]])
    metrics["w1_rebuild_max_abs_m_per_day"] = float(rebuild.max())
    metrics["w1_rebuild_max_rel"] = float(rebuild.max() / np.abs(np.asarray(w1["value"])).max())
    return metrics


def digitise_all(grid, tw) -> tuple[dict, dict]:
    import shapely

    digi, diag = {}, {}
    act_rc = np.argwhere(grid.mask)
    cx = grid.x0 + (act_rc[:, 1] + 0.5) * grid.dx
    cy = grid.y0 + (act_rc[:, 0] + 0.5) * grid.dx
    maj = np.full(len(cx), -1, np.int32)
    for i, geom in enumerate(tw.geometry):
        maj[shapely.contains_xy(geom, cx, cy) & (maj < 0)] = i
    for off in OFFICE:
        d = cm.digitise(off)
        d["towns"] = pixel_town(d["points"], tw)
        d["cell_majority_town"] = maj
        digi[off] = d
        county = {}
        for k in d["points"]:
            for i, n in zip(*np.unique(d["towns"][k], return_counts=True), strict=True):
                c = tw.COUNTYNAME.iloc[i] if i >= 0 else "outside"
                county[c] = county.get(c, 0.0) + float(n) * d["px_area_m2"] / 1e4
        diag[off] = {"geo": {k: v for k, v in d["geo"].items()},
                     "class_area_ha": d["class_area_ha"], "area_by_county_ha": county}
    return digi, diag


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[argv.index("--out-root") + 1]) if "--out-root" in argv else DATAPREP
    grid = load_grid()
    tw = towns_3826()
    digi, diag = digitise_all(grid, tw)
    stats = ia_stats()
    for off in OFFICE:
        s = stats.get(off, {})
        ratio = {}
        for yr, rec in s.items():
            ratio[yr] = {k: diag[off]["class_area_ha"].get(k, 0.0) / rec[k]
                         for k in ("double", "single", "rotation") if rec.get(k)}
        diag[off]["published_ha"] = s
        diag[off]["digitised_over_published"] = ratio
    rice, rice_files = rice_table()
    st = pd.read_parquet(STATIONS, columns=["sid", "CountyName", "LocationByTWD97"])
    xy = st["LocationByTWD97"].astype(str).str.split(expand=True).astype(float)
    st["x"] = np.where(xy[0] < 1e6, xy[0], xy[1])
    st["y"] = np.where(xy[0] < 1e6, xy[1], xy[0])
    results = {}
    for variant, years, w1_dir in (("preorigin", (2012, 2022), W1_PRE),
                                   ("full", (2012, 2025), W1_FULL)):
        out = manifest.new_version_dir(f"{PROD_CANAL}/{variant}", root=root)
        met = build(variant, years, w1_dir, out, digi, grid, tw, rice, st)
        gate_ok = all(
            0.6 <= r <= 1.15 for off in OFFICE
            for k, r in diag[off]["digitised_over_published"].get("2025", {}).items())
        placement = {off: diag[off]["area_by_county_ha"].get(OFFICE[off]["county"], 0.0)
                     / sum(diag[off]["area_by_county_ha"].values()) for off in OFFICE}
        gate_ok = gate_ok and all(v >= 0.9 for v in placement.values()) \
            and met["w1_rebuild_max_rel"] < 1e-6
        val = {"schema": "x4-canal-validation/1", "variant": variant, "years": list(years),
               "digitisation": diag, "placement_in_main_county": placement,
               "metrics": met, "gate_passed": bool(gate_ok),
               "gate_rule": "digitised class areas within 0.60-1.15 of the 2025 published "
                            "office areas per regime; >= 90 % of digitised service area in "
                            "the office's main county; W1 field rebuilt from its own shares "
                            "to < 1e-6 relative. A data-integrity gate (no sub-district "
                            "delivery observations exist to score the allocation)."}
        (out / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                        default=float), encoding="utf-8")
        code = [Path(__file__), Path(cm.__file__), HERE_CFG,
                Path(__file__).with_name("ods_read.py")]
        inputs = [L0_CANAL / "chc_system_map_2024.pdf", L0_CANAL / "yl_system_map_2024.pdf",
                  *IA_STATS.values(), TOWNS, GRID_NPZ, w1_dir / "grid_sw_v4.npz",
                  w1_dir / "district_monthly.csv", w1_dir / "MANIFEST.json"]
        inputs += [Path(f) for f in rice_files
                   if variant == "full" or int(Path(f).stem.split("_")[1]) <= 111]
        inputs += [REPO / f for f in met["season_table_files"]]
        if variant == "preorigin":
            m = manifest.build_manifest(
                out, product=f"{PROD_CANAL}/preorigin", layer="L2", inputs=inputs, code=code,
                parameters={"years": list(years), "duty": DUTY,
                            "classifier": {"blend_tol": cm.BLEND_TOL,
                                           "alpha_range": cm.ALPHA_RANGE,
                                           "density_win": cm.DENSITY_WIN,
                                           "density_thr": cm.DENSITY_THR,
                                           "legend_rgb": cm.LEGEND_RGB},
                            "rice_years": [years[0], years[1]],
                            "w1_series": rel_to_repo(w1_dir)},
                fit_window_end="2024-12", non_exempt_fit_window_end=ORIGIN,
                exemptions=[{
                    "input": rel_to_repo(L0_CANAL / "chc_system_map_2024.pdf") + " ; "
                             + rel_to_repo(L0_CANAL / "yl_system_map_2024.pdf"),
                    "data_end": "2024-12",
                    "reason": "the irrigation-regime maps were published with the 2024 "
                              "irrigation-system notices (cadastre version 2023-10-15); no "
                              "pre-origin map of the service area is public. The service area "
                              "is a legal, slowly changing extent (the 2020-11 notice defines "
                              "the same office areas by cadastral section)",
                    "ledger_number_id": "X4.canal.preorigin.service_map_exemption",
                    "sensitivity": "the W1 LUIMAP allocation is the alternative; "
                                   "validation.json metrics give the moved fraction, the "
                                   "cell-share correlation and the in-fan fraction change",
                    "uses": ["forcing_frozen", "model_design", "design_time_evidence"]}],
                validation_report="validation.json",
                gate={"passed": bool(gate_ok), "criteria": val["gate_rule"]},
                allowed_uses=["forcing_frozen", "model_design", "design_time_evidence",
                              "after_the_fact_analysis"],
                forbidden_uses=["scoring_target", "baseline", "datum_fit", "calibration_target",
                                "prospective_scorer", "validator", "selection",
                                "initial_conditions", "idw_field", "prior",
                                "prior_width_input", "forcing_recorded_arm"],
                personal_data="none (cadastral parcel numbers in the L0 registers are not "
                              "read; maps hold no names)",
                extra={"wp": "X4", "flags": "5 (inferred allocation of a flag-5 series)",
                       "note": "an alternative allocation for an A/B rung; not validated "
                               "against sub-district deliveries (none are public)"})
        else:
            m = manifest.build_manifest(
                out, product=f"{PROD_CANAL}/full", layer="L2", inputs=inputs, code=code,
                parameters={"years": list(years), "duty": DUTY,
                            "w1_series": rel_to_repo(w1_dir)},
                fit_window_end="2026-09", validation_report="validation.json",
                gate={"passed": bool(gate_ok), "criteria": val["gate_rule"]},
                allowed_uses=["forcing_recorded_arm", "statistical_layer_history",
                              "after_the_fact_analysis"],
                forbidden_uses=["scoring_target", "baseline", "datum_fit", "calibration_target",
                                "prospective_scorer", "validator", "initial_conditions",
                                "idw_field", "prior", "prior_width_input", "model_design",
                                "selection", "forcing_frozen", "design_time_evidence"],
                personal_data="none",
                extra={"wp": "X4", "note": "U4: post-origin forcing feeds the recorded-forcing "
                                           "arm only"})
        manifest.write_manifest(out, m)
        results[variant] = {"dir": rel_to_repo(out), "gate": gate_ok, "metrics": met,
                            "placement": placement}
        print(json.dumps({"variant": variant, "dir": rel_to_repo(out), "gate": gate_ok},
                         ensure_ascii=False))
    print(json.dumps({"built": dt.datetime.now().isoformat(timespec="seconds"),
                      "digitisation": {k: {"class_area_ha": v["class_area_ha"],
                                           "ratio": v["digitised_over_published"]}
                                       for k, v in diag.items()}},
                     ensure_ascii=False, indent=1, default=float))
    return 0


HERE_CFG = Path(__file__).with_name("config.py")

if __name__ == "__main__":
    raise SystemExit(main())
