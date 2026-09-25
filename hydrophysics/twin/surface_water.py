"""Canal irrigation deliveries per cell and month: the forcing the flow model has never seen.

    python -m hydrophysics.twin.surface_water --out results/twin/surface_water.npz

The fan is irrigated from the Jiji weir through the Changhua (north of the Choushui) and
Yunlin (south of it) irrigation districts: 20-31 x 10^8 m3 a year, delivered onto roughly
40 % of the fan, about as much water per irrigated cell as rain supplies. Through 2022 the
model knew only rain minus ET0 and the pumps. This module builds the delivery field from
public, unauthenticated sources, caching every download under the (gitignored) data dir:

- **How much, per district and year.** MOA open data 35644, "actual irrigation water per
  management office" (農田水利署各管理處歷年實際灌溉用水量), in 10^8 m3: per crop season
  through ROC 104 (2015), per year from 2016. Whole-year values are split into the two
  crops with each district's 2012-2015 mean share (Changhua ~48/52 %, Yunlin ~35/65 %).
  Caveat kept in the output: Changhua steps from ~15.5 to ~13.0 at 2016, the year the
  reporting switched from per-season to per-year, which may be accounting rather than
  water.
- **When, within a crop.** The rotation calendar: first crop February to mid-June, land
  preparation and puddling (February-March) weighted 1.5x; second crop July to
  mid-November, July weighted 1.5x; December and January zero (canal maintenance, 歲修).
- **Where.** Irrigable farmland from the NLSC land-use survey tiles (``LUIMAP``, the same
  public WMTS ``basemap.py`` uses; the 2024 survey), sampled on a 50 m lattice: paddy
  (水田, code 010101, RGB (171, 220, 97)) and dry field (旱田, 010102, (198, 230, 150)).
  **The colour classes are confirmed by NLSC's own colour table** for the ROC 109 survey
  (``LEGEND_URL``), which also gives canals (溝渠, 040104, (166, 205, 213)), orchards
  (010103) and aquaculture (0102); canal area per cell is stored (``canal_m2``) for a
  canal-seepage variant. 水田 is a field type, not a record of rice planted in a season.
  Both areas are stored separately so a different reading is a re-weighting, not a
  re-fetch. Districts come from the local county polygons
  (``TW_COUNTY.gpkg``, TWD67, reprojected): Changhua County -> Changhua district, Yunlin
  County -> Yunlin district; each district's delivery is spread over its farmland inside
  the model grid's bounding box. Cells in other counties (the Nantou apex) get none.

Depth delivered to a cell in month m of year y (m/day, per the recharge field's
mean-daily-rate convention):

    sw[c, m] = V_d(y, season) * w(m) / sum_season(w) * A_farm[c, d] / A_farm_total[d]
               / cell_area / days(m)

Check on the colour reading (2026-09-23 build, z14, 50 m lattice): Changhua County has
24.2k ha paddy + 19.4k ha dry field in the bounding box against the WRA's ~28-31k ha rice
+ 12-14k ha other irrigated crops (~42k ha); Yunlin 18.7k + 39.2k = 58k ha against
20-25k + 22-32k (42-57k). Totals agree; the paddy/dry split is less certain than the sum,
which is why both are irrigated at the same weight by default (``--dry-weight 1``). The
resulting field delivers 22.5 x 10^8 m3/yr inside the fan, 1,046 mm/yr fan-mean, onto
93 % of cells.

This is *delivered* water. The share that reaches the water table is the calibration's
``log_sw_scale`` (literature prior ~0.25: Liu et al. 2001, 2005).

Version 2 (2026-09-23, ``--percolation`` / ``--drought-duty``; keys added, the v1 field
is untouched unless ``--drought-duty`` is given):

- **Paddy flood percolation** (``perc_m_per_day``, the calibration's second
  ``--sw-components`` field). Recharge from paddies is capped by how fast water
  percolates, not by how much is delivered; water beyond ~4 mm/d leaves through the
  drains, so a recharge that scales with delivered volume over-reacts to accounting steps
  such as Changhua's at 2016. ``perc[c, m] = i_tex(zone) * A_paddy[c] * r(d, y, crop) *
  f_flood(m) * duty(d, y, m) / cell_area`` with ``i_tex`` 4.4 / 3.8 / 3.2 mm/d on the
  proximal (sand) / mid (loam) / distal (silty clay) fan (Liu, Tan & Huang 2005: 3.2-4.4
  mm/d by texture), ``f_flood`` the flooded fraction of the month (crop 1: Feb 0.7,
  Mar-May 1, Jun 0.5; crop 2: Jul 0.8, Aug-Oct 1, Nov 0.3; Dec-Jan 0), and ``r`` the
  district's rice area planted that year and crop over its 2018-2020 mean (WRA open data
  58701, per irrigation association and crop season, fetched without credentials and
  cached; associations 12 and 13 are read as Changhua and Yunlin from their rice areas,
  ~31k and ~25k ha, which is an inference the table's code list does not state; the
  table ends in 2020, so 2021-2022 hold 2020's area). The calibration's fraction of it is
  ``k_p`` (prior ~0.25, Liu et al. 2001).
- **The 2021 drought duty** (``--drought-duty``). There was no irrigation stoppage in
  Changhua or Yunlin in 2021, but first-crop canal water ran on rotation from March to
  harvest: Changhua 4 days in 10 (Babao system; 31 deep wells opened), Yunlin 3 on / 3
  off (Luchangke) and 2-3 in 5 (Linnei), the independent canals switching to wells
  (Agriharvest 2021; Legislative Yuan brief 2021-03-18). The annual MOA totals hide this
  (Yunlin 2021 equals 2020), so March-May 2021 deliveries and flooding are multiplied by
  0.4 (Changhua) and 0.5 (Yunlin). The water not delivered is stored as
  ``sw_deficit_m3``; it is NOT added to the pumping forcing, because the forcing is
  metered electricity, which already contains the wells that replaced it.

Not represented, and recorded in the npz: the Jiji weir diversion history (only current
values are public), per-canal duty beyond the 2021 proxy, the HSR-corridor well sealing
(no public map), and township-level fallow changes (the district rice area stands in).
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np
import pandas as pd

MOA_URL = ("https://data.moa.gov.tw/Service/OpenData/DataFileService.aspx"
           "?UnitId=691&FOTT=CSV&IsTransData=1")
SOURCE = ("MOA open data 35644 (農田水利署各管理處歷年實際灌溉用水量); land use: NLSC "
          "LUIMAP (國土利用現況調查成果圖) WMTS; counties: GADM TW_COUNTY")
DISTRICTS = ("changhua", "yunlin")
OFFICE = {"changhua": "彰化", "yunlin": "雲林"}          # 處別 in the MOA table
COUNTY = {"changhua": "Changhua", "yunlin": "Yulin"}    # NAME_2 in TW_COUNTY.gpkg
PADDY_RGB = (171, 220, 97)      # 水田 010101
DRY_RGB = (198, 230, 150)       # 旱田 010102
ORCHARD_RGB = (99, 192, 59)     # 果園 010103
CANAL_RGB = (166, 205, 213)     # 溝渠 040104
AQUA_RGB = (138, 255, 218)      # 水產養殖 0102
LEGEND_URL = "https://maps.nlsc.gov.tw/demo/109年版國土利用色碼表.png"
# WRA open data 58701 (灌溉面積與灌溉用水量統計), per association and crop season
WRA_URL = ("https://opendata.wra.gov.tw/api/v2/b0206d05-f54a-4a14-94a1-cb0b2a17a00b"
           "?sort=_importdate%20asc&format=CSV")
WRA_ASSOC = {"changhua": 12, "yunlin": 13}     # inferred from rice areas, see the doc
RICE_REF_YEARS = (2018, 2019, 2020)
# flooded fraction of each calendar month (paddy percolation)
F_FLOOD = {2: 0.7, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.5, 7: 0.8, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.3}
# infiltration under ponding by fan zone (proximal, mid, distal), mm/day (Liu et al. 2005)
I_TEX_MM_D = (4.4, 3.8, 3.2)
# 2021 first-crop canal rotation: (district, year) -> {month: duty}
DROUGHT_DUTY = {("changhua", 2021): {3: 0.4, 4: 0.4, 5: 0.4},
                ("yunlin", 2021): {3: 0.5, 4: 0.5, 5: 0.5}}
# rotation calendar: month -> (crop 1 or 2, weight); months absent get no water
CALENDAR = {2: (1, 1.5), 3: (1, 1.5), 4: (1, 1.0), 5: (1, 1.0), 6: (1, 0.5),
            7: (2, 1.5), 8: (2, 1.0), 9: (2, 1.0), 10: (2, 1.0), 11: (2, 0.5)}
DEFAULT_CACHE = os.path.join("chou-shui-data", "data", "water", "irrigation")


# ---------------------------------------------------------------------------------------
# how much: district deliveries
# ---------------------------------------------------------------------------------------
def fetch_deliveries(cache_dir: str = DEFAULT_CACHE, refresh: bool = False,
                     log=print) -> pd.DataFrame:
    """The MOA delivery table, cached as ``moa_35644_actual.csv`` under ``cache_dir``."""
    path = os.path.join(cache_dir, "moa_35644_actual.csv")
    if refresh or not os.path.exists(path):
        import requests

        os.makedirs(cache_dir, exist_ok=True)
        r = requests.get(MOA_URL, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(r.content)
        log(f"  fetched {MOA_URL} -> {path} ({len(r.content)} bytes)")
    return pd.read_csv(path, encoding="utf-8-sig")


def district_seasons(df: pd.DataFrame, years: range) -> dict[str, pd.DataFrame]:
    """``{district: DataFrame(index=year AD, columns=[crop1, crop2, split]) in m3}``.

    ``split`` is ``"reported"`` where the table has the two crops and ``"share"`` where a
    whole-year value was divided with the district's per-season-years mean share."""
    df = df.rename(columns=lambda c: str(c).strip())
    out = {}
    for d in DISTRICTS:
        sub = df[df["處別"].astype(str).str.strip() == OFFICE[d]].copy()
        sub["year"] = sub["年別"].astype(int) + 1911
        sub["v"] = sub["灌溉用水量_億噸"].astype(float) * 1e8
        sub["crop"] = sub["期作"].astype(str).str.strip()
        by = sub.pivot_table(index="year", columns="crop", values="v", aggfunc="sum")
        both = by.dropna(subset=[c for c in ("一", "二") if c in by.columns])
        seasonal = both[(both.index >= 2012) & (both.index <= 2015)]
        if seasonal.empty:
            seasonal = both
        share1 = float((seasonal["一"] / (seasonal["一"] + seasonal["二"])).mean())
        rows = []
        for y in years:
            if y in both.index and np.isfinite(both.loc[y, "一"]):
                rows.append((y, both.loc[y, "一"], both.loc[y, "二"], "reported"))
            elif "全" in by.columns and y in by.index and np.isfinite(by.loc[y, "全"]):
                tot = by.loc[y, "全"]
                rows.append((y, share1 * tot, (1 - share1) * tot, "share"))
            else:
                raise ValueError(f"{d}: no delivery recorded for {y}")
        t = pd.DataFrame(rows, columns=["year", "crop1", "crop2", "split"]).set_index("year")
        t.attrs["share_crop1"] = share1
        out[d] = t
    return out


def month_weights() -> dict[int, float]:
    """Within-crop share of a crop's volume delivered in each calendar month."""
    tot = {1: 0.0, 2: 0.0}
    for _m, (c, w) in CALENDAR.items():
        tot[c] += w
    return {m: w / tot[c] for m, (c, w) in CALENDAR.items()}


# ---------------------------------------------------------------------------------------
# where: farmland per cell and district
# ---------------------------------------------------------------------------------------
def fetch_landuse(grid, cache_dir: str = DEFAULT_CACHE, zoom: int = 14,
                  refresh: bool = False, log=print):
    """LUIMAP mosaic over the grid's bounding box, cached as a PNG plus its lon/lat box."""
    from PIL import Image
    from pyproj import Transformer

    from .basemap import fetch_mosaic

    png = os.path.join(cache_dir, f"luimap_z{zoom}.png")
    meta = png.replace(".png", ".box.npy")
    if not refresh and os.path.exists(png) and os.path.exists(meta):
        Image.MAX_IMAGE_PIXELS = None
        return Image.open(png).convert("RGB"), tuple(np.load(meta))
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    lon0, lat0 = to_ll.transform(grid.x0, grid.y0)
    lon1, lat1 = to_ll.transform(grid.x0 + grid.nx * grid.dx, grid.y0 + grid.ny * grid.dx)
    img, box = fetch_mosaic(lon0, lat0, lon1, lat1, "LUIMAP", z=zoom, log=log)
    os.makedirs(cache_dir, exist_ok=True)
    img.save(png, optimize=True)
    np.save(meta, np.array(box, dtype="float64"))
    log(f"  cached {png} ({os.path.getsize(png) / 1e6:.1f} MB)")
    return img, box


def farmland_by_cell(grid, img, box, county_gpkg: str, step_m: float = 50.0,
                     log=print) -> dict:
    """Sample the land-use mosaic on a ``step_m`` lattice over the grid's bounding box.

    Returns per district: ``paddy``/``dry`` area per active cell (m2, (A,)) and the
    district's total paddy/dry area inside the bounding box (m2), plus the cell's
    district share."""
    import geopandas as gpd
    import shapely
    from pyproj import Transformer

    west, south, east, north = box
    xs = grid.x0 + (np.arange(int(grid.nx * grid.dx / step_m)) + 0.5) * step_m
    ys = grid.y0 + (np.arange(int(grid.ny * grid.dx / step_m)) + 0.5) * step_m
    XX, YY = np.meshgrid(xs, ys)
    X, Y = XX.ravel(), YY.ravel()
    lon, lat = Transformer.from_crs(3826, 4326, always_xy=True).transform(X, Y)

    def merc_y(a):
        return np.log(np.tan(np.pi / 4 + np.radians(a) / 2))

    src = np.asarray(img)
    H, W = src.shape[:2]
    px = np.clip(((lon - west) / (east - west) * W).astype("int64"), 0, W - 1)
    py = np.clip(((merc_y(north) - merc_y(lat)) / (merc_y(north) - merc_y(south)) * H)
                 .astype("int64"), 0, H - 1)
    rgb = src[py, px]
    is_paddy = np.all(rgb == np.array(PADDY_RGB, dtype=rgb.dtype), axis=1)
    is_dry = np.all(rgb == np.array(DRY_RGB, dtype=rgb.dtype), axis=1)
    is_canal = np.all(rgb == np.array(CANAL_RGB, dtype=rgb.dtype), axis=1)

    counties = gpd.read_file(county_gpkg).to_crs(3826)
    col = ((X - grid.x0) // grid.dx).astype("int64")
    row = ((Y - grid.y0) // grid.dx).astype("int64")
    inside = grid.mask[row, col]
    active = -np.ones((grid.ny, grid.nx), dtype="int64")
    r_, c_ = np.nonzero(grid.mask)
    active[r_, c_] = np.arange(r_.size)
    cell = active[row, col]
    a_pt = step_m * step_m
    out = {"step_m": step_m, "n_points": int(X.size),
           "frac_paddy_bbox": float(is_paddy.mean()), "frac_dry_bbox": float(is_dry.mean())}
    canal = np.zeros(grid.n_active)
    sc = is_canal & inside
    np.add.at(canal, cell[sc], step_m * step_m)
    out["canal_m2"] = canal
    for d in DISTRICTS:
        poly = counties[counties["NAME_2"] == COUNTY[d]].geometry.union_all()
        in_d = shapely.contains_xy(poly, X, Y)
        res = {}
        for name, flag in (("paddy", is_paddy), ("dry", is_dry)):
            sel = in_d & flag
            res[f"{name}_total_m2"] = float(sel.sum() * a_pt)
            per = np.zeros(grid.n_active)
            s2 = sel & inside
            np.add.at(per, cell[s2], a_pt)
            res[name] = per
        res["cell_share"] = np.zeros(grid.n_active)
        np.add.at(res["cell_share"], cell[in_d & inside], a_pt / grid.dx ** 2)
        out[d] = res
        log(f"  {d}: paddy {res['paddy_total_m2'] / 1e4:,.0f} ha, dry {res['dry_total_m2'] / 1e4:,.0f} "
            f"ha in the bounding box; {res['paddy'].sum() / 1e4:,.0f} + "
            f"{res['dry'].sum() / 1e4:,.0f} ha inside the fan")
    return out


# ---------------------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------------------
def fetch_rice_area(cache_dir: str = DEFAULT_CACHE, refresh: bool = False,
                    log=print) -> pd.DataFrame:
    """WRA 58701 (irrigated rice / other-crop area and water per association and crop
    season, ROC 84-109), cached as ``wra_58701.csv``."""
    path = os.path.join(cache_dir, "wra_58701.csv")
    if refresh or not os.path.exists(path):
        import requests

        os.makedirs(cache_dir, exist_ok=True)
        r = requests.get(WRA_URL, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(r.content)
        log(f"  fetched WRA 58701 -> {path} ({len(r.content)} bytes)")
    return pd.read_csv(path, encoding="utf-8-sig")


def rice_area_ratio(df: pd.DataFrame, years: range,
                    ref_years: tuple[int, ...] = RICE_REF_YEARS) -> dict[str, pd.DataFrame]:
    """``{district: DataFrame(index=year, columns=[crop1, crop2, rice_ha1, rice_ha2,
    source])}``: rice area planted over its ``ref_years`` mean, per crop. Years past the
    table hold its last year ("held"); years before it hold its first."""
    out = {}
    for d in DISTRICTS:
        sub = df[df["irrigationassociation"] == WRA_ASSOC[d]].copy()
        sub["year"] = sub["year"].astype(int) + 1911
        sub = sub.groupby("year")[["firstphasericeirrigationarea",
                                   "secondphasericeirrigationarea"]].max()
        if sub.empty:
            raise ValueError(f"WRA 58701: no rows for association {WRA_ASSOC[d]} ({d})")
        ref = sub.loc[[y for y in ref_years if y in sub.index]].mean()
        rows = []
        for y in years:
            yy = min(max(y, int(sub.index.min())), int(sub.index.max()))
            a1 = float(sub.loc[yy, "firstphasericeirrigationarea"])
            a2 = float(sub.loc[yy, "secondphasericeirrigationarea"])
            rows.append((y, a1 / float(ref.iloc[0]), a2 / float(ref.iloc[1]), a1, a2,
                         "table" if yy == y else "held"))
        out[d] = pd.DataFrame(rows, columns=["year", "crop1", "crop2", "rice_ha1",
                                             "rice_ha2", "source"]).set_index("year")
    return out


def duty_factor(district: str, day: pd.Timestamp, duty: dict | None) -> float:
    """Canal duty of ``district`` in the month of ``day`` (1 unless ``duty`` says)."""
    if not duty:
        return 1.0
    return float(duty.get((district, day.year), {}).get(day.month, 1.0))


def build_percolation(grid, farm: dict, dates: pd.DatetimeIndex, ratios: dict | None,
                      zone_of_cell: np.ndarray, duty: dict | None = None,
                      i_tex_mm_d: tuple[float, float, float] = I_TEX_MM_D) -> np.ndarray:
    """Paddy flood percolation potential ``(A, T)``, m/day of the cell (see the module
    doc): texture-limited infiltration over the flooded paddy area, NOT scaled by the
    delivered volume."""
    A, T = grid.n_active, len(dates)
    area = float(grid.dx) ** 2
    i_cell = np.asarray(i_tex_mm_d, dtype="float64")[np.asarray(zone_of_cell)] / 1000.0
    perc = np.zeros((A, T))
    for d in DISTRICTS:
        frac = farm[d]["paddy"] / area                      # paddy share of each cell
        for t, day in enumerate(dates):
            f = F_FLOOD.get(day.month, 0.0)
            if f == 0.0:
                continue
            crop = CALENDAR[day.month][0]
            r = float(ratios[d].loc[day.year, f"crop{crop}"]) if ratios else 1.0
            perc[:, t] += i_cell * frac * r * f * duty_factor(d, day, duty)
    return perc


def build_field(grid, seasons: dict[str, pd.DataFrame], farm: dict,
                dates: pd.DatetimeIndex, dry_weight: float = 1.0,
                duty: dict | None = None) -> dict:
    """``sw_m_per_day`` (A, T) plus the per-district monthly volumes that built it.
    ``duty`` (``DROUGHT_DUTY``) scales the months a rotation cut; the water it removes
    is returned as ``deficit_m3`` (A, T). ``None`` is the v1 field."""
    wm = month_weights()
    A, T = grid.n_active, len(dates)
    sw = np.zeros((A, T))
    deficit = np.zeros((A, T))
    area = float(grid.dx) ** 2
    vol_fan = np.zeros((len(DISTRICTS), T))
    for di, d in enumerate(DISTRICTS):
        f = farm[d]
        a_cell = f["paddy"] + dry_weight * f["dry"]
        a_tot = f["paddy_total_m2"] + dry_weight * f["dry_total_m2"]
        if a_tot <= 0:
            continue
        share = a_cell / a_tot                               # fraction of the district
        tab = seasons[d]
        for t, day in enumerate(dates):
            m = day.month
            if m not in CALENDAR:
                continue
            crop = CALENDAR[m][0]
            V = float(tab.loc[day.year, f"crop{crop}"]) * wm[m]    # m3 this month
            k = duty_factor(d, day, duty)
            days = day.days_in_month
            sw[:, t] += k * V * share / area / days
            deficit[:, t] += (1.0 - k) * V * share
            vol_fan[di, t] = k * V * share.sum()
    return {"sw_m_per_day": sw, "vol_fan_m3": vol_fan, "deficit_m3": deficit}


def main(argv=None) -> None:
    from .calibrate_flow import DEFAULT_PATHS
    from .grid import build_grid

    ap = argparse.ArgumentParser(description="canal irrigation deliveries for the twin")
    ap.add_argument("--polygon", default=DEFAULT_PATHS["polygon"])
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--county-gpkg", default=os.path.join(
        os.environ.get("HYDROMIND_GW_DATA", os.path.join("chou-shui-data", "data")),
        "water", "Taiwan_countyV02", "TW_COUNTY.gpkg"))
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--zoom", type=int, default=14)
    ap.add_argument("--step-m", type=float, default=50.0)
    ap.add_argument("--dry-weight", type=float, default=1.0,
                    help="weight of dry-field farmland relative to paddy in the map")
    ap.add_argument("--t0", default="2012-01-01")
    ap.add_argument("--t1", default="2023-01-01")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--drought-duty", action="store_true",
                    help="v2: scale March-May 2021 deliveries and flooding by the "
                         "first-crop canal rotation (Changhua 0.4, Yunlin 0.5)")
    ap.add_argument("--no-percolation", action="store_true",
                    help="skip the v2 paddy percolation field (and its WRA download)")
    ap.add_argument("--zone-boundaries", default="205,182",
                    help="proximal/mid and mid/distal eastings (km) for the texture zones")
    ap.add_argument("--out", default="results/twin/surface_water.npz")
    args = ap.parse_args(argv)

    grid = build_grid(args.polygon, dx=args.dx)
    dates = pd.date_range(args.t0, args.t1, freq="MS", inclusive="left")
    print("deliveries:", flush=True)
    seasons = district_seasons(fetch_deliveries(args.cache_dir, args.refresh),
                               range(dates[0].year, dates[-1].year + 1))
    for d, t in seasons.items():
        print(f"  {d}: crop-1 share {t.attrs['share_crop1']:.2f}; annual 10^8 m3 "
              + " ".join(f"{y}:{(r.crop1 + r.crop2) / 1e8:.2f}" for y, r in t.iterrows()),
              flush=True)
    print("land use:", flush=True)
    img, box = fetch_landuse(grid, args.cache_dir, zoom=args.zoom, refresh=args.refresh)
    farm = farmland_by_cell(grid, img, box, args.county_gpkg, step_m=args.step_m)
    duty = DROUGHT_DUTY if args.drought_duty else None
    f = build_field(grid, seasons, farm, dates, dry_weight=args.dry_weight, duty=duty)
    sw = f["sw_m_per_day"]
    extra: dict = {"canal_m2": farm["canal_m2"], "drought_duty": np.array(str(duty or {}))}
    if duty:
        extra["sw_deficit_m3"] = f["deficit_m3"]
        print(f"2021 rotation: {f['deficit_m3'].sum() / 1e8:.2f} x 10^8 m3 of canal water "
              "not delivered inside the fan (recorded, not added to pumping)", flush=True)
    if not args.no_percolation:
        from .zones import fan_zones

        prox, dist = (float(v) for v in args.zone_boundaries.split(","))
        zoc = fan_zones(grid.centroids(), proximal_km=prox, distal_km=dist)
        print("rice area (WRA 58701):", flush=True)
        ratios = rice_area_ratio(fetch_rice_area(args.cache_dir, args.refresh),
                                 range(dates[0].year, dates[-1].year + 1))
        for d, t in ratios.items():
            print(f"  {d}: crop-1 ratio " + " ".join(
                f"{y}:{r.crop1:.2f}{'*' if r.source == 'held' else ''}"
                for y, r in t.iterrows()), flush=True)
        perc = build_percolation(grid, farm, dates, ratios, zoc, duty=duty)
        days_ = np.array([d.days_in_month for d in dates], dtype="float64")
        pmm = (perc * days_).reshape(grid.n_active, -1, 12).sum(axis=2).mean(axis=1) * 1000
        vol = (perc * days_).sum(axis=1).sum() * grid.dx ** 2 / (len(dates) / 12) / 1e8
        print(f"paddy percolation potential: fan-mean {pmm.mean():.0f} mm/yr, p95 "
              f"{np.percentile(pmm, 95):.0f}; {vol:.2f} x 10^8 m3/yr before k_p", flush=True)
        buf2 = io.StringIO()
        pd.concat({d: t for d, t in ratios.items()}).to_csv(buf2)
        extra.update(perc_m_per_day=perc, rice_ratio_csv=np.array(buf2.getvalue()),
                     i_tex_mm_d=np.array(I_TEX_MM_D), f_flood=np.array(str(F_FLOOD)),
                     wra_assoc=np.array(str(WRA_ASSOC)))
    days = np.array([d.days_in_month for d in dates], dtype="float64")
    yearly_mm = (sw * days).reshape(grid.n_active, -1, 12).sum(axis=2).mean(axis=1) * 1000
    print(f"field: fan-mean delivered {yearly_mm.mean():.0f} mm/yr; "
          f"{(yearly_mm > 0).mean() * 100:.0f} % of cells receive water; p95 "
          f"{np.percentile(yearly_mm, 95):.0f} mm/yr; delivered inside the fan "
          f"{f['vol_fan_m3'].sum() / len(dates) * 12 / 1e8:.2f} x 10^8 m3/yr", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    buf = io.StringIO()
    pd.concat({d: t for d, t in seasons.items()}).to_csv(buf)
    np.savez_compressed(
        args.out, sw_m_per_day=sw.astype("float64"),
        dates=np.array([d.strftime("%Y-%m-01") for d in dates]),
        dx=float(grid.dx), n_active=int(grid.n_active),
        districts=np.array(DISTRICTS),
        paddy_m2=np.stack([farm[d]["paddy"] for d in DISTRICTS]),
        dry_m2=np.stack([farm[d]["dry"] for d in DISTRICTS]),
        district_share=np.stack([farm[d]["cell_share"] for d in DISTRICTS]),
        vol_fan_m3=f["vol_fan_m3"], seasons_csv=np.array(buf.getvalue()),
        dry_weight=float(args.dry_weight), zoom=int(args.zoom), step_m=float(args.step_m),
        paddy_rgb=np.array(PADDY_RGB), dry_rgb=np.array(DRY_RGB), source=np.array(SOURCE),
        calendar=np.array(str(CALENDAR)),
        legend_url=np.array(LEGEND_URL), **extra,
        caveats=np.array("LUIMAP classes per the NLSC ROC 109 colour table; 2024 land use "
                         "for every year (percolation rescaled by district rice area, WRA "
                         "associations 12/13 read as Changhua/Yunlin, 2021-22 held at "
                         "2020); Changhua 2016 step may be a reporting change; delivered "
                         "water, not recharge; 2021 rotation only if drought_duty is set; "
                         "no Jiji diversion history"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
