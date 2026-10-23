"""W3 repair round (W3r): the DTM against surveyed ground at monitoring wells.

Why this test exists (diagnosis in ``results/dataprep/reports/W3.repair.md``, sections 1-4):
the registered W3 gate compared the 1 km DTM cell mean with leveling-benchmark marks. Those
marks sit on locally raised structures (cell-scale siting -0.42 m, DTM only; site-type
dependent point residual), so the test cannot measure bare-ground elevation. Agency
GroundHeight at a groundwater monitoring well is a survey of the ground itself, made by WRA
in the same height system as the benchmarks. If the -0.38 m point residual at benchmarks
were a vertical-datum offset, the wells would show it too; if it is monument siting, they
would not.

Registered design (pre-registration ``W3r.ground_at_wells.v1``; every constant below is
recorded there and checked by the scorer):

* unit = site (``sid[:6]``); site residual = median over its eligible wells; statistic =
  median over sites;
* well residual r = H_pt + [u(t_s) - u(t_sheet)] - GroundHeight, where H_pt is the bilinear
  20 m DTM at the station at its sheet epoch t_sheet (the published ground_elevation/v3
  mosaic; for fan stations the v3 well table's value), t_s is the station's EstablishDate (the
  survey epoch of its metadata), and u is cumulative leveling displacement interpolated by
  inverse distance (k 8, power 2) from the WRA main campaigns (April-July) 2005-2021 of every
  year with at least 400 benchmarks; an increment larger than 0.30 m per year of separation
  is a benchmark reset and is dropped (the W3 rule, scaled to multi-year gaps);
* eligibility, from metadata and the DTM only (never from a residual):
  E1 finite coordinates and DTM point; E2 GroundHeight finite and not the 0.0 missing code;
  E3 |WellElevation - GroundHeight| <= 5 m when WellElevation is finite (H1b's consistency
  rule, both signs); E4 EstablishDate in 2005-01-01..2022-12-31 (TWVD2001 era, dense
  leveling, pre-origin); E5 DTM sd in the 5 x 5 (100 m) window <= 2 m (point comparison in
  flat terrain); E6 ModificationDate <= 2022-12-31 (no post-origin metadata, U7);
* T1 (gate of the corrected test): fan stations (``AMP_V2/data/fan_stations.parquet``, the
  twin's wells); |median| <= 0.30 m; evaluable with at least 20 sites. These units are a
  SECOND LOOK (W3's diagnostics and H1b computed GroundHeight minus DTM before), so a pass is
  conditional;
* T2 (fresh check, reported): WRA stations outside the fan set inside the DTM coverage
  (never compared with the DTM); too few sites to gate.

Run with ``python -I``; reads only non-personal station columns (never LocationAddress,
Remarks or contact fields).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))           # dataprep_staging
sys.path.insert(0, str(HERE.parents[2]))           # repository (read-only imports)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import hashing  # noqa: E402
from common import manifest as mf  # noqa: E402
from common.paths import DATAPREP, L0_NEW, REPO, TWIN_GRID_NPZ, rel_to_repo  # noqa: E402
from w3_dtm import leveling, mosaic, validation  # noqa: E402

V3 = DATAPREP / "ground_elevation" / "v3"
V3_FILES = ("dtm20m_mosaic.npz", "well_ground_elevation.parquet",
            "ground_elevation_cells.parquet", "sheets.csv", "ground_elevation_grid.npz")
FAN = REPO / "AMP_V2" / "data" / "fan_stations.parquet"
WRA_LIST = L0_NEW / "gw-wra-gw10min-obs-stations.parquet"
LEVELING = REPO / "chou-shui-data" / "data" / "ls_cache" / "ls-wra-lsp-obs__choushui_panel.parquet"
META_COLS = ["sid", "LocationByTWD97", "GroundHeight", "WellElevation", "EstablishDate",
             "ModificationDate", "FinishDate"]

# ------------------------------------------------------------------ registered constants
DESIGN = {
    "unit": "site = sid[:6]; site residual = median of its eligible wells",
    "statistic": "median over sites of the site residual",
    "threshold_m": 0.30,
    "min_sites_T1": 20,
    "establish_window": ["20050101", "20221231"],
    "modification_max": "20221231",
    "stickup_abs_max_m": 5.0,
    "local_sd_max_m": 2.0,
    "local_window_points": 5,
    "displacement": {"first_year": 2005, "last_year": 2021, "main_months": [4, 5, 6, 7],
                     "min_benchmarks_per_campaign": 400, "idw_k": 8, "idw_power": 2.0,
                     "reset_m_per_year_of_gap": 0.30, "n_last_rate": 4},
    "boot_n": 2000, "boot_seed": 1000, "block_m": 10000.0,
}
THRESHOLD_M = DESIGN["threshold_m"]
LOCAL_HALF = 2


def ymd_to_decimal(v) -> float:
    """YYYYMMDD (int, float or str) -> decimal year; NaN if missing or unparseable."""
    try:
        s = str(int(float(v)))
    except (TypeError, ValueError):
        return float("nan")
    return leveling_decimal(s)


def leveling_decimal(s: str) -> float:
    from w3_dtm.dtm_io import decimal_year
    return decimal_year(s)


# ------------------------------------------------------------------ inputs
def verify_v3() -> dict:
    """sha256 of the v3 outputs this test reads, checked against the v3 manifest."""
    m, _ = mf.load_manifest(V3 / "MANIFEST.json")
    rec = {o["path"]: o["sha256"] for o in m["outputs"]}
    out = {}
    for f in V3_FILES:
        got = hashing.sha256_file(V3 / f)
        if rec.get(f) != got:
            raise SystemExit(f"ground_elevation/v3/{f}: sha256 {got} != manifest {rec.get(f)}")
        out[f] = got
    return out


def load_mosaic() -> mosaic.Mosaic:
    with np.load(V3 / "dtm20m_mosaic.npz") as z:
        H, ep, sh = z["H"], z["epoch"], z["sheet"]
        ext, step = tuple(float(v) for v in z["extent"]), float(z["step"])
    sheets = pd.read_csv(V3 / "sheets.csv", dtype={"acquisition_date": str, "dtm_date": str})
    return mosaic.Mosaic(extent=ext, step=step, H=H, epoch=ep, sheet=sh, sheets=sheets)


def local_sd(mos: mosaic.Mosaic, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """sd of the DTM in the 5 x 5 lattice window around each point (w3_dtm.build.local_sd)."""
    out = np.full(len(x), np.nan)
    for i, (xx, yy) in enumerate(zip(x, y, strict=True)):
        if not (np.isfinite(xx) and np.isfinite(yy)):
            continue
        j = int(round((xx - mos.extent[0]) / mos.step))
        k = int(round((yy - mos.extent[2]) / mos.step))
        if j < 0 or k < 0 or j >= mos.H.shape[1] or k >= mos.H.shape[0]:
            continue
        w = mos.H[max(k - LOCAL_HALF, 0):k + LOCAL_HALF + 1,
                  max(j - LOCAL_HALF, 0):j + LOCAL_HALF + 1].astype("float64")
        w = w[np.isfinite(w)]
        if len(w) > 1:
            out[i] = float(w.std(ddof=1))
    return out


def parse_xy(s) -> tuple[float, float]:
    """``LocationByTWD97`` 'x y' (or 'y x', detected by range) -> metres; NaN if invalid."""
    try:
        p = str(s).split()
        a, b = float(p[0]), float(p[1])
    except (ValueError, IndexError, TypeError):
        return float("nan"), float("nan")
    if 140000 <= a <= 240000 and 2580000 <= b <= 2700000:
        return a, b
    if 140000 <= b <= 240000 and 2580000 <= a <= 2700000:
        return b, a
    return float("nan"), float("nan")


def station_table(mos: mosaic.Mosaic | None = None) -> pd.DataFrame:
    """Candidate stations with metadata, DTM point, epoch and local sd. Fan stations take the
    published v3 well-table values; WRA stations outside the fan set are interpolated on the
    same mosaic. No residual is computed here."""
    mos = mos or load_mosaic()
    w3 = pd.read_parquet(V3 / "well_ground_elevation.parquet",
                         columns=["sid", "x", "y", "xy_status", "model_well", "dtm_point_sheet_m",
                                  "dtm_point_epoch", "dtm_local_sd_m", "cell"])
    w3["sid"] = w3["sid"].astype(str)
    fan = pd.read_parquet(FAN, columns=META_COLS)
    fan["sid"] = fan["sid"].astype(str)
    a = w3.merge(fan.drop(columns=["LocationByTWD97"]), on="sid", how="left", validate="1:1")
    a["set"] = "fan"
    wra = pd.read_parquet(WRA_LIST, columns=META_COLS)
    wra["sid"] = wra["sid"].astype(str)
    wra = wra[~wra["sid"].isin(set(a["sid"]))].copy()
    xy = np.array([parse_xy(s) for s in wra["LocationByTWD97"]], dtype="float64")
    wra["x"], wra["y"] = xy[:, 0], xy[:, 1]
    hp, hep = mos.bilinear(wra["x"].to_numpy(), wra["y"].to_numpy())
    wra["dtm_point_sheet_m"], wra["dtm_point_epoch"] = hp, hep
    wra["dtm_local_sd_m"] = local_sd(mos, wra["x"].to_numpy(), wra["y"].to_numpy())
    wra["xy_status"] = np.where(np.isfinite(wra["x"]), "wra_list", "missing")
    wra["model_well"] = False
    with np.load(TWIN_GRID_NPZ, allow_pickle=False) as z:
        mask = z["mask"].astype(bool)
        x0, y0, dx = float(z["x0"]), float(z["y0"]), float(z["dx"])
    wra["cell"] = validation.cell_of_point(np.nan_to_num(wra["x"], nan=-1.0),
                                           np.nan_to_num(wra["y"], nan=-1.0), mask, x0, y0, dx)
    wra = wra[np.isfinite(wra["dtm_point_sheet_m"])].copy()     # inside the DTM coverage
    wra["set"] = "nonfan"
    cols = ["sid", "set", "x", "y", "xy_status", "model_well", "cell", "dtm_point_sheet_m",
            "dtm_point_epoch", "dtm_local_sd_m", "GroundHeight", "WellElevation",
            "EstablishDate", "ModificationDate", "FinishDate"]
    t = pd.concat([a[cols], wra[cols]], ignore_index=True)
    t["site"] = t["sid"].str[:6]
    t["t_establish"] = [ymd_to_decimal(v) for v in t["EstablishDate"]]
    t["t_modification"] = [ymd_to_decimal(v) for v in t["ModificationDate"]]
    return t


def eligibility(t: pd.DataFrame) -> pd.DataFrame:
    """Rules E1-E6 (metadata and DTM only)."""
    t = t.copy()
    est = pd.to_numeric(t["EstablishDate"], errors="coerce")
    mod = pd.to_numeric(t["ModificationDate"], errors="coerce")
    gh = pd.to_numeric(t["GroundHeight"], errors="coerce")
    we = pd.to_numeric(t["WellElevation"], errors="coerce")
    lo, hi = (int(v) for v in DESIGN["establish_window"])
    t["E1_coords_dtm"] = np.isfinite(t["x"]) & np.isfinite(t["y"]) & np.isfinite(
        t["dtm_point_sheet_m"]) & np.isfinite(t["dtm_point_epoch"])
    t["E2_groundheight"] = np.isfinite(gh) & (gh != 0.0)
    t["E3_stickup"] = ~np.isfinite(we) | ((we - gh).abs() <= DESIGN["stickup_abs_max_m"])
    t["E4_establish"] = (est >= lo) & (est <= hi)
    t["E5_flat"] = t["dtm_local_sd_m"] <= DESIGN["local_sd_max_m"]
    t["E6_no_postorigin_edit"] = mod.isna() | (mod <= int(DESIGN["modification_max"]))
    rules = ["E1_coords_dtm", "E2_groundheight", "E3_stickup", "E4_establish", "E5_flat",
             "E6_no_postorigin_edit"]
    t["eligible"] = t[rules].all(axis=1)
    return t


def eligibility_counts(t: pd.DataFrame) -> dict:
    out = {}
    for s in ("fan", "nonfan"):
        d = t[t["set"] == s]
        rec = {"stations": int(len(d))}
        keep = np.ones(len(d), bool)
        for r in ("E1_coords_dtm", "E2_groundheight", "E3_stickup", "E4_establish", "E5_flat",
                  "E6_no_postorigin_edit"):
            keep &= d[r].to_numpy()
            rec[f"after_{r}"] = int(keep.sum())
        e = d[d["eligible"]]
        rec["eligible_wells"] = int(len(e))
        rec["eligible_sites"] = int(e["site"].nunique())
        out[s] = rec
    return out


# ------------------------------------------------------------------ displacement
class GapScaledDisplacement(leveling.DisplacementModel):
    """``leveling.DisplacementModel`` over campaign years that may skip years: the reset
    threshold scales with the gap (0.30 m per year of separation). With consecutive years it
    equals the W3 model exactly (checked by :func:`check_equals_w3_model`)."""

    def __init__(self, camp: pd.DataFrame, years: list[int], k: int = leveling.IDW_K,
                 power: float = leveling.IDW_POWER, reset_per_year: float = 0.30):
        from scipy.spatial import cKDTree
        self.years = list(years)
        self.k, self.power = k, power
        self.T = np.array([float(camp.loc[camp["year"] == y, "t"].median())
                           for y in self.years])
        piv = camp.pivot_table(index="sid", columns="year", values="elev")
        xy = camp.groupby("sid")[["x", "y"]].first()
        self.inc = []
        self.n_dropped = 0
        for y0, y1 in zip(self.years[:-1], self.years[1:], strict=True):
            d = (piv[y1] - piv[y0]).dropna()
            bad = d.abs() > reset_per_year * (y1 - y0)
            self.n_dropped += int(bad.sum())
            d = d[~bad]
            self.inc.append((xy.loc[d.index].to_numpy(), d.to_numpy()))
        self.trees = [cKDTree(p) if len(p) else None for p, _ in self.inc]


def campaign_years(obs: pd.DataFrame) -> tuple[list[int], dict]:
    dd = DESIGN["displacement"]
    m = obs[obs["month"].isin(dd["main_months"])
            & obs["year"].between(dd["first_year"], dd["last_year"])]
    n = m.groupby("year")["sid"].nunique()
    years = [int(y) for y, c in n.items() if c >= dd["min_benchmarks_per_campaign"]]
    return years, {int(y): int(c) for y, c in n.items()}


def displacement_model() -> tuple[GapScaledDisplacement, dict]:
    obs = leveling.load_panel(str(LEVELING))
    years, counts = campaign_years(obs)
    camp = leveling.campaigns(obs, range(min(years), max(years) + 1))
    camp = camp[camp["year"].isin(years)]
    dm = GapScaledDisplacement(camp, years,
                               reset_per_year=DESIGN["displacement"]["reset_m_per_year_of_gap"])
    info = {"years": years, "main_campaign_benchmarks": counts, "epochs": [float(v)
                                                                           for v in dm.T],
            "increments_per_pair": dm.n_increments, "resets_dropped": dm.n_dropped}
    return dm, info


def check_equals_w3_model(pts: np.ndarray, t: np.ndarray) -> float:
    """max |u_gap-scaled - u_W3| over points and epochs for the W3 years 2011-2021."""
    obs = leveling.load_panel(str(LEVELING))
    camp = leveling.campaigns(obs, validation.DISP_YEARS)
    a = leveling.DisplacementModel(camp, validation.DISP_YEARS)
    b = GapScaledDisplacement(camp, list(validation.DISP_YEARS))
    ua, _ = a.u(pts, t)
    ub, _ = b.u(pts, t)
    return float(np.nanmax(np.abs(ua - ub)))


# ------------------------------------------------------------------ residuals and statistics
def corrections(dm, t: pd.DataFrame, epoch_col: str) -> np.ndarray:
    """u(t_s) - u(t_sheet) at each station (NaN where an epoch is missing)."""
    out = np.full(len(t), np.nan)
    ok = np.isfinite(t[epoch_col].to_numpy()) & np.isfinite(t["dtm_point_epoch"].to_numpy()) \
        & np.isfinite(t["x"].to_numpy())
    if ok.any():
        P = t.loc[ok, ["x", "y"]].to_numpy(dtype="float64")
        U, r_first, r_last = dm.cumulative(P)
        ts = t.loc[ok, epoch_col].to_numpy(dtype="float64")
        te = t.loc[ok, "dtm_point_epoch"].to_numpy(dtype="float64")
        vals = np.empty(len(P))
        for i in range(len(P)):
            u = dm.interp(U[i], r_first[i], r_last[i], np.array([ts[i], te[i]]))
            vals[i] = u[0] - u[1]
        out[ok] = vals
    return out


def well_residuals(t: pd.DataFrame, dm, groundheight: np.ndarray | None = None) -> pd.DataFrame:
    """Residual columns for every station (computed for all; the gate uses eligible ones).
    ``groundheight`` overrides the agency values (synthetic checks only)."""
    t = t.copy()
    gh = pd.to_numeric(t["GroundHeight"], errors="coerce").to_numpy(dtype="float64") \
        if groundheight is None else np.asarray(groundheight, dtype="float64")
    t["gh_used"] = gh
    t["corr_establish"] = corrections(dm, t, "t_establish")
    t["corr_modification"] = corrections(dm, t, "t_modification")
    t["r_primary"] = t["dtm_point_sheet_m"] + t["corr_establish"] - gh
    t["r_modification_epoch"] = t["dtm_point_sheet_m"] + t["corr_modification"] - gh
    t["r_uncorrected"] = t["dtm_point_sheet_m"] - gh
    return t


def site_values(t: pd.DataFrame, col: str) -> pd.DataFrame:
    d = t[np.isfinite(t[col])]
    return d.groupby("site").agg(r=(col, "median"), x=("x", "median"), y=("y", "median"),
                                 n_wells=("sid", "size")).reset_index()


def summarize(v) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    q25, q75 = np.percentile(v, [25, 75])
    med = float(np.median(v))
    return {"n": int(len(v)), "median": med, "mean": float(v.mean()),
            "iqr": [float(q25), float(q75)],
            "mad_sd": float(1.4826 * np.median(np.abs(v - med))),
            "frac_abs_gt_0.30": float(np.mean(np.abs(v) > 0.30)),
            "frac_abs_gt_1m": float(np.mean(np.abs(v) > 1.0)),
            "share_positive": float(np.mean(v > 0))}


def block_bootstrap(sites: pd.DataFrame, x0: float, y0: float) -> list[float]:
    """Block-bootstrap 90 % interval of the median over sites (10 km blocks anchored at the
    twin grid origin; 2000 draws; seed 1000)."""
    b = validation.block_ids(sites["x"].to_numpy(), sites["y"].to_numpy(), x0, y0)
    groups = {k: sites.loc[b == k, "r"].to_numpy() for k in np.unique(b)}
    keys = np.array(sorted(groups))
    rng = np.random.default_rng(DESIGN["boot_seed"])
    meds = []
    for _ in range(DESIGN["boot_n"]):
        pick = rng.choice(keys, size=len(keys), replace=True)
        meds.append(float(np.median(np.concatenate([groups[k] for k in pick]))))
    return [float(np.percentile(meds, 5)), float(np.percentile(meds, 95))]


def evaluate(t: pd.DataFrame, x0: float, y0: float) -> dict:
    """T1 (fan, second look) and T2 (non-fan, fresh) on eligible stations."""
    e = t[t["eligible"]]
    out = {}
    for name, s in (("T1_fan_secondlook", "fan"), ("T2_nonfan_fresh", "nonfan")):
        d = e[e["set"] == s]
        sv = site_values(d, "r_primary")
        rec = {"sites": summarize(sv["r"]), "wells": summarize(d["r_primary"]),
               "n_sites": int(len(sv)), "n_wells": int(np.isfinite(d["r_primary"]).sum())}
        if s == "fan":
            rec["evaluable"] = bool(len(sv) >= DESIGN["min_sites_T1"])
            med = rec["sites"].get("median", float("nan"))
            rec["passed"] = bool(rec["evaluable"] and abs(med) <= THRESHOLD_M)
            rec["block_bootstrap_90"] = block_bootstrap(sv, x0, y0) if len(sv) >= 2 else None
            rec["n_blocks"] = int(len(np.unique(validation.block_ids(
                sv["x"].to_numpy(), sv["y"].to_numpy(), x0, y0)))) if len(sv) else 0
        else:
            med = rec["sites"].get("median", float("nan"))
            rec["consistent_with_threshold"] = bool(np.isfinite(med) and abs(med) <= THRESHOLD_M)
            rec["per_site"] = sv.assign(r=sv["r"].round(4)).to_dict(orient="records")
        out[name] = rec
    return out


SECONDARY = (
    "block-bootstrap 90 % interval of T1 (10 km blocks at the twin grid origin, 2000 draws, "
    "seed 1000)",
    "T1 per well instead of per site",
    "T1 with the survey epoch at ModificationDate instead of EstablishDate",
    "T1 without epoch correction",
    "T1 with the W3 product's own displacement model (2011-2021), stations established "
    "2011-01-01 or later",
    "T1 with the containing 1 km cell mean (v3 z_sheet_mean) instead of the DTM point "
    "(documents cell-scale siting at wells)",
    "T1 on every establishment date up to 2022-12-31 (displacement extrapolated before the "
    "2005 campaign with the first increment's rate; pre-TWVD2001 surveys included)",
    "T1 strata: Changhua (sid 07), Yunlin (sid 09), other counties; model wells or not",
    "agency stick-up WellElevation - GroundHeight of the eligible stations (metadata)",
    "T2 per site",
)


def secondary(t: pd.DataFrame, dm_w3, x0: float, y0: float) -> dict:
    """Registered secondary metrics (never gating). ``t`` carries the residual columns of
    :func:`well_residuals` and the eligibility columns."""
    out = {}
    e = t[t["eligible"] & (t["set"] == "fan")].copy()
    sv = site_values(e, "r_primary")
    out["T1_block_bootstrap_90"] = block_bootstrap(sv, x0, y0) if len(sv) >= 2 else None
    out["T1_per_well"] = summarize(e["r_primary"])
    out["T1_modification_epoch"] = summarize(site_values(e, "r_modification_epoch")["r"])
    out["T1_uncorrected"] = summarize(site_values(e, "r_uncorrected")["r"])
    # the product's own model, interpolation range only
    e11 = e[e["t_establish"] >= 2011.0].copy()
    e11["corr_w3"] = corrections(dm_w3, e11, "t_establish")
    e11["r_w3model"] = e11["dtm_point_sheet_m"] + e11["corr_w3"] - e11["gh_used"]
    out["T1_w3_model_2011plus"] = summarize(site_values(e11, "r_w3model")["r"])
    # containing cell mean
    cells = pd.read_parquet(V3 / "ground_elevation_cells.parquet", columns=["cell",
                                                                            "z_sheet_mean"])
    zc = cells.set_index("cell")["z_sheet_mean"]
    ec = e[e["cell"] >= 0].copy()
    ec["r_cell"] = zc.reindex(ec["cell"]).to_numpy() + ec["corr_establish"] - ec["gh_used"]
    out["T1_cell_mean"] = summarize(site_values(ec, "r_cell")["r"])
    # every establishment date (E4 lower bound dropped; E4 upper bound and E1-E3, E5, E6 kept)
    rules = ["E1_coords_dtm", "E2_groundheight", "E3_stickup", "E5_flat",
             "E6_no_postorigin_edit"]
    est = pd.to_numeric(t["EstablishDate"], errors="coerce")
    ea = t[(t["set"] == "fan") & t[rules].all(axis=1)
           & (est <= int(DESIGN["establish_window"][1]))]
    out["T1_all_establish_dates"] = summarize(site_values(ea, "r_primary")["r"])
    out["T1_all_establish_dates_pre2005_only"] = summarize(site_values(
        ea[pd.to_numeric(ea["EstablishDate"], errors="coerce")
           < int(DESIGN["establish_window"][0])], "r_primary")["r"])
    strata = {}
    for lab, sel in (("changhua_sid07", e["sid"].str.startswith("07")),
                     ("yunlin_sid09", e["sid"].str.startswith("09")),
                     ("other_counties", ~e["sid"].str[:2].isin(["07", "09"])),
                     ("model_well", e["model_well"].astype(bool)),
                     ("not_model_well", ~e["model_well"].astype(bool))):
        strata[lab] = summarize(site_values(e[sel.to_numpy()], "r_primary")["r"])
    out["T1_strata_sites"] = strata
    we = pd.to_numeric(e["WellElevation"], errors="coerce")
    out["T1_stickup_metadata"] = summarize(we - e["gh_used"])
    out["T1_survey_epoch_years"] = summarize(e["t_establish"])
    out["T1_epoch_correction_m"] = summarize(e["corr_establish"])
    return out


def grid_origin() -> tuple[float, float]:
    with np.load(TWIN_GRID_NPZ, allow_pickle=False) as z:
        return float(z["x0"]), float(z["y0"])


def synthetic_check(t: pd.DataFrame, dm, delta: float = 0.25) -> dict:
    """Sign-convention check with synthetic agency values: GroundHeight := DTM point moved to
    the survey epoch, minus ``delta``. The primary statistic must return +delta exactly and
    the uncorrected variant must differ from it by the correction. No agency value is used."""
    corr = corrections(dm, t, "t_establish")
    synth = t["dtm_point_sheet_m"].to_numpy() + corr - delta
    r = well_residuals(t, dm, groundheight=synth)
    r["eligible"] = t["eligible"].to_numpy() & np.isfinite(synth)
    x0, y0 = grid_origin()
    ev = evaluate(r, x0, y0)
    return {"delta": delta, "T1_median": ev["T1_fan_secondlook"]["sites"].get("median"),
            "T2_median": ev["T2_nonfan_fresh"]["sites"].get("median"),
            "max_abs_dev_primary": float(np.nanmax(np.abs(r.loc[r["eligible"], "r_primary"]
                                                          - delta)))}


def code_files() -> list[Path]:
    return [HERE, HERE.parent / "leveling.py", HERE.parent / "mosaic.py",
            HERE.parent / "validation.py", HERE.parent / "dtm_io.py"]


def input_files() -> list[Path]:
    return [*(V3 / f for f in V3_FILES), V3 / "MANIFEST.json", FAN, WRA_LIST, LEVELING,
            TWIN_GRID_NPZ]


def rel(p) -> str:
    return rel_to_repo(p)
