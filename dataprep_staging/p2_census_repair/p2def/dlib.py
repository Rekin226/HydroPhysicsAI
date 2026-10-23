"""WP P2def (plan WP P2 parts d, e, f): shared library.

P2d  post-2016 synthetic census layer (zero / low / high members, unweighted) with the
     mid-2013 freeze back-test, for every in-grid county (Yunlin, Changhua, in-grid Chiayi);
P2e  extension of the census past its end (2026-07..2026-08) from TPC d007012 farm-sector
     sales, recorded-forcing arm only (U4);
P2f  local-only layer-response evidence at nested head sites with the ``cwa`` theta and
     regional placebos.

Everything builds on WP C0's pump_electricity v3 products (cleaned census, model units) and
WP P2abc's yunlin_electricity v2 products; nothing here edits them. The round-1/2/3 P2abc code
in ``..`` stays unchanged; this package only adds files.

Time axis: C0's, index 0 = 2007-01 (pre-origin T = 192 to 2022-12, full T = 235 to 2026-07).
Grid: the 2148 active cells of ``results/twin_forward/datum_gate_innov.npz``.
Classes C = (irrigation, aquaculture, livestock, domestic, industry, other); counties
K = (yunlin, changhua, chiayi). Units: kWh/month in model units (cleaned census, duty cap 1.0).

Privacy: pump uids and coordinates are read into memory only (to place pumps and build local
energy); no output carries a uid, coordinate, name, address or raw meter number.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parents[2]
for _p in (REPO, REPO / "dataprep_staging"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import guard  # noqa: E402

WP = "P2def"
DATAPREP = REPO / "results" / "dataprep"
C0 = {"census_L1": DATAPREP / "pump_electricity" / "census_L1" / "v3",
      "preorigin": DATAPREP / "pump_electricity" / "preorigin" / "v3",
      "full": DATAPREP / "pump_electricity" / "full" / "v3"}
P2ABC = {"preorigin": DATAPREP / "yunlin_electricity" / "preorigin" / "v2",
         "full": DATAPREP / "yunlin_electricity" / "full" / "v2"}
HEADS_L1 = DATAPREP / "heads" / "L1_clean" / "v3"
TPC_CSV = REPO / "data_fetch_api" / "new_measurements" / "tpc_d007012" / "2026-10-12" / \
    "d007012.csv"
RAIN = REPO / "data_fetch_api" / "new_measurements" / "cwa_codis_rain" / \
    "rf_timeseries_codis.csv"
RAIN_ST = REPO / "chou-shui-data" / "data" / "rf_stations.csv"
FWD_CWA = REPO / "results" / "twin_forward" / "datum_gate_cwa.npz"
FWD_CHC = REPO / "results" / "twin_forward" / "datum_gate_chc.npz"
THETA_CWA = REPO / "results" / "twin_runs" / "stage3_datum_gate_cwa" / "stage3_theta.json"
PILOT_PHYS = DATAPREP / "_pilots" / "physics_inversion_hooks"
GRID_NPZ = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"

WORK = DATAPREP / "census_post2016" / "_work_p2def"
EVID = DATAPREP / "reports" / "P2def_evidence"

COUNTIES = ("yunlin", "changhua", "chiayi")
FILETAG = {"yunlin": "yunlin", "changhua": "changhua", "chiayi": "chiayi_ingrid"}
COUNTY_ZH = {"yunlin": "雲林縣", "changhua": "彰化縣", "chiayi": "嘉義縣"}
CLASSES = ("irrigation", "aquaculture", "livestock", "domestic", "industry", "other")
DATES = pd.date_range("2007-01-01", "2026-07-01", freq="MS")      # C0's axis (235)
T_PRE = 192            # 2007-01..2022-12
T_FULL = 235           # 2007-01..2026-07
A_CELLS = 2148
Z90 = 1.6448536269514722

# ----------------------------------------------------------------------------- P2d constants
#: layer start (first month after the census freeze, plan P2d) and back-test freeze
LAYER_START = "2016-07"
BT_FREEZE = "2013-07"                 # census frozen at mid-2013: installs <= 2013-06 known
BT_RATE_YEARS = tuple(range(2007, 2013))          # 6 calendar years before the freeze year
BT_COHORT = ("2014-01", "2016-06")    # the 2014-16 cohorts the census holds
BT_KWH_END = "2022-11"                # kWh scored to 2022-11 (2022-12 holds C0's flag-4 fill)
HIGH_YEARS = tuple(range(2010, 2016))             # plan: 2010-15, 2013 surge included
LOW_YEARS = tuple(y for y in HIGH_YEARS if y != 2013)   # plan: 2013 excluded
MAX_AGE = 6                           # the ramp runs over six years, then holds
MIN_CLASS_COHORT = 200                # class-specific ramp needs >= 200 cohort pumps
MIN_TOWN_ACTUAL = 10                  # townships scored on count error need >= 10 actual


def tidx(month: str) -> int:
    return int(DATES.get_loc(pd.Timestamp(month + "-01")))


def ym(i: int) -> str:
    return DATES[int(i)].strftime("%Y-%m")


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha(*paths) -> dict:
    return {rel(p): sha256(p) for p in paths}


def p2def_code() -> dict:
    return code_sha(*sorted(CODE_DIR.glob("*.py")))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return rel(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def _clean_nan(o):
    if isinstance(o, dict):
        return {str(k): _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    return o


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_nan(obj), f, indent=1, ensure_ascii=False, default=_default)
        f.write("\n")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- inputs
def open_c0(variant: str, use: str):
    return guard.open_product(C0[variant], use)


L1_COLS = ("uid", "county", "town", "cell", "in_grid", "kept", "purpose_class",
           "install_month", "install_valid", "pump_hp", "meter_idx", "meter_hp_sum",
           "hp_share", "well_depth_m", "billing_code")


def l1_table(use: str = "prior", cols=L1_COLS, with_xy: bool = False) -> pd.DataFrame:
    """census_L1 v3 attribute columns (no personal column besides uid and, on request, the
    pump coordinates, which stay in memory)."""
    p = open_c0("census_L1", use)
    cols = list(cols) + (["x_twd97", "y_twd97"] if with_xy else [])
    return pd.read_parquet(p.path("census_pumps.parquet"), columns=cols)


def month_index(s: pd.Series) -> np.ndarray:
    """'YYYY-MM' strings -> index on C0's 2007-01 axis (may be negative or > T); invalid -> a
    large negative sentinel."""
    out = np.full(len(s), -10**6, dtype=np.int64)
    v = s.astype(str).to_numpy()
    ok = np.array([len(x) == 7 and x[4] == "-" and x[:4].isdigit() and x[5:].isdigit()
                   for x in v])
    yy = np.array([int(x[:4]) if o else 0 for x, o in zip(v, ok, strict=False)])
    mm = np.array([int(x[5:]) if o else 0 for x, o in zip(v, ok, strict=False)])
    out[ok] = (yy[ok] - 2007) * 12 + (mm[ok] - 1)
    return out


class Pumps:
    """Kept pumps of one county and variant: C0 per-pump energy (recorded arm) joined with the
    census_L1 attributes (install month, township, class, cell)."""

    def __init__(self, variant: str, county: str, use: str, l1: pd.DataFrame | None = None,
                 with_xy: bool = False):
        prod = open_c0(variant, use)
        z = np.load(prod.path(f"pumps_{FILETAG[county]}.npz"), allow_pickle=False)
        self.variant, self.county = variant, county
        self.uid = z["uid"]
        self.cell = z["cell"].astype(np.int64)
        self.cls = z["purpose_class"].astype(str)
        self.share = z["hp_share"].astype(float)
        self.hp = z["hp"].astype(float)
        self.meter_idx = z["meter_idx"].astype(np.int64)
        self.value = z["value"]                 # (P, T) kWh/month, NaN = missing reading
        self.flag = z["flag"]
        self.T = int(self.value.shape[1])
        if l1 is None:
            l1 = l1_table(with_xy=with_xy)
        sub = l1[l1["county"] == county].set_index("uid")
        s = sub.reindex(self.uid)
        if s["county"].isna().any():
            raise ValueError(f"{county}: {int(s['county'].isna().sum())} array pumps missing "
                             f"from census_L1")
        if not s["kept"].astype(bool).all():
            raise ValueError(f"{county}: array pumps that census_L1 does not keep")
        if not (s["cell"].to_numpy() == self.cell).all():
            raise ValueError(f"{county}: cell mismatch between C0 arrays and census_L1")
        self.town = s["town"].fillna("<none>").astype(str).to_numpy()
        self.inst_valid = s["install_valid"].fillna(False).astype(bool).to_numpy()
        self.inst = np.where(self.inst_valid, month_index(s["install_month"]), -10**6)
        self.meter_hp_sum = s["meter_hp_sum"].to_numpy(dtype=float)
        self.well_depth = s["well_depth_m"].to_numpy(dtype=float)
        self.billing = s["billing_code"].astype(str).to_numpy()
        if with_xy:
            self.x = s["x_twd97"].to_numpy(dtype=float)
            self.y = s["y_twd97"].to_numpy(dtype=float)

    def energy(self) -> np.ndarray:
        """Energy for aggregates: missing readings count as 0 (C0's aggregate convention)."""
        return np.nan_to_num(self.value, nan=0.0)


# ----------------------------------------------------------------------------- P2d core
def install_year(inst: np.ndarray) -> np.ndarray:
    return np.where(inst > -10**5, 2007 + np.floor_divide(inst, 12), -1)


def rates_by_town_class(town, cls, inst, years) -> pd.DataFrame:
    """Installs per year by township x class over ``years`` (kept pumps with a valid install
    month): count / len(years)."""
    iy = install_year(inst)
    sel = np.isin(iy, list(years))
    df = pd.DataFrame({"town": town[sel], "cls": cls[sel]})
    out = df.groupby(["town", "cls"]).size().rename("n_window").reset_index()
    out["rate_per_yr"] = out["n_window"] / float(len(years))
    return out


def cell_weights(town, cell, inst, years) -> pd.DataFrame:
    """Placement weights: share of each township's window installs (all classes) per 1 km
    cell; cell -1 = outside the active grid."""
    iy = install_year(inst)
    sel = np.isin(iy, list(years))
    df = pd.DataFrame({"town": town[sel], "cell": cell[sel]})
    w = df.groupby(["town", "cell"]).size().rename("n").reset_index()
    w["w"] = w["n"] / w.groupby("town")["n"].transform("sum")
    return w


def per_pump_index(E: np.ndarray, cls: np.ndarray, inst: np.ndarray, member: np.ndarray,
                   T: int) -> dict:
    """P(c, t): mean energy per installed pump of class c in month t over the pumps in
    ``member`` installed by t (pumps without a valid install month count as installed from the
    start). Returns {class: (T,) array} (NaN where no pump is installed)."""
    out = {}
    start = np.where(inst > -10**5, np.maximum(inst, 0), 0)
    for c in CLASSES:
        m = member & (cls == c)
        if not m.any():
            out[c] = np.full(T, np.nan)
            continue
        tot = E[m, :T].sum(0)
        n_inst = np.bincount(np.minimum(start[m], T), minlength=T + 1)[:T].cumsum()
        out[c] = np.where(n_inst > 0, tot / np.maximum(n_inst, 1), np.nan)
    return out


def ramp(E: np.ndarray, cls: np.ndarray, inst: np.ndarray, cohort: np.ndarray, P: dict,
         data_end: int, max_age: int = MAX_AGE, min_class: int = MIN_CLASS_COHORT) -> dict:
    """Relative age ramp g(c, a), a = 1..max_age: cohort energy over the age-a year divided by
    the contemporaneous per-pump index P(c, t) summed over the same months (so a cohort of n
    pumps of age a draws n * g(c, a) * P(c, t) in month t). Only complete age windows ending
    by ``data_end`` (index, inclusive) count. Classes with fewer than ``min_class`` cohort
    pumps with a complete window use the pooled (all-class) ramp of that age."""
    num = {c: np.zeros(max_age) for c in CLASSES}
    den = {c: np.zeros(max_age) for c in CLASSES}
    npump = {c: np.zeros(max_age, dtype=np.int64) for c in CLASSES}
    idx = np.flatnonzero(cohort)
    for a in range(1, max_age + 1):
        lo = inst[idx] + 12 * (a - 1)
        hi = lo + 12                                   # exclusive
        ok = (lo >= 0) & (hi - 1 <= data_end)
        for c in CLASSES:
            sel = idx[ok & (cls[idx] == c)]
            if not len(sel):
                continue
            Pc = np.nan_to_num(P[c], nan=0.0)
            cP = np.concatenate([[0.0], np.cumsum(Pc)])
            l0 = inst[sel] + 12 * (a - 1)
            # energy over each pump's own age window
            e = np.array([E[i, lo_: lo_ + 12].sum() for i, lo_ in zip(sel, l0, strict=True)])
            num[c][a - 1] += e.sum()
            den[c][a - 1] += (cP[l0 + 12] - cP[l0]).sum()
            npump[c][a - 1] += len(sel)
    def hold(v: np.ndarray) -> np.ndarray:
        """Ages without a complete window take the last earlier age's value (the ramp holds)."""
        v = v.copy()
        for a in range(1, len(v)):
            if not np.isfinite(v[a]):
                v[a] = v[a - 1]
        return v

    den_all = np.array([sum(den[c][a] for c in CLASSES) for a in range(max_age)])
    num_all = np.array([sum(num[c][a] for c in CLASSES) for a in range(max_age)])
    pooled = hold(np.where(den_all > 0, num_all / np.where(den_all > 0, den_all, 1.0), np.nan))
    g, src = {}, {}
    for c in CLASSES:
        gc = np.where(den[c] > 0, num[c] / np.where(den[c] > 0, den[c], 1.0), np.nan)
        use_cls = npump[c] >= min_class
        g[c] = hold(np.where(use_cls, gc, pooled))
        src[c] = ["class" if u else ("pooled" if den_all[a] > 0 else "held")
                  for a, u in enumerate(use_cls)]
    return {"g": g, "pooled": pooled, "source": src,
            "n_pumps": {c: npump[c].tolist() for c in CLASSES},
            "class_raw": {c: (num[c] / np.where(den[c] > 0, den[c], np.nan)).tolist()
                          for c in CLASSES}}


def synth_monthly(rate: pd.DataFrame, weights: pd.DataFrame, g: dict, P: dict,
                  first_cohort: int, last_cohort: int, t0: int, t1: int,
                  max_age: int = MAX_AGE) -> dict:
    """Synthetic layer: monthly cohorts of rate/12 pumps per township x class installed in
    months first_cohort..last_cohort (inclusive), placed by ``weights``; energy in months
    t0..t1 (inclusive): sum over cohorts installed by t of n * g(c, age) * P(c, t), age =
    floor((t - m) / 12) + 1 capped at max_age.

    Returns per class: 'cells' (A, n_t) in-grid energy, 'county' (n_t,) county energy (cells
    outside the grid included), 'n_cells' (A, n_t) expected in-grid pump counts, 'n_county'
    (n_t,) expected county counts."""
    ts = np.arange(t0, t1 + 1)
    nt = len(ts)
    out = {}
    wt = {tw: (g_["cell"].to_numpy(), g_["w"].to_numpy())
          for tw, g_ in weights.groupby("town")}
    for c in CLASSES:
        # cohort factor per target month: sum over installed cohorts of g(age)/12 (per unit
        # annual rate) and the count factor
        fac = np.zeros(nt)
        cnt = np.zeros(nt)
        for j, t in enumerate(ts):
            ms = np.arange(first_cohort, min(last_cohort, t) + 1)
            if not len(ms):
                continue
            age = np.minimum((t - ms) // 12 + 1, max_age)
            fac[j] = g[c][age - 1].sum() / 12.0
            cnt[j] = len(ms) / 12.0
        Pc = np.nan_to_num(P[c][ts], nan=0.0)
        e_unit = fac * Pc                        # energy per unit annual rate
        cells = np.zeros((A_CELLS, nt))
        ncell = np.zeros((A_CELLS, nt))
        county = np.zeros(nt)
        ncounty = np.zeros(nt)
        rc = rate[rate["cls"] == c]
        for tw, r in zip(rc["town"], rc["rate_per_yr"], strict=False):
            county += r * e_unit
            ncounty += r * cnt
            if tw not in wt:
                continue
            cl, w = wt[tw]
            inside = cl >= 0
            if inside.any():
                cells[cl[inside]] += np.outer(r * w[inside], e_unit)
                ncell[cl[inside]] += np.outer(r * w[inside], cnt)
        out[c] = {"cells": cells, "county": county, "n_cells": ncell, "n_county": ncounty,
                  "fac": fac, "cnt": cnt}
    return out


# ----------------------------------------------------------------------------- TPC (P2e)
def read_tpc(path=TPC_CSV) -> pd.DataFrame:
    """TPC d007012 (L0, untrusted: parsed as text, never executed): county x sector monthly
    kWh. Columns month ('YYYY-MM'), county_zh, sector, kwh (float)."""
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    raw = raw.iloc[:, :4]
    raw.columns = ["roc_ym", "county_zh", "sector", "kwh"]
    raw = raw.dropna(subset=["roc_ym", "county_zh", "sector", "kwh"])
    r = raw["roc_ym"].str.strip()
    ok = r.str.fullmatch(r"\d{4,5}")
    raw = raw[ok].copy()
    r = raw["roc_ym"].str.strip()
    yy = r.str[:-2].astype(int) + 1911
    mm = r.str[-2:].astype(int)
    raw["month"] = [f"{y:04d}-{m:02d}" for y, m in zip(yy, mm, strict=False)]
    raw["kwh"] = raw["kwh"].str.replace(",", "", regex=False).str.strip().astype(float)
    raw["county_zh"] = raw["county_zh"].str.strip().str.replace("台", "臺")
    raw["sector"] = raw["sector"].str.strip()
    return raw[["month", "county_zh", "sector", "kwh"]].reset_index(drop=True)


def tpc_farm(county: str, path=TPC_CSV) -> pd.Series:
    """Monthly TPC 農林漁牧 kWh of a county (index 'YYYY-MM')."""
    t = read_tpc(path)
    zh = COUNTY_ZH[county]
    s = t[(t["county_zh"] == zh) & (t["sector"] == "農林漁牧")].set_index("month")["kwh"]
    if s.index.duplicated().any():
        raise ValueError(f"duplicate TPC months for {zh}")
    return s.sort_index()
