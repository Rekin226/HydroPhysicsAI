"""WP C0: the Taiwan Power pump-electricity census for every county with pumps in the twin's
active cells (Yunlin, Changhua, the in-grid Chiayi pumps), cleaned exactly as the model cleans
Yunlin, on the twin's 1 km grid.

Conventions (fixed before any acceptance number was computed; see the C0 pre-registration):

* **L0.** ``tpc_changhua_yunlin_chiayi_pingtung_2026/parquet/<county>_census.parquet`` and
  ``<county>_kwh_long.parquet`` only. ``raw/`` and ``extracted/`` (names, addresses) are never
  opened. Census columns are read with explicit lists that never include 戶名1, 用電地址2 or
  郵遞區號 (plan 2.6); the raw meter number is used in memory for de-duplication and never
  written (outputs carry a county-scoped integer ``meter_idx``). The AMP_V2 Yunlin cache (used
  only by the reproduction checks) is read with the same kind of explicit column list.
* **Cleaning** = ``hydrophysics.twin.pumping.clean_census`` as the model applies it: the kept
  set is ``clean_census(cap_duty=1.0, t0=2012-01-01, t1=2023-01-01)`` (shared meters counted
  once, energy split by rated HP, meters whose mean 2012-2022 draw exceeds 1.0 x HP x 0.746 kW
  x 730 h dropped whole); the kept pumps' record is then re-shared with ``cap_duty=None``
  (the convention of ``forcing_experiment.observed_energy``), which equals the model's own
  allocation on 2012-2022 because the HP shares depend on the kept table only.
* **Pre-install coding (decision recorded for P1's open issue).** The duty statistic sees the
  L0 readings as delivered (NaN months are left out of the mean, zero months enter it), which
  is what the model does for Yunlin. The Yunlin sheet codes most pre-install months 0, the
  Changhua and Chiayi sheets NaN, so Changhua's duty mean runs over its post-install months
  only. The alternative (every pre-install month coded 0 before the duty statistic, in every
  county) is reported as a sensitivity (kept pumps and energy), not used: it changes the
  Yunlin kept set by 2 meters and would break the reproduction of the calibrated forcing.
  In the product every pre-install month is a structural zero (value 0, flag 0, method
  ``pre_install``), never imputed.
* **Classes** = ``scenario.purpose_class`` of the raw purpose string (Chiayi: 用水標的).
* **Cells** = ``pumping._cell_lookup`` on the grid of ``results/twin_forward/datum_gate_innov.npz``
  (pumps outside the grid or mask are dropped from cell arrays, never snapped).
* **Variants.** ``preorigin``: the kWh table is truncated to <= 2022-12 before cleaning; every
  fitted or classified quantity (duty statistic, billing-phase detection, termination) uses data
  <= 2022-12. ``full``: every month to 2026-07; 2026-07 is masked (incomplete billing pairs NaN,
  flag 8); termination and phase detection use months <= 2026-06. A phase-B (M1) meter's
  December value is half of a December-January bill (the TPC even split, as in the model's
  forcing), so the pre-origin variant's 2022-12 carries half a month of January 2023
  electricity for those meters; this is the model's own convention and is disclosed, not
  changed.
* **Flags** (plan 2.4) and per pump-month ``method`` codes are in :data:`METHODS`.

Privacy: no output of this WP carries a name, an address, a postal code or a raw meter number.
Pump coordinates appear only in the gitignored L1 table; well numbers (``uid``) appear in the
L1 table and in the gitignored per-pump L2 arrays (plan 2.6).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "dataprep_staging"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hydrophysics.twin.pumping import (  # noqa: E402
    HOURS_PER_MONTH,
    KW_PER_HP,
    METER_COL,
    _cell_lookup,
    clean_census,
)
from hydrophysics.twin.scenario import CLASSES, purpose_class  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
L0_DIR = REPO / "data_fetch_api" / "new_measurements" / "tpc_changhua_yunlin_chiayi_pingtung_2026"
PARQ = L0_DIR / "parquet"
SOURCE_JSON = L0_DIR / "SOURCE.json"
AMP_CENSUS = REPO / "AMP_V2" / "data" / "tpc_pumps.parquet"
AMP_KWH = REPO / "AMP_V2" / "data" / "pump_kwh_all.parquet"
GRID_NPZ = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"
MODEL_INPUTS_PKL = REPO / "results" / "twin_runs" / "stage3_datum_gate_cwa" / "inputs.local.pkl"
DATAPREP = REPO / "results" / "dataprep"
PRODUCT_ROOT = DATAPREP / "pump_electricity"
WORK = PRODUCT_ROOT / "_build_c0"
EVIDENCE = DATAPREP / "reports" / "C0_evidence"
P1_CENSUS_CHECK = DATAPREP / "county_electricity" / "census_check" / "v2"
P1_TPC_OBSERVED = DATAPREP / "county_electricity" / "tpc_observed" / "v1"
PREREG_ID = "C0.census_product.v1"
WP = "C0"

DATES = pd.date_range("2007-01-01", "2026-07-01", freq="MS")      # 235 months, L0 time base
T_ALL = len(DATES)
DUTY_T0, DUTY_T1 = "2012-01-01", "2023-01-01"                      # the model's duty window
CAP_DUTY = 1.0
#: a cell-month is flagged 9 when more than this fraction of the installed HP of its kept
#: pumps has a missing (NaN) reading after installation (fixed a priori, not fitted)
HP_MISSING_FLAG_FRAC = 0.10
TERMINATION_MIN_RUN = 12                                            # months (plan P2c)
PAIR_ATOL = 0.51                                                    # kWh, identical-pair test

VARIANTS = {
    "preorigin": {"end": "2022-12", "last_complete": "2022-12",
                  "phase_window": ("2012-01", "2022-12"), "mask_last": False},
    "full": {"end": "2026-07", "last_complete": "2026-06",
             "phase_window": ("2012-01", "2026-06"), "mask_last": True},
}

COUNTIES = ("yunlin", "changhua", "chiayi")
COUNTY_ZH = {"yunlin": "雲林縣", "changhua": "彰化縣", "chiayi": "嘉義縣"}
FORBIDDEN_COLS = ("戶名1", "用電地址2", "郵遞區號")

#: explicit L0 column lists and the harmonised field each one maps to (plan 2.6)
SCHEMA = {
    "yunlin": {"sid": "WELL_NO", "meter": "電號1", "hp": "PUMP_HP", "purpose": "PURPOSE",
               "x": "TWD97_X", "y": "TWD97_Y", "town": "TOWN", "install": "新設日期1",
               "billing": "每期用電度數1", "station": "工作站", "district": "灌區",
               "tube": "W_TUBE_DIA", "depth": None},
    "changhua": {"sid": "WELL_NO", "meter": "電號1", "hp": "PUMP_HP", "purpose": "PURPOSE",
                 "x": "TWD97_X", "y": "TWD97_Y", "town": "TOWN", "install": "新設日期1",
                 "billing": "每期用電度數1", "station": "工作站", "district": "灌區",
                 "tube": "W_TUBE_DIA", "depth": None},
    # Chiayi has no WELL_NO, no PURPOSE and no irrigation station columns; 電號 == 電號2 on
    # every row (checked in build); the purpose is 用水標的 and the rated power HP.
    "chiayi": {"sid": None, "meter": "電號", "hp": "HP", "purpose": "用水標的",
               "x": "TWD97_X", "y": "TWD97_Y", "town": "TOWN", "install": "新設日期1",
               "billing": "每期用電度數1", "station": None, "district": None,
               "tube": None, "depth": "深度(m)"},
}

#: per pump-month method codes (priority order in :func:`pump_methods`)
METHODS = {
    0: ("observed", 0),
    1: ("pre_install", 0),                 # structural zero before the meter's first install
    2: ("observed_before_install_date", 0),  # positive reading before the recorded install
    3: ("missing", 8),                     # NaN reading after installation; not filled
    4: ("incomplete_pair_2026_07", 8),     # phase-A (M2) bill Jul-Aug 2026 not yet issued
    5: ("partial_reading_cycle_2026_07", 9),  # monthly meters (M0/M9) in the partial month
    6: ("terminated_trailing_run", 0),     # trailing zero run >= 12 months at the variant end
    7: ("bill_in_second_month", 0),        # bimonthly bill posted wholly to the 2nd month
    8: ("uneven_pair", 0),                 # bimonthly pair recorded with unequal halves
}
METHOD_CODE = {v[0]: k for k, v in METHODS.items()}


# ----------------------------------------------------------------------------- small helpers
def tidx(month: str) -> int:
    """Index of 'YYYY-MM' on :data:`DATES`."""
    return int(DATES.get_loc(pd.Timestamp(month + "-01")))


def ym(i: int) -> str:
    return DATES[int(i)].strftime("%Y-%m")


def save_json(obj, path) -> None:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=default)


def save_npz(path, **arrays) -> None:
    """``np.savez_compressed`` with fixed zip timestamps and entry order, so the same arrays
    give the same bytes (byte-reproducible products)."""
    import zipfile
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name in sorted(arrays):
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(arrays[name]), allow_pickle=False)
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rel(p) -> str:
    """Repository-relative posix path."""
    return Path(p).resolve().relative_to(REPO).as_posix()


def load_grid() -> SimpleNamespace:
    g = np.load(GRID_NPZ, allow_pickle=False)
    mask = g["mask"].astype(bool)
    return SimpleNamespace(mask=mask, nx=int(g["nx"]), ny=int(g["ny"]), dx=float(g["dx"]),
                           x0=float(g["x0"]), y0=float(g["y0"]), n_active=int(mask.sum()))


def grid_meta(grid) -> dict:
    from common.hashing import grid_hash, mask_hash
    return {"nx": grid.nx, "ny": grid.ny, "dx": grid.dx, "x0": grid.x0, "y0": grid.y0,
            "n_active": grid.n_active, "mask_hash": mask_hash(grid.mask),
            "grid_hash": grid_hash(grid.nx, grid.ny, grid.dx, grid.x0, grid.y0, grid.mask),
            "source": "results/twin_forward/datum_gate_innov.npz",
            "active_order": "row-major over masked cells (pumping._cell_lookup)"}


def roc_install_month(v: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """ROC yyymmdd (string or number) -> (month index on DATES, may be negative or > T;
    valid mask). Month 0 is read as January (as the billing pilot did); months > 12 and
    unparsable values are invalid."""
    x = pd.to_numeric(v, errors="coerce").to_numpy(dtype="float64")
    ok = np.isfinite(x) & (x > 0)
    xi = np.where(ok, x, 0).astype("int64")
    y = xi // 10000 + 1911
    m = (xi // 100) % 100
    m = np.where(m == 0, 1, m)
    ok &= (m >= 1) & (m <= 12) & (y >= 1911) & (y <= 2100)
    t = (y - 2007) * 12 + (m - 1)
    return np.where(ok, t, np.iinfo("int64").min // 2), ok


# ----------------------------------------------------------------------------- census
def read_census_l0(tag: str) -> pd.DataFrame:
    """The county census with an explicit, privacy-safe column list."""
    s = SCHEMA[tag]
    cols = ["row_idx"] + [c for c in (s["sid"], s["meter"], s["hp"], s["purpose"], s["x"],
                                      s["y"], s["town"], s["install"], s["billing"],
                                      s["station"], s["district"], s["tube"], s["depth"])
                          if c is not None]
    if tag == "chiayi":
        cols.append("電號2")
    assert not set(cols) & set(FORBIDDEN_COLS)
    df = pd.read_parquet(PARQ / f"{tag}_census.parquet", columns=cols)
    if not (df["row_idx"].to_numpy() == np.arange(len(df))).all():
        raise SystemExit(f"{tag}: row_idx is not 0..n-1")
    return df


def _str_or_none(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.where(~out.isin(["", "nan", "None", "NaN", "<NA>"]), pd.NA)


def harmonise(tag: str, raw: pd.DataFrame, grid) -> pd.DataFrame:
    """Harmonised pump table (one row per census row) plus the model's own column names
    (``sid``, ``PUMP_HP``, ``電號1``, ``PURPOSE``, ``TWD97_X``, ``TWD97_Y``) used to call
    ``clean_census`` / ``_cell_lookup``. The ``電號1`` column holds the raw meter number in
    memory only; :func:`l1_columns` never selects it."""
    s = SCHEMA[tag]
    n = len(raw)
    if s["sid"] is not None:
        sid = raw[s["sid"]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    else:
        sid = pd.Series([f"r{i}" for i in range(n)])
    if sid.duplicated().any():
        raise SystemExit(f"{tag}: sid not unique")
    meter_raw = raw[s["meter"]]
    if tag == "chiayi" and not (raw["電號"].fillna(-1) == raw["電號2"].fillna(-1)).all():
        raise SystemExit("chiayi: 電號 differs from 電號2")
    df = pd.DataFrame({
        "county": tag,
        "county_zh": COUNTY_ZH[tag],
        "row_idx": raw["row_idx"].to_numpy(),
        "sid": sid.to_numpy(),
    })
    df["uid"] = tag + ":" + df["sid"]
    df["town"] = _str_or_none(raw[s["town"]]).to_numpy()
    df["work_station"] = (_str_or_none(raw[s["station"]]).to_numpy() if s["station"]
                          else pd.array([pd.NA] * n, dtype="string"))
    df["irrigation_district"] = (_str_or_none(raw[s["district"]]).to_numpy() if s["district"]
                                 else pd.array([pd.NA] * n, dtype="string"))
    df["x_twd97"] = pd.to_numeric(raw[s["x"]], errors="coerce").to_numpy(dtype="float64")
    df["y_twd97"] = pd.to_numeric(raw[s["y"]], errors="coerce").to_numpy(dtype="float64")
    df["pump_hp_raw"] = _str_or_none(raw[s["hp"]].astype(str)).to_numpy()
    df["pump_hp"] = pd.to_numeric(raw[s["hp"]], errors="coerce").to_numpy(dtype="float64")
    df["purpose_raw"] = raw[s["purpose"]].astype(str).to_numpy()
    df["purpose_class"] = raw[s["purpose"]].map(purpose_class).to_numpy()
    df["billing_code"] = _str_or_none(raw[s["billing"]]).to_numpy()
    df["install_raw"] = _str_or_none(raw[s["install"]].astype(str)).to_numpy()
    t_inst, ok = roc_install_month(raw[s["install"]])
    df["install_month"] = [ym(t) if (v and 0 <= t < T_ALL) else
                           (f"{2007 + t // 12:04d}-{t % 12 + 1:02d}" if v else None)
                           for t, v in zip(t_inst, ok, strict=True)]
    df["install_valid"] = ok
    df["tube_dia_raw"] = (_str_or_none(raw[s["tube"]].astype(str)).to_numpy() if s["tube"]
                          else pd.array([pd.NA] * n, dtype="string"))
    df["well_depth_m"] = (pd.to_numeric(raw[s["depth"]], errors="coerce").to_numpy()
                          if s["depth"] else np.nan)
    # meter key exactly as clean_census builds it: str(meter), replaced by the sid where the
    # string is 'nan' or empty (clean_census tests the string, not the raw value)
    mstr = meter_raw.astype(str)
    mkey = mstr.where((mstr != "nan") & (mstr != ""), df["sid"])
    df["meter_idx"] = pd.factorize(mkey)[0].astype("int64")
    df["meter_n_pumps"] = df.groupby("meter_idx")["sid"].transform("size").to_numpy()
    df["meter_first_pump"] = ~df["meter_idx"].duplicated().to_numpy()
    # grid
    cell_of = _cell_lookup(pd.DataFrame({"sid": df["sid"], "TWD97_X": raw[s["x"]],
                                         "TWD97_Y": raw[s["y"]]}), grid)
    df["cell"] = df["sid"].map(cell_of).fillna(-1).astype("int64").to_numpy()
    col = np.floor((df["x_twd97"] - grid.x0) / grid.dx)
    row = np.floor((df["y_twd97"] - grid.y0) / grid.dx)
    inb = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < grid.nx) & (row >= 0) \
        & (row < grid.ny)
    df["grid_col"] = np.where(inb, col, -1).astype("int64")
    df["grid_row"] = np.where(inb, row, -1).astype("int64")
    df["in_grid"] = df["cell"] >= 0
    # model columns (memory only)
    df["PUMP_HP"] = raw[s["hp"]].to_numpy()
    df[METER_COL] = meter_raw.to_numpy()
    df["PURPOSE"] = raw[s["purpose"]].to_numpy()
    df["TWD97_X"] = raw[s["x"]].to_numpy()
    df["TWD97_Y"] = raw[s["y"]].to_numpy()
    df["_t_install"] = t_inst
    return df


MODEL_COLS = ["sid", "PUMP_HP", METER_COL, "PURPOSE", "TWD97_X", "TWD97_Y"]


def model_pumps(df: pd.DataFrame) -> pd.DataFrame:
    """The columns ``clean_census`` and ``_cell_lookup`` read, in census order."""
    return df[MODEL_COLS].copy()


def l1_columns() -> list[str]:
    """Columns written to the L1 census table (no raw meter number, no personal field)."""
    return ["county", "county_zh", "row_idx", "sid", "uid", "town", "work_station",
            "irrigation_district", "x_twd97", "y_twd97", "grid_row", "grid_col", "cell",
            "in_grid", "pump_hp_raw", "pump_hp", "purpose_raw", "purpose_class",
            "billing_code", "install_raw", "install_month", "install_valid", "tube_dia_raw",
            "well_depth_m", "meter_idx", "meter_n_pumps", "meter_first_pump"]


# ----------------------------------------------------------------------------- kWh
def read_kwh_l0(tag: str, sids: np.ndarray, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    """(long table for clean_census: datetime, electricity_kwh, pump; dense (n, T_ALL) L0
    matrix with NaN where L0 is NaN)."""
    k = pd.read_parquet(PARQ / f"{tag}_kwh_long.parquet",
                        columns=["row_idx", "datetime", "electricity_kwh"])
    k["datetime"] = pd.to_datetime(k["datetime"])
    ri = k["row_idx"].to_numpy()
    ti = ((k["datetime"].dt.year - 2007) * 12 + k["datetime"].dt.month - 1).to_numpy()
    if ti.min() < 0 or ti.max() >= T_ALL:
        raise SystemExit(f"{tag}: kWh months outside {DATES[0]}..{DATES[-1]}")
    L = np.full((n, T_ALL), np.nan)
    L[ri, ti] = k["electricity_kwh"].to_numpy(dtype="float64")
    if len(k) != n * T_ALL or np.unique(ri * T_ALL + ti).size != len(k):
        raise SystemExit(f"{tag}: kWh table is not one row per pump-month")
    k["pump"] = np.asarray(sids, dtype=object)[ri]
    return k[["datetime", "electricity_kwh", "pump"]], L


#: AMP_V2 Yunlin cache columns read by the reproduction checks (no personal column; the cache
#: also holds 戶名1, 郵遞區號 and 用電地址2, which are never read)
AMP_COLS = ["sid", "TOWN", "工作站", "灌區", "TWD97_X", "TWD97_Y", "PUMP_HP", "W_TUBE_DIA",
            "PURPOSE", "電號1", "新設日期1", "每期用電度數1"]


def read_amp_census() -> pd.DataFrame:
    """The AMP_V2 Yunlin census (explicit columns, cache row order) renamed to the L0 Yunlin
    schema so that :func:`harmonise` applies unchanged."""
    assert not set(AMP_COLS) & set(FORBIDDEN_COLS)
    a = pd.read_parquet(AMP_CENSUS, columns=AMP_COLS).rename(columns={"sid": "WELL_NO"})
    a.insert(0, "row_idx", np.arange(len(a), dtype="int64"))
    return a


def read_amp_kwh(sids: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """(the AMP_V2 kWh long table as the model reads it; dense (n, T_ALL) matrix aligned on
    ``sids`` with NaN where the cache has no row or a NaN; info)."""
    k = pd.read_parquet(AMP_KWH, columns=["datetime", "electricity_kwh", "pump"])
    k["datetime"] = pd.to_datetime(k["datetime"])
    k["pump"] = k["pump"].astype(str)
    pi = pd.Index(np.asarray(sids, dtype=object)).get_indexer(k["pump"])
    ti = ((k["datetime"].dt.year - 2007) * 12 + k["datetime"].dt.month - 1).to_numpy()
    ok = (pi >= 0) & (ti >= 0) & (ti < T_ALL)
    key = pi[ok].astype("int64") * T_ALL + ti[ok]
    L = np.full((len(sids), T_ALL), np.nan)
    L[pi[ok], ti[ok]] = k["electricity_kwh"].to_numpy(dtype="float64")[ok]
    info = {"rows": int(len(k)), "rows_unmatched_pump": int((pi < 0).sum()),
            "rows_outside_dates": int(((pi >= 0) & ~ok).sum()),
            "duplicate_pump_months": int(len(key) - np.unique(key).size),
            "first_month": str(k["datetime"].min())[:7], "last_month": str(k["datetime"].max())[:7]}
    return k, L, info


def dense_from_long(kc: pd.DataFrame, sids: np.ndarray, T: int) -> np.ndarray:
    """(P, T) matrix of a clean_census kWh output (NaN where it has no row)."""
    pi = pd.Index(np.asarray(sids, dtype=object)).get_indexer(kc["pump"].astype(str))
    dt = pd.to_datetime(kc["datetime"])
    ti = ((dt.dt.year - 2007) * 12 + dt.dt.month - 1).to_numpy()
    if (pi < 0).any() or ti.min() < 0 or ti.max() >= T:
        raise ValueError("clean_census output does not map onto the kept pumps / months")
    V = np.full((len(sids), T), np.nan)
    V[pi, ti] = kc["electricity_kwh"].to_numpy(dtype="float64")
    if np.unique(pi.astype("int64") * T + ti).size != len(kc):
        raise ValueError("duplicate pump-months in clean_census output")
    return V


def hp_numeric(pump_hp) -> np.ndarray:
    """``_hp`` exactly as clean_census computes it."""
    return pd.to_numeric(pd.Series(pump_hp), errors="coerce").fillna(0.0).clip(lower=0.0) \
        .to_numpy(dtype="float64")


def hp_shares(meter_idx: np.ndarray, hp: np.ndarray) -> np.ndarray:
    """HP share of each pump on its meter (even split when the meter's HP sum is 0), the
    rule of clean_census applied to the kept table."""
    s = pd.Series(hp)
    tot = s.groupby(meter_idx).transform("sum").to_numpy()
    n = pd.Series(np.ones(len(hp))).groupby(meter_idx).transform("sum").to_numpy()
    return np.where(tot > 0, hp / np.where(tot > 0, tot, 1.0), 1.0 / n)


def meter_duty(L_first: np.ndarray, hp_sum: np.ndarray) -> np.ndarray:
    """Mean monthly kWh over 2012-01..2022-12 (non-NaN months) / (HP x 0.746 x 730); NaN for
    a meter with no reading or zero HP (clean_census keeps those)."""
    w = slice(tidx("2012-01"), tidx("2022-12") + 1)
    X = L_first[:, w]
    cnt = np.isfinite(X).sum(1)
    mean = np.where(cnt > 0, np.nansum(X, 1) / np.maximum(cnt, 1), np.nan)
    cap = hp_sum * KW_PER_HP * HOURS_PER_MONTH
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cap > 0, mean / np.where(cap > 0, cap, 1.0), np.nan)


# ----------------------------------------------------------------------------- billing
def phase_from_code(code) -> np.ndarray:
    """M2 -> 'A' (pairs Jan-Feb, Mar-Apr, ...), M1 -> 'B' (Feb-Mar, ...), M0/M9 -> 'monthly',
    anything else 'unknown'."""
    c = pd.Series(code).astype("string")
    return np.select([c == "M2", c == "M1", c.isin(["M0", "M9"])], ["A", "B", "monthly"],
                     "unknown").astype(object)


def pair_starts(phase: str, t_lo: int, t_hi: int) -> np.ndarray:
    """Pair starts of a phase lying entirely inside [t_lo, t_hi) on DATES. Phase A pairs
    start in odd calendar months (index even, 2007-01 = 0), phase B in even months."""
    s = 0 if phase == "A" else 1
    st = np.arange(s, T_ALL - 1, 2)
    return st[(st >= t_lo) & (st + 1 < t_hi)]


def detect_phase(X: np.ndarray, t_lo: int, t_hi: int) -> pd.DataFrame:
    """Billing phase from the identical-pair fractions over months [t_lo, t_hi) (the billing
    pilot's rule with its fixed thresholds: a phase needs >= 0.8 identical pairs while the
    other has <= 0.3; both < 0.3 is monthly; both >= 0.8 'both-flat'; < 6 valid pairs
    'insufficient')."""
    out = {}
    for ph in ("A", "B"):
        st = pair_starts(ph, t_lo, t_hi)
        a, b = X[:, st], X[:, st + 1]
        valid = np.isfinite(a) & np.isfinite(b) & ((a > 0) | (b > 0))
        eq = valid & np.isclose(a, b, rtol=0, atol=PAIR_ATOL)
        out[f"n{ph}"] = valid.sum(1)
        out[f"f{ph}"] = np.where(valid.sum(1) > 0, eq.sum(1) / np.maximum(valid.sum(1), 1),
                                 np.nan)
    fA, fB = out["fA"], out["fB"]
    hi, lo = np.fmax(fA, fB), np.fmin(fA, fB)
    ph = np.full(len(fA), "mixed", dtype=object)
    ph[lo >= 0.8] = "both-flat"
    ph[hi < 0.3] = "monthly"
    ab = (hi >= 0.8) & (lo <= 0.3)
    ph[ab & (fA > fB)] = "A"
    ph[ab & ~(fA > fB)] = "B"
    ph[(out["nA"] < 6) | (out["nB"] < 6)] = "insufficient"
    return pd.DataFrame({"fA": fA, "fB": fB, "nA": out["nA"], "nB": out["nB"],
                         "phase_detected": ph})


def termination(X: np.ndarray, t_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Per meter, over months [0, t_end]: (terminated, first month index of the trailing run
    of non-positive or missing months, -1 if none). A meter is terminated when that run is
    at least TERMINATION_MIN_RUN months long and the meter used energy before it."""
    pos = np.nan_to_num(X[:, :t_end + 1], nan=0.0) > 0
    any_pos = pos.any(1)
    last = np.where(any_pos, t_end - np.argmax(pos[:, ::-1], axis=1), -1)
    run_from = np.where(any_pos, last + 1, -1)
    run_len = np.where(any_pos, t_end - last, 0)
    term = any_pos & (run_len >= TERMINATION_MIN_RUN)
    return term, np.where(term, run_from, -1)


def pair_methods(X: np.ndarray, phase_code: np.ndarray, t_hi: int) -> np.ndarray:
    """(M, t_hi) uint8 codes for bimonthly pairs inside [0, t_hi): 7 where a pair with both
    months finite has the first month 0 and the second > 0 (bill posted to the second month),
    8 where the halves differ by more than PAIR_ATOL otherwise; 0 elsewhere. The phase is
    the billing-code phase (A for M2, B for M1)."""
    out = np.zeros((X.shape[0], t_hi), dtype=np.uint8)
    for ph in ("A", "B"):
        rows = np.where(phase_code == ph)[0]
        if rows.size == 0:
            continue
        st = pair_starts(ph, 0, t_hi)
        a, b = X[np.ix_(rows, st)], X[np.ix_(rows, st + 1)]
        fin = np.isfinite(a) & np.isfinite(b)
        second = fin & (a == 0) & (b > 0)
        uneven = fin & ~second & ~np.isclose(a, b, rtol=0, atol=PAIR_ATOL)
        for mask_, code in ((second, 7), (uneven, 8)):
            r, k = np.nonzero(mask_)
            out[rows[r], st[k]] = code
            out[rows[r], st[k] + 1] = code
    return out


def env_info() -> dict:
    import platform
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS")}


__all__ = ["CLASSES", "clean_census", "_cell_lookup"]
