"""Component builders of the L3 release.

Every component is an L3 product version directory ``<release>/<variant>/<name>/v1/`` with
its outputs, ``validation.json`` and a ``MANIFEST.json`` (schema of plan 2.3, layer L3), so
the A0 loader guard enforces its allowed uses. Bytes come only through
``guard.open_product`` on the upstream version with a declared use:

* ``copy``   - the upstream file, byte for byte (its sha256 equals the upstream manifest's);
* ``repack`` - an npz re-written deterministically: pickled members are never unpickled
  (W1's 'dates' are reconstructed), the release's value/sd/flag keys are added where the
  upstream file lacks them (P6 member, H2 IC field), every upstream key is kept;
* ``table``  - ``Product.read_table`` (time window and per-use flag filter applied by the
  guard), written as parquet, with the standard long-table columns added where missing.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from common import flags as F
from common import guard, hashing, manifest
from common import uses as U
from common.paths import DATAPREP, ORIGIN, REPO, rel_to_repo

from l3_release_b import config as C
from l3_release_b import io_det

CODE_FILES = [
    "dataprep_staging/l3_release_b/__init__.py",
    "dataprep_staging/l3_release_b/config.py",
    "dataprep_staging/l3_release_b/io_det.py",
    "dataprep_staging/l3_release_b/components.py",
    "dataprep_staging/l3_release_b/build.py",
    "dataprep_staging/l3_release_b/ledger_rows.py",
    "dataprep_staging/common/__init__.py",
    "dataprep_staging/common/guard.py",
    "dataprep_staging/common/manifest.py",
    "dataprep_staging/common/hashing.py",
    "dataprep_staging/common/flags.py",
    "dataprep_staging/common/uses.py",
    "dataprep_staging/common/paths.py",
    "dataprep_staging/common/gitinfo.py",
    C.GRID_CODE,
]

PRE_STATION_COLS = ["sid", "site_id", "WellIdentifier", "WellName", "CountyName", "TownName",
                    "x_twd97", "y_twd97", "lon", "lat", "xy_flag", "xy_method", "layer_code",
                    "layer_flag", "layer_method", "WellDepth", "FinishDepth", "WellElevation",
                    "EstablishDate"]
FULL_STATION_EXTRA = ["DisuseDate", "start_datetime_from_dataset", "end_datetime_from_dataset",
                      "in_hist_cache", "in_new_network"]
POST_ORIGIN_DATE = 20221231            # EstablishDate (yyyymmdd) after the origin
#: pre-origin tables: no value of these columns may pass the origin (publication and vintage)
PRE_PUBLICATION_COLS = ("report_published", "reprint_published", "vintage_month",
                        "reprint_vintage_month", "report_caption_month")
H1A_METHOD = {0: "observed", 1: "observed_corrected",
              3: "step_corrected_neighbour_regression"}


class ComponentError(RuntimeError):
    """A component check failed in a way the release cannot accept."""


# ----------------------------------------------------------------------------- grid
def load_grid():
    """(FanGrid, info) from the twin forward file's geometry keys (nx, ny, dx, x0, y0, mask)."""
    from hydrophysics.twin.grid import FanGrid
    p = REPO / C.GRID_SOURCE
    with np.load(p, allow_pickle=False) as z:
        nx, ny = int(z["nx"]), int(z["ny"])
        dx, x0, y0 = float(z["dx"]), float(z["x0"]), float(z["y0"])
        mask = np.asarray(z["mask"]).astype(bool)
    g = FanGrid(nx=nx, ny=ny, dx=dx, x0=x0, y0=y0, mask=mask)
    info = {"nx": nx, "ny": ny, "dx": dx, "x0": x0, "y0": y0, "n_active": g.n_active,
            "mask_hash": hashing.mask_hash(mask),
            "grid_hash": hashing.grid_hash(nx, ny, dx, x0, y0, mask),
            "crs": "EPSG:3826 (TWD97 / TM2 zone 121), metres",
            "cell_order": ("active cells in row-major order of the (ny, nx) mask, row 0 at the "
                           "southern edge y0, column 0 at the western edge x0 "
                           "(hydrophysics.twin.grid.FanGrid.active_index / centroids)"),
            "source": C.GRID_SOURCE, "source_sha256": hashing.sha256_file(p)}
    return g, info


def grid_crosscheck(g) -> dict:
    """Rebuild the grid from the fan polygon with the twin's build_grid and compare."""
    from hydrophysics.twin.grid import build_grid
    b = build_grid(str(REPO / C.GRID_POLYGON), dx=g.dx)
    same_shape = (b.nx, b.ny) == (g.nx, g.ny)
    dmax = max(abs(b.x0 - g.x0), abs(b.y0 - g.y0))
    mask_equal = bool(same_shape and np.array_equal(b.mask, g.mask))
    return {"polygon": C.GRID_POLYGON, "polygon_sha256": hashing.sha256_file(
                REPO / C.GRID_POLYGON),
            "nx_ny_equal": bool(same_shape), "max_origin_diff_m": float(dmax),
            "mask_equal": mask_equal, "passed": bool(same_shape and mask_equal and dmax < 1e-6)}


# ----------------------------------------------------------------------------- helpers
def _ym(v) -> str:
    return str(v)[:7]


def _flag_counts(a) -> dict:
    vals, cnt = np.unique(np.asarray(a), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, cnt, strict=True)}


def _check_triple(arrays: dict, suffix: str, n_active: int, dates) -> tuple[list[str], dict]:
    """Errors and a summary for one value/sd/flag triple (sd may be absent)."""
    v = arrays["value" + suffix]
    f = arrays["flag" + suffix]
    s = arrays.get("sd" + suffix)
    errs = []
    if v.ndim < 2 or v.shape[-2] != n_active:
        errs.append(f"value{suffix} shape {v.shape}: second-to-last axis must be A={n_active}")
    if s is not None:
        errs += [f"value{suffix}: {e}" for e in F.validate_grid(v, s, f, dates)]
    else:
        if v.shape != f.shape:
            errs.append(f"value{suffix}/flag{suffix} shape mismatch {v.shape} {f.shape}")
        if f.dtype != np.uint8:
            errs.append(f"flag{suffix} dtype {f.dtype} != uint8")
        bad = sorted(set(np.unique(f).tolist()) - F.VALID_FLAGS)
        if bad:
            errs.append(f"flag{suffix} invalid values {bad}")
        if np.isin(f, sorted(F.NON_OBSERVED_FLAGS)).any():
            errs.append(f"value{suffix}: flags 2-6 present but no sd{suffix} array")
        if dates is not None and v.shape[-1] != len(dates):
            errs.append(f"value{suffix} last axis {v.shape[-1]} != len(dates) {len(dates)}")
    ok8 = f == 8
    n_nan_unflagged = int((~ok8 & ~np.isfinite(v)).sum()) if v.dtype.kind == "f" else 0
    if n_nan_unflagged:
        errs.append(f"value{suffix}: {n_nan_unflagged} non-finite values without flag 8")
    summ = {"shape": list(v.shape), "dtype": str(v.dtype), "flags": _flag_counts(f),
            "sd": "present" if s is not None else "absent"}
    return errs, summ


def _grid_meta_errors(arrays: dict, ginfo: dict) -> list[str]:
    errs = []
    for k in ("mask_hash", "grid_hash"):
        if k not in arrays:
            errs.append(f"missing key {k}")
        elif str(arrays[k]) != ginfo[k]:
            errs.append(f"{k} {str(arrays[k])[:12]}... != twin grid {ginfo[k][:12]}...")
    return errs


def _dates_in_window(dates, variant: str) -> list[str]:
    if variant != "preorigin" or dates is None or len(dates) == 0:
        return []
    last = max(_ym(d) for d in dates)
    return [] if last <= ORIGIN else [f"dates reach {last} > {ORIGIN} in the pre-origin variant"]


def check_grid_file(path: Path, ginfo: dict, variant: str) -> tuple[list[str], dict]:
    arrays, skipped = io_det.read_npz(path)
    errs = [f"pickled members {skipped}"] if skipped else []
    errs += _grid_meta_errors(arrays, ginfo)
    dates = arrays.get("dates")
    if dates is None:
        errs.append("missing dates")
    errs += _dates_in_window(dates, variant)
    triples = {}
    for k in arrays:
        if k.startswith("value") and ("flag" + k[5:]) in arrays:
            e, s = _check_triple(arrays, k[5:], ginfo["n_active"], dates)
            errs += e
            triples[k] = s
    if not triples:
        errs.append("no value/flag triple")
    summ = {"kind": "grid", "triples": triples,
            "dates": ([_ym(dates[0]), _ym(dates[-1]), int(len(dates))]
                      if dates is not None and len(dates) else None),
            "keys": sorted(arrays)}
    for k in ("unit", "units", "sd_label"):
        if k in arrays:
            summ[k] = str(arrays[k])
    return errs, summ


def check_members_file(path: Path, variant: str) -> tuple[list[str], dict]:
    arrays, skipped = io_det.read_npz(path)
    errs = [f"pickled members {skipped}"] if skipped else []
    dates = arrays.get("dates")
    if dates is None:
        errs.append("missing dates")
    errs += _dates_in_window(dates, variant)
    shapes = {k: list(a.shape) for k, a in arrays.items() if k.startswith("draws")}
    for k, sh in shapes.items():
        if dates is not None and sh[-1] != len(dates):
            errs.append(f"{k} last axis {sh[-1]} != len(dates) {len(dates)}")
        if not np.isfinite(arrays[k]).all():
            errs.append(f"{k}: non-finite draws")
    return errs, {"kind": "members", "draws": shapes,
                  "dates": [_ym(dates[0]), _ym(dates[-1]), int(len(dates))]
                  if dates is not None else None}


def check_points_file(path: Path, variant: str) -> tuple[list[str], dict]:
    arrays, skipped = io_det.read_npz(path)
    errs = [f"pickled members {skipped}"] if skipped else []
    et, fl, dates = arrays.get("et0"), arrays.get("flag"), arrays.get("dates")
    if et is None or fl is None or dates is None:
        return errs + ["needs et0, flag and dates"], {"kind": "points"}
    if et.shape != fl.shape or et.shape[-1] != len(dates):
        errs.append(f"shapes et0 {et.shape} flag {fl.shape} dates {len(dates)}")
    bad = sorted(set(np.unique(fl).tolist()) - F.VALID_FLAGS)
    if bad:
        errs.append(f"invalid flags {bad}")
    if int((~np.isfinite(et) & (fl != 8)).sum()):
        errs.append("non-finite et0 without flag 8")
    errs += _dates_in_window(dates, variant)
    return errs, {"kind": "points", "n_points": int(et.shape[0]),
                  "dates": [str(dates[0]), str(dates[-1]), int(len(dates))],
                  "flags": _flag_counts(fl), "keys": sorted(arrays)}


def _time_col(df) -> str | None:
    for c in ("month", "date"):
        if c in df.columns:
            return c
    return None


def check_table(df: pd.DataFrame, variant: str, strict_columns: bool) -> tuple[list[str], dict]:
    errs = F.validate_long_table(df, require_all_columns=strict_columns) if (
        "flag" in df.columns) else []
    tc = _time_col(df)
    rng = None
    if tc is not None and len(df):
        keys = sorted({_ym(x) for x in df[tc].astype(str)})
        rng = [keys[0], keys[-1]]
        if variant == "preorigin" and keys[-1] > ORIGIN:
            errs.append(f"{tc} reaches {keys[-1]} > {ORIGIN} in the pre-origin variant")
    latest = {}
    if variant == "preorigin":
        for c in PRE_PUBLICATION_COLS:
            if c in df.columns:
                vals = sorted({_ym(x) for x in df[c].dropna().astype(str) if str(x)})
                if vals:
                    latest[c] = vals[-1]
                    if vals[-1] > ORIGIN:
                        errs.append(f"{c} reaches {vals[-1]} > {ORIGIN} in the pre-origin "
                                    f"variant (information published after the origin)")
        if "EstablishDate" in df.columns and len(df):
            mx = int(df["EstablishDate"].astype("int64").max())
            latest["EstablishDate"] = mx
            if mx > POST_ORIGIN_DATE:
                errs.append(f"EstablishDate reaches {mx} > {POST_ORIGIN_DATE} in the "
                            f"pre-origin variant")
    summ = {"kind": "table", "rows": int(len(df)), "columns": list(map(str, df.columns)),
            "time_column": tc, "range": rng}
    if latest:
        summ["latest_dated_values"] = latest
    if "flag" in df.columns:
        summ["flags"] = _flag_counts(df["flag"].to_numpy())
    for idc in ("sid", "sensor_id", "district", "county"):
        if idc in df.columns:
            summ[f"n_{idc}"] = int(df[idc].nunique())
            break
    if {"in_grid", "n_months_in_table"} <= set(df.columns):
        has = df["n_months_in_table"] > 0
        summ["stations_with_rows_in_grid"] = int((has & df["in_grid"]).sum())
        summ["stations_with_rows_off_grid"] = int((has & ~df["in_grid"]).sum())
    return errs, summ


def read_any_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


# ----------------------------------------------------------------------------- repackers
def _months_from_district_csv(p) -> list[str]:
    df = pd.read_csv(p.path("district_monthly.csv"), usecols=["date"])
    months = sorted({_ym(d) for d in df["date"].astype(str)})
    # consecutive months check
    y0, m0 = int(months[0][:4]), int(months[0][5:7])
    want = []
    y, m = y0, m0
    for _ in months:
        want.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    if months != want:
        raise ComponentError("district_monthly.csv months are not consecutive")
    return months


def repack_dates_from_district_monthly(arrays, skipped, order, p, spec, ginfo, g):
    if [k for k in skipped if k != "dates"]:
        raise ComponentError(f"pickled members other than 'dates': {skipped}")
    months = _months_from_district_csv(p)
    n = None
    for k in ("value", "draws_changhua"):
        if k in arrays:
            n = arrays[k].shape[-1]
            break
    if n != len(months):
        raise ComponentError(f"reconstructed {len(months)} months != array length {n}")
    out = {}
    for k in order:
        out[k] = np.array(months, dtype="<U7") if k == "dates" else arrays[k]
    note = ("dates reconstructed by the release from district_monthly.csv (consecutive months "
            f"{months[0]}..{months[-1]}, {len(months)}); the upstream 'dates' member is a "
            "pickled object array and was not unpickled")
    return out, [note]


def repack_member_value_flag(arrays, skipped, order, p, spec, ginfo, g):
    if skipped:
        raise ComponentError(f"pickled members {skipped}")
    v = arrays["rate_phys_m3d__primary"]
    if not np.isfinite(v).all():
        raise ComponentError("non-finite primary member")
    out = {"value": v, "sd": np.full(v.shape, np.nan, dtype=v.dtype),
           "flag": np.full(v.shape, 5, dtype=np.uint8)}
    for k in order:
        out[k] = arrays[k]
    note = ("value = rate_phys_m3d__primary (physical m3/day, 4 twin layers x A x T); flag 5 "
            "(inferred) on every cell-month; sd NaN: the variants are the uncertainty "
            "representation (plan 4.8 'an sd or members'), uncalibrated")
    return out, [note]


def repack_ic_field(arrays, skipped, order, p, spec, ginfo, g):
    if skipped:
        raise ComponentError(f"pickled members {skipped}")
    gp, sd = arrays["gp_bf"], arrays["gp_bf_sd"]
    cent = arrays["cell_centroids"]
    dmax = float(np.max(np.abs(cent - g.centroids())))
    if gp.shape != (4, ginfo["n_active"]) or dmax > 1e-6:
        raise ComponentError(f"IC field shape {gp.shape} or centroid mismatch {dmax}")
    fin = np.isfinite(gp) & np.isfinite(sd)
    out = {"value": gp[:, :, None], "sd": sd[:, :, None],
           "flag": np.where(fin, 3, 8).astype(np.uint8)[:, :, None],
           "dates": np.array(["2012-01"], dtype="<U7"),
           "layers": np.array([1, 2, 3, 4], dtype=np.int64)}
    for k in order:
        out[k] = arrays[k]
    note = (f"value/sd/flag (L=4, A, T=1) from gp_bf/gp_bf_sd; flag 3 (field from other "
            f"sites) on {int(fin.sum())} of {fin.size} layer-cells; cell_centroids equal the "
            f"twin grid centroids (max |diff| {dmax:.1e} m)")
    return out, [note]


REPACKERS = {
    "dates_from_district_monthly": repack_dates_from_district_monthly,
    "member_value_flag": repack_member_value_flag,
    "ic_field": repack_ic_field,
}


# ----------------------------------------------------------------------------- table kinds
def transform_heads_monthly(df: pd.DataFrame, variant: str, g, ctx: dict) -> tuple:
    flags = set(int(x) for x in df["flag"].unique())
    if not flags <= set(H1A_METHOD):
        raise ComponentError(f"unexpected head flags {sorted(flags - set(H1A_METHOD))}")
    out = df.reset_index(drop=True).copy()
    fl = out["flag"].to_numpy()
    out["sd"] = np.where(np.isin(fl, [0, 1]), 0.0, np.nan)
    out["method"] = [H1A_METHOD[int(x)] for x in fl]
    out["product_version"] = "heads/L1_clean/v3"
    out["source_ids"] = "heads/L1_clean/v3:own_record"
    ctx["months_by_sid"] = out.groupby("sid").size().to_dict()
    note = ("columns added by the release: sd (0 for observed months; NaN for flag 3, none "
            "given upstream), method (from the flag), product_version, source_ids")
    return out, [note], True


def _cell_index(g, x, y) -> tuple[int, int, int]:
    if not (np.isfinite(x) and np.isfinite(y)):
        return -1, -1, -1
    rc = g.cell_of(float(x), float(y))
    if rc is None:
        return -1, -1, -1
    return int(g.active_index(float(x), float(y))), int(rc[0]), int(rc[1])


def transform_stations(df: pd.DataFrame, variant: str, g, ctx: dict) -> tuple:
    cols = PRE_STATION_COLS + (FULL_STATION_EXTRA if variant == "full" else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ComponentError(f"station table lacks {missing}")
    out = df[cols].reset_index(drop=True).copy()
    n_late = 0
    if variant == "preorigin":
        late = out["EstablishDate"].astype("int64") > POST_ORIGIN_DATE
        n_late = int(late.sum())
        out = out[~late].reset_index(drop=True)
    idx = [_cell_index(g, x, y) for x, y in zip(out["x_twd97"], out["y_twd97"], strict=True)]
    out["cell_index"] = np.array([i[0] for i in idx], dtype=np.int64)
    out["grid_row"] = np.array([i[1] for i in idx], dtype=np.int64)
    out["grid_col"] = np.array([i[2] for i in idx], dtype=np.int64)
    out["in_grid"] = out["cell_index"] >= 0
    months = ctx.get("months_by_sid", {})
    out["n_months_in_table"] = np.array([int(months.get(s, 0)) for s in out["sid"]],
                                        dtype=np.int64)
    out["established_after_origin"] = out["EstablishDate"].astype("int64") > POST_ORIGIN_DATE
    note = ((f"pre-origin variant: dropped the {n_late} stations established after 2022-12 "
             "(EstablishDate > 20221231); ") if variant == "preorigin" else "") + (
            "curated non-personal columns (GroundHeight and free-text Remarks not carried); "
            "cell_index/grid_row/grid_col from x_twd97, y_twd97 with hydrophysics.twin.grid."
            "FanGrid (-1 off the grid or without coordinates); n_months_in_table = rows of "
            "heads_monthly.parquet in this component; established_after_origin = EstablishDate "
            "> 2022-12-31")
    return out, [note], False


def transform_identity(df: pd.DataFrame, variant: str, g, ctx: dict) -> tuple:
    return df.reset_index(drop=True), [], True


def _roc_to_ym(v: str) -> str:
    y, m = v.strip().split("/")
    return f"{int(y) + 1911:04d}-{int(m):02d}"


def pile_publication() -> dict[str, str]:
    """{report file: publication month YYYY-MM} from the agency listing (發布年月) recorded in
    the 007001 SOURCE.json; a file linked from several listing rows takes the latest month."""
    src = json.loads((REPO / C.PILE_PUBLICATION_SOURCE).read_text(encoding="utf-8"))
    out = {}
    for f in src["files"]:
        if "published_roc" not in f:            # listing pages, not reports
            continue
        months = [_roc_to_ym(f["published_roc"])]
        months += [_roc_to_ym(x) for x in
                   re.findall(r"'published_roc': '([0-9]+/[0-9]+)'", f.get("note", ""))]
        out[f["file"]] = max(months)
    return out


_PRINTINGS = re.compile(r"of (\d+) printing\(s\)")


def transform_piles_preorigin(df: pd.DataFrame, variant: str, g, ctx: dict) -> tuple:
    """Keep only information published by the origin (L3.repair.md 2.3 a)."""
    if variant != "preorigin":
        raise ComponentError("piles_preorigin is a pre-origin transform")
    pub = pile_publication()
    out = df.reset_index(drop=True).copy()
    missing = sorted(set(out["report"]) - set(pub))
    rp = out["reprint_report"].fillna("").astype(str)
    missing += sorted({r for r in rp if r and r not in pub})
    if missing:
        raise ComponentError(f"no publication month for reports {missing}")
    if int(out["n_printings"].max()) > 2:
        raise ComponentError("more than two printings: the recount below assumes at most two")
    out["report_published"] = out["report"].map(pub)
    out["reprint_published"] = [pub[r] if r else None for r in rp]
    late = out["report_published"] > ORIGIN
    late_rp = out["reprint_published"].fillna("") > ORIGIN
    rescue = int((late & out["reprint_published"].notna() & ~late_rp).sum())
    if rescue:
        raise ComponentError(f"{rescue} dropped rows have a pre-origin second printing")
    n_drop = int(late.sum())
    drop_months = sorted({str(m) for m in out.loc[late, "month"]})
    drop_reports = sorted(set(out.loc[late, "report"]))
    out = out[~late].reset_index(drop=True)
    late_rp = out["reprint_published"].fillna("") > ORIGIN
    n_rp = int(late_rp.sum())
    # recount printings over those published by the origin (at most two printings per row)
    reprint_verified = (out["n_verified_printings"]
                        - (out["flag"] == 0).astype(int)) >= 1
    out.loc[late_rp, "n_printings"] = out.loc[late_rp, "n_printings"] - 1
    out.loc[late_rp, "n_verified_printings"] = (
        out.loc[late_rp, "n_verified_printings"] - reprint_verified[late_rp].astype(int))
    for c in ("reprint_value", "reprint_diff_mm"):
        out.loc[late_rp, c] = np.nan
    for c in ("reprint_report", "reprint_vintage_month", "reprint_published"):
        out[c] = out[c].astype(object)
        out.loc[late_rp, c] = None
    out["method"] = [_PRINTINGS.sub(f"of {int(n)} printing(s) published by {ORIGIN}", m)
                     for m, n in zip(out["method"], out["n_printings"], strict=True)]
    if not all(_PRINTINGS.search(m) is None or "published by" in m for m in out["method"]):
        raise ComponentError("method text printing count not rewritten")
    ctx["pile_publication"] = {"dropped_rows": n_drop, "dropped_months": drop_months,
                               "dropped_reports": drop_reports,
                               "reprints_removed": n_rp, "rows_kept": int(len(out))}
    note = (f"pre-origin publication rule (L3.repair.md 2.3 a): kept rows whose primary report "
            f"was published (agency listing month, 發布年月, {C.PILE_PUBLICATION_SOURCE}) by "
            f"{ORIGIN}; dropped {n_drop} rows (data months {', '.join(drop_months)}; reports "
            f"{', '.join(drop_reports)}); removed {n_rp} second printings published after "
            f"{ORIGIN} (reprint_* null); n_printings, n_verified_printings and the method text "
            f"count printings published by {ORIGIN}; columns report_published and "
            f"reprint_published added; {len(out)} rows kept")
    return out, [note], True


TABLE_KINDS = {"heads_monthly": transform_heads_monthly, "stations": transform_stations,
               "piles": transform_identity, "piles_preorigin": transform_piles_preorigin}


# ----------------------------------------------------------------------------- uses
def _grantable(up_m: dict, use: str, window_end, fname: str) -> tuple[bool, str]:
    try:
        acc = guard.check_use(up_m, use, window_end=window_end,
                              require_gate=True if use in U.GATE_EXEMPT_USES else None)
    except (guard.UseNotAllowed, guard.LeakageError, guard.GateNotPassed, ValueError) as exc:
        return False, type(exc).__name__
    if acc.outputs is not None and fname not in acc.outputs:
        return False, "output not readable for this use (per-output window)"
    return True, "granted" + (f" (window_end {window_end})" if window_end else "")


def uses_check(spec: dict, up_m: dict) -> tuple[dict, list[str]]:
    """Every release use must be grantable by the upstream guard for every file read."""
    res, errs = {}, []
    for use in spec["allowed_uses"]:
        per = {}
        for rd in spec["reads"]:
            w = rd["window_end"]
            for f in rd["files"]:
                ok, why = _grantable(up_m, use, w if (w and use in U.PRE_ORIGIN_USES) else None,
                                     f["name"])
                if not ok and w and use not in U.PRE_ORIGIN_USES:
                    ok, why = _grantable(up_m, use, w, f["name"])
                per[f["name"]] = why if ok else f"REFUSED: {why}"
                if not ok:
                    errs.append(f"use {use!r} not grantable upstream for {f['name']}: {why}")
        res[use] = per
    if spec["variant"] == "full":
        pre = sorted(set(spec["allowed_uses"]) & U.PRE_ORIGIN_USES)
        if pre:
            errs.append(f"full variant allows pre-origin uses {pre}")
    return res, errs


def inherited_exemptions(spec: dict, up_m: dict) -> list[dict]:
    allowed = set(spec["allowed_uses"])
    out = []
    for x in up_m.get("exemptions") or []:
        y = dict(x)
        y["uses"] = [u for u in x["uses"] if u in allowed]
        if not y["uses"]:
            raise ComponentError(f"exemption {x.get('ledger_number_id')} covers no allowed use")
        y["inherited_from"] = f"{up_m['product']}/{up_m['version']}"
        out.append(y)
    return out


# ----------------------------------------------------------------------------- upstream check
CODE_ARCHIVE_GLOB = "results/dataprep/reports/*_evidence/code_as_registered/{sha12}/{name}"


def _git_blob_sha256(commit: str, path: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                           capture_output=True, check=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    return hashlib.sha256(r.stdout).hexdigest()


def _pinned_source(m: dict, rec: dict) -> dict | None:
    """Where the recorded bytes of a moved-on code file are reproduced (L3.repair.md 2.4 h)."""
    want = rec["sha256"]
    if rec.get("git_state") == "committed" and m.get("git_commit"):
        got = _git_blob_sha256(m["git_commit"], rec["path"])
        if got == want:
            return {"path": rec["path"], "sha256": want, "source": "git",
                    "ref": f"{m['git_commit']}:{rec['path']}"}
        return None
    if rec.get("git_state") == "untracked":
        name = Path(rec["path"]).name
        hits = sorted(REPO.glob(CODE_ARCHIVE_GLOB.format(sha12=want[:12], name=name)))
        for h in hits:
            if hashing.sha256_file(h) == want:
                return {"path": rec["path"], "sha256": want, "source": "archive",
                        "ref": rel_to_repo(h)}
    return None


def assert_releasable_pinned(up_dir: Path) -> tuple[dict, list[dict]]:
    """common.guard.assert_releasable, except that a code file whose working-tree hash moved on
    is accepted when its recorded sha256 is reproduced from the manifest's git commit
    (committed files) or from the archived registered bytes (untracked files). Inputs and
    outputs must reproduce from disk. Returns (manifest, pinned code files)."""
    m, d = manifest.load_manifest(up_dir)
    errs = manifest.validate_manifest(m, d)
    if errs:
        raise manifest.ManifestError(f"{d}: " + "; ".join(errs))
    if m["layer"] not in ("L1", "L2"):
        raise guard.GateNotPassed(f"{d}: layer {m['layer']!r} cannot enter a release")
    if not m["gate"]["passed"]:
        raise guard.GateNotPassed(f"{d}: validation gate not passed ({m['gate']})")
    pinned, bad = [], []
    for prob in manifest.verify_hashes(m, d):
        rec = next((c for c in m.get("code", []) if c["path"] == prob.get("path")), None)
        src = (_pinned_source(m, rec) if prob.get("section") == "code" and rec is not None
               and prob.get("problem") in ("sha256", "missing") else None)
        if src is None:
            bad.append(prob)
        else:
            pinned.append(src)
    if bad:
        raise manifest.HashMismatch(f"{d}: {bad}")
    return m, sorted(pinned, key=lambda x: x["path"])


# ----------------------------------------------------------------------------- component
def component_product(spec: dict) -> str:
    return f"model_ready/{C.RELEASE}/{spec['variant']}/{spec['name']}"


def build_component(spec: dict, variant_dir: Path, ginfo: dict, g, created: str) -> dict:
    """Build one component; returns its record (for RELEASE.json and the card)."""
    up_dir = DATAPREP / spec["upstream"]
    up_m, pinned = assert_releasable_pinned(up_dir)
    up_manifest_sha = hashing.sha256_file(up_dir / manifest.MANIFEST_NAME)
    comp_dir = variant_dir / spec["name"] / "v1"
    comp_dir.mkdir(parents=True, exist_ok=False)
    inputs = [rel_to_repo(up_dir / manifest.MANIFEST_NAME)] + list(spec.get("extra_inputs",
                                                                         []))
    up_out = {o["path"]: o for o in up_m["outputs"]}
    reads, outputs, errs = [], [], []
    ctx: dict = {}
    for rd in spec["reads"]:
        use, wend = rd["use"], rd["window_end"]
        p = guard.open_product(up_dir, use, window_end=wend,
                               require_gate=True if use in U.GATE_EXEMPT_USES else None)
        reads.append({"use": use, "window_end": wend, "guard_notes": list(p.notes),
                      "files": [f["name"] for f in rd["files"]]})
        for f in rd["files"]:
            name, mode, kind = f["name"], f["mode"], f["kind"]
            out_name = f.get("rename", name)
            if mode == "table" and not out_name.endswith(".parquet"):
                out_name = Path(out_name).stem + ".parquet"
            dst = comp_dir / out_name
            rec = {"path": out_name, "upstream_file": name, "mode": mode, "kind": kind,
                   "upstream_sha256": up_out[name]["sha256"], "read_use": use,
                   "window_end": wend, "notes": []}
            if mode == "copy":
                io_det.copy_file(p.path(name), dst)
            elif mode == "repack":
                arrays, skipped = io_det.read_npz(p.path(name))
                with np.load(p.path(name), allow_pickle=False) as z:
                    order = list(z.files)
                new, rnotes = REPACKERS[f["repack"]](arrays, skipped, order, p, spec, ginfo, g)
                new["release_component"] = np.array(component_product(spec))
                new["upstream_product"] = np.array(f"{up_m['product']}/{up_m['version']}")
                new["upstream_file_sha256"] = np.array(up_out[name]["sha256"])
                if "mask_hash" not in new and kind == "grid":
                    new["mask_hash"] = np.array(ginfo["mask_hash"])
                    new["grid_hash"] = np.array(ginfo["grid_hash"])
                io_det.write_npz(dst, new)
                rec["notes"] += rnotes
            elif mode == "table":
                df = p.read_table(name)
                rec["rows_granted"] = int(len(df))
                df, tnotes, strict = TABLE_KINDS[kind](df, spec["variant"], g, ctx)
                io_det.write_parquet(dst, df)
                rec["notes"] += tnotes
                rec["strict_columns"] = strict
            else:
                raise ComponentError(f"unknown mode {mode}")
            inputs.append(rel_to_repo(up_dir / name))
            rec["sha256"] = hashing.sha256_file(dst)
            rec["bytes"] = dst.stat().st_size
            if mode == "copy":
                rec["identical_to_upstream"] = rec["sha256"] == rec["upstream_sha256"]
                if not rec["identical_to_upstream"]:
                    errs.append(f"{out_name}: copy differs from the upstream manifest hash")
            if kind == "grid":
                e, s = check_grid_file(dst, ginfo, spec["variant"])
            elif kind == "members":
                e, s = check_members_file(dst, spec["variant"])
            elif kind == "points":
                e, s = check_points_file(dst, spec["variant"])
            else:
                e, s = check_table(read_any_table(dst), spec["variant"],
                                   bool(rec.get("strict_columns", False)))
            rec["checks"] = {"errors": e, "summary": s}
            errs += [f"{out_name}: {x}" for x in e]
            outputs.append(rec)
    use_res, use_errs = uses_check(spec, up_m)
    errs += use_errs
    # accepted exceptions (declared in config before the build)
    accepted, failed = [], []
    for e in errs:
        hit = None
        for ax in spec.get("accepted_exceptions", []):
            fname = ax["check"].split(":", 1)[-1]
            if e.startswith(fname + ":") and ax["contains"] in e:
                hit = ax
        if hit:
            accepted.append({"error": e, "reason": hit["reason"]})
        else:
            failed.append(e)
    if spec["fit_window_end"] == "inherit":
        fwe = up_m["fit_window_end"]
        ne = up_m.get("non_exempt_fit_window_end")
        exemptions = inherited_exemptions(spec, up_m)
    else:
        fwe, ne, exemptions = spec["fit_window_end"], None, []
    if spec["variant"] == "preorigin" and not ((fwe <= ORIGIN) or (ne and ne <= ORIGIN)):
        failed.append(f"pre-origin component fit_window_end {fwe} without exempt coverage")
    passed = not failed
    upstream = {"product": up_m["product"], "version": up_m["version"],
                "dir": rel_to_repo(up_dir), "manifest_sha256": up_manifest_sha,
                "layer": up_m["layer"], "fit_window_end": up_m["fit_window_end"],
                "non_exempt_fit_window_end": up_m.get("non_exempt_fit_window_end"),
                "gate": up_m["gate"],
                "assert_releasable": (
                    "ok (gate passed; every input, code and output hash reproduced)" if not pinned
                    else f"ok (gate passed; every input and output hash reproduced from disk; "
                         f"{len(pinned)} code file(s) moved on in the working tree, their recorded "
                         f"sha256 reproduced from the recorded git commit or archive: "
                         f"code_pinned)"),
                "code_pinned": pinned, "wp": spec["wp"],
                "wp_final_status": C.WP_STATUS[spec["wp"]],
                "inclusion_basis": C.inclusion_basis(spec["upstream"]),
                "rule_R": C.INCLUSION[spec["upstream"]]}
    validation = {
        "schema": "dataprep-l3-component-validation/1",
        "release": C.RELEASE, "component": spec["name"], "variant": spec["variant"],
        "product": component_product(spec), "version": "v1", "title": spec["title"],
        "upstream": upstream, "reads": reads, "outputs": outputs,
        "uses": {"allowed": spec["allowed_uses"], "grantable_upstream": use_res},
        "exemptions_inherited": [x.get("ledger_number_id") for x in exemptions],
        "sd_status": spec["sd_status"], "notes": spec["notes"],
        "pile_publication": ctx.get("pile_publication"),
        "checks_failed": failed, "accepted_exceptions": accepted, "passed": passed,
    }
    io_det.write_json(comp_dir / "validation.json", validation)
    forbidden = sorted(set(U.USES) - set(spec["allowed_uses"]))
    gate = {"passed": passed,
            "criteria": ("L3 release component (dataprep_staging/l3_release): upstream "
                         "assert_releasable (gate passed, every hash reproduced); every byte "
                         "read through guard.open_product with a declared use; every release "
                         "use grantable by the upstream guard; per-file checks (copy identity, "
                         "twin grid and mask hash, value/sd/flag validity, dates within the "
                         "variant window); see validation.json"),
            "upstream_product": f"{up_m['product']}/{up_m['version']}",
            "upstream_gate_passed": bool(up_m["gate"]["passed"]),
            "wp": spec["wp"], "wp_final_status": C.WP_STATUS[spec["wp"]]}
    m = manifest.build_manifest(
        comp_dir, product=component_product(spec), layer="L3",
        inputs=inputs, code=CODE_FILES,
        parameters={"release": C.RELEASE, "variant": spec["variant"],
                    "component": spec["name"], "title": spec["title"],
                    "upstream": f"{up_m['product']}/{up_m['version']}",
                    "reads": [{"use": r["use"], "window_end": r["window_end"],
                               "files": r["files"]} for r in reads],
                    "sd_status": spec["sd_status"], "notes": spec["notes"],
                    "grid_hash": ginfo["grid_hash"], "mask_hash": ginfo["mask_hash"]},
        fit_window_end=fwe, validation_report="validation.json", gate=gate,
        allowed_uses=spec["allowed_uses"], forbidden_uses=forbidden,
        personal_data=spec["personal_data"], exemptions=exemptions,
        non_exempt_fit_window_end=ne if exemptions else None, created=created,
        version="v1")
    manifest.write_manifest(comp_dir, m)
    return {"spec": spec, "dir": comp_dir, "manifest": m, "validation": validation,
            "upstream": upstream}


def build_grid_component(release_dir: Path, g, ginfo: dict, created: str) -> dict:
    comp_dir = release_dir / "grid" / "v1"
    comp_dir.mkdir(parents=True, exist_ok=False)
    rows, cols = np.nonzero(g.mask)
    cent = g.centroids()
    io_det.write_npz(comp_dir / "grid.npz", {
        "mask": g.mask.astype(bool), "nx": np.int64(g.nx), "ny": np.int64(g.ny),
        "dx": np.float64(g.dx), "x0": np.float64(g.x0), "y0": np.float64(g.y0),
        "n_active": np.int64(g.n_active), "rows": rows.astype(np.int64),
        "cols": cols.astype(np.int64), "centroids": cent.astype(np.float64),
        "mask_hash": np.array(ginfo["mask_hash"]), "grid_hash": np.array(ginfo["grid_hash"]),
        "crs": np.array(ginfo["crs"]), "cell_order": np.array(ginfo["cell_order"])})
    xc = grid_crosscheck(g)
    errs = [] if xc["passed"] else ["build_grid(polygon) does not reproduce the twin grid"]
    if g.n_active != 2148:
        errs.append(f"n_active {g.n_active} != 2148")
    validation = {"schema": "dataprep-l3-component-validation/1", "release": C.RELEASE,
                  "component": "grid", "product": f"model_ready/{C.RELEASE}/grid",
                  "version": "v1", "grid": ginfo, "polygon_crosscheck": xc,
                  "checks_failed": errs, "passed": not errs}
    io_det.write_json(comp_dir / "validation.json", validation)
    gate = {"passed": not errs,
            "criteria": ("geometry read from the twin forward file (nx, ny, dx, x0, y0, mask); "
                         "the twin's build_grid on the fan polygon reproduces it; 2148 active "
                         "cells; see validation.json")}
    m = manifest.build_manifest(
        comp_dir, product=f"model_ready/{C.RELEASE}/grid", layer="L3",
        inputs=[C.GRID_SOURCE, C.GRID_POLYGON], code=CODE_FILES,
        parameters={"release": C.RELEASE, "keys_read": ["nx", "ny", "dx", "x0", "y0", "mask"],
                    "fit_window_end_note": ("no fitted parameter: the grid is the fan polygon's "
                                            "mask, time-invariant; 2012-01 is the start of the "
                                            "model window"),
                    "grid_hash": ginfo["grid_hash"], "mask_hash": ginfo["mask_hash"]},
        fit_window_end="2012-01", validation_report="validation.json", gate=gate,
        allowed_uses=C.GRID_USES, forbidden_uses=sorted(set(U.USES) - set(C.GRID_USES)),
        personal_data="none", created=created, version="v1")
    manifest.write_manifest(comp_dir, m)
    return {"dir": comp_dir, "manifest": m, "validation": validation}


def probe_guard(comp_dir: Path, allowed: list[str]) -> tuple[dict, list[str]]:
    """Open the component for every use: allowed uses must open, every other use must be
    refused. Returns ({use: 'granted' | exception class}, mismatches)."""
    res, bad = {}, []
    for use in sorted(U.USES):
        try:
            guard.open_product(comp_dir, use,
                               require_gate=True if use in U.GATE_EXEMPT_USES else None)
            res[use] = "granted"
        except (guard.UseNotAllowed, guard.LeakageError, guard.GateNotPassed) as exc:
            res[use] = f"refused:{type(exc).__name__}"
        if (res[use] == "granted") != (use in allowed):
            bad.append(use)
    return res, bad


def today() -> str:
    return _dt.date.today().isoformat()
