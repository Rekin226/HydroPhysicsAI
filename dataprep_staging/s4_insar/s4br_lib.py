"""WP S4b repair round 2 (S4br) library (hydro env): load the repaired LiCSBAS runs (filtered
series, M2), apply the stated off-fan reference (M3), NGL L0 readers for the fresh re-test,
and the two-frame merge of the full product (M5).

Conventions are round 1's (``s4b_lib``): LiCSBAS ``cum`` is LOS towards the satellite (mm,
positive = uplift), zero at the first epoch; pixels are placed by index on the window's
reference geotransform (``geom_<frame>.npz``); vertical-equivalent = referenced LOS / U.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C0  # noqa: E402
import s4b_lib as Lb  # noqa: E402
import s4br_config as R  # noqa: E402

_MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                                        "SEP", "OCT", "NOV", "DEC"])}


# ----------------------------------------------------------------------------- runs
def geom_path(frame: str) -> Path:
    return (C0.RUN_DIR if frame == C0.PRIMARY else R.RUN_DIR) / f"geom_{frame}.npz"


def load_geom(frame: str) -> dict:
    g = np.load(geom_path(frame), allow_pickle=False)
    return {k: g[k] for k in g.files}


def load_run(name: str, cum_file: str = R.CUM_FILE) -> dict:
    """Repaired run: filtered cum (T, L, W) float32 mm, dates, LiCSBAS15 mask, per-pixel gap
    intervals (from cum.h5; LiCSBAS16 keeps the network) and the network-level gaps."""
    ts = R.RUN_DIR / name / "TS_GEOCml2"
    with h5py.File(ts / cum_file, "r") as f:
        cum = f["cum"][()].astype(np.float32)
        dates = pd.to_datetime([str(d) for d in f["imdates"][()]], format="%Y%m%d")
    with h5py.File(ts / "cum.h5", "r") as f:
        gap = f["gap"][()].astype(bool) if "gap" in f else None
        d2 = pd.to_datetime([str(d) for d in f["imdates"][()]], format="%Y%m%d")
    if not d2.equals(dates):
        raise RuntimeError(f"{name}: cum.h5 and {cum_file} epochs differ")
    L, W = cum.shape[1:]
    m = np.fromfile(ts / "results" / "mask", dtype=np.float32).reshape(L, W)
    return {"name": name, "frame": R.RUNS[name][0], "cum": cum, "dates": dates,
            "mask": np.isfinite(m) & (m == 1), "gap": gap, "ts_dir": ts,
            "network_gaps": network_gaps(ts)}


def network_gaps(ts: Path) -> list[list[str]]:
    out, on = [], False
    p = ts / "info" / "12network_gap_info.txt"
    for line in p.read_text().splitlines():
        if line.startswith("Gaps between"):
            on = True
            continue
        if line.startswith("Connected"):
            break
        q = line.split()
        if on and len(q) == 2:
            out.append(q)
    return out


def processed(name: str, terms=R.REF_TERMS, u_const=None):
    """(run, geom, vertical-equivalent (T, L, W), per-epoch reference table) with the stated
    off-fan reference (M3) on the round-1 stable strip."""
    run = load_run(name)
    g = load_geom(run["frame"])
    los, tab = Lb.reference(run, g, terms)
    return run, g, Lb.vertical(los, g, u_const), tab


# ----------------------------------------------------------------------------- NGL (L0)
def _day(s: str) -> pd.Timestamp:
    yy = int(s[:2])
    return pd.Timestamp(year=(1900 + yy) if yy >= 90 else (2000 + yy), month=_MON[s[2:5]],
                        day=int(s[5:7]))


def ngl_holdings() -> pd.DataFrame:
    """Station, lat, lon (deg, -180..180), height (m), first and last day from NGL's
    DataHoldings.txt (L0; positions are metadata, no displacement is read)."""
    rows = []
    with open(R.NGL_HOLDINGS, encoding="ascii", errors="replace") as f:
        next(f)
        for line in f:
            q = line.split()
            if len(q) < 9:
                continue
            try:
                rows.append({"station": q[0], "lat": float(q[1]),
                             "lon": ((float(q[2]) + 180.0) % 360.0) - 180.0,
                             "hgt_m": float(q[3]), "first_day": pd.Timestamp(q[7]),
                             "last_day": pd.Timestamp(q[8])})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows).drop_duplicates("station").set_index("station")


def ngl_tenv3_path(st: str) -> Path | None:
    return next((d / f"{st}.tenv3" for d in R.NGL_DIRS if (d / f"{st}.tenv3").is_file()), None)


def read_ngl_up(st: str) -> pd.Series | None:
    """Daily up (m, u0 + up) of an NGL station from its tenv3 file (pure text parsing)."""
    p = ngl_tenv3_path(st)
    if p is None:
        return None
    rows = []
    with open(p, encoding="ascii", errors="replace") as f:
        hdr = next(f).split()
        ix = {h: i for i, h in enumerate(hdr)}
        for line in f:
            q = line.split()
            if len(q) < len(hdr):
                continue
            try:
                rows.append((_day(q[ix["YYMMMDD"]]),
                             float(q[ix["u0(m)"]]) + float(q[ix["____up(m)"]])))
            except (ValueError, KeyError):
                continue
    s = pd.Series(dict(rows)).sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s[~s.index.duplicated(keep="first")]


def ngl_steps(st: str, lo: pd.Timestamp, hi: pd.Timestamp, types=("1",)) -> list[str]:
    """NGL steps.txt entries of a station inside [lo, hi] whose type is in ``types``
    (1 = equipment change; 2 = earthquake within NGL's magnitude-dependent radius)."""
    out = []
    with open(R.NGL_STEPS, encoding="ascii", errors="replace") as f:
        for line in f:
            q = line.split()
            if len(q) >= 3 and q[0] == st and q[2] in types:
                try:
                    d = _day(q[1])
                except (KeyError, ValueError):
                    continue
                if lo <= d <= hi:
                    out.append(f"{d.date()} type {q[2]}")
    return out


# ----------------------------------------------------------------------------- merge (M5)
def tie_and_merge(vP: np.ndarray, dP: pd.DatetimeIndex, mP: np.ndarray,
                  vS: np.ndarray, dS: pd.DatetimeIndex, mS: np.ndarray):
    """Two-frame merge on the primary multilooked grid (S2 placed by index; the two LiCSAR
    grids differ by 1/6 multilooked pixel E-W and 1/36 N-S). S2 is tied, pixel by pixel, to F2
    at their first common epoch t*: S2' = S2 + F2(t*). Product epochs = union of both epoch
    lists. Value = mean of F2 and S2' where both are valid, F2 or S2' alone where one is.
    Returns (value (T, L, W), source (T, L, W) int8: 1 = primary only, 2 = second only (tied),
    3 = mean of both, 4 = second only and NOT tied (primary invalid at t*: relative to the
    second frame's first epoch), dates)."""
    dates = dP.union(dS)
    T = len(dates)
    L, W = vP.shape[1:]
    common = dP.intersection(dS)
    if common.empty:
        raise RuntimeError("no common epoch between the frames")
    t0 = common.min()
    iP0, iS0 = int(np.flatnonzero(dP == t0)[0]), int(np.flatnonzero(dS == t0)[0])
    both0 = mP & np.isfinite(vP[iP0]) & mS & np.isfinite(vS[iS0])
    off = np.where(both0, vP[iP0] - vS[iS0], np.nan)
    tied = np.isfinite(off)
    val = np.full((T, L, W), np.nan, np.float32)
    src = np.zeros((T, L, W), np.int8)
    for k, d in enumerate(dates):
        a = np.full((L, W), np.nan, np.float32)
        b = np.full((L, W), np.nan, np.float32)
        if d in dP:
            i = int(np.flatnonzero(dP == d)[0])
            a = np.where(mP, vP[i], np.nan)
        if d in dS:
            j = int(np.flatnonzero(dS == d)[0])
            b = np.where(mS, vS[j], np.nan)
            b = np.where(tied, b + np.nan_to_num(off), b)
        fa, fb = np.isfinite(a), np.isfinite(b)
        both = fa & fb
        val[k] = np.where(both, 0.5 * (a + b), np.where(fa, a, b))
        src[k] = np.where(both, 3, np.where(fa, 1, np.where(fb & tied, 2,
                                                            np.where(fb, 4, 0))))
    return val, src, dates, str(t0.date())


def jdump(obj, path: Path) -> None:
    Lb.jdump(obj, path)


def jload(path: Path):
    return json.loads(Path(path).read_text())
