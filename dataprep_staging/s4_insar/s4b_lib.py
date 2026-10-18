"""WP S4b library (hydro env): LiCSBAS run loading, off-fan referencing, vertical conversion,
twin-cell aggregation, seasonal harmonic fit and the GNSS comparison helpers.

Conventions
-----------
* LiCSBAS ``cum`` (mm) is LOS displacement towards the satellite (positive = uplift), zero at
  the first epoch for every pixel. Pixels are placed by array index on the window's reference
  geotransform (``geom_<frame>.npz`` from ``s4b_geom.py``); the cum.h5 geotransform is ignored.
* Referencing (plan S4 step 5, ``s4b_config.REF_TERMS``): per epoch, a robust least-squares fit
  of offset + east/north plane + height term on the stable pixels only (off-fan, land,
  100-600 m, LiCSBAS15 mask); the fitted surface is subtracted from every pixel. No on-fan
  GNSS, ring, pile or leveling value enters the reference.
* Vertical-equivalent (plan S4 step 6): referenced LOS / U, with U the per-pixel up component.
* Twin cells: median of the valid multilooked pixels whose centre lies in the active cell.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402

MAD = 1.4826


# ----------------------------------------------------------------------------- loading
def load_geom(frame: str) -> dict:
    g = np.load(C.RUN_DIR / f"geom_{frame}.npz", allow_pickle=False)
    return {k: g[k] for k in g.files}


def run_frame(name: str) -> str:
    return C.RUNS[name][0]


def load_run(name: str) -> dict:
    """cum (T, L, W) float32 mm, dates (DatetimeIndex), mask (L, W) bool (LiCSBAS15 keep),
    coh_avg, refarea and the LiCSBAS-recorded geotransform (for the record only)."""
    ts = C.RUN_DIR / name / "TS_GEOCml2"
    with h5py.File(ts / "cum.h5", "r") as f:
        cum = f["cum"][()].astype(np.float32)
        dates = pd.to_datetime([str(d) for d in f["imdates"][()]], format="%Y%m%d")
        coh = f["coh_avg"][()]
        rec = {"corner_lat": float(f["corner_lat"][()]), "corner_lon": float(f["corner_lon"][()]),
               "post_lat": float(f["post_lat"][()]), "post_lon": float(f["post_lon"][()]),
               "refarea": f["refarea"][()].decode() if isinstance(f["refarea"][()], bytes)
               else str(f["refarea"][()])}
    L, W = cum.shape[1:]
    m = np.fromfile(ts / "results" / "mask", dtype=np.float32).reshape(L, W)
    mask = np.isfinite(m) & (m == 1)
    return {"name": name, "frame": run_frame(name), "cum": cum, "dates": dates, "mask": mask,
            "coh_avg": coh, "cumh5_georef": rec, "ts_dir": ts}


# ----------------------------------------------------------------------------- grid
def twin_grid():
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    mask = np.asarray(z["mask"]).astype(bool)
    return mask, float(z["dx"]), float(z["x0"]), float(z["y0"])


def cell_index(x: np.ndarray, y: np.ndarray, mask, dx, x0, y0) -> np.ndarray:
    """Active-cell index (np.nonzero(mask) order, as heads/units) of points, -1 outside."""
    ny, nx = mask.shape
    col = np.floor((x - x0) / dx).astype(int)
    row = np.floor((y - y0) / dx).astype(int)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    lut = -np.ones(mask.shape, dtype=int)
    r, c = np.nonzero(mask)
    lut[r, c] = np.arange(r.size)
    out = -np.ones(x.shape, dtype=int)
    out[inside] = lut[row[inside], col[inside]]
    return out



# ----------------------------------------------------------------------------- referencing
def ref_design(geom: dict, terms=C.REF_TERMS) -> np.ndarray:
    """Design columns (L, W, k) of the reference surface; x, y relative to the stable-pixel
    centroid (km), height in km."""
    st = geom["stable_geom"]
    xc, yc = float(np.mean(geom["x"][st])), float(np.mean(geom["y"][st]))
    cols = []
    for t in terms:
        if t == "offset":
            cols.append(np.ones_like(geom["x"]))
        elif t == "east_km":
            cols.append((geom["x"] - xc) / 1000.0)
        elif t == "north_km":
            cols.append((geom["y"] - yc) / 1000.0)
        elif t == "hgt_km":
            cols.append(np.nan_to_num(geom["hgt"]) / 1000.0)
        else:
            raise ValueError(t)
    return np.stack(cols, -1)


def robust_lstsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Two-pass LS: fit, reject |r| > 3 * 1.4826 * MAD, refit. Returns (beta, keep, robust sd
    of the kept residuals, n kept)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    s = MAD * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r - np.median(r)) <= 3 * max(s, 1e-9)
    beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    r2 = y[keep] - X[keep] @ beta
    sd = float(MAD * np.median(np.abs(r2 - np.median(r2))))
    return beta, keep, sd, int(keep.sum())


def reference(run: dict, geom: dict, terms=C.REF_TERMS) -> tuple[np.ndarray, pd.DataFrame]:
    """Referenced LOS cum (T, L, W) float32 and the per-epoch reference table."""
    cum = run["cum"]
    X = ref_design(geom, terms)
    stable = geom["stable_geom"] & run["mask"]
    out = np.empty_like(cum)
    rows = []
    for t in range(cum.shape[0]):
        c = cum[t]
        sel = stable & np.isfinite(c)
        n = int(sel.sum())
        if n >= len(terms) + 2:
            beta, keep, sd, nk = robust_lstsq(X[sel].astype(float), c[sel].astype(float))
        else:
            beta, sd, nk = np.full(len(terms), np.nan), np.nan, 0
        surf = (X @ np.nan_to_num(beta)).astype(np.float32)
        out[t] = c - surf if np.all(np.isfinite(beta)) else np.nan
        rows.append({"date": run["dates"][t].strftime("%Y-%m-%d"), "n_stable": n,
                     "n_kept": nk, "resid_sd_los_mm": sd,
                     **{f"coef_{k}": float(b) for k, b in zip(terms, beta, strict=True)}})
    tab = pd.DataFrame(rows)
    tab["flag"] = np.where(tab["n_kept"] >= C.STABLE_MIN_PIXELS, 0, 9)
    return out, tab


def vertical(los: np.ndarray, geom: dict, u_const: float | None = None) -> np.ndarray:
    U = np.full_like(geom["U"], u_const) if u_const is not None else geom["U"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (los / U[None]).astype(np.float32)


# ----------------------------------------------------------------------------- twin cells
def cell_groups(geom: dict) -> tuple[np.ndarray, np.ndarray]:
    """(flat pixel indices sorted by cell, their cell ids) for pixels inside active cells."""
    cell = geom["cell"].ravel()
    idx = np.flatnonzero(cell >= 0)
    order = np.argsort(cell[idx], kind="stable")
    return idx[order], cell[idx][order]


def to_cells(field: np.ndarray, valid: np.ndarray, geom: dict, n_cells: int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median over valid pixels per active cell for each epoch of ``field`` (T, L, W) or a
    single map (L, W). Returns (value (T, A), robust se (T, A), n_valid (A,)); cells with
    fewer than CELL_MIN_PIXELS valid pixels are NaN."""
    f = field[None] if field.ndim == 2 else field
    pix, cid = cell_groups(geom)
    T = f.shape[0]
    flat = f.reshape(T, -1)[:, pix]
    ok = valid.ravel()[pix][None] & np.isfinite(flat)
    df_n = pd.Series(ok[0].astype(int)).groupby(cid).sum()
    nval = np.zeros(n_cells, int)
    nval[df_n.index.to_numpy()] = df_n.to_numpy()
    val = np.full((T, n_cells), np.nan, np.float32)
    se = np.full((T, n_cells), np.nan, np.float32)
    for t in range(T):
        v = np.where(ok[t], flat[t], np.nan)
        s = pd.Series(v)
        g = s.groupby(cid)
        med = g.median()
        mad = g.apply(lambda z: np.nanmedian(np.abs(z - np.nanmedian(z)))
                      if np.isfinite(z).any() else np.nan)
        cnt = g.count()
        ids = med.index.to_numpy()
        val[t, ids] = med.to_numpy()
        se[t, ids] = (1.2533 * MAD * mad / np.sqrt(np.maximum(cnt, 1))).to_numpy()
    low = nval < C.CELL_MIN_PIXELS
    val[:, low] = np.nan
    se[:, low] = np.nan
    return (val[0], se[0], nval) if field.ndim == 2 else (val, se, nval)


def n_active() -> int:
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    return int(np.asarray(z["mask"]).sum())


# ----------------------------------------------------------------------------- seasonal fit
def decimal_years(dates: pd.DatetimeIndex) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return np.asarray(d.year + (d.dayofyear - 0.5) / np.where(d.is_leap_year, 366.0, 365.0),
                      dtype=float)


def harmonic_fit(series: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """Per-pixel LS of a + b t + annual + semiannual on (T, ...) data; pixels need every epoch
    finite. Returns amplitude (annual, same units), phase (day of year of the annual
    maximum), rate (per year) and the residual sd."""
    t = decimal_years(dates)
    t0 = t - t.mean()
    w = 2 * np.pi * t
    X = np.column_stack([np.ones_like(t), t0, np.cos(w), np.sin(w), np.cos(2 * w),
                         np.sin(2 * w)])
    shp = series.shape[1:]
    Y = series.reshape(series.shape[0], -1).astype(float)
    ok = np.isfinite(Y).all(0)
    amp = np.full(Y.shape[1], np.nan)
    ph = np.full(Y.shape[1], np.nan)
    rate = np.full(Y.shape[1], np.nan)
    rsd = np.full(Y.shape[1], np.nan)
    if ok.any():
        B, *_ = np.linalg.lstsq(X, Y[:, ok], rcond=None)
        res = Y[:, ok] - X @ B
        amp[ok] = np.hypot(B[2], B[3])
        ph[ok] = (np.degrees(np.arctan2(B[3], B[2])) % 360.0) / 360.0 * 365.25
        rate[ok] = B[1]
        rsd[ok] = res.std(0, ddof=X.shape[1])
    return {k: v.reshape(shp) for k, v in (("amp", amp), ("phase_doy", ph), ("rate", rate),
                                            ("resid_sd", rsd))}


# ----------------------------------------------------------------------------- GNSS
WRA_META_COLS = ["_station_id", "Latitude_4326", "Longitude_4326", "X_3826", "Y_3826"]


def wra_meta() -> pd.DataFrame:
    """WRA GNSS station positions (explicit columns; no address or contact field)."""
    p = C.REPO / "chou-shui-data" / "data" / "ls_cache" / "ls-wra-gnss-obs__meta.parquet"
    m = pd.read_parquet(p, columns=WRA_META_COLS)
    return m.rename(columns={"_station_id": "station"}).set_index("station")


def wra_files() -> dict[str, Path]:
    d = C.REPO / "chou-shui-data" / "data" / "ls_cache"
    return {p.name.split("__", 1)[1][:-8]: p for p in sorted(d.glob("ls-wra-gnss-obs__*.parquet"))
            if not p.name.endswith("__meta.parquet")}


def read_wra(path: Path) -> pd.Series:
    """Daily ellipsoidal height (m) of one WRA station (raw vendor cache, L0)."""
    df = pd.read_parquet(path, columns=["value"])
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    return s[~s.index.duplicated(keep="first")].sort_index().dropna()


def smooth_daily(s: pd.Series, days: int = C.GNSS_SMOOTH_DAYS) -> pd.Series:
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    return full.rolling(days, center=True, min_periods=3).median()


def at_epochs(s: pd.Series, dates: pd.DatetimeIndex, tol_d: int = C.GNSS_MATCH_TOL_D
              ) -> pd.Series:
    s = s.dropna()
    return s.reindex(pd.DatetimeIndex(dates), method="nearest",
                     tolerance=pd.Timedelta(days=tol_d))


def station_pixel(geom: dict, x: float, y: float) -> tuple[int, int, float]:
    d2 = (geom["x"] - x) ** 2 + (geom["y"] - y) ** 2
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return int(i), int(j), float(np.sqrt(d2[i, j]))


def station_series(field: np.ndarray, valid: np.ndarray, i: int, j: int,
                   half: int = C.GNSS_PIX_HALF, min_valid: int = C.GNSS_PIX_MIN_VALID
                   ) -> np.ndarray:
    """Median of the (2 half + 1)^2 window per epoch; NaN when fewer than min_valid."""
    sl = (slice(None), slice(max(i - half, 0), i + half + 1), slice(max(j - half, 0),
                                                                    j + half + 1))
    w = np.where(valid[sl[1:]][None], field[sl], np.nan).reshape(field.shape[0], -1)
    n = np.isfinite(w).sum(1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(w, 1)
    med[n < min_valid] = np.nan
    return med


def window_mean(s: pd.Series, centre: str, half_d: int = C.GNSS_HALF_WINDOW_D) -> tuple[float,
                                                                                     int]:
    c = pd.Timestamp(centre)
    w = s.loc[c - pd.Timedelta(days=half_d):c + pd.Timedelta(days=half_d)].dropna()
    return (float(w.mean()) if len(w) else np.nan), int(len(w))


def loso(Ins: pd.DataFrame, Gn: pd.DataFrame, t0: str, t1: str,
         half_d: int = C.GNSS_HALF_WINDOW_D, min_ep: int = C.GNSS_MIN_EPOCHS_PER_WINDOW,
         tie: str = "loso") -> pd.DataFrame:
    """14-month change error per station. ``Ins`` and ``Gn`` (mm, epochs x stations) hold the
    InSAR vertical-equivalent and the GNSS height at the same epochs. ``tie``:

    * ``loso``: per epoch, subtract the mean over the OTHER stations of (Ins - Gn) (each
      station's Ins - Gn demeaned over the analysis epochs: an offset-only tie, no rate or
      plane);
    * ``none``: no GNSS tie (the stable-area reference only)."""
    lo = pd.Timestamp(t0) - pd.Timedelta(days=half_d)
    hi = pd.Timestamp(t1) + pd.Timedelta(days=half_d)
    sel = (Ins.index >= lo) & (Ins.index <= hi)
    Ins, Gn = Ins.loc[sel], Gn.loc[sel]
    D = Ins - Gn
    D = D - D.mean()
    rows = []
    for k in Ins.columns:
        if tie == "loso":
            others = [o for o in Ins.columns if o != k]
            off = D[others].mean(axis=1)
            cal = Ins[k] - off
        else:
            cal = Ins[k]
        both = cal.notna() & Gn[k].notna()
        c, g = cal.where(both), Gn[k].where(both)
        c0, n0 = window_mean(c, t0, half_d)
        c1, n1 = window_mean(c, t1, half_d)
        g0, _ = window_mean(g, t0, half_d)
        g1, _ = window_mean(g, t1, half_d)
        ok = n0 >= min_ep and n1 >= min_ep
        rows.append({"station": k, "n_ep_w0": n0, "n_ep_w1": n1, "evaluable": ok,
                     "insar_change_mm": c1 - c0 if ok else np.nan,
                     "gnss_change_mm": g1 - g0 if ok else np.nan,
                     "err_mm": (c1 - c0) - (g1 - g0) if ok else np.nan,
                     "ts_rms_mm": float(np.sqrt(np.nanmean(((c - g) - (c - g).mean()) ** 2)))
                     if both.any() else np.nan})
    return pd.DataFrame(rows)


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_js))


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)
