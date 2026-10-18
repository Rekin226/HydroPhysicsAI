"""WP S1b repair round r3 (S1br): guarded reads of the REPAIRED products.

All three gates failed, so each product is opened for ``after_the_fact_analysis`` only:

* ``rings_bridged/v3`` (S1a r3): ring_corrections.csv and the per-site survey files (the raw
  depths are those of v2; the bridged values, P(A) and members are v3's);
* ``gnss_clean/full/v4`` (S3r): WRA and NGL daily heights, stations, steps, common modes (same
  columns as v2);
* ``insar_069A/v3`` (S4b r2): the merged multilooked field (``vertical_ml.npz``) and the cell
  field (``vertical_cells.npz``, both reference variants).

The functions mirror ``s1b_io`` (registered round-1 module, unchanged) so that the registered
computation functions of ``s1b_run`` run unchanged through :class:`Adapter`. The only rule that
had to be restated is the InSAR pixel rule, because v3 has no network segments:

    a pixel of the 3 x 3 window around the nearest multilooked pixel is usable when it lies in
    the primary frame's LiCSBAS15 mask and carries no bridged epoch at any epoch (v2's rule:
    segment ids equal to the network's at every epoch); at an epoch its value counts when it is
    finite and its source code is 1, 2 or 3 (primary, tied second frame, or their mean; code 4,
    an untied second-frame value, is excluded); the station value is the median of >= 4 such
    values. The network has one segment, so ``network_segment`` is all zeros.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import guard  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_io as IO1  # noqa: E402

USE = "after_the_fact_analysis"
RINGS_V3 = K.DATAPREP / "rings_bridged" / "v3"
GNSS_FULL_V4 = K.DATAPREP / "gnss_clean" / "full" / "v4"
INSAR_V3 = K.DATAPREP / "insar_069A" / "v3"
RINGS_V2 = K.RINGS_V2
OK_SOURCES = (1, 2, 3)


# ----------------------------------------------------------------------------- rings
def rings(product_dir: Path = RINGS_V3, *, values: bool = True) -> tuple[dict, list[str]]:
    """``s1b_io.rings`` for any rings_bridged version (same rule: the top-to-bottom interval of
    the raw depths at surveys where both rings are read; zero at the first survey)."""
    prod = guard.open_product(product_dir, USE)
    rc = prod.read_table("ring_corrections.csv")
    out = {}
    for _, r in rc.iterrows():
        site = r["site"]
        df = prod.read_table(IO1._site_file(prod, site),
                             columns=["datetime", "ring", "depth_raw_m", "flag"])
        p = df.pivot_table(index="datetime", columns="ring", values="depth_raw_m",
                           aggfunc="mean")
        p.index = pd.DatetimeIndex(pd.to_datetime(p.index)).astype("datetime64[ns]")
        p = p.sort_index()
        S = (p[r["bottom_ring"]] - p[r["top_ring"]]).dropna()
        t_lp, t_fp = pd.Timestamp(r["last_pre_survey"]), pd.Timestamp(r["first_post_survey"])
        s_lp = S.loc[:t_lp + pd.Timedelta(hours=23, minutes=59)]
        s_fp = S.loc[t_fp:]
        S_lp, S_fp = float(s_lp.iloc[-1]), float(s_fp.iloc[0])
        c = -1000.0 * (S - S.iloc[0])
        raw_mm = 1000.0 * (S_lp - S_fp)
        out[site] = {
            "site": site, "t": S.index, "c": c.to_numpy(float) if values else None,
            "t_lp": s_lp.index[-1], "t_fp": s_fp.index[0],
            "L_mm": 1000.0 * float(r["L_m"]), "sd_s_mm": float(r["sd_s_mm"]),
            "gap_yr": float(r["gap_yr"]), "p_A": float(r["p_A"]), "primary": r["primary"],
            "assigned": r["assigned"] if isinstance(r["assigned"], str) else "",
            "members": r["members_carried"],
            "C": ({"primary": 10.0 * float(r["C_gap_primary_cm"]),
                   "A": 10.0 * float(r["C_gap_A_cm"]), "B": 10.0 * float(r["C_gap_B_cm"]),
                   "raw": raw_mm} if values else None),
            "raw_check_mm": (raw_mm - (-10.0 * float(r["dS_obs_cm"]))) if values else None,
            "b_A_ppm": float(r["b_A_ppm"]), "b_B_ppm": float(r["b_B_ppm"]),
            "sigma_C_primary_mm": 10.0 * float(r["sigma_C_primary_cm"]),
        }
    return out, list(prod.notes)


def ring_resid_sd(r: dict, win=K.PRE_WIN) -> float:
    """Residual sd (mm) of the ring's own trend + annual fit inside ``win`` clipped to the
    last pre-gap survey (the ring noise level of the verifier's sensitivity; ring values
    only, no instrument)."""
    from s1_rings.s1b import s1b_core as C
    t = pd.DatetimeIndex(r["t"])
    hi = min(pd.Timestamp(win[1]), pd.Timestamp(r["t_lp"]))
    m = (t >= pd.Timestamp(win[0])) & (t <= hi) & np.isfinite(r["c"])
    y_mid = C.years([pd.Timestamp(r["t_lp"])])[0]
    f = C.fit_pre(C.years(t[m]), np.asarray(r["c"])[m], y_mid, True)
    return float(f.resid_sd) if f is not None else float("nan")


# ----------------------------------------------------------------------------- GNSS
def gnss(product_dir: Path = GNSS_FULL_V4) -> dict:
    prod = guard.open_product(product_dir, USE)
    cols = ["family", "station", "date", "value", "value_raw", "correction_m", "flag"]
    daily = pd.concat([prod.read_table("wra/daily.parquet", columns=cols),
                       prod.read_table("ngl/daily.parquet", columns=cols)], ignore_index=True)
    daily["date"] = pd.DatetimeIndex(pd.to_datetime(daily["date"])).astype("datetime64[ns]")
    return {"daily": daily, "stations": prod.read_table("stations.csv"),
            "steps": prod.read_table("steps.csv"),
            "common_mode": prod.read_table("common_mode.parquet"), "notes": list(prod.notes)}


def gnss_availability(product_dir: Path = GNSS_FULL_V4) -> dict:
    """Dates and flags of the daily series (no height column), stations and steps."""
    prod = guard.open_product(product_dir, USE)
    cols = ["family", "station", "date", "flag"]
    daily = pd.concat([prod.read_table("wra/daily.parquet", columns=cols),
                       prod.read_table("ngl/daily.parquet", columns=cols)], ignore_index=True)
    daily["date"] = pd.DatetimeIndex(pd.to_datetime(daily["date"])).astype("datetime64[ns]")
    return {"daily": daily, "stations": prod.read_table("stations.csv"),
            "steps": prod.read_table("steps.csv"), "notes": list(prod.notes)}


# ----------------------------------------------------------------------------- InSAR
def insar(product_dir: Path = INSAR_V3, *, values: bool = True) -> dict:
    """The merged multilooked field with the v3 pixel-rule inputs, and both cell variants
    (``value`` = M3 reference, ``value_round1_plane_reference``); cell values with a flag
    other than 0 are set to NaN (S4b: use flag-0 values for changes)."""
    prod = guard.open_product(product_dir, USE)
    ml = np.load(prod.path("vertical_ml.npz"), allow_pickle=False)
    cells = np.load(prod.path("vertical_cells.npz"), allow_pickle=False)
    dates = pd.DatetimeIndex(pd.to_datetime(ml["dates"])).astype("datetime64[ns]")
    out = {"x": ml["x"], "y": ml["y"], "licsbas15_mask": ml["licsbas15_mask_primary"],
           "any_bridged": ml["bridged"].any(axis=0), "dates": dates,
           "network_segment": np.zeros(len(dates), dtype=int),
           "tie_epoch": str(ml["tie_epoch"])}
    cflag = np.asarray(cells["flag"])
    cd = pd.DatetimeIndex(pd.to_datetime(cells["dates"])).astype("datetime64[ns]")
    out["cell_dates"] = cd
    out["cell_segment"] = np.zeros(cflag.shape, dtype=int)
    if values:
        out["value"] = ml["value"]
        out["source"] = ml["source"]
        for key, name in (("cell_value", "value"),
                          ("cell_value_plane", "value_round1_plane_reference")):
            v = np.asarray(cells[name], dtype=float).copy()
            v[cflag != 0] = np.nan
            out[key] = v
    else:
        out["finite"] = np.isfinite(ml["value"])
        out["source"] = ml["source"]
    out["notes"] = list(prod.notes)
    return out


def with_plane_cells(ins: dict) -> dict:
    """A shallow copy of ``ins`` whose cell values are the plane-reference variant."""
    o = dict(ins)
    o["cell_value"] = ins["cell_value_plane"]
    return o


def pixel_rule(value, source, mask, any_bridged, i: int, j: int, *, half: int = 1,
               min_valid: int = 4, values: bool = True) -> dict:
    """The v3 station rule at multilooked pixel (i, j) (see module docstring). ``value``
    (T, ny, nx) up-positive mm (or a finite mask when ``values=False``), ``source`` (T, ny,
    nx) int, ``mask`` and ``any_bridged`` (ny, nx) bool."""
    sl = (slice(max(i - half, 0), i + half + 1), slice(max(j - half, 0), j + half + 1))
    pix_ok = np.asarray(mask[sl], bool) & ~np.asarray(any_bridged[sl], bool)
    src = np.asarray(source[(slice(None),) + sl])
    v = np.asarray(value[(slice(None),) + sl])
    fin = np.isfinite(v) if values else v.astype(bool)
    good = fin & pix_ok[None] & np.isin(src, OK_SOURCES)
    T = good.shape[0]
    n = good.reshape(T, -1).sum(1)
    out = {"n_pix_ok": int(pix_ok.sum()), "n_valid": n,
           "n_source4": int((np.isfinite(v) if values else v.astype(bool))[
               np.asarray(src) == 4].sum())}
    if values:
        w = np.where(good, v.astype(float), np.nan).reshape(T, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(w, 1)
        med[n < min_valid] = np.nan
        out["down_mm"] = -med
    return out


def insar_pixel_series(ins: dict, x: float, y: float, *, values: bool = True) -> dict:
    d2 = (ins["x"] - x) ** 2 + (ins["y"] - y) ** 2
    i, j = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
    val = ins["value"] if values else ins["finite"]
    px = pixel_rule(val, ins["source"], ins["licsbas15_mask"], ins["any_bridged"], int(i),
                    int(j), half=K.INSAR_PIX_HALF, min_valid=K.INSAR_PIX_MIN_VALID,
                    values=values)
    px.update({"i": int(i), "j": int(j), "dist_m": float(math.sqrt(d2[i, j]))})
    return px


def insar_cell_series(ins: dict, cell: int) -> dict:
    if cell < 0:
        return {"down_mm": np.full(len(ins["cell_dates"]), np.nan),
                "segment": np.zeros(len(ins["cell_dates"]), dtype=int)}
    v = np.asarray(ins["cell_value"][cell], dtype=float)
    return {"down_mm": -v, "segment": np.zeros(len(v), dtype=int)}


class Adapter:
    """The IO interface ``s1b_run`` and ``s1b_synth`` call (``IO.<name>``), backed by the
    repaired products. Installed by :func:`install` for the duration of a run."""
    pairs = staticmethod(IO1.pairs)
    gnss_series = staticmethod(IO1.gnss_series)
    gnss_end_var = staticmethod(IO1.gnss_end_var)
    twin_cell = staticmethod(IO1.twin_cell)
    site_xy = staticmethod(IO1.site_xy)
    _site_file = staticmethod(IO1._site_file)
    insar_pixel_series = staticmethod(insar_pixel_series)
    insar_cell_series = staticmethod(insar_cell_series)

    def __init__(self, rings_dir: Path = RINGS_V3, gnss_dir: Path = GNSS_FULL_V4,
                 insar_dir: Path = INSAR_V3):
        self.rings_dir, self.gnss_dir, self.insar_dir = rings_dir, gnss_dir, insar_dir

    def rings(self):
        return rings(self.rings_dir)

    def gnss(self):
        return gnss(self.gnss_dir)

    def insar(self):
        return insar(self.insar_dir)


def install(adapter: Adapter) -> dict:
    """Point the registered modules' ``IO`` name at ``adapter``; returns the previous
    bindings for :func:`restore`."""
    from s1_rings.s1b import s1b_run, s1b_synth
    prev = {"s1b_run": s1b_run.IO, "s1b_synth": s1b_synth.IO}
    s1b_run.IO = adapter
    s1b_synth.IO = adapter
    return prev


def restore(prev: dict) -> None:
    from s1_rings.s1b import s1b_run, s1b_synth
    s1b_run.IO = prev["s1b_run"]
    s1b_synth.IO = prev["s1b_synth"]
