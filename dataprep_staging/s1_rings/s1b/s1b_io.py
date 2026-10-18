"""Guarded reads of the three products WP S1b checks against each other.

All three gates failed, so every product is opened for ``after_the_fact_analysis`` (the only
use the guard opens on a gate-failed product besides design-time evidence):

* ``rings_bridged/v2`` (S1a): ring_corrections.csv and the per-site survey files;
* ``gnss_clean/full/v2`` (S3): WRA and NGL daily heights, stations, steps, common modes;
* ``insar_069A/v2`` (S4b): the multilooked vertical-equivalent field with its segment ids,
  and the per-cell field (report-only variant).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import guard  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402

USE = "after_the_fact_analysis"


# ----------------------------------------------------------------------------- rings
def _site_file(prod, site: str) -> str:
    outs = prod.outputs
    for name in (f"rings_corrected_{site}.parquet", f"rings_corrected_{site}_A.parquet"):
        if name in outs:
            return name
    raise KeyError(f"no corrected-ring file for {site}")


def rings() -> tuple[dict, list[str]]:
    """Per site: the product's gap values (mm) and the raw total-interval series ``c(t)``
    (mm, shortening positive, zero at the first survey), from the surveys where both the top
    and the bottom ring are read (depth_raw_m; the post-gap scale changes rates by 2e-4)."""
    prod = guard.open_product(K.RINGS_V2, USE)
    rc = prod.read_table("ring_corrections.csv")
    out = {}
    for _, r in rc.iterrows():
        site = r["site"]
        df = prod.read_table(_site_file(prod, site),
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
        t_lp_s, t_fp_s = s_lp.index[-1], s_fp.index[0]
        c = -1000.0 * (S - S.iloc[0])
        raw_mm = 1000.0 * (S_lp - S_fp)
        out[site] = {
            "t": S.index, "c": c.to_numpy(float), "t_lp": t_lp_s, "t_fp": t_fp_s,
            "L_mm": 1000.0 * float(r["L_m"]), "sd_s_mm": float(r["sd_s_mm"]),
            "gap_yr": float(r["gap_yr"]), "p_A": float(r["p_A"]), "primary": r["primary"],
            "assigned": r["assigned"] if isinstance(r["assigned"], str) else "",
            "members": r["members_carried"],
            "C": {"primary": 10.0 * float(r["C_gap_primary_cm"]),
                  "A": 10.0 * float(r["C_gap_A_cm"]), "B": 10.0 * float(r["C_gap_B_cm"]),
                  "raw": raw_mm},
            "raw_check_mm": raw_mm - (-10.0 * float(r["dS_obs_cm"])),
            "b_A_ppm": float(r["b_A_ppm"]), "b_B_ppm": float(r["b_B_ppm"]),
            "sigma_C_primary_mm": 10.0 * float(r["sigma_C_primary_cm"]),
        }
    return out, list(prod.notes)


def site_xy() -> dict:
    d = pd.read_csv(K.RING_STATIONS)
    return {r.sub_id: (float(r.x), float(r.y)) for r in d.itertuples()}


# ----------------------------------------------------------------------------- GNSS
def gnss() -> dict:
    prod = guard.open_product(K.GNSS_FULL_V2, USE)
    cols = ["family", "station", "date", "value", "value_raw", "correction_m", "flag"]
    daily = pd.concat([prod.read_table("wra/daily.parquet", columns=cols),
                       prod.read_table("ngl/daily.parquet", columns=cols)], ignore_index=True)
    daily["date"] = pd.DatetimeIndex(pd.to_datetime(daily["date"])).astype("datetime64[ns]")
    return {"daily": daily, "stations": prod.read_table("stations.csv"),
            "steps": prod.read_table("steps.csv"),
            "common_mode": prod.read_table("common_mode.parquet"), "notes": list(prod.notes)}


def pairs(stations: pd.DataFrame, xy: dict) -> dict:
    """Ring-site / GNSS-station pairs by the declared distance rules (metadata only)."""
    st = stations[stations["family"].isin(["wra", "ngl"])].copy()
    st = st[pd.to_datetime(st["first_day"]) <= pd.Timestamp(K.PRE_GAP_END)]
    st = st[pd.to_datetime(st["last_day"]) >= pd.Timestamp(K.POST_WIN[0])]
    out = {"wra_colocated": [], "ngl_near": [], "neighbour": []}
    for site in K.SITES:
        x, y = xy[site]
        d = np.hypot(st["x_twd97"] - x, st["y_twd97"] - y) / 1000.0
        s = st.assign(d_km=d).sort_values("d_km")
        co = s[(s.family == "wra") & (s.d_km <= K.COLOCATED_KM)]
        for r in co.itertuples():
            out["wra_colocated"].append({"site": site, "family": "wra", "station": r.station,
                                         "d_km": round(float(r.d_km), 3)})
        for r in s[(s.family == "ngl") & (s.d_km <= K.NGL_NEAR_KM)].itertuples():
            out["ngl_near"].append({"site": site, "family": "ngl", "station": r.station,
                                    "d_km": round(float(r.d_km), 3)})
        if co.empty:
            nb = s[s.d_km <= K.NEIGHBOUR_KM]
            if len(nb):
                r = nb.iloc[0]
                out["neighbour"].append({"site": site, "family": r.family,
                                         "station": r.station, "d_km": round(float(r.d_km), 3)})
    return out


def gnss_series(g: dict, family: str, station: str, *, flags=K.GNSS_FLAGS,
                steps_mode: str = "product", gap=None, common_mode: str | None = None
                ) -> tuple[pd.DatetimeIndex, np.ndarray, float]:
    """Daily subsidence (mm, + down) of one station, and the variance (mm^2) of the step
    corrections the series carries inside ``gap`` = (t_lp, t_fp): the sum of se_m^2 of every
    corrected step dated strictly between the two surveys. ``steps_mode='catalogue_in_gap'``
    adds back every step dated inside the gap whose basis is not metadata_or_catalogue
    (value = value_raw - correction_m, correction = sum of step sizes after each step date);
    those steps then carry no variance term. ``common_mode``: a column of common_mode.parquet
    to subtract."""
    if steps_mode not in ("product", "catalogue_in_gap"):
        raise ValueError(steps_mode)
    d = g["daily"]
    x = d[(d.family == family) & (d.station == station)].sort_values("date")
    x = x[x["flag"].isin(list(flags))]
    h = x["value"].to_numpy(float).copy()
    t = pd.DatetimeIndex(x["date"])
    var_steps = 0.0
    st = g["steps"]
    st = st[(st.family == family) & (st.station == station)]
    for r in st.itertuples():
        ts = pd.Timestamp(r.date)
        if not (gap is not None and gap[0] < ts < gap[1]):
            continue
        if steps_mode == "catalogue_in_gap" and r.basis != "metadata_or_catalogue":
            h = h + float(r.size_m) * (t >= ts)
        else:
            se = float(r.se_m) if np.isfinite(float(r.se_m)) else 0.0
            var_steps += (1000.0 * se) ** 2
    if common_mode is not None:
        cm = g["common_mode"][["date", common_mode]].copy()
        cm["date"] = pd.DatetimeIndex(pd.to_datetime(cm["date"])).astype("datetime64[ns]")
        cmv = cm.set_index("date")[common_mode].reindex(t).to_numpy(float)
        h = h - cmv
    return t, -1000.0 * h, var_steps


def gnss_end_var(stations: pd.DataFrame, family: str, station: str, n_lp: int,
                 n_fp: int) -> float:
    """Variance (mm^2) of the difference of two endpoint means of n daily heights each:
    sigma_hf^2 (1 + rho)/(1 - rho) / n per endpoint."""
    r = stations[(stations.family == family) & (stations.station == station)].iloc[0]
    s = 1000.0 * float(r["sigma_hf_m"])
    rho = min(max(float(r["rho_hf"]), 0.0), 0.95)
    f = (1 + rho) / (1 - rho)
    return s ** 2 * f * (1.0 / max(n_lp, 1) + 1.0 / max(n_fp, 1))


# ----------------------------------------------------------------------------- InSAR
def insar() -> dict:
    prod = guard.open_product(K.INSAR_V2, USE)
    ml = np.load(prod.path("vertical_ml.npz"), allow_pickle=False)
    cells = np.load(prod.path("vertical_cells.npz"), allow_pickle=False)
    out = {k: ml[k] for k in ("value", "x", "y", "licsbas15_mask", "pix_segment")}
    out["dates"] = pd.DatetimeIndex(pd.to_datetime(ml["dates"])).astype("datetime64[ns]")
    out["network_segment"] = np.asarray(cells["network_segment"])
    out["cell_value"] = cells["value"]
    out["cell_segment"] = cells["segment"]
    out["cell_dates"] = pd.DatetimeIndex(pd.to_datetime(cells["dates"])).astype("datetime64[ns]")
    out["notes"] = list(prod.notes)
    return out


def twin_cell(x: float, y: float) -> int:
    z = np.load(K.TWIN_GRID, allow_pickle=False)
    mask = np.asarray(z["mask"]).astype(bool)
    dx, x0, y0 = float(z["dx"]), float(z["x0"]), float(z["y0"])
    col, row = int(math.floor((x - x0) / dx)), int(math.floor((y - y0) / dx))
    if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]) or not mask[row, col]:
        return -1
    lut = -np.ones(mask.shape, dtype=int)
    r, c = np.nonzero(mask)
    lut[r, c] = np.arange(r.size)
    return int(lut[row, col])


def insar_pixel_series(ins: dict, x: float, y: float, *, values: bool = True) -> dict:
    """S4b's station rule at a ring site: the median over the 3 x 3 multilooked pixels around
    the nearest pixel (>= 4 valid: LiCSBAS15 mask, finite, and segment ids equal to the
    network's at every epoch). ``values=False`` returns the valid-count series only."""
    d2 = (ins["x"] - x) ** 2 + (ins["y"] - y) ** 2
    i, j = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
    h = K.INSAR_PIX_HALF
    sl = (slice(max(i - h, 0), i + h + 1), slice(max(j - h, 0), j + h + 1))
    seg = ins["pix_segment"][(slice(None),) + sl]                    # (T, 3, 3)
    netw = ins["network_segment"][:, None, None]
    pix_ok = ins["licsbas15_mask"][sl] & np.all(seg == netw, axis=0)
    v = ins["value"][(slice(None),) + sl].astype(float)
    good = np.isfinite(v) & pix_ok[None]
    n = good.reshape(len(v), -1).sum(1)
    out = {"i": int(i), "j": int(j), "dist_m": float(np.sqrt(d2[i, j])),
           "n_pix_ok": int(pix_ok.sum()), "n_valid": n}
    if values:
        w = np.where(good, v, np.nan).reshape(len(v), -1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(w, 1)
        med[n < K.INSAR_PIX_MIN_VALID] = np.nan
        out["down_mm"] = -med                                          # + = subsidence
    return out


def insar_cell_series(ins: dict, cell: int) -> dict:
    if cell < 0:
        return {"down_mm": np.full(len(ins["cell_dates"]), np.nan)}
    v = ins["cell_value"][cell].astype(float)
    return {"down_mm": -v, "segment": ins["cell_segment"][cell]}
