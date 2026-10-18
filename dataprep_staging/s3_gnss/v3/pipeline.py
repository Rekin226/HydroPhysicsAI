"""WP S3 family pipelines (v3 code, repair round 3 S3r, 2026-10-18): WRA (+ NCKU) and NGL, with
the detection common modes and the product common modes.

v3 change against v2 (R6, design-time evidence, repair report Part 2): the WRA DETECTION common
mode is the median of the 5 nearest member stations (leave-one-out), not of every member. The
residual noise has a spatially correlated part that a neighbourhood median removes better: at
the injection-eligible days of the pre-origin base runs the share of step-size errors beyond
5 mm falls from 8.0 % to 7.0 % (level difference, +-10 d) and from 12.3 % to 11.1 % (v2 model).
The product common modes (``product_cms``), the NGL detection common mode, the station
policies and the passes are unchanged.

v2 change against v1: the WRA station policy depends on the variant (``wra_policy``).
The plan drops LNJS and YSLL and flags ANES on full-period evidence (LNJS +53.6 cm/yr
with sd 1.3 m, YSLL 14 steps, ANES 68 jumps), all of it in 2023-25: in 2020-22 the three
stations are as clean as the rest (LNJS 5, YSLL 10, ANES 6 day-to-day jumps > 3 cm, against
33 at GFES and 25 at TKJS). A pre-origin product may not rest on that evidence (U3/U7), so
the pre-origin policy keeps every station and uses all of them in the common modes; the
full (after-the-fact) variant applies the plan's drops and flag. Likewise for NGL
(``ngl_policy``): the pre-origin variant holds only stations chosen by criteria valid at the
origin (the fan box, the twin grid) and draws its off-fan reference from the plan's height-
based candidates inside the box; LSB0 and the regional continuity set (chosen for their
records after 2024-04) are in the full variant only.

Pass structure per family: (1) every station cleaned without a common mode; (2) the
detection common mode is computed from the pass-1 residuals and every station is cleaned
again with it subtracted inside detection and the local step estimates; (3) once more with
the detection common mode recomputed from pass 2. The output values never have a common
mode removed; the product common modes are separate series.

Detection common modes (used only inside detection):
* WRA (v3): daily median of the pass residuals of the 5 nearest member stations (v2: every
  member; the pilot's on-fan median, full band). Network-coherent motion, real (the 2021
  drought-year subsidence) or not, is therefore never "corrected" as a station step.
* NGL: the off-fan reference median (leave-one-out for the reference stations themselves).

Product common modes (``common_mode.parquet``; design evidence in ``tune.py`` on 2020-22):
* ``ngl_offfan``: daily median residual of the off-fan NGL reference stations (>= 2).
* ``wra_hf``: the WRA network median residual minus its centred running mean over
  ``Params.hp_window`` days, i.e. the high-frequency part only. The window is the largest
  whose high-frequency part has |corr| <= 0.05 with both the NGL off-fan and the NGL
  fan-station medians on 2020-22 (``tune.py`` writes it into the tuned parameters; in v1 the
  rule selected 21 d with the final detector but the product kept round r1's 31 d).
* ``wra_onfan_fullband``: the pilot's on-fan median (diagnostic only: it contains real
  fan-wide deformation and must not be removed from the series).
* ``ngl_offfan`` applied to WRA is also scored (it does not reduce WRA scatter on 2020-22).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import io_gnss as io
import model as M
import numpy as np
import pandas as pd

ORIGIN_END = pd.Timestamp("2022-12-31")

#: plan WP S3 drops and flag; full-period evidence (2023-25), so applied to the full
#: (after-the-fact) variant only (``wra_policy``)
WRA_DROP = {
    "LNJS": "plan WP S3: fitted rate +53.6 cm/yr and residual sd 1.3 m (pilot, full period; "
            "the pathology is in 2023-25); dropped from the full variant only",
    "YSLL": "plan WP S3: 14 steps (pilot, full period; the pathology is in 2023-25); dropped "
            "from the full variant only",
}
WRA_FLAG = {"ANES": "plan WP S3: flagged (68 day-to-day jumps > 3 cm in the audit, full period; "
                    "they start in 2024-07); full variant only: excluded from the common modes "
                    "and jump-cluster days flag 9"}


def wra_policy(window_end: pd.Timestamp | None) -> tuple[dict[str, str], dict[str, str]]:
    """(drop, flag) for a family run: none for a pre-origin run (series cut at or before
    2022-12-31), the plan's for the full records."""
    if _is_preorigin(window_end):
        return {}, {}
    return dict(WRA_DROP), dict(WRA_FLAG)
#: plan WP S3 step catalogue (dates as printed; refined +-10 days if detection misses them)
WRA_CATALOGUE = {
    "TKJS": [("2022-06-17", "plan catalogue +13.2 cm on 2022-06-17"),
             ("2024-01-01", "plan catalogue -9.5 cm around 2024-01-01")],
    "SJES": [("2022-03-29", "plan catalogue +10.0 cm on 2022-03-29")],
    "GFES": [("2020-07-10", "plan catalogue -5.6 cm on 2020-07-10")],
}
NGL_FAN = ["CHUA", "LUKN", "PINT", "S103", "ERLN", "PAKU", "FUNY", "GUKN", "DNAN", "HUWE",
           "PUS2"]
NGL_NORTH = ["LSB0"]
NGL_REGIONAL = ["CKSV", "JUNA", "TCMS", "S01R"]
#: inside an active twin cell (939) but east of the plan's box (120.68E); added 2026-10-12
NGL_GRID_EXTRA = ["JSAN"]
NGL_ALL = NGL_FAN + NGL_GRID_EXTRA + NGL_NORTH + NGL_REGIONAL
#: v2: station sets by variant. The fan-box stations (a spatial criterion) and JSAN (inside an
#: active twin cell) are valid at the origin. LSB0 was singled out by the plan for its record
#: to 2026-09, and v1 added CKSV, JUNA, TCMS and S01R because they still produce after
#: 2024-04: both choices rest on post-origin availability, so these five stations are in the
#: full variant only.
NGL_PREORIGIN = NGL_FAN + NGL_GRID_EXTRA
#: plan WP S3 off-fan candidates inside the fan box, chosen by station height (unverified)
NGL_REF_CANDIDATES_PRE = ["PINT", "PAKU", "S103", "GUKN", "CHUA"]
#: full variant: plus LSB0 (plan) and the regional continuity set (v1)
NGL_REF_CANDIDATES = NGL_REF_CANDIDATES_PRE + ["LSB0"] + NGL_REGIONAL


def _is_preorigin(window_end) -> bool:
    return window_end is not None and pd.Timestamp(window_end) <= ORIGIN_END


def ngl_policy(window_end) -> tuple[list[str], list[str]]:
    """(stations, reference candidates) of the NGL family for a pre-origin or full run."""
    if _is_preorigin(window_end):
        return list(NGL_PREORIGIN), list(NGL_REF_CANDIDATES_PRE)
    return list(NGL_ALL), list(NGL_REF_CANDIDATES)
#: reference rule (fixed on pre-origin data): off the twin grid and |pre-origin rate| <= 5
REF_MAX_RATE = 0.005
REF_MIN_YEARS = 2.0


@dataclass
class Family:
    name: str
    results: dict[str, M.StationResult]
    detect_cm: dict[str, pd.Series] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def resid_frame(self, stations: list[str] | None = None) -> pd.DataFrame:
        cols = {}
        for k, r in self.results.items():
            if stations is not None and k not in stations:
                continue
            ok = r.ok & ~r.untied
            cols[k] = pd.Series(np.where(ok, r.fit.resid, np.nan), index=r.dates)
        return pd.DataFrame(cols)


def _loo_median(R: pd.DataFrame, members: list[str]) -> dict[str, pd.Series]:
    """Leave-one-out daily median over ``members`` for every column of R (members excluded
    from their own)."""
    out = {}
    base = R[[m for m in members if m in R]]
    for k in R:
        cols = [m for m in base if m != k]
        out[k] = base[cols].median(axis=1) if cols else pd.Series(0.0, index=R.index)
    return out


#: v3 R6: neighbours in the WRA detection common mode
KNN_WRA = 5


def wra_station_xy() -> dict[str, tuple[float, float]]:
    """TWD97 TM2 coordinates of the WRA stations (vendor meta; no contact field is read)."""
    m = io.wra_meta()
    return {k: (float(r.X_3826), float(r.Y_3826)) for k, r in m.iterrows()}


def _knn_median(R: pd.DataFrame, members: list[str], xy: dict[str, tuple[float, float]],
                k: int = KNN_WRA) -> dict[str, pd.Series]:
    """For every column of R: the daily median over the ``k`` nearest ``members`` (itself
    excluded); falls back to every member when coordinates are missing."""
    out = {}
    base = [m for m in members if m in R]
    for s in R:
        cand = [m for m in base if m != s]
        if s in xy and all(m in xy for m in cand):
            cand = sorted(cand, key=lambda m: np.hypot(xy[m][0] - xy[s][0],
                                                       xy[m][1] - xy[s][1]))[:k]
        out[s] = R[cand].median(axis=1) if cand else pd.Series(0.0, index=R.index)
    return out


def _inject(s: pd.Series, inj: list[tuple[pd.Timestamp, float]] | None) -> pd.Series:
    if not inj:
        return s
    s = s.copy()
    for d, c in inj:
        s.loc[s.index >= pd.Timestamp(d)] += c
    return s


# ----------------------------------------------------------------------------- WRA
def wra_series(window_end: pd.Timestamp | None) -> dict[str, pd.Series]:
    out = {}
    for k in io.wra_files():
        s = io.read_wra(k)
        if window_end is not None:
            s = s.loc[:window_end]
        if len(s) >= 60:
            out[k] = s
    return out


def run_wra(p: M.Params, window_end: pd.Timestamp | None = None,
            inject: dict[str, list] | None = None, stations: list[str] | None = None,
            n_pass: int = 3, series: dict[str, pd.Series] | None = None) -> Family:
    ser = series if series is not None else wra_series(window_end)
    if stations is not None:
        ser = {k: v for k, v in ser.items() if k in stations}
    cat = {k: [(d, t) for d, t in v if window_end is None or pd.Timestamp(d) <= window_end]
           for k, v in WRA_CATALOGUE.items()}
    drop, flag = wra_policy(window_end)
    proc = {k: v for k, v in ser.items() if k not in drop}
    members = [k for k in proc if k not in flag]
    xy = wra_station_xy()
    fam = Family("wra", {})
    cm = None
    for ipass in range(n_pass):
        res = {}
        for k, s in proc.items():
            s2 = _inject(s, (inject or {}).get(k))
            res[k] = M.process_station(s2, p, cm=None if cm is None else cm.get(k),
                                       catalogue=cat.get(k))
        fam.results = res
        if ipass < n_pass - 1:
            cm = _knn_median(fam.resid_frame(), members, xy)
    fam.detect_cm = cm or {}
    fam.info = {"members": members, "dropped": drop, "flagged": flag, "catalogue": cat,
                "policy": "preorigin" if not drop and not flag else "full (plan drops/flag)",
                "detection_cm": f"median of the {KNN_WRA} nearest members (v3 R6)"}
    return fam


def run_ncku(p: M.Params, window_end: pd.Timestamp | None = None) -> Family:
    res = {}
    for st in io.ncku_meta().index:
        s = io.read_ncku(st)
        if window_end is not None:
            s = s.loc[:window_end]
        if len(s) >= 60:
            res[f"NCKU_{st}"] = M.process_station(s, p)
    return Family("ncku", res, info={"note": "NCKU daily solutions, 2023-04..2025-03; no "
                                             "common mode (4 stations, own processing)"})


# ----------------------------------------------------------------------------- NGL
def ngl_series(window_end: pd.Timestamp | None, stations: list[str] | None = None
               ) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    if stations is None:
        stations = ngl_policy(window_end)[0]
    ser, raw = {}, {}
    for k in stations:
        df = io.read_tenv3(io.ngl_tenv3_path(k))
        if window_end is not None:
            df = df.loc[:window_end]
        raw[k] = df
        s = df["up"].copy()
        s.name = k
        ser[k] = s
    return ser, raw


def ngl_metadata(stations: list[str], window_end: pd.Timestamp | None
                 ) -> tuple[dict[str, list], dict[str, list]]:
    st = io.read_ngl_steps(io.NGL_S3 / "steps.txt", set(stations))
    eq, ev = {}, {}
    for k in stations:
        sub = st[st.station == k] if len(st) else st
        if window_end is not None and len(sub):
            sub = sub[sub.date <= window_end]
        ev[k] = [(r.date, r.detail) for r in sub.itertuples() if r.code == 1]
        eq[k] = [(r.date, r.detail) for r in sub.itertuples() if r.code == 2]
    return ev, eq


def run_ngl(p: M.Params, ref: list[str], window_end: pd.Timestamp | None = None,
            inject: dict[str, list] | None = None, stations: list[str] | None = None,
            n_pass: int = 3, series: dict[str, pd.Series] | None = None) -> Family:
    if stations is None:
        stations = ngl_policy(window_end)[0]
    ser = series if series is not None else ngl_series(window_end, stations)[0]
    ser = {k: v for k, v in ser.items() if k in stations}
    equip, quakes = ngl_metadata(list(ser), window_end)
    fam = Family("ngl", {})
    cm = None
    for ipass in range(n_pass):
        res = {}
        for k, s in ser.items():
            s2 = _inject(s, (inject or {}).get(k))
            res[k] = M.process_station(s2, p, cm=None if cm is None else cm.get(k),
                                       equipment=equip.get(k), earthquakes=quakes.get(k))
        fam.results = res
        if ipass < n_pass - 1:
            cm = _loo_median(fam.resid_frame(), ref)
    fam.detect_cm = cm or {}
    fam.info = {"reference": list(ref), "equipment": {k: [(str(d.date()), t) for d, t in v]
                                                      for k, v in equip.items()}}
    return fam


def select_reference(fam_pre: Family, grid_npz, candidates: list[str]
                     ) -> tuple[list[str], pd.DataFrame]:
    """Off-fan reference rule applied to a PRE-ORIGIN family run: candidate, off the twin
    grid, >= 2 years of data, |rate| <= 5 mm/yr."""
    rows = []
    for k in candidates:
        r = fam_pre.results.get(k)
        if r is None:
            continue
        df = io.read_tenv3(io.ngl_tenv3_path(k))
        x, y = io.lonlat_to_twd97(np.array([df.lon.median()]), np.array([df.lat.median()]))
        cell = int(io.grid_cells(np.column_stack([x, y]), grid_npz)[0])
        n_years = float(r.ok.sum()) / 365.25
        rate = r.fit.rate
        keep = cell < 0 and n_years >= REF_MIN_YEARS and abs(rate) <= REF_MAX_RATE
        rows.append({"station": k, "cell": cell, "years_preorigin": round(n_years, 2),
                     "rate_preorigin_mm_yr": round(1000 * rate, 2),
                     "annual_amp_mm": round(1000 * float(np.hypot(*r.fit.beta[2:4])), 2),
                     "sigma_hf_mm": round(1000 * r.sigma_hf, 2), "reference": keep})
    tab = pd.DataFrame(rows)
    return tab.loc[tab.reference, "station"].tolist(), tab


# ----------------------------------------------------------------------------- CMs
def product_cms(wra: Family | None, ngl: Family | None, ref: list[str], p: M.Params
                ) -> pd.DataFrame:
    parts = []
    if ngl is not None:
        c = M.median_cm(ngl.resid_frame(ref), min_n=2)
        parts.append(c.rename(columns={"cm": "ngl_offfan", "n": "ngl_offfan_n",
                                       "sd": "ngl_offfan_sd"}))
    if wra is not None:
        members = wra.info["members"]
        R = wra.resid_frame(members)
        hf = M.highpass_cm(R, p.hp_window, min_n=5)
        parts.append(hf.rename(columns={"cm": "wra_hf", "n": "wra_hf_n", "sd": "wra_hf_sd"}))
        fb = M.median_cm(R, min_n=5)
        parts.append(fb[["cm"]].rename(columns={"cm": "wra_onfan_fullband"}))
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "date"
    return out
