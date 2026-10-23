"""Deep benchmarks (深層樁) as a first-class series: pure functions for WP S2.

Two feeds measure the same nine sensors:

* the **vendor feed** (``chou-shui-data/data/ls_cache/ls-wra-dbm-obs__*.parquet``, WiseEnvr
  cache, 2016-06..2025-03), whose processing removed sensor resets by forcing continuity at
  discontinuities, and
* the **raw agency feed** (WRA open data, data.gov.tw 58339; two 100,000-row dumps archived in
  ``data_fetch_api/new_measurements/wra_opendata_dbm_58339/``), which keeps every reset.

Wherever both cover the same hours the vendor reading equals the raw reading plus a piecewise
constant offset (or minus it, after a sign inversion): :func:`identity` measures that and lists
the points where the offset changes, i.e. where the vendor stitched.

The step catalogue treats every discontinuity of a feed (an outage longer than ``GAP_H`` hours,
or a jump inside continuous recording) the same way: the change across it (or its residual
against the change the local rate predicts) is compared with an envelope ``Q(d)``, the 99.9 %
quantile of the same statistic on artificial gaps placed in continuous **pre-origin** data
(2022-12 or earlier). A value above ``max(Q(d), FLOOR_MM)`` is a sensor step and is removed
(flag 1 after it); a smaller one is kept as real movement. At each duration the statistic
(rate-adjusted residual or plain change) with the smaller pre-origin envelope is used.

Readings are in mm, rising = compression. Printed monthly values (cm) are the change of the
daily mean between the last days of consecutive months; :func:`monthly_from_daily` uses the
same convention. Nothing here reads personal fields: :data:`META_COLS` excludes the contact
names, phones and addresses of the station metadata.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO / "chou-shui-data" / "data" / "ls_cache"
VENDOR_KEYS = ("TKJS_281_29", "TKJS_282_29", "TKJS_283_29", "NTUH_281_29", "NTUH_282_29",
               "SJES", "STES", "WNES", "DLIO")
META_PARQUET = VENDOR_DIR / "ls-wra-dbm-obs__meta.parquet"
#: the only metadata columns ever read (never 管理人, 管理人電話, 聯絡人, 聯絡人電話, 學校地址)
META_COLS = ["站 名", "檔名", "資料欄位順序", "設置日期", "深度", "設置單位", "資料頻率",
             "start_datetime_from_dataset", "end_datetime_from_dataset", "_station_id"]
OPEN_DIR = REPO / "data_fetch_api" / "new_measurements" / "wra_opendata_dbm_58339"
OPEN_FILES = ("res58339_default_order.json", "res58339_latest_desc.json")
PRINTED_CSV = REPO / "results" / "twin" / "report_extract" / "deep_piles" / "deep_piles.csv"
PRINTED_REPORT = PRINTED_CSV.parent / "report.json"
SITES_CSV = REPO / "chou-shui-data" / "data" / "subsidence" / "deep_pile_sites.csv"
GRID_NPZ = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"

ORIGIN_MONTH = "2022-12"
ORIGIN_END = pd.Timestamp("2022-12-31 23:00")
DURS_H = (1, 2, 3, 6, 12, 24, 48, 72, 120, 168, 240, 360, 480, 720, 1080, 1440)
SIDE_H = 72            # rate window on each side of a discontinuity (hours)
MIN_SIDE_PTS = 24      # valid hours a rate window needs
GAP_H = 2              # consecutive valid hours further apart than this = an outage
MERGE_H = 3            # candidate discontinuities closer than this merge into one event
FLOOR_MM = 1.0         # never call a change below 1 mm a step (a priori)
Q_LEVEL = 0.999        # envelope quantile: 0.1 % false positives on natural changes
INJECT_MM = (2.0, 5.0, 10.0, 20.0)
INJECT_DURS_H = (1, 6, 24, 72, 168)
DAY_MIN_HOURS = 12     # a day with fewer valid hours is a partial day (flag 2)
BRIDGE_DAYS = 3        # a month end may be taken from up to 3 days earlier (flag 2)

#: printed (pile, depth) -> sensor id of the agency feed (identifier without the BM_ prefix)
PRINTED_TO_SENSOR = {("土庫國中", 300): "TKJS(1)", ("土庫國中", 130): "TKJS(2)",
                     ("土庫國中", 400): "TKJS(3)", ("中科虎尾", 300): "NTUH(1)",
                     ("中科虎尾", 70): "NTUH(2)", ("溪州國小", 300): "SJES",
                     ("秀潭國小", 70): "STES", ("灣內國小", 100): "WNES",
                     ("頂寮安檢所", 150): "DLIO"}


# ------------------------------------------------------------------------------- loading
def as_ns(s: pd.Series) -> pd.Series:
    """Copy of ``s`` with a nanosecond DatetimeIndex.

    pandas 2.3.3 (the hydro env) mis-aligns two regular (freq-set) DatetimeIndexes of a
    non-nanosecond unit: ``pd.date_range(..., freq="h", unit="us").union(...)`` returns a
    handful of wrong timestamps (verified 2026-10-12; the 2027 dates in the subsidence pilot's
    dbm_daily_raw.csv come from it). The vendor parquet files store datetime64[us], so every
    series is converted to ns before it is resampled or aligned."""
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).as_unit("ns")
    return s


def vendor_raw(key: str) -> pd.Series:
    """Vendor samples (mm) with the cache's zero = missing marker removed."""
    s = pd.read_parquet(VENDOR_DIR / f"ls-wra-dbm-obs__{key}.parquet")["value"].sort_index()
    s = s[~s.index.duplicated(keep="first")].astype("float64")
    return as_ns(s.where(s != 0))


def to_hourly(s: pd.Series) -> pd.Series:
    """Regular hourly series (mean of the samples in each hour; NaN where none), ns index."""
    s = as_ns(s.dropna())
    if s.empty:
        return s
    return s.resample("h").mean()


def load_open(paths: list[Path]) -> tuple[dict[str, pd.Series], dict]:
    """Raw agency dumps -> ({identifier: hourly reading mm}, {identifier: name}, stats).

    Returns (series, info) where info carries names, duplicate and conflict counts."""
    rows = []
    for p in paths:
        rows.extend(json.loads(Path(p).read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    df["t"] = pd.DatetimeIndex(pd.to_datetime(df["observationtime"])).as_unit("ns")
    df["v"] = pd.to_numeric(df["depth"], errors="coerce")
    n_all = len(df)
    dup = df.duplicated(["identifier", "t"], keep=False)
    conflicts = int(df[dup].groupby(["identifier", "t"])["v"].nunique().gt(1).sum())
    df = df.drop_duplicates(["identifier", "t"], keep="first")
    names = df.groupby("identifier")["deepbenchmarkchinesename"].first().to_dict()
    out = {str(k): to_hourly(g.set_index("t")["v"].sort_index()) for k, g in df.groupby("identifier")}
    info = {"rows": n_all, "duplicate_rows": int(dup.sum() - df.duplicated(["identifier", "t"]).sum()),
            "conflicting_duplicates": conflicts, "names": names,
            "nonnumeric": int(df["v"].isna().sum()),
            "span": {k: [str(v.dropna().index.min()), str(v.dropna().index.max()),
                         int(v.notna().sum())] for k, v in out.items()}}
    return out, info


def read_meta() -> pd.DataFrame:
    """Vendor station metadata, non-personal columns only."""
    return pd.read_parquet(META_PARQUET, columns=META_COLS)


# ------------------------------------------------------------------------------- identity
def identity(v: pd.Series, r: pd.Series, tol: float = 0.015) -> dict:
    """How the vendor hourly series ``v`` relates to the raw hourly series ``r``.

    On common consecutive hours, ``v - r`` is constant where the vendor copies the raw feed
    (same sign) and ``v + r`` where it copies it with the sign inverted. Hours where ``v``
    does not move are uninformative. Returns shares, the sign segments (runs of at least 48
    informative hours) and the stitch points (the offset changes by more than 0.05 mm between
    consecutive common hours, gaps included)."""
    j = pd.concat([v, r], axis=1, keys=["v", "r"]).dropna()
    out = {"n_common_hours": int(len(j))}
    if len(j) < 48:
        return {**out, "n_informative_hours": 0, "share_same": float("nan"),
                "share_inverted": float("nan"), "share_union": float("nan"),
                "segments": [], "stitches": []}
    step1 = j.index.to_series().diff() == pd.Timedelta(hours=1)
    dv = j["v"].diff().abs()
    dd = (j["v"] - j["r"]).diff().abs()
    ds = (j["v"] + j["r"]).diff().abs()
    inf = step1 & (dv > tol)
    same = (dd <= tol) & inf
    inv = (ds <= tol) & inf
    n_inf = int(inf.sum())
    out["n_informative_hours"] = n_inf
    out["share_same"] = float(same.sum() / n_inf) if n_inf else float("nan")
    out["share_inverted"] = float(inv.sum() / n_inf) if n_inf else float("nan")
    # the two conventions exclude each other on informative hours (|dv| > tol), so their
    # union is the share of hours that keep "an unchanged offset or sum" (prereg wording)
    out["share_union"] = float((same | inv).sum() / n_inf) if n_inf else float("nan")
    lab = pd.Series(np.where(same & ~inv, 1, np.where(inv & ~same, -1, 0)), index=j.index)
    lab = lab[inf & (lab != 0)]
    segs = []
    if len(lab):
        sm = lab.rolling(25, center=True, min_periods=5).median().fillna(lab)
        sgn = np.sign(sm.to_numpy())
        start = 0
        for i in range(1, len(sgn) + 1):
            if i == len(sgn) or sgn[i] != sgn[start]:
                if i - start >= 48:
                    segs.append({"sign": int(sgn[start]), "start": str(sm.index[start]),
                                 "end": str(sm.index[i - 1]), "hours": int(i - start)})
                start = i
    out["segments"] = segs
    # stitch points: offset change in the active sign convention
    sign = sign_series(j.index, segs)
    off = np.where(sign == 1, (j["v"] - j["r"]).to_numpy(), (j["v"] + j["r"]).to_numpy())
    doff = np.diff(off)
    st = []
    for k in np.nonzero(np.abs(doff) > 0.05)[0]:
        t0, t1 = j.index[k], j.index[k + 1]
        kind = "sign_change" if sign[k] != sign[k + 1] else "offset_change"
        st.append({"t_before": str(t0), "t_after": str(t1),
                   "gap_h": float((t1 - t0) / pd.Timedelta(hours=1)),
                   "raw_change_mm": float(j["r"].iloc[k + 1] - j["r"].iloc[k]),
                   "vendor_change_mm": float(j["v"].iloc[k + 1] - j["v"].iloc[k]),
                   "offset_change_mm": float(doff[k]), "kind": kind})
    out["stitches"] = st
    return out


def sign_series(index: pd.DatetimeIndex, segments: list[dict]) -> np.ndarray:
    """+1/-1 per hour of ``index``: -1 from the start of an inverted segment until the start
    of a later same-sign segment (or the end of the record)."""
    sgn = np.ones(len(index))
    for s in sorted(segments, key=lambda x: x["start"]):
        t = pd.Timestamp(s["start"])
        sgn[index >= t] = s["sign"]
    return sgn


# ------------------------------------------------------------------------------- slopes
class Prefix:
    """O(1) least-squares slopes (mm/h) of an hourly array over index windows [a, b)."""

    def __init__(self, x: np.ndarray):
        m = np.isfinite(x)
        t = np.arange(len(x), dtype="float64")
        mu = float(np.nanmean(x)) if m.any() else 0.0
        xx = np.where(m, x - mu, 0.0)
        z = np.zeros(1)
        self.c = np.concatenate([z, np.cumsum(m)])
        self.st = np.concatenate([z, np.cumsum(np.where(m, t, 0.0))])
        self.stt = np.concatenate([z, np.cumsum(np.where(m, t * t, 0.0))])
        self.sx = np.concatenate([z, np.cumsum(xx)])
        self.stx = np.concatenate([z, np.cumsum(np.where(m, t * xx, 0.0))])
        self.n = len(x)

    def slope(self, a, b):
        a = np.clip(np.asarray(a, dtype="int64"), 0, self.n)
        b = np.clip(np.asarray(b, dtype="int64"), 0, self.n)
        b = np.maximum(a, b)
        n = self.c[b] - self.c[a]
        st = self.st[b] - self.st[a]
        stt = self.stt[b] - self.stt[a]
        sx = self.sx[b] - self.sx[a]
        stx = self.stx[b] - self.stx[a]
        den = n * stt - st * st
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (n * stx - st * sx) / den
        return np.where((n >= MIN_SIDE_PTS) & (den > 0), s, np.nan)


def runs(valid_idx: np.ndarray, gap_h: int = GAP_H) -> np.ndarray:
    """Run id per valid index: a new run starts where consecutive valid hours are more than
    ``gap_h`` apart."""
    if len(valid_idx) == 0:
        return np.zeros(0, dtype="int64")
    return np.concatenate([[0], np.cumsum(np.diff(valid_idx) > gap_h)])


# ------------------------------------------------------------------------------- envelope
def artificial_gaps(x: np.ndarray, durs=DURS_H, n_per: int = 3000, seed: int = 2026,
                    last_index: int | None = None) -> pd.DataFrame:
    """Artificial discontinuities in continuous stretches of an hourly array.

    For each duration d, up to ``n_per`` start hours t0 are drawn such that x is valid at t0
    and t0+d and the whole span [t0-SIDE_H, t0+d+SIDE_H] lies in one continuous run (no gap
    longer than GAP_H) ending by ``last_index``. Returns one row per draw with the observed
    change across the span (``glob``) and its residual against the rate predicted from the two
    side windows (``rate``)."""
    rng = np.random.default_rng(seed)
    valid = np.nonzero(np.isfinite(x))[0]
    if last_index is not None:
        valid = valid[valid <= last_index]
    if len(valid) < 4 * SIDE_H:
        return pd.DataFrame(columns=["d", "glob", "rate"])
    rid = runs(valid)
    run_start = pd.Series(valid).groupby(rid).transform("min").to_numpy()
    run_end = pd.Series(valid).groupby(rid).transform("max").to_numpy()
    isvalid = np.isfinite(x)
    pre = Prefix(x)
    rows = []
    for d in durs:
        t0 = valid
        ok = (t0 - SIDE_H >= run_start) & (t0 + d + SIDE_H <= run_end)
        t1 = t0 + d
        ok &= (t1 < len(x))
        ok[ok] &= isvalid[t1[ok]]
        cand = t0[ok]
        if last_index is not None:
            cand = cand[cand + d + SIDE_H <= last_index]
        if len(cand) == 0:
            continue
        pick = rng.choice(cand, size=min(n_per, len(cand)), replace=False)
        rb = pre.slope(pick - SIDE_H + 1, pick + 1)
        ra = pre.slope(pick + d, pick + d + SIDE_H)
        r = np.nanmean(np.vstack([rb, ra]), axis=0)
        r = np.where(np.isfinite(r), r, 0.0)
        dobs = x[pick + d] - x[pick]
        rows.append(pd.DataFrame({"d": d, "glob": dobs, "rate": dobs - r * d}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["d", "glob", "rate"])


def envelope_table(samples: pd.DataFrame) -> pd.DataFrame:
    """Per duration and estimator: n, Q (q99.9 of |residual|) and the robust sd of the
    residual (sigma_fill, 1.4826 x MAD)."""
    out = []
    for d, g in samples.groupby("d"):
        for est in ("rate", "glob"):
            e = g[est].to_numpy()
            med = np.median(e)
            out.append({"d_h": int(d), "estimator": est, "n": int(len(e)),
                        "q999_mm": float(np.quantile(np.abs(e), Q_LEVEL)),
                        "sigma_fill_mm": float(1.4826 * np.median(np.abs(e - med))),
                        "median_mm": float(med)})
    return pd.DataFrame(out)


def interp_dur(table: pd.DataFrame, estimator: str, col: str, d: float) -> float:
    """Log-log interpolation of ``col`` over duration (flat below 1 h; sqrt growth beyond the
    longest tabulated duration)."""
    t = table[table.estimator == estimator].sort_values("d_h")
    xs, ys = t["d_h"].to_numpy(float), np.maximum(t[col].to_numpy(float), 1e-6)
    d = max(float(d), xs[0])
    if d > xs[-1]:
        return float(ys[-1] * math.sqrt(d / xs[-1]))
    return float(np.exp(np.interp(np.log(d), np.log(xs), np.log(ys))))


def best_estimator(table: pd.DataFrame, d: float) -> str:
    """Pre-registered rule: at duration d use the estimator whose pre-origin envelope Q(d) is
    smaller (equal false-positive level, tighter natural spread). Ties -> 'rate'."""
    qr = interp_dur(table, "rate", "q999_mm", d)
    qg = interp_dur(table, "glob", "q999_mm", d)
    return "rate" if qr <= qg else "glob"


def injected_recall(samples: pd.DataFrame, table: pd.DataFrame, sizes=INJECT_MM,
                    durs=INJECT_DURS_H, floor: float = FLOOR_MM) -> pd.DataFrame:
    """Recall of injected steps: a step of size S (alternating sign) added across an artificial
    discontinuity of duration d is detected when |residual + S| exceeds max(Q(d), floor).
    Rows for each estimator and for the per-duration choice ('best'), plus the observed
    false-positive share on the uninjected samples (size 0)."""
    out = []
    for est in ("rate", "glob", "best"):
        for d in durs:
            use = best_estimator(table, d) if est == "best" else est
            e = samples.loc[samples.d == d, use].to_numpy()
            if len(e) == 0:
                continue
            thr = max(interp_dur(table, use, "q999_mm", d), floor)
            out.append({"estimator": est, "uses": use, "d_h": d, "size_mm": 0.0,
                        "n": int(len(e)), "threshold_mm": thr,
                        "detected_share": float(np.mean(np.abs(e) > thr))})
            sgn = np.where(np.arange(len(e)) % 2 == 0, 1.0, -1.0)
            for sz in sizes:
                det = np.abs(e + sgn * sz) > thr
                out.append({"estimator": est, "uses": use, "d_h": d, "size_mm": sz,
                            "n": int(len(e)), "threshold_mm": thr,
                            "detected_share": float(det.mean())})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------- detection
@dataclass
class Detector:
    """Thresholds from the pre-origin envelope; the estimator is chosen per duration."""
    table: pd.DataFrame
    floor: float = FLOOR_MM

    def est(self, d: float) -> str:
        return best_estimator(self.table, d)

    def thr(self, d: float) -> float:
        return max(interp_dur(self.table, self.est(d), "q999_mm", d), self.floor)

    def sigma(self, d: float) -> float:
        return interp_dur(self.table, self.est(d), "sigma_fill_mm", d)


def detect(h: pd.Series, det: Detector) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Step catalogue of one hourly series.

    Returns (events, corrected hourly series, hourly flag series). Event kinds: ``step``
    (residual above threshold; removed from every later value), ``spike`` (flagged jumps that
    cancel; interior hours removed), ``outage`` (gap longer than GAP_H whose change is kept as
    real movement), ``noise`` (a flagged jump that falls below the threshold once its own
    span is excluded from the rate windows; no action). Hour flags: 0 observed, 1 after a
    removed step, 9 removed (spike or mid-step)."""
    idx = h.index
    x = h.to_numpy(dtype="float64").copy()
    valid = np.nonzero(np.isfinite(x))[0]
    cols = ["kind", "estimator", "a", "b", "t_before", "t_after", "duration_h", "delta_obs_mm",
            "delta_pred_mm", "resid_mm", "threshold_mm", "rate_mm_per_day", "rate_ok",
            "corrected", "correction_mm", "sd_mm", "n_flagged_pairs"]
    if len(valid) < 2:
        return pd.DataFrame(columns=cols), h.copy(), pd.Series(8, index=idx, dtype="int64")
    pre = Prefix(x)
    a0, b0 = valid[:-1], valid[1:]
    gap = b0 - a0
    # rate-adjusted residual of every consecutive pair (windows outside the pair)
    rb = pre.slope(a0 - SIDE_H + 1, a0 + 1)
    ra = pre.slope(b0, b0 + SIDE_H)
    r = np.nanmean(np.vstack([rb, ra]), axis=0)
    r = np.where(np.isfinite(r), r, 0.0)
    dobs = x[b0] - x[a0]
    ug = np.unique(gap)
    est_g = {int(g): det.est(g) for g in ug}
    thr_g = {int(g): det.thr(g) for g in ug}
    use_rate = np.array([est_g[int(g)] == "rate" for g in gap])
    e = np.where(use_rate, dobs - r * gap, dobs)
    thr_pair = np.array([thr_g[int(g)] for g in gap])
    flagged = (np.abs(e) > thr_pair) & (gap <= GAP_H)
    outage = gap > GAP_H
    cand = np.nonzero(flagged | outage)[0]
    # merge candidates closer than MERGE_H (in hours between them)
    groups: list[list[int]] = []
    for k in cand:
        if groups and a0[k] - b0[groups[-1][-1]] < MERGE_H:
            groups[-1].append(k)
        else:
            groups.append([k])
    ev = []
    for gk in groups:
        a, b = int(a0[gk[0]]), int(b0[gk[-1]])
        ev.append({"a": a, "b": b, "n_flagged_pairs": int(flagged[gk].sum()),
                   "has_outage": bool(outage[gk].any())})
    # classify each event with rate windows bounded by the neighbouring events
    for i, E in enumerate(ev):
        a, b = E["a"], E["b"]
        lo = ev[i - 1]["b"] if i > 0 else 0
        hi = ev[i + 1]["a"] + 1 if i + 1 < len(ev) else len(x)
        sb = pre.slope(max(a - SIDE_H + 1, lo), a + 1)
        sa = pre.slope(b, min(b + SIDE_H, hi))
        rates = [float(s) for s in (sb, sa) if np.isfinite(s)]
        rate = float(np.mean(rates)) if rates else 0.0
        d = float(b - a)
        dob = float(x[b] - x[a])
        est = det.est(d)
        dpred = rate * d if est == "rate" else 0.0
        res = dob - dpred
        thr = det.thr(d)
        if abs(res) > thr:
            kind = "step"
        elif E["n_flagged_pairs"] > 0 and not E["has_outage"]:
            interior_dev = np.nanmax(np.abs(x[a + 1:b] - (x[a] + (x[b] - x[a])
                                                             * (np.arange(a + 1, b) - a) / d))) \
                if b - a > 1 else 0.0
            kind = "spike" if interior_dev > thr else "noise"
        elif E["has_outage"]:
            kind = "outage"
        else:
            kind = "noise"
        E.update({"kind": kind, "estimator": est, "t_before": idx[a], "t_after": idx[b],
                  "duration_h": d,
                  "delta_obs_mm": dob, "delta_pred_mm": dpred, "resid_mm": res,
                  "threshold_mm": thr, "rate_mm_per_day": rate * 24.0, "rate_ok": bool(rates),
                  "corrected": kind == "step", "correction_mm": res if kind == "step" else 0.0,
                  "sd_mm": det.sigma(d) if kind == "step" else 0.0})
    flags = np.zeros(len(x), dtype="int64")
    flags[~np.isfinite(x)] = 8
    xc = x.copy()
    offset = 0.0
    corr = np.zeros(len(x))
    for E in ev:
        a, b = E["a"], E["b"]
        if E["kind"] in ("step", "spike") and b - a > 1:
            inner = np.arange(a + 1, b)
            inner = inner[np.isfinite(x[inner])]
            xc[inner] = np.nan
            flags[inner] = 9
        if E["kind"] == "step":
            offset += E["correction_mm"]
            corr[b:] = offset
    xc = xc - corr
    after = np.zeros(len(x), dtype=bool)
    steps = [E for E in ev if E["kind"] == "step"]
    if steps:
        after[steps[0]["b"]:] = True
    flags[(flags == 0) & after] = 1
    evdf = pd.DataFrame(ev, columns=cols + ["has_outage"]) if ev else pd.DataFrame(columns=cols)
    return evdf, pd.Series(xc, index=idx), pd.Series(flags, index=idx)


# ------------------------------------------------------------------------------- daily
def daily_from_hourly(xc: pd.Series, hflag: pd.Series, raw: pd.Series | None = None,
                      sigma_day: float = float("nan")) -> pd.DataFrame:
    """Daily mean of the valid hours, hour count and day flag (0 observed; 1 the level carries
    a removed step; 2 partial day, fewer than DAY_MIN_HOURS hours; 8 no valid hour)."""
    ok = xc.notna()
    g = xc.groupby(xc.index.floor("D"))
    val = g.mean()
    n = ok.groupby(xc.index.floor("D")).sum().astype("int64")
    f1 = (hflag == 1).groupby(hflag.index.floor("D")).any()
    flag = np.where(n == 0, 8, np.where(n < DAY_MIN_HOURS, 2, np.where(f1.reindex(n.index), 1, 0)))
    out = pd.DataFrame({"date": n.index, "value": val.reindex(n.index).to_numpy(),
                        "n_hours": n.to_numpy(), "flag": flag})
    if raw is not None:
        rv = raw.groupby(raw.index.floor("D")).mean()
        out["raw_value"] = rv.reindex(n.index).to_numpy()
    out["sd"] = np.where(out["flag"] == 8, np.nan,
                         sigma_day * np.sqrt(np.maximum(DAY_MIN_HOURS / np.maximum(out["n_hours"], 1), 1.0)))
    return out


def sigma_day_preorigin(daily: pd.DataFrame) -> float:
    """Robust sd of daily means about a centred 7-day linear fit, 2022-12 or earlier, full
    days only (the day-to-day noise of a month-end level)."""
    d = daily[(daily["flag"].isin([0, 1])) & (pd.to_datetime(daily["date"]) <= ORIGIN_END)]
    s = pd.Series(d["value"].to_numpy(), index=pd.to_datetime(d["date"])).asfreq("D")
    if s.notna().sum() < 60:
        return float("nan")
    res = []
    v = s.to_numpy()
    for i in range(3, len(v) - 3):
        w = v[i - 3:i + 4]
        if np.isfinite(w).sum() == 7:
            t = np.arange(7) - 3.0
            p = np.polyfit(t, w, 1)
            res.append(w[3] - p[1])
    res = np.asarray(res)
    if len(res) < 30:
        return float("nan")
    return float(1.4826 * np.median(np.abs(res - np.median(res))) / math.sqrt(1 - 1 / 7.0))


# ------------------------------------------------------------------------------- monthly
def month_end_levels(daily: pd.DataFrame) -> pd.DataFrame:
    """Per month: the level of the last day (flag 0/1), or of the latest day with a value up
    to BRIDGE_DAYS earlier or a partial last day (endpoint flag 2, lag in days)."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    rows = []
    for per, g in d.groupby(d.index.to_period("M")):
        last = per.to_timestamp(how="end").normalize()
        lev, fl, lag, day = np.nan, 8, np.nan, pd.NaT
        for k in range(0, BRIDGE_DAYS + 1):
            t = last - pd.Timedelta(days=k)
            if t in g.index and g.at[t, "flag"] in (0, 1, 2) and np.isfinite(g.at[t, "value"]):
                lev = float(g.at[t, "value"])
                fl = 0 if (k == 0 and g.at[t, "flag"] in (0, 1)) else 2
                lag, day = k, t
                break
        rows.append({"month": str(per), "level_end_mm": lev, "end_flag": fl, "end_lag_days": lag,
                     "end_day": day,
                     "mean_level_mm": float(g.loc[g["flag"].isin([0, 1, 2]), "value"].mean())
                     if g["flag"].isin([0, 1, 2]).any() else np.nan,
                     "n_days": int(g["flag"].isin([0, 1, 2]).sum())})
    return pd.DataFrame(rows)


def monthly_from_daily(daily: pd.DataFrame, events: pd.DataFrame, det: Detector,
                       sigma_day: float, suspect_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly increments (cm) between month-end levels with flag and sd.

    Flag: 8 missing endpoint; 9 a suspect event inside the interval (``suspect_events``:
    rows with t_before/t_after); 2 a bridged or partial endpoint; 1 a removed step inside
    the interval; else 0. sd (cm): sqrt(2) sigma_day, plus sigma_fill(d) of every removed
    step inside and of every bridged endpoint, in quadrature."""
    me = month_end_levels(daily)
    full = pd.period_range(me["month"].iloc[0], me["month"].iloc[-1], freq="M").astype(str)
    me = me.set_index("month").reindex(full)
    me["end_flag"] = me["end_flag"].fillna(8)
    st = events[events["kind"] == "step"] if len(events) else events
    rows = []
    prev = None
    for m, r in me.iterrows():
        if prev is None:
            prev = (m, r)
            rows.append({"month": m, "inc_cm": np.nan, "flag": 8, "sd_cm": np.nan,
                         "step_ids": "", "n_steps": 0, **_lev(r)})
            continue
        _, pr = prev
        prev = (m, r)
        if not (np.isfinite(r["level_end_mm"]) and np.isfinite(pr["level_end_mm"])):
            rows.append({"month": m, "inc_cm": np.nan, "flag": 8, "sd_cm": np.nan,
                         "step_ids": "", "n_steps": 0, **_lev(r)})
            continue
        # an event affects the increment if it lies between the two end days' daily means
        t1 = pd.Timestamp(r["end_day"]) + pd.Timedelta(hours=23, minutes=59)
        t0s = pd.Timestamp(pr["end_day"])
        var = 2.0 * sigma_day ** 2
        ids = []
        if len(st):
            inside = st[(pd.to_datetime(st["t_after"]) > t0s) & (pd.to_datetime(st["t_before"]) < t1)]
            for k, e in inside.iterrows():
                var += float(e["sd_mm"]) ** 2
                ids.append(str(e.get("event_id", k)))
        flag = 0
        for rr in (pr, r):
            if rr["end_flag"] == 2:
                flag = 2
                lag = rr["end_lag_days"] if np.isfinite(rr["end_lag_days"]) else 0
                var += det.sigma(max(24.0 * lag, 12.0)) ** 2
        if ids and flag == 0:
            flag = 1
        if suspect_events is not None and len(suspect_events):
            sus = suspect_events[(pd.to_datetime(suspect_events["t_after"]) > t0s)
                                 & (pd.to_datetime(suspect_events["t_before"]) < t1)]
            if len(sus):
                flag = 9
        inc = (float(r["level_end_mm"]) - float(pr["level_end_mm"])) / 10.0
        rows.append({"month": m, "inc_cm": inc, "flag": flag, "sd_cm": math.sqrt(var) / 10.0,
                     "step_ids": ";".join(ids), "n_steps": len(ids), **_lev(r)})
    return pd.DataFrame(rows)


def _lev(r) -> dict:
    return {"level_end_mm": r["level_end_mm"], "end_flag": int(r["end_flag"]),
            "end_lag_days": r["end_lag_days"], "mean_level_mm": r["mean_level_mm"],
            "n_days": int(r["n_days"]) if np.isfinite(r["n_days"]) else 0}


# ------------------------------------------------------------------------------- printed
def read_printed() -> pd.DataFrame:
    """Verified printed values with the sensor each (pile, printed depth) maps to and the
    rounding sd of each value (half the last printed decimal / sqrt(3))."""
    p = pd.read_csv(PRINTED_CSV)
    p["depth_m"] = p["depth_m"].astype("Int64")
    p["sensor_id"] = [PRINTED_TO_SENSOR.get((a, int(b)) if pd.notna(b) else (a, -1))
                      for a, b in zip(p["pile"], p["depth_m"], strict=True)]

    def rsd(v) -> float:
        if pd.isna(v):
            return float("nan")
        s = f"{abs(float(v)):.6f}".rstrip("0")
        dec = len(s.split(".")[1]) if "." in s and s.split(".")[1] else 0
        unit = 10.0 ** (-max(dec, 1))
        return unit / 2.0 / math.sqrt(3.0)

    p["month_round_sd_cm"] = [rsd(v) for v in p["month_cm"]]
    p["year_round_sd_cm"] = [rsd(v) for v in p["year_cm"]]
    return p


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------------------- cumulative
def cumulative(months: list[str], incs: pd.DataFrame, years: pd.DataFrame,
               ref: str = ORIGIN_MONTH) -> pd.DataFrame:
    """Cumulative compression (cm) at month ends by weighted least squares.

    ``incs``: month, value, sd, flag (c(m) - c(m-1) = value). ``years``: month, value, sd
    (c(m) - c(m-12) = value; printed trailing-twelve-month values at an unchanged sensor).
    The reference is ``ref`` when it lies in ``months`` and is connected to observed
    constraints, else the first node of the component holding the latest node. Two passes:
    nodes reached through flag 0/1 constraints only (values from those alone) get flag 0;
    nodes reached once flag-2 constraints are added get flag 2; the rest are NaN (8)."""
    pos = {m: i for i, m in enumerate(months)}
    n = len(months)

    def build(allowed_flags):
        rows = []
        for r in incs.itertuples():
            if r.flag in allowed_flags and np.isfinite(r.value) and r.month in pos:
                pm = str(pd.Period(r.month, freq="M") - 1)
                if pm in pos:
                    rows.append((pos[r.month], pos[pm], float(r.value), max(float(r.sd), 1e-4)))
        for r in years.itertuples():
            if np.isfinite(r.value) and r.month in pos:
                pm = str(pd.Period(r.month, freq="M") - 12)
                if pm in pos:
                    rows.append((pos[r.month], pos[pm], float(r.value), max(float(r.sd), 1e-4)))
        return rows

    def components(rows):
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j, _, _ in rows:
            parent[find(i)] = find(j)
        return np.array([find(i) for i in range(n)])

    def solve(rows, anchor, nodes):
        idx = {k: c for c, k in enumerate([k for k in nodes if k != anchor])}
        if not idx:
            return {anchor: (0.0, 0.0)}
        A = np.zeros((len(rows), len(idx)))
        b = np.zeros(len(rows))
        w = np.zeros(len(rows))
        for k, (i, j, v, s) in enumerate(rows):
            if i in idx:
                A[k, idx[i]] += 1.0
            if j in idx:
                A[k, idx[j]] -= 1.0
            b[k], w[k] = v, 1.0 / s ** 2
        N = A.T @ (A * w[:, None])
        cov = np.linalg.pinv(N)
        x = cov @ (A.T @ (w * b))
        out = {anchor: (0.0, 0.0)}
        for k, c in idx.items():
            out[k] = (float(x[c]), float(math.sqrt(max(cov[c, c], 0.0))))
        return out

    res = {"value": np.full(n, np.nan), "sd": np.full(n, np.nan), "flag": np.full(n, 8)}
    rows01 = build({0, 1})
    rows012 = build({0, 1, 2})
    comp012 = components(rows012)
    if ref in pos and (comp012 == comp012[pos[ref]]).sum() > 1:
        anchor = pos[ref]
    else:
        anchor = int(np.nonzero(comp012 == comp012[n - 1])[0][0])
    comp01 = components(rows01)
    nodes01 = [k for k in range(n) if comp01[k] == comp01[anchor]]
    sol01 = solve([r for r in rows01 if comp01[r[0]] == comp01[anchor]], anchor, nodes01) \
        if len(nodes01) > 1 else {anchor: (0.0, 0.0)}
    nodes012 = [k for k in range(n) if comp012[k] == comp012[anchor]]
    sol012 = solve([r for r in rows012 if comp012[r[0]] == comp012[anchor]], anchor, nodes012) \
        if len(nodes012) > 1 else {anchor: (0.0, 0.0)}
    for k in nodes012:
        if k in sol01:
            res["value"][k], res["sd"][k] = sol01[k]
            res["flag"][k] = 0
        else:
            res["value"][k], res["sd"][k] = sol012[k]
            res["flag"][k] = 2
    return pd.DataFrame({"month": months, "value": res["value"], "sd": res["sd"],
                         "flag": res["flag"].astype("int64"), "ref_month": months[anchor]})


# ------------------------------------------------------------------------------- grid
def grid_cell(xy: tuple[float, float], npz_path: Path = GRID_NPZ) -> tuple[int, int, int]:
    """(active-cell index, row, col) of the nearest active cell centre within dx, else -1s
    (mirrors hydrophysics.twin.mechanism.ring_cells)."""
    z = np.load(npz_path, allow_pickle=False)
    mask = np.asarray(z["mask"])
    dx = float(z["dx"])
    rows, cols = np.nonzero(mask)
    cent = np.column_stack([float(z["x0"]) + (cols + 0.5) * dx, float(z["y0"]) + (rows + 0.5) * dx])
    d2 = ((cent - np.asarray(xy, float)) ** 2).sum(1)
    c = int(np.argmin(d2))
    if math.sqrt(d2[c]) <= dx:
        return c, int(rows[c]), int(cols[c])
    return -1, -1, -1
