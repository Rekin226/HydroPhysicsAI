"""Per-well, per-month head-signature indices (plan WP P5 step 1), barometrically corrected.

The index definitions are the head-signature pilot's, copied unchanged in substance from
``results/dataprep/_pilots/head_pumping_signatures/scripts/well_features.py`` and
``well_features2.py`` (the vectorised version): ``astro``, ``nodal``, ``tidal_leak``,
``month_harmonics``, ``events`` and the per-month loop of ``process``.

  level     : monthly mean head, OLS slope of daily means (m/day)
  dr_*      : daily range after removing a within-day linear trend (hourly means), median/p90
  dday/dngt : median head change 06->18 h and 18->06 h (day / night); DN = dngt - dday
  harmonic  : per-month least squares on hourly means with a cubic B-spline trend (knots
              every 3 days) + Q1, O1, S1, N2, M2, S2, S3, S4. K1+P1 (inside the 1-cpd fit)
              and S2grav+K2 (inside the 2-cpd fit) are predicted from the fitted O1 / M2 via
              astronomical arguments and a diurnal / semidiurnal admittance ratio, then removed
              (A_S1c_eq / A_S1c_strain, A_S2c).
  profile   : 24-value mean solar-hour profile of (hourly - centred 24 h mean)
  events    : 10-minute months only: runs of 30-min drops / rises beyond 5 cm and 20 cm

What changes against the pilot (declared deviations, plan WP P5 step 6 and WP H1):

* Cleaning: the samples are H1a's (WP H1a, ``heads/L1_clean/v3``): its stage-1 screen is
  re-run from L0 with ``pipeline.process_well`` and the published barometric responses fixed
  (``be_override``). A sample is kept when its stage-1 base screens are clean, it lies
  outside every sample-removing L1 v3 event (flag-9 offset episodes, dropouts, steps and
  rejected-step runs; the flag-3 step) and its day has an L1 v3 daily flag of 0 or 1: the
  sample set behind L1's flag-0/1 daily values (``assemble_v2.apply_part_v2``; checked day by
  day against L1, ``daily_check``). The pilot's own screens (12 m outliers, flat dropouts,
  1 m spikes) are not used.
* Barometric loading: removed sample by sample (flag 1) with H1a's accepted hourly impulse
  response per logger epoch; wells or epochs without an accepted response are uncorrected
  (flag 0). Every index is also computed on the uncorrected samples (``*_raw`` columns).
* Pre-origin purity: samples before 2023-01-01 and from 2023-01-01 on are processed
  separately, so no 2022-12 index uses a 2023 sample (the pilot's centred 24-hour mean and
  night change crossed the origin).
* Added: O1 and M2 phases relative to their astronomical arguments (``gO1``, ``gM2``; H1a's
  ``tides.o1_phases`` convention), the loading's own S1 amplitude (``A_S1_baro``, the
  difference of the corrected and uncorrected S1 coefficients, exact by linearity), DN and
  A_daily (pilot ``analysis.py``) as columns.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from p5_head_signatures import config as C

# speeds in degrees per hour (pilot)
SPEED = {"Q1": 13.3986609, "O1": 13.9430356, "S1": 15.0, "N2": 28.4397295,
         "M2": 28.9841042, "S2": 30.0, "S3": 45.0, "S4": 60.0}
CONS = list(SPEED)
A_EQ = {"O1": 0.3769, "K1": 0.5305, "P1": 0.1755, "M2": 0.9081, "S2": 0.4229, "K2": 0.1151}
K1_FAC = {"eq": 1.0, "strain": 0.74}
P1_FAC = {"eq": 1.0, "strain": 0.93}
#: columns also kept for the uncorrected variant (suffix ``_raw``)
RAW_COLS = ("mean", "dr_med", "dr_p90", "dsd_med", "dday_med", "dngt_med", "DN", "A_S1",
            "A_S1c_strain", "A_S1c_eq", "A_S2c", "A_S2", "A_O1", "A_M2", "gO1", "gM2",
            "re_S1c_strain", "im_S1c_strain", "re_S1", "im_S1", "res_sd", "A_daily")


def astro(t: pd.Timestamp):
    """Mean longitudes s, h (deg) and node N (deg) at UT time t (pilot)."""
    d = (t - pd.Timestamp("2000-01-01 12:00")).total_seconds() / 86400.0
    s = (218.3164 + 13.17639648 * d) % 360
    h = (280.4661 + 0.98564736 * d) % 360
    N = (125.0445 - 0.05295377 * d) % 360
    return s, h, N


def nodal(N):
    n = np.radians(N)
    f = {"O1": 1.0089 + 0.1871 * np.cos(n) - 0.0147 * np.cos(2 * n) + 0.0014 * np.cos(3 * n),
         "K1": 1.0060 + 0.1150 * np.cos(n) - 0.0088 * np.cos(2 * n) + 0.0006 * np.cos(3 * n),
         "M2": 1.0007 - 0.0373 * np.cos(n) + 0.0002 * np.cos(2 * n),
         "K2": 1.0241 + 0.2863 * np.cos(n) + 0.0083 * np.cos(2 * n) - 0.0015 * np.cos(3 * n),
         "P1": 1.0, "S2": 1.0}
    u = {"O1": 10.80 * np.sin(n) - 1.34 * np.sin(2 * n) + 0.19 * np.sin(3 * n),
         "K1": -8.86 * np.sin(n) + 0.68 * np.sin(2 * n) - 0.07 * np.sin(3 * n),
         "M2": -2.14 * np.sin(n),
         "K2": -17.74 * np.sin(n) + 0.68 * np.sin(2 * n) - 0.04 * np.sin(3 * n),
         "P1": 0.0, "S2": 0.0}
    return f, u


def tidal_leak(cO1, cM2, tref_local):
    """Predicted complex amplitudes inside the S1 and S2 fits from fitted O1 and M2 (pilot)."""
    t_ut = tref_local - pd.Timedelta(hours=8)
    s, h, N = astro(t_ut)
    f, u = nodal(N)
    out = {}
    for var in ("eq", "strain"):
        rk = A_EQ["K1"] * f["K1"] / (A_EQ["O1"] * f["O1"]) * K1_FAC[var]
        rp = A_EQ["P1"] * f["P1"] / (A_EQ["O1"] * f["O1"]) * P1_FAC[var]
        phk = np.radians(2 * s + 180.0 + u["K1"] - u["O1"])
        php = np.radians(2 * s - 2 * h - u["O1"])
        out["S1leak_" + var] = cO1 * (rk * np.exp(1j * phk) + rp * np.exp(1j * php))
    rs = A_EQ["S2"] / (A_EQ["M2"] * f["M2"])
    rk2 = A_EQ["K2"] * f["K2"] / (A_EQ["M2"] * f["M2"])
    out["S2leak"] = cM2 * (rs * np.exp(1j * np.radians(2 * s - 2 * h - u["M2"]))
                           + rk2 * np.exp(1j * np.radians(2 * s + u["K2"] - u["M2"])))
    return out


def astro_phases(cO1: complex, cM2: complex, tref_local: pd.Timestamp) -> tuple[float, float]:
    """O1 and M2 phase (rad) relative to their astronomical arguments V + u (H1a
    ``tides.o1_phases`` convention; Greenwich arguments, no longitude term)."""
    tu = tref_local - pd.Timedelta(hours=8)
    s, h, N = astro(tu)
    _, u = nodal(N)
    Th = 180.0 + 15.0 * (tu.hour + tu.minute / 60.0)
    VO1 = Th - 2 * s + h + 90.0 + u["O1"]
    VM2 = 2 * Th - 2 * s + 2 * h + u["M2"]
    return (float(np.radians(VO1) - np.angle(cO1)), float(np.radians(VM2) - np.angle(cM2)))


def month_harmonics(hh: pd.Series, mstart: pd.Timestamp, mend: pd.Timestamp):
    """Pilot ``month_harmonics``: B-spline trend (3-day knots) + 8 harmonics, least squares."""
    t = (hh.index - mstart).total_seconds().to_numpy() / 3600.0  # hours
    T = (mend - mstart).total_seconds() / 3600.0
    tref = T / 2.0
    k = 3
    inner = np.arange(72.0, T - 36.0, 72.0)
    knots = np.r_[[0.0] * (k + 1), inner, [T] * (k + 1)]
    B = BSpline.design_matrix(np.clip(t, 0, T - 1e-6), knots, k).toarray()
    cols = [B]
    for c in CONS:
        w = np.radians(SPEED[c]) * (t - tref)
        cols += [np.cos(w)[:, None], np.sin(w)[:, None]]
    X = np.hstack(cols)
    y = hh.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    nb = B.shape[1]
    out = {}
    for i, c in enumerate(CONS):
        a, b = beta[nb + 2 * i], beta[nb + 2 * i + 1]
        out[c] = complex(a, -b)   # Re{c e^{iw(t-tref)}} = a cos + b sin
    return out, float(np.std(res)), mstart + pd.Timedelta(hours=tref)


def events(s10: pd.Series, thr: float):
    """Pilot ``events``: runs of 30-minute drops / rises beyond ``thr`` on a 10-min grid."""
    d30 = (s10.shift(-3) - s10).to_numpy()
    out = {}
    for name, mask in (("drop", d30 < -thr), ("rise", d30 > thr)):
        m = np.nan_to_num(mask.astype(float), nan=0.0).astype(bool)
        starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
        out[name] = len(starts)
        if len(starts):
            ends = np.flatnonzero(m & ~np.r_[m[1:], False])
            mags = [np.nanmax(np.abs(d30[a:b + 1])) for a, b in zip(starts, ends, strict=True)]
            out[name + "_mag"] = float(np.median(mags))
        else:
            out[name + "_mag"] = np.nan
    return out


def daily_matrix(hh: pd.Series):
    d0 = hh.index.min().normalize()
    d1 = hh.index.max().normalize() + pd.Timedelta(days=1)
    grid = pd.date_range(d0, d1, freq="h", inclusive="left")
    H = hh.reindex(grid).to_numpy().reshape(-1, 24)
    days = pd.date_range(d0, d1, freq="D", inclusive="left")
    return days, H, grid


def month_features(s: pd.Series, months: pd.PeriodIndex) -> pd.DataFrame:
    """The pilot's per-month loop (``well_features2.process`` after cleaning) on the cleaned
    sample series ``s`` (DatetimeIndex, local time, m). One row per month with >= 10 days of
    hourly means."""
    if s.empty:
        return pd.DataFrame()
    hh = s.resample("h").mean()
    days, H, grid = daily_matrix(hh)
    hp_full = (hh - hh.rolling(24, center=True, min_periods=20).mean()).reindex(grid)
    HP = hp_full.to_numpy().reshape(-1, 24)
    nval = np.isfinite(H).sum(1)
    good = nval >= C.DAY_MIN_HOURS
    x = np.tile(np.arange(24.0), (len(H), 1))
    m = np.isfinite(H)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        xm = np.where(m, x, np.nan)
        xbar = np.nanmean(xm, 1, keepdims=True)
        ybar = np.nanmean(H, 1, keepdims=True)
        sxy = np.nansum((xm - xbar) * (H - ybar), 1, keepdims=True)
        sxx = np.nansum((xm - xbar) ** 2, 1, keepdims=True)
        b = sxy / np.where(sxx > 0, sxx, np.nan)
        res = H - (ybar + b * (xm - xbar))
        dr = np.nanmax(res, 1) - np.nanmin(res, 1)
        dsd = np.nanstd(res, 1)
    dmean = ybar[:, 0]
    dday = H[:, 18] - H[:, 6]
    dngt = np.r_[H[1:, 6], np.nan] - H[:, 18]
    D = pd.DataFrame({"mean": dmean, "dr": dr, "dsd": dsd, "dday": dday, "dngt": dngt},
                     index=days)[good]
    hv = hh.dropna()
    hv_t = hv.index.to_numpy()
    s_t = s.index.to_numpy()
    dts = np.r_[np.nan, np.diff(s_t).astype("timedelta64[s]").astype(float)]
    D_t = D.index.to_numpy()
    rows = []
    for mth in months:
        m0, m1 = mth.start_time, (mth + 1).start_time
        a, z = np.searchsorted(hv_t, np.datetime64(m0)), np.searchsorted(hv_t, np.datetime64(m1))
        if z - a < C.MONTH_MIN_HOURS:
            continue
        hm_ = hv.iloc[a:z]
        da, dz = np.searchsorted(D_t, np.datetime64(m0)), np.searchsorted(D_t, np.datetime64(m1))
        Dm = D.iloc[da:dz]
        row = {"month": str(mth), "n_hours": int(z - a), "n_days": len(Dm),
               "mean": float(hm_.mean()),
               "min_daily": float(Dm["mean"].min()) if len(Dm) else np.nan,
               "max_daily": float(Dm["mean"].max()) if len(Dm) else np.nan}
        if len(Dm) >= C.DAILY_STATS_MIN_DAYS:
            xd = (Dm.index - m0).days.to_numpy().astype(float)
            row["slope"] = float(np.polyfit(xd, Dm["mean"].to_numpy(), 1)[0])
            row["dr_med"] = float(Dm["dr"].median())
            row["dr_p90"] = float(Dm["dr"].quantile(0.9))
            row["dsd_med"] = float(Dm["dsd"].median())
            row["dday_med"] = float(Dm["dday"].median())
            row["dngt_med"] = float(Dm["dngt"].median())
            row["DN"] = row["dngt_med"] - row["dday_med"]
        if z - a >= C.HARM_MIN_HOURS:
            c, rsd, tref = month_harmonics(hm_, m0, m1)
            row["res_sd"] = rsd
            th = (tref - m0).total_seconds() / 3600.0

            def clk(v, cname, th=th):
                return v * np.exp(-1j * np.radians(SPEED[cname]) * th)
            for k_, v in c.items():
                row["A_" + k_] = abs(v)
                vv = clk(v, k_) if k_ in ("S1", "S2", "S3", "S4") else v
                row["ph_" + k_] = float(np.degrees(np.angle(vv)))
            leak = tidal_leak(c["O1"], c["M2"], tref)
            for var in ("eq", "strain"):
                cs1 = clk(c["S1"] - leak["S1leak_" + var], "S1")
                row["A_S1c_" + var] = abs(cs1)
                row["ph_S1c_" + var] = float(np.degrees(np.angle(cs1)))
                row["A_S1leak_" + var] = abs(leak["S1leak_" + var])
                row["re_S1c_" + var], row["im_S1c_" + var] = cs1.real, cs1.imag
            cs2 = clk(c["S2"] - leak["S2leak"], "S2")
            row["A_S2c"] = abs(cs2)
            row["ph_S2c"] = float(np.degrees(np.angle(cs2)))
            row["A_S2leak"] = abs(leak["S2leak"])
            row["re_S1"], row["im_S1"] = c["S1"].real, c["S1"].imag
            row["re_O1"], row["im_O1"] = c["O1"].real, c["O1"].imag
            row["re_M2"], row["im_M2"] = c["M2"].real, c["M2"].imag
            row["re_S2"], row["im_S2"] = c["S2"].real, c["S2"].imag
            row["gO1"], row["gM2"] = astro_phases(c["O1"], c["M2"], tref)
            row["A_daily"] = float(np.sqrt(row["A_S1c_strain"] ** 2 + row["A_S2c"] ** 2
                                           + row["A_S3"] ** 2 + row["A_S4"] ** 2))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prof = np.nanmean(HP[(days >= m0) & (days < m1)], 0)
            for hr in range(24):
                row[f"p{hr:02d}"] = float(prof[hr])
        sa, sz = np.searchsorted(s_t, np.datetime64(m0)), np.searchsorted(s_t, np.datetime64(m1))
        sm = s.iloc[sa:sz]
        dm = dts[sa:sz]
        frac10 = float(np.nanmean(dm <= 660)) if len(dm) else 0.0
        row["frac10"] = frac10
        if frac10 > C.EV_FRAC10_MIN and len(sm) > 144 * 10:
            g10 = pd.date_range(m0, m1, freq="10min", inclusive="left")
            s10 = sm.reindex(g10)
            ndays = s10.notna().sum() / 144.0
            for thr, tag in C.EV_THRESHOLDS:
                e = events(s10, thr)
                row["ev_drop" + tag] = e["drop"] / ndays
                row["ev_rise" + tag] = e["rise"] / ndays
                row["ev_dropmag" + tag] = e["drop_mag"]
                row["ev_risemag" + tag] = e["rise_mag"]
            dd = np.abs(np.diff(sm.to_numpy()))
            dd = dd[dd > 0]
            row["res_step"] = float(np.quantile(dd, 0.01)) if len(dd) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- H1a samples
def be_override(be_rows: pd.DataFrame) -> dict | None:
    """``pipeline.process_well`` override built from the published barometric table (one
    row per logger epoch). Accepted epochs carry their hourly impulse response; others none."""
    if be_rows is None or be_rows.empty:
        return None
    be_rows = be_rows.sort_values("epoch")
    segs = []
    for _, r in be_rows.iterrows():
        a = None
        if bool(r["accepted"]) and isinstance(r["impulse_response_a_0_48h"], str):
            a = [float(x) for x in json.loads(r["impulse_response_a_0_48h"])]
        start = int(pd.Timestamp(r["epoch_start"]).value // 10**9)
        end = int(pd.Timestamp(r["epoch_end"]).value // 10**9)
        segs.append({"start": start, "end": end, "a": a, "epoch": int(r["epoch"])})
    return {"segments": segs, "p_ref_hpa": float(be_rows["p_ref_hpa"].iloc[0])}


#: L1 v3 events that remove samples (H1a ``assemble_v2.apply_part_v2``): interval
#: semantics [t1, t2) for episodes and dropouts, [t_lo, t_hi] for steps and step-candidate runs,
#: [from, until) for carried steps; flags 9 (suspect) and 3 (neighbour-corroborated correction,
#: not a flag-0/1 value).
OBJECT_KINDS = {"offset_episode": "half_open", "dropout": "half_open", "step": "closed",
                "excursion": "closed", "jump_edge": "closed", "step_carried": "half_open"}
OBJECT_FLAGS = (3.0, 9.0)


def object_windows(events: pd.DataFrame) -> list[tuple[int, int, bool]]:
    """(start s, end s, closed) of the sample-removing L1 v3 events of one well."""
    out = []
    if events is None or events.empty:
        return out
    e = events[events["kind"].isin(list(OBJECT_KINDS)) & events["flag"].isin(OBJECT_FLAGS)]
    for _, r in e.iterrows():
        a = int(pd.Timestamp(r["t_start"]).value // 10**9)
        te = r["t_end"]
        b = 2**62 if (not isinstance(te, str) or not te) else int(pd.Timestamp(te).value // 10**9)
        out.append((a, b, OBJECT_KINDS[r["kind"]] == "closed"))
    return out


def well_samples(sid: str, meta: dict, P: pd.DataFrame, be_rows: pd.DataFrame | None,
                 good_days: np.ndarray, events: pd.DataFrame | None = None) -> dict:
    """H1a samples of one well with the published barometric response: for each period
    ('pre' < 2023-01-01 <= 'post'), times (s), raw values, loading L and keep mask. A sample
    is kept when its stage-1 base screens are clean (``kind_base`` without artefact or
    sentinel bits), it lies outside every sample-removing L1 v3 event (flag 9 or 3), and its
    day has an L1 v3 daily flag in GOOD_DAY_FLAGS: the sample set behind L1's flag-0/1 daily
    values."""
    from h1_heads_qc import pipeline, screens
    from h1_heads_qc.assemble_v2 import ART_BITS_V2
    ov = be_override(be_rows)
    if ov is None:
        ov = {"segments": [], "p_ref_hpa": np.nan}
    res = pipeline.process_well(sid, meta, P, be_override=ov, o1=False)
    out = {"be_status": res.get("be", {}).get("status"),
           "baro_accepted": bool(any(x.get("a") is not None for x in ov["segments"])),
           "segments": [{"epoch": x["epoch"], "start": x["start"], "end": x["end"],
                         "accepted": x["a"] is not None} for x in ov["segments"]]}
    gd = np.asarray(good_days, dtype="datetime64[D]")
    wins = object_windows(events)
    bad_bits = int(ART_BITS_V2) | int(screens.K_SENTINEL)
    for per in ("pre", "post"):
        if per not in res:
            continue
        p = res[per]
        t = np.asarray(p["t"], dtype=np.int64)
        day = t.astype("datetime64[s]").astype("datetime64[D]")
        base_ok = (np.asarray(p["kind_base"]).astype(np.int64) & bad_bits) == 0
        in_obj = np.zeros(len(t), bool)
        for a, b, closed in wins:
            in_obj |= (t >= a) & ((t <= b) if closed else (t < b))
        keep = base_ok & ~in_obj & np.isin(day, gd)
        out[per] = {"t": t, "v_raw": np.asarray(p["v_raw"], float),
                    "L": np.asarray(p["L"], float), "keep": keep,
                    "n_object_samples": int(in_obj.sum()),
                    "n_base_flagged": int((~base_ok).sum())}
    return out


def to_series(t: np.ndarray, v: np.ndarray) -> pd.Series:
    return pd.Series(v, index=pd.to_datetime(t, unit="s"))


def daily_check(smp: dict, daily_ref: pd.DataFrame) -> dict:
    """Construction check against L1 v3: the daily mean of hourly means of the kept,
    corrected samples must equal L1's ``value - corr_m`` (and the loading L1's ``baro_m``)
    on every flag-0/1 day of L1 that the kept samples cover."""
    parts = []
    for per in ("pre", "post"):
        if per not in smp:
            continue
        p = smp[per]
        k = p["keep"]
        if not k.any():
            continue
        s = to_series(p["t"][k], p["v_raw"][k] - p["L"][k])
        lo = to_series(p["t"][k], p["L"][k])
        hv = s.resample("h").mean().dropna()
        hl = lo.resample("h").mean().dropna()
        parts.append(pd.DataFrame({"mine": hv.groupby(hv.index.normalize()).mean(),
                                   "mine_baro": hl.groupby(hl.index.normalize()).mean()}))
    if not parts:
        return {"n_days_compared": 0}
    m = pd.concat(parts)
    r = daily_ref.set_index("date")
    j = m.join(r, how="inner")
    if j.empty:
        return {"n_days_compared": 0}
    d = (j["mine"] - (j["value"] - j["corr_m"].fillna(0))).abs()
    db = (j["mine_baro"] - j["baro_m"].fillna(0)).abs()
    return {"n_days_compared": int(len(j)), "max_abs_diff_m": float(d.max()),
            "n_days_diff_gt_1mm": int((d > 1e-3).sum()), "max_abs_diff_baro_m": float(db.max()),
            "n_l1_days_not_covered": int(len(r.index.difference(m.index)))}


def extract_well(sid: str, meta: dict, P: pd.DataFrame, be_rows: pd.DataFrame | None,
                 good_days: np.ndarray, events: pd.DataFrame | None = None,
                 daily_ref: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Well-month features of one well (corrected primary columns plus ``*_raw``)."""
    smp = well_samples(sid, meta, P, be_rows, good_days, events)
    frames, diag = [], {"sid": sid, "be_status": smp["be_status"],
                        "baro_accepted": smp["baro_accepted"], "segments": smp["segments"]}
    if daily_ref is not None:
        diag["daily_check"] = daily_check(smp, daily_ref)
    spans = {"pre": pd.period_range(C.PRE_MONTHS[0], C.PRE_MONTHS[1], freq="M"),
             "post": pd.period_range(C.POST_MONTHS[0], C.POST_MONTHS[1], freq="M")}
    for per, months in spans.items():
        if per not in smp:
            continue
        p = smp[per]
        k = p["keep"]
        diag[f"n_samples_{per}"] = int(len(k))
        diag[f"n_kept_{per}"] = int(k.sum())
        if k.sum() < 24 * 10:
            continue
        t = p["t"][k]
        v_raw = p["v_raw"][k]
        Lk = p["L"][k]
        v_cor = v_raw - Lk
        Fc = month_features(to_series(t, v_cor), months)
        if Fc.empty:
            continue
        Fr = month_features(to_series(t, v_raw), months)
        keep_cols = [c for c in RAW_COLS if c in Fr.columns]
        Fr = Fr[["month"] + keep_cols].rename(columns={c: c + "_raw" for c in keep_cols})
        F = Fc.merge(Fr, on="month", how="left")
        # loading diagnostics per month
        Ls = to_series(t, Lk)
        lm = Ls.groupby(Ls.index.to_period("M")).agg(["mean", "std",
                                                     lambda x: float(np.mean(np.abs(x) > 0))])
        lm.columns = ["baro_mean_m", "baro_sd_m", "baro_frac_corrected"]
        lm.index = lm.index.astype(str)
        F = F.merge(lm, left_on="month", right_index=True, how="left")
        if "re_S1" in F and "re_S1_raw" in F:
            cdiff = (F["re_S1_raw"] - F["re_S1"]) + 1j * (F["im_S1_raw"] - F["im_S1"])
            F["A_S1_baro"] = np.abs(cdiff)
        F["period"] = per
        frames.append(F)
    if not frames:
        return pd.DataFrame(), diag
    F = pd.concat(frames, ignore_index=True)
    F.insert(0, "sid", sid)
    corr = F["baro_frac_corrected"].fillna(0) > 0
    F["flag"] = np.where(corr, 1, 0).astype(np.uint8)
    F["method"] = np.where(corr, "p5x1_h1a_screen_baro_removed", "p5x1_h1a_screen_no_baro")
    return F, diag
