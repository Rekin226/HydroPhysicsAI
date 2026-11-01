"""Census-only county audit (round 7 rank 8): pure functions, no file access.

Two bookkeeping questions about the census forcing, asked by county, with no head, leveling or
township value anywhere:

1. **Bimonthly even split.** TPC bills about 92 % of kept Changhua meters (94.5 % Yunlin) every
   two months and the product splits each bill evenly over its two months (PRODUCT_CARD of
   pump_electricity/preorigin/v3). Phase A pairs start in odd calendar months (Jan-Feb, ...),
   phase B pairs in even months (..., Dec-Jan); ``dataprep_staging/c0_pump_census/lib.py``
   ``pair_starts``. :func:`reshape_pairs` re-splits every even-split pair in proportion to a
   reference monthly shape (the county-class series of the monthly-billed meters), which keeps
   every pair total exactly. The audit compares the annual harmonic, the period-2 sawtooth and
   the climatology range of the delivered and the re-split county series.
2. **Duty-cap retention.** ``pumping.clean_census`` counts a shared meter once and drops a meter
   whose mean monthly draw exceeds ``cap_duty`` x its rated capacity (HP x 0.746 kW x 730 h).
   :func:`retention` splits de-duplicated energy into kept and dropped by county, class, scope
   (all pumps, in-grid pumps) and duty bucket, with the same semantics (a meter with no rated
   HP has no duty and is kept).

The decision rules (:func:`decide`) are fixed here, before the audit runs on the products.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

PAIR_ATOL_KWH = 0.51            # C0's identity tolerance for an even-split pair
KW_PER_HP = 0.746
HOURS_PER_MONTH = 730.0
CAP_DUTY = 1.0                  # the model's cap (pumping.clean_census default)
CAP_SENSITIVITY = (1.5, 2.0, 3.0, None)
DUTY_BUCKETS = (1.0, 1.5, 3.0, 10.0, math.inf)
N_MIN_REF = 30                  # monthly meters needed for a class-specific reference shape

# decision thresholds (fixed before the audit runs)
# D1 materiality. A fully even-split series carries the annual harmonic at (cos 15 deg)^2 =
# 0.933 of the true amplitude (pair mean x two-month hold), so with Changhua's 91.6 % bimonthly
# kept meters the expected ratio is about 0.94 BEFORE the audit runs (expected_amp_ratio). The
# bound is set against the seasonal-amplitude bands the Changhua gates use ([1/1.5, 1.5]): a
# forcing bias under 10 % is recorded as a known bias (README Limits), not a forcing change.
AMP_TOL = 0.10                  # |amp_delivered / amp_resplit - 1| for D1
PHASE_TOL_DAYS = 10.0           # |peak_delivered - peak_resplit| for D1
# D2. Smoothing both observed and modelled series cannot undo a multiplicative attenuation
# (smooth2 scales both by the same 0.933), so it is decided by the month-alternating artefact
# only; the climatology-range ratio is reported beside it.
SAW_TOL = 0.05                  # period-2 amplitude / annual amplitude for D2
RANGE_TOL = 0.05                # |range_delivered / range_resplit - 1|: report only
CAP_NEAR_TOL = 0.05             # irrigation in-grid energy dropped at duty (1, 1.5] / kept
ANNUAL_TOL = 0.01               # max relative change of a county's annual sum under the re-split
PARITY_TOL = 0.02               # |preorigin v3 in-grid / model_census v1 kept in-grid - 1|,
                                # Changhua (rank 16's regressor against R1c's forcing)
PHASES = ("A", "B", "monthly")  # resolvable billing phases (C0: code, else detected)
MEMORY_BAND = (0.06, 0.10)      # '~8 % kept' memory figure
DECISION_COUNTY = "changhua"
DECISION_CLASSES = ("all", "irrigation")

DAYS_PER_MONTH = 365.25 / 12.0


# ----------------------------------------------------------------------------- pairs
def pair_starts(phase: str, months: np.ndarray, t_lo: int, t_hi: int) -> np.ndarray:
    """Pair-start indices of a billing phase lying entirely inside ``[t_lo, t_hi)``.

    ``months`` holds the calendar month (1-12) of every index. Phase A pairs start in odd
    calendar months, phase B pairs in even months (C0 ``pair_starts``)."""
    if phase not in ("A", "B"):
        raise ValueError(f"phase {phase!r}")
    months = np.asarray(months)
    want = 1 if phase == "A" else 0
    t = np.arange(max(t_lo, 0), min(t_hi, len(months)) - 1)
    return t[(months[t] % 2) == want]


def even_split_mask(X: np.ndarray, starts: np.ndarray, atol: float = PAIR_ATOL_KWH
                    ) -> np.ndarray:
    """(M, len(starts)) True where both halves are finite, equal within ``atol`` and the pair
    total is positive."""
    a, b = X[:, starts], X[:, starts + 1]
    return np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= atol) & ((a + b) > 0)


def reshape_pairs(X: np.ndarray, phase: np.ndarray, group: np.ndarray, ref: np.ndarray,
                  months: np.ndarray, t_lo: int, t_hi: int, atol: float = PAIR_ATOL_KWH
                  ) -> tuple[np.ndarray, dict]:
    """Re-split every even-split bimonthly pair by a reference monthly shape.

    ``X`` (M, T) monthly kWh (NaN = missing), ``phase`` (M,) 'A', 'B' or anything else
    (left unchanged), ``group`` (M,) row of ``ref`` (G, T) each entity takes its shape from.
    A pair (t, t+1) gets ``P * ref[t] / (ref[t] + ref[t+1])`` and ``P * ref[t+1] / (...)``,
    ``P`` its total; a non-positive or non-finite reference sum leaves the pair even.
    Returns the re-split copy and counts (pairs re-split, their energy)."""
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    phase = np.asarray(phase).astype(str)
    group = np.asarray(group, dtype=np.int64)
    ref = np.asarray(ref, dtype=np.float64)
    n_pairs, e_pairs, n_even = 0, 0.0, 0
    for ph in ("A", "B"):
        rows = np.flatnonzero(phase == ph)
        if rows.size == 0:
            continue
        st = pair_starts(ph, months, t_lo, t_hi)
        if st.size == 0:
            continue
        sub = X[rows]
        ev = even_split_mask(sub, st, atol)
        ra, rb = ref[group[rows]][:, st], ref[group[rows]][:, st + 1]
        rs = ra + rb
        ok = ev & np.isfinite(rs) & (rs > 0) & np.isfinite(ra) & (ra >= 0) & (rb >= 0)
        tot = sub[:, st] + sub[:, st + 1]
        w = np.where(ok, ra / np.where(ok, rs, 1.0), 0.5)
        new_a = np.where(ok, tot * w, sub[:, st])
        new_b = np.where(ok, tot - new_a, sub[:, st + 1])
        blk = out[rows]
        blk[:, st] = new_a
        blk[:, st + 1] = new_b
        out[rows] = blk
        n_pairs += int(ok.sum())
        n_even += int(ev.sum())
        e_pairs += float(np.where(ok, tot, 0.0).sum())
    return out, {"pairs_even_split": n_even, "pairs_resplit": n_pairs,
                 "energy_resplit_kwh": e_pairs}


def even_split_share(X: np.ndarray, phase: np.ndarray, months: np.ndarray, t_lo: int,
                     t_hi: int, atol: float = PAIR_ATOL_KWH) -> dict:
    """Energy share of the bimonthly pairs (both halves finite) that are even splits."""
    phase = np.asarray(phase).astype(str)
    e_even = e_all = 0.0
    for ph in ("A", "B"):
        rows = np.flatnonzero(phase == ph)
        st = pair_starts(ph, months, t_lo, t_hi)
        if rows.size == 0 or st.size == 0:
            continue
        sub = X[rows]
        a, b = sub[:, st], sub[:, st + 1]
        fin = np.isfinite(a) & np.isfinite(b)
        tot = np.where(fin, a + b, 0.0)
        e_all += float(tot.sum())
        e_even += float(np.where(even_split_mask(sub, st, atol), tot, 0.0).sum())
    return {"energy_even_split_share": e_even / e_all if e_all > 0 else float("nan"),
            "energy_bimonthly_pairs_kwh": e_all}


# ----------------------------------------------------------------------------- seasonal shape
def _year_normalised(s: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Each calendar year divided by its own mean (removes interannual level changes)."""
    s = np.asarray(s, dtype=np.float64)
    out = np.full_like(s, np.nan)
    for y in np.unique(years):
        m = years == y
        mu = np.nanmean(s[m]) if np.isfinite(s[m]).any() else np.nan
        if np.isfinite(mu) and mu > 0:
            out[m] = s[m] / mu
    return out


def harmonic(s: np.ndarray, months: np.ndarray, years: np.ndarray) -> dict:
    """Annual harmonic of the year-normalised series: relative amplitude (fraction of the
    year's mean) and the peak day of year (calendar month m centred at (m - 0.5) months)."""
    z = _year_normalised(s, years)
    ok = np.isfinite(z)
    ph = 2 * np.pi * (np.asarray(months, float) - 0.5) / 12.0
    A = np.column_stack([np.ones(ok.sum()), np.cos(ph[ok]), np.sin(ph[ok])])
    if ok.sum() < 12:
        return {"amp_rel": float("nan"), "peak_day": float("nan")}
    c, *_ = np.linalg.lstsq(A, z[ok] - 1.0, rcond=None)
    amp = float(math.hypot(c[1], c[2]))
    ang = math.atan2(c[2], c[1]) % (2 * math.pi)
    return {"amp_rel": amp, "peak_day": float(ang / (2 * math.pi) * 365.25)}


def sawtooth(s: np.ndarray, months: np.ndarray, years: np.ndarray) -> dict:
    """The month-alternating artefact of a held two-month split: after the annual and
    semi-annual harmonics, the amplitude at 5/12 cycles per month (the alias of the annual
    cycle under a two-month hold; 7/12 is the same sampled frequency) and at 1/2 (the
    period-2 sawtooth), combined in quadrature, and its ratio to the annual amplitude. A single
    billing phase carries it; equal A and B phases cancel it."""
    z = _year_normalised(s, years)
    ok = np.isfinite(z)
    t = np.arange(len(z), dtype=float)
    ph = 2 * np.pi * (np.asarray(months, float) - 0.5) / 12.0
    f5 = 2 * np.pi * 5.0 / 12.0 * t
    cols = [np.ones(len(z)), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph),
            np.cos(f5), np.sin(f5), (-1.0) ** t]
    A = np.column_stack(cols)[ok]
    if ok.sum() < 24:
        return {"saw_amp_rel": float("nan"), "saw_to_annual": float("nan")}
    c, *_ = np.linalg.lstsq(A, z[ok] - 1.0, rcond=None)
    annual = math.hypot(c[1], c[2])
    saw = math.sqrt(c[5] ** 2 + c[6] ** 2 + c[7] ** 2)
    return {"saw_amp_rel": saw, "saw_to_annual": saw / annual if annual > 0 else float("nan")}


def clim_range(s: np.ndarray, months: np.ndarray, years: np.ndarray) -> float:
    """Max minus min of the 12-month climatology of the year-normalised series."""
    z = _year_normalised(s, years)
    clim = np.array([np.nanmean(z[np.asarray(months) == m]) for m in range(1, 13)])
    return float(np.nanmax(clim) - np.nanmin(clim))


def smooth2(s: np.ndarray) -> np.ndarray:
    """The declared 2-month smoother: the mean of the backward and forward 2-month averages,
    i.e. weights (1/4, 1/2, 1/4), zero phase; it removes the period-2 component exactly, damps
    the 5/12 alias to 0.067 and
    multiplies the annual harmonic by (1 + cos 30 deg) / 2 = 0.933. Ends are NaN."""
    s = np.asarray(s, dtype=np.float64)
    out = np.full_like(s, np.nan)
    out[1:-1] = 0.25 * s[:-2] + 0.5 * s[1:-1] + 0.25 * s[2:]
    return out


def series_metrics(s: np.ndarray, months: np.ndarray, years: np.ndarray) -> dict:
    return {**harmonic(s, months, years), **sawtooth(s, months, years),
            "clim_range": clim_range(s, months, years),
            "mean_kwh_per_month": float(np.nanmean(s))}


def resolve_phase(phase_code, phase_detected) -> np.ndarray:
    """C0's convention (c0_pump_census/checks.py, repair3_build.py): the billing code's phase,
    and where it is 'unknown' the detected phase if that is A, B or monthly; else 'unknown'."""
    pc = np.asarray(phase_code).astype(str)
    pdet = np.asarray(phase_detected).astype(str)
    return np.where(np.isin(pc, PHASES), pc, np.where(np.isin(pdet, PHASES), pdet, "unknown"))


def expected_amp_ratio(bimonthly_energy_share: float) -> float:
    """Annual-harmonic ratio delivered / true expected from even splitting alone."""
    return 1.0 - bimonthly_energy_share * (1.0 - math.cos(math.pi / 12.0) ** 2)


def circ_days(a: float, b: float) -> float:
    """Signed circular difference a - b in days on a 365.25-day year."""
    d = (a - b) % 365.25
    return float(d - 365.25 if d > 365.25 / 2 else d)


# ----------------------------------------------------------------------------- retention
def meter_table(pumps: pd.DataFrame, kwh: pd.DataFrame, t0: str, t1: str,
                meter_col: str = "電號1") -> pd.DataFrame:
    """Per-pump bookkeeping with ``pumping.clean_census``'s semantics, over [t0, t1).

    ``pumps``: sid, PUMP_HP, the meter column, county, cls, cell (-1 = not in an active cell).
    ``kwh``: pump, datetime, electricity_kwh. Returns one row per pump with its meter, the
    meter's duty, ``e_raw`` (the pump's own, possibly duplicated, series), ``e_dedup`` (its HP
    share of its meter's once-counted energy) and ``dropped`` (meter duty > CAP_DUTY)."""
    p = pumps.copy()
    p["sid"] = p["sid"].astype(str)
    p["_hp"] = pd.to_numeric(p["PUMP_HP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    meter = p[meter_col].astype(str) if meter_col in p else p["sid"]
    p["meter"] = meter.where(meter.notna() & (meter != "nan") & (meter != ""), p["sid"])
    k = kwh[["pump", "datetime", "electricity_kwh"]].copy()
    k["pump"] = k["pump"].astype(str)
    k["datetime"] = pd.to_datetime(k["datetime"])
    k = k[(k["datetime"] >= pd.Timestamp(t0)) & (k["datetime"] < pd.Timestamp(t1))]
    k = k.dropna(subset=["electricity_kwh"])
    e_raw = k.groupby("pump")["electricity_kwh"].sum()
    first = p.groupby("meter")["sid"].first()
    km = k[k["pump"].isin(set(first.values))].copy()
    km["meter"] = km["pump"].map(p.set_index("sid")["meter"])
    em = km.groupby("meter")["electricity_kwh"].agg(["mean", "sum"])
    hp_m = p.groupby("meter")["_hp"].sum()
    capm = hp_m.reindex(em.index).fillna(0.0) * KW_PER_HP * HOURS_PER_MONTH
    duty = em["mean"] / capm.replace(0.0, np.nan)
    hp_share = p["_hp"] / p.groupby("meter")["_hp"].transform("sum").replace(0.0, np.nan)
    n_on = p.groupby("meter")["sid"].transform("size")
    p["share"] = hp_share.fillna(1.0 / n_on)
    p["duty"] = p["meter"].map(duty)
    p["e_raw"] = p["sid"].map(e_raw).fillna(0.0)
    p["e_dedup"] = p["share"] * p["meter"].map(em["sum"]).fillna(0.0)
    p["dropped"] = p["duty"] > CAP_DUTY
    return p.drop(columns=["_hp"])


def _bucket_label(lo: float, hi: float) -> str:
    return f"({lo:g}, {'inf' if math.isinf(hi) else f'{hi:g}'}]"


def retention(pt: pd.DataFrame, years: float) -> dict:
    """Kept and dropped de-duplicated energy (GWh/yr) by county, scope, class and duty bucket,
    the kept energy at the sensitivity caps, and the share of the dropped energy held by the
    10 largest dropped meters (no meter id is reported)."""
    out: dict = {}
    for county, g in pt.groupby("county"):
        out[county] = {}
        for scope in ("all", "in_grid"):
            s = g if scope == "all" else g[g["cell"] >= 0]
            sc: dict = {}
            classes = ["all"] + sorted(s["cls"].dropna().unique().tolist())
            for cls in classes:
                c = s if cls == "all" else s[s["cls"] == cls]
                dedup = float(c["e_dedup"].sum())
                kept = float(c.loc[~c["dropped"], "e_dedup"].sum())
                row = {"raw_GWh_yr": float(c["e_raw"].sum()) / 1e6 / years,
                       "dedup_GWh_yr": dedup / 1e6 / years,
                       "kept_GWh_yr": kept / 1e6 / years,
                       "kept_frac_of_dedup": kept / dedup if dedup > 0 else float("nan"),
                       "kept_frac_of_raw": (kept / float(c["e_raw"].sum())
                                            if c["e_raw"].sum() > 0 else float("nan")),
                       "n_pumps": int(len(c)), "n_meters": int(c["meter"].nunique()),
                       "n_meters_dropped": int(c.loc[c["dropped"], "meter"].nunique())}
                bk = {}
                for lo, hi in zip(DUTY_BUCKETS[:-1], DUTY_BUCKETS[1:], strict=True):
                    m = (c["duty"] > lo) & (c["duty"] <= hi)
                    bk[_bucket_label(lo, hi)] = {
                        "GWh_yr": float(c.loc[m, "e_dedup"].sum()) / 1e6 / years,
                        "frac_of_dedup": (float(c.loc[m, "e_dedup"].sum()) / dedup
                                          if dedup > 0 else float("nan")),
                        "n_meters": int(c.loc[m, "meter"].nunique())}
                row["dropped_by_duty"] = bk
                row["kept_GWh_yr_at_cap"] = {
                    ("none" if cap is None else f"{cap:g}"):
                        float(c.loc[~(c["duty"] > cap) if cap is not None
                                    else slice(None), "e_dedup"].sum()) / 1e6 / years
                    for cap in CAP_SENSITIVITY}
                dm = c[c["dropped"]].groupby("meter")["e_dedup"].sum().sort_values(
                    ascending=False)
                row["top10_dropped_meters_share_of_dropped"] = (
                    float(dm.iloc[:10].sum() / dm.sum()) if dm.sum() > 0 else float("nan"))
                sc[cls] = row
            out[county][scope] = sc
    return out


def memory_figure(ret: dict, county: str = DECISION_COUNTY) -> dict:
    """Which ratio, if any, reproduces the '~8 % of de-duplicated Changhua energy kept'
    memory figure (data-audit-2026-10-25-extraction-map)."""
    r = ret[county]
    cands = {
        "kept/dedup, all pumps, all classes": r["all"]["all"]["kept_frac_of_dedup"],
        "kept/dedup, in-grid, all classes": r["in_grid"]["all"]["kept_frac_of_dedup"],
        "kept/raw, all pumps, all classes": r["all"]["all"]["kept_frac_of_raw"],
        "kept/raw, in-grid, all classes": r["in_grid"]["all"]["kept_frac_of_raw"],
    }
    for scope in ("all", "in_grid"):
        if "irrigation" in r[scope]:
            cands[f"kept/dedup, {scope.replace('_', '-')}, irrigation"] = (
                r[scope]["irrigation"]["kept_frac_of_dedup"])
    lo, hi = MEMORY_BAND
    hits = sorted(k for k, v in cands.items() if np.isfinite(v) and lo <= v <= hi)
    return {"band": list(MEMORY_BAND), "candidates": cands, "reproduced_by": hits,
            "reading": ("reproduced by: " + "; ".join(hits)) if hits else
            "not reproduced by any declared ratio: the '~8 % kept' figure is withdrawn from "
            "the record and the in-grid kept/dedup ratio replaces it"}


def forcing_parity(ret: dict, extra: dict, county: str = DECISION_COUNTY) -> dict:
    """The regressor rank 16 builds (preorigin v3 in-grid kept pumps, all classes) against R1c's
    forcing (model_census v1, clean_census kept, in-grid, all classes), GWh/yr 2012-2022."""
    mc = ret[county]["in_grid"]["all"]["kept_GWh_yr"]
    po = extra[county]["in_grid_GWh_yr_delivered"]
    rel = po / mc - 1.0 if mc > 0 else float("nan")
    return {"model_census_v1_kept_in_grid_GWh_yr": mc, "preorigin_v3_in_grid_GWh_yr": po,
            "rel_diff": rel, "tolerance": PARITY_TOL,
            "within": bool(np.isfinite(rel) and abs(rel) <= PARITY_TOL)}


# ----------------------------------------------------------------------------- decisions
def decide(seasonal: dict, ret: dict, annual_change: dict) -> dict:
    """D1 (the fixed Changhua forcing version), D1b (duty-cap member), D2 (2-month smoothing
    for seasonal-amplitude gates). ``seasonal[county][class][variant]`` holds
    :func:`series_metrics` for variants 'delivered' and 'resplit'; ``annual_change[county]``
    is the max relative change of a calendar-year county sum under the re-split."""
    c = DECISION_COUNTY
    d1_rows, d2_rows = {}, {}
    for cls in DECISION_CLASSES:
        s = seasonal[c][cls]
        dl, rs = s["delivered"], s["resplit"]
        amp_ratio = dl["amp_rel"] / rs["amp_rel"] if rs["amp_rel"] > 0 else float("nan")
        dphi = circ_days(dl["peak_day"], rs["peak_day"])
        d1_rows[cls] = {"amp_ratio": amp_ratio, "phase_diff_days": dphi,
                        "expected_amp_ratio_theory": s.get("expected_amp_ratio_from_even_split"),
                        "within": bool(np.isfinite(amp_ratio) and abs(amp_ratio - 1) <= AMP_TOL
                                       and abs(dphi) <= PHASE_TOL_DAYS)}
        rr = dl["clim_range"] / rs["clim_range"] if rs["clim_range"] > 0 else float("nan")
        saw = dl["saw_to_annual"]
        d2_rows[cls] = {"saw_to_annual": saw, "range_ratio_report_only": rr,
                        "range_outside_tol_report_only": bool(not np.isfinite(rr)
                                                              or abs(rr - 1) > RANGE_TOL),
                        "smooth": bool(not np.isfinite(saw) or saw > SAW_TOL)}
    d1_keep = all(r["within"] for r in d1_rows.values())
    irr = ret[c]["in_grid"].get("irrigation")
    near = (irr["dropped_by_duty"][_bucket_label(1.0, 1.5)]["GWh_yr"] / irr["kept_GWh_yr"]
            if irr and irr["kept_GWh_yr"] > 0 else float("nan"))
    d1b = bool(np.isfinite(near) and near > CAP_NEAR_TOL)
    d2 = any(r["smooth"] for r in d2_rows.values())
    ann = annual_change.get(c, float("nan"))
    return {
        "D1_changhua_forcing_version": {
            "decision": "model_census_v1_as_is" if d1_keep else "resplit_product_required",
            "rows": d1_rows, "thresholds": {"amp_tol": AMP_TOL,
                                            "phase_tol_days": PHASE_TOL_DAYS},
            "nature": ("a measured known bias, not an open choice: an even split damps the "
                       "annual harmonic to at most (cos 15 deg)^2 = 0.933 and leaves the peak "
                       "in place by pair symmetry, inside AMP_TOL, so 'as is' was expected "
                       "unless the monthly-meter shape is unusual; expected_amp_ratio_theory "
                       "sits beside the realised amp_ratio"),
            "meaning": ("every Changhua route uses pump_electricity/model_census v1 (R1c's "
                        "forcing, even-split as delivered), cites this audit and records the "
                        "measured amplitude damping and peak shift as a known forcing bias "
                        "(README Limits)" if d1_keep else
                        "every Changhua route waits for a governed re-split product (Changhua "
                        "pumps re-split by reshape_pairs; Yunlin's registered forcing "
                        "unchanged, the parity break disclosed)")},
        "D1b_duty_cap": {
            "decision": "cap_member_required" if d1b else "cap_immaterial",
            "irrigation_in_grid_dropped_at_duty_1_to_1.5_over_kept": near,
            "threshold": CAP_NEAR_TOL,
            "meaning": ("every Changhua route carries a report-only cap-1.5 member; the primary "
                        "stays cap 1.0 (the registered convention shared with Yunlin)" if d1b
                        else "the cap-1.0 primary stands with no extra member")},
        "D2_seasonal_amplitude_gates": {
            "decision": "smooth2" if d2 else "unsmoothed",
            "rows": d2_rows, "thresholds": {"saw_tol": SAW_TOL,
                                            "range_tol_report_only": RANGE_TOL},
            "meaning": ("Changhua seasonal-amplitude gates compare smooth2() of the observed and "
                        "modelled monthly series (weights 1/4, 1/2, 1/4)" if d2 else
                        "Changhua seasonal-amplitude gates use the unsmoothed monthly series")},
        "annual_sum_max_rel_change": {"changhua": ann, "tolerance": ANNUAL_TOL,
                                      "within": bool(np.isfinite(ann) and ann <= ANNUAL_TOL)},
    }
