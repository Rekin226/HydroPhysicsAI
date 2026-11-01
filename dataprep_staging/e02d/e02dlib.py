"""Shared constants and pure functions of round 7 ranks 4-6 (bands): DR.relative_drift.v1,
E02d.decomposed_prospective_2026-12.v1, U1.decomp_power.v1 and the year-2/3 windows.

Nothing here opens a file. Every function works on arrays (wells x months) or frames the
callers build; the scripts that read the frozen model output or, for the lead, the pre-origin
fitted-month residuals, live beside this module and are bound by sha256 in the registrations.

The relative (per-well) estimand. A head residual r[w, t] = observed - predicted (the frozen
E02 centre, per-well datum included) is split into a part shared by the wells of a group in a
month and a per-well part. The per-well part is

    e[w, t] = r[w, t] - median{ r[v, t] : v in group(w), v != w, v reports in month t }

(the leave-one-out median, LOO), defined only when at least ``MIN_OTHERS`` other wells of the
group report that month. A shock shared by every well of a group cancels exactly; a shock
shared by a subset of the group (another layer, another zone) survives in part. The same
operator builds the in-sample spread (``sd_fit_rel``, fitted months 2012-01..2022-12), the
ensemble's relative member spread and the prospective score, so the interval and the test
measure the same quantity.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- identities
DR_ID = "DR.relative_drift.v1"
DR_WP = "DR"
DR_UNIT = "DR:fit_residuals_2012-01_2022-12_wells158"
DR_SCOPE = "heads:2012-01..2022-12"
POWER_ID = "U1.decomp_power.v1"
POWER_WP = "U1"
E02D_ID = "E02d.decomposed_prospective_2026-12.v1"
E02D_WP = "E02d"
E02D_TEST = "E02dP.G_idio_2026-12"
E02Y_ID = "E02y.years23_2027-11_2029-10.v1"
E02Y_WP = "E02y"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-01b.json"
RULING_IDS = {"dr": "R7.e02d_residual_read", "years23": "R7.heads_years23"}

E02_DIR = "results/twin/prospective_2026-11"
E02_PRED_SHA = "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32"
E02_PROTOCOL_SHA = "8a96c0ea9330fd79432eb17a666640ad526728c838b62bd8459cd41de0d908b9"
FORWARD = "results/twin_forward/datum_gate_innov.npz"
MODEL_SHA = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
STATIONS = "AMP_V2/data/fan_stations.parquet"
WELLS_DIR = "AMP_V2/data/wells"
THETA_DIR = "results/twin_runs/stage3_datum_gate"

# ----------------------------------------------------------------------------- calendar
ORIGIN = "2022-12"
FIT = ("2012-01", "2022-12")
EARLY = ("2012-01", "2016-12")
LATE = ("2018-01", "2022-12")
DELTA_YEARS = 6.0                 # distance between the two halves' mid-points
WINDOWS = {
    "A": {"first": "2026-12", "last": "2027-10", "not_before": "2028-01-15",
          "no_later_than": "2028-01-31", "unit": "E02d:heads_2026-12_2027-10_wells158",
          "role": "year 1 (E02d)"},
    "B": {"first": "2027-11", "last": "2028-10", "not_before": "2029-01-15",
          "no_later_than": "2029-01-31", "unit": "E02y:heads_2027-11_2028-10_wells158",
          "role": "year 2"},
    "C": {"first": "2028-11", "last": "2029-10", "not_before": "2030-01-15",
          "no_later_than": "2030-01-31", "unit": "E02y:heads_2028-11_2029-10_wells158",
          "role": "year 3"},
}
ABS_DIRS = {"A": E02_DIR, "B": "results/twin/prospective_2027-11_y2",
            "C": "results/twin/prospective_2028-11_y3"}
ABS_CHECK_DIR = "results/twin_runs/e02d/register_check_2026-12"
#: twin.prospective.register refuses a window that has started, so the reproduction check
#: runs on 2026-12..2027-10 (E02's rows of those months), before 2026-12-01

# ----------------------------------------------------------------------------- the rules
COUNTY = {"07": "Changhua", "08": "Nantou", "09": "Yunlin"}
COUNTIES = ("Yunlin", "Changhua")
N_UNIT_WELLS = 158
Z90 = 1.6448536269514722
LAYER_GROUP = {1: "L1", 2: "L2", 3: "L34", 4: "L34"}
LAYER_ORDER = ("L1", "L2", "L34")
ZONE_NAMES = ("prox", "mid", "dist")      # twin.zone_county.column_zone_map ids 0, 1, 2
MIN_OTHERS = 5                    # LOO median needs >= 5 other wells of the group that month
MIN_GROUP_WELLS = 8               # a group of the partition has >= 8 of the unit's wells
ICC_SWITCH = 0.05                 # a sub-county partition is used only above this ICC
MIN_HALF_MONTHS = 24
INCLUSION_MIN_SHARE = 0.60
DRIFT_RULE_FRACTION = 0.10
LITERAL_LEAD_YEARS = 4.0          # the judge's wording, reported beside the declared rule
#: where the drift lead is measured from (chosen by the lead at DR's --write, recorded in the
#: declaration): each well's datum centre (mean fitted month) or the forecast origin
LEAD_DEFINITIONS = ("datum_centre", "origin")
ORIGIN_ANCHOR_YEAR = 2023.0       # the end of the origin month 2022-12
ZONE_IDS = (0, 1, 2)              # column_zone_map ids DR accepts (else engineering failure)
# block bootstrap of the remaining sub-group ICC (bounds U1.decomp_power.v1's a_s grid)
ICC_BOOT_B = 400
ICC_BOOT_SEED = 20261105
ICC_BOOT_BLOCK = 12               # months per moving block
ICC_BOOT_Q = 0.90
AR_CLIP = (0.0, 0.95)
MIN_AR_PAIRS = 20
MIN_FIT_MONTHS_SD = 12
BOOT_B = 2000
BOOT_SEED = 20261102
# scoring (G-idio)
TARGET, GATE_LEVEL = 0.90, 0.95
POINT_WINDOW = (0.85, 0.95)
N_EFF_WINDOW_MIN = 30.0
WINDOW_CONF = 0.95
GATE_METHOD = "upper90"
MIN_MONTHS = 6
MIN_WELLS_COUNTY = 20
MIN_EXT_WELLS_MONTH = 8
# fold-member rerun
MEMBER_MEAN_TOL_M = 1e-3
DATUM_TOL_M = 1e-6
ABS_REPRO_TOL_M = 1e-9


def county_of(sid) -> str:
    return COUNTY.get(str(sid)[:2], "other")


def month_index(first: str, last: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{first}-01", f"{last}-01", freq="MS")


def year_frac(dates) -> np.ndarray:
    """Decimal year of month starts (mid-month: + 0.5 / 12)."""
    d = pd.DatetimeIndex(dates)
    return d.year.to_numpy(float) + (d.month.to_numpy(float) - 0.5) / 12.0


def window_mid(win: str) -> float:
    m = month_index(WINDOWS[win]["first"], WINDOWS[win]["last"])
    return float(year_frac(m).mean())


def lead_anchor(definition: str, t_centre) -> np.ndarray:
    """Decimal year from which the drift lead is measured, per well: the datum centre
    (``t_centre``) or the forecast origin (ORIGIN_ANCHOR_YEAR for every well)."""
    t = np.asarray(t_centre, dtype=float)
    if definition == "datum_centre":
        return t
    if definition == "origin":
        return np.full(t.shape, ORIGIN_ANCHOR_YEAR)
    raise ValueError(f"lead definition {definition!r} not in {LEAD_DEFINITIONS}")


# ----------------------------------------------------------------------------- LOO median
def loo_median(X: np.ndarray, min_others: int = MIN_OTHERS) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out median over the wells axis (-2) of ``X`` (..., W, T), NaN = missing.

    Returns (median of the OTHER finite values of the same month, number of those others);
    the median is NaN where the entry itself is missing or fewer than ``min_others`` others
    report."""
    X = np.asarray(X, dtype=float)
    W = X.shape[-2]
    fin = np.isfinite(X)
    n = fin.sum(axis=-2, keepdims=True)
    s = np.sort(X, axis=-2)                       # NaN sorted last
    order = np.argsort(np.where(fin, X, np.inf), axis=-2, kind="stable")
    rank = np.empty_like(order)
    ar = np.broadcast_to(np.arange(W).reshape((W, 1)), order.shape)
    np.put_along_axis(rank, order, ar, axis=-2)
    m = np.broadcast_to(n - 1, X.shape)
    lo = (m - 1) // 2
    hi = m // 2
    i_lo = np.clip(lo + (lo >= rank), 0, W - 1)
    i_hi = np.clip(hi + (hi >= rank), 0, W - 1)
    med = 0.5 * (np.take_along_axis(s, i_lo, axis=-2) + np.take_along_axis(s, i_hi, axis=-2))
    others = np.where(fin, m, 0)
    med = np.where(fin & (m >= max(1, int(min_others))), med, np.nan)
    return med, others


def relative(X: np.ndarray, groups, min_others: int = MIN_OTHERS) -> np.ndarray:
    """Per-well part of ``X`` (..., W, T): X minus the LOO median of its group in the same
    month (NaN where undefined). ``groups``: (W,) labels."""
    X = np.asarray(X, dtype=float)
    g = np.asarray(groups)
    out = np.full(X.shape, np.nan)
    for lab in pd.unique(g):
        idx = np.flatnonzero(g == lab)
        sub = X[..., idx, :]
        med, _ = loo_median(sub, min_others)
        out[..., idx, :] = sub - med
    return out


# ----------------------------------------------------------------------------- ICC
def anova_icc(values, cells) -> dict:
    """One-way random-effects ANOVA ICC of finite ``values`` grouped by ``cells`` (unequal
    sizes): (MSB - MSW) / (MSB + (n0 - 1) MSW), n0 = (N - sum n_i^2 / N) / (k - 1). Returns
    the raw estimate and the estimate clipped to [0, 1] (``icc``)."""
    v = np.asarray(values, dtype=float).ravel()
    c = np.asarray(cells).ravel()
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if v.size < 3:
        return {"icc": float("nan"), "icc_raw": float("nan"), "k": 0, "n": int(v.size)}
    codes, inv = np.unique(c, return_inverse=True)
    k = codes.size
    N = v.size
    if k < 2 or N - k < 1:
        return {"icc": float("nan"), "icc_raw": float("nan"), "k": int(k), "n": int(N)}
    n_i = np.bincount(inv).astype(float)
    mean_i = np.bincount(inv, weights=v) / n_i
    grand = v.mean()
    ssb = float((n_i * (mean_i - grand) ** 2).sum())
    ssw = float(((v - mean_i[inv]) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    den = msb + (n0 - 1.0) * msw
    raw = (msb - msw) / den if den > 0 else float("nan")
    return {"icc": float(min(1.0, max(0.0, raw))) if math.isfinite(raw) else float("nan"),
            "icc_raw": float(raw), "k": int(k), "n": int(N), "msb": msb, "msw": msw,
            "n0": float(n0)}


def icc_boot_upper(E: np.ndarray, parts: list, B: int = ICC_BOOT_B,
                   seed: int = ICC_BOOT_SEED, block: int = ICC_BOOT_BLOCK,
                   q: float = ICC_BOOT_Q) -> dict:
    """Well x moving-month-block bootstrap of the group-month ICC of ``E`` (W, T): wells
    resampled with replacement (keeping their labels), months as moving blocks of ``block``
    consecutive months; per replicate the larger ICC over ``parts`` (each a (W,) label
    array). Captures sampling uncertainty only (not in-sample damping). Returns the
    ``q``-quantile (``upper``) and the median of the replicate ICCs (clipped to [0, 1])."""
    E = np.asarray(E, dtype=float)
    W, T = E.shape
    codes = [pd.factorize(np.asarray(p, dtype=str))[0] for p in parts]
    rng = np.random.default_rng(seed)
    nb = -(-T // block)
    reps = np.empty(B)
    for b in range(B):
        wi = rng.integers(0, W, size=W)
        starts = rng.integers(0, max(1, T - block + 1), size=nb)
        ti = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        ti = np.minimum(ti, T - 1)
        Eb = E[np.ix_(wi, ti)]
        col = np.arange(T)[None, :]
        best = float("nan")
        for c in codes:
            v = anova_icc(Eb, c[wi][:, None] * T + col)["icc"]
            if math.isfinite(v) and not (best >= v):
                best = v
        reps[b] = best
    ok = reps[np.isfinite(reps)]
    if ok.size == 0:
        return {"upper": float("nan"), "median": float("nan"), "q": q, "B": B,
                "block_months": block, "seed": seed, "finite": 0}
    return {"upper": float(np.quantile(ok, q)), "median": float(np.median(ok)), "q": q,
            "B": B, "block_months": block, "seed": seed, "finite": int(ok.size),
            "label": "sampling uncertainty only (wells and month blocks); in-sample damping "
                     "is not covered"}


def cell_labels(part, W: int, T: int) -> np.ndarray:
    """(W, T) labels '<partition group>|<month index>' for an ANOVA over group-months."""
    p = np.asarray(part, dtype=str).reshape(W, 1)
    t = np.arange(T).astype(str).reshape(1, T)
    return np.char.add(np.char.add(np.broadcast_to(p, (W, T)), "|"),
                       np.broadcast_to(t, (W, T)))


# ----------------------------------------------------------------------------- partitions
def merge_small(labels, order, min_wells: int = MIN_GROUP_WELLS) -> tuple[np.ndarray, list]:
    """Merge groups of a partition (declared ``order``) with fewer than ``min_wells`` wells:
    the first under-size group (in order) joins the next group in order, or the previous
    one if it is the last; repeat. Returns (merged labels, merge log)."""
    lab = np.asarray(labels, dtype=object).copy()
    seq = [g for g in order if (lab == g).any()]
    log = []
    while len(seq) > 1:
        small = [g for g in seq if (lab == g).sum() < min_wells]
        if not small:
            break
        g = small[0]
        i = seq.index(g)
        into = seq[i + 1] if i + 1 < len(seq) else seq[i - 1]
        new = f"{into}+{g}" if i + 1 >= len(seq) else f"{g}+{into}"
        lab[(lab == g) | (lab == into)] = new
        seq[seq.index(into)] = new
        seq.remove(g)
        log.append({"merged": g, "into": into, "as": new})
    return lab.astype(str), log


def choose_grouping(icc_layer: float, icc_zone: float, n_layer_groups: int,
                    n_zone_groups: int) -> dict:
    """Declared grouping rule (per county): use the sub-county partition (layer group or
    zone, after merging groups below MIN_GROUP_WELLS) whose county-relative group-month ICC
    is larger, if that ICC is at least ICC_SWITCH and the partition still has at least two
    groups; otherwise the county. A tie goes to the layer partition."""
    cand = []
    if n_layer_groups >= 2 and math.isfinite(icc_layer):
        cand.append(("layer", icc_layer))
    if n_zone_groups >= 2 and math.isfinite(icc_zone):
        cand.append(("zone", icc_zone))
    cand.sort(key=lambda x: (-x[1], x[0] != "layer"))
    if cand and cand[0][1] >= ICC_SWITCH:
        return {"grouping": cand[0][0], "icc": cand[0][1], "rule": "max ICC >= switch"}
    return {"grouping": "county", "icc": max([c[1] for c in cand], default=float("nan")),
            "rule": "no sub-county partition at or above the switch"}


# ----------------------------------------------------------------------------- drift
def var_factor(n: int, phi: float) -> float:
    """Variance factor of the mean of n consecutive stationary AR(1) values:
    1 + 2 sum_{k=1}^{n-1} (1 - k/n) phi^k (gaps are ignored: n = observed count)."""
    if n <= 1 or phi <= 0:
        return 1.0
    k = np.arange(1, n)
    return float(1.0 + 2.0 * ((1.0 - k / n) * phi ** k).sum())


def ar1_phi(x: np.ndarray, halves: np.ndarray) -> tuple[float, int]:
    """Lag-1 autocorrelation of one well's half-demeaned series on consecutive month pairs
    inside the same half: sum x_t x_{t+1} / sum (x_t^2 + x_{t+1}^2) / 2, clipped to AR_CLIP.
    ``halves``: (T,) 0 early, 1 late, -1 neither. Returns (phi, n_pairs)."""
    x = np.asarray(x, dtype=float).copy()
    for h in (0, 1):
        m = (halves == h) & np.isfinite(x)
        if m.any():
            x[m] -= x[m].mean()
    a, b = x[:-1], x[1:]
    same = (halves[:-1] == halves[1:]) & (halves[:-1] >= 0)
    ok = same & np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    den = 0.5 * float((a[ok] ** 2 + b[ok] ** 2).sum())
    phi = float((a[ok] * b[ok]).sum() / den) if den > 0 else 0.0
    return float(min(AR_CLIP[1], max(AR_CLIP[0], phi))), n


def drift_county(E: np.ndarray, dates, county_mask: np.ndarray, seed: int = BOOT_SEED,
                 B: int = BOOT_B) -> dict:
    """Split-half drift of the relative residuals ``E`` (W, T fitted months) of one county.

    d_i = mean(late half) - mean(early half) for wells with >= MIN_HALF_MONTHS months in
    each half; expected var under no drift E_i = s_i^2 (f(n_e, phi_i) / n_e + f(n_l, phi_i)
    / n_l) with s_i^2 the pooled within-half variance and phi_i the in-sample AR(1) (county
    median where fewer than MIN_AR_PAIRS pairs); linear-trend model: tau^2 = max(0,
    mean(d^2 - E)) / DELTA_YEARS^2 (m^2 per year^2), a LOWER bound (theta and the datum were
    fitted on both halves)."""
    d = pd.DatetimeIndex(dates)
    t = d.to_period("M")
    halves = np.full(len(d), -1)
    halves[(t >= pd.Period(EARLY[0], "M")) & (t <= pd.Period(EARLY[1], "M"))] = 0
    halves[(t >= pd.Period(LATE[0], "M")) & (t <= pd.Period(LATE[1], "M"))] = 1
    idx = np.flatnonzero(county_mask)
    rows = []
    for w in idx:
        x = E[w]
        fe = np.isfinite(x) & (halves == 0)
        fl = np.isfinite(x) & (halves == 1)
        ne, nl = int(fe.sum()), int(fl.sum())
        phi, npairs = ar1_phi(x, halves)
        rec = {"w": int(w), "n_early": ne, "n_late": nl, "phi_raw": phi, "pairs": npairs,
               "included": ne >= MIN_HALF_MONTHS and nl >= MIN_HALF_MONTHS}
        if rec["included"]:
            me, ml = x[fe].mean(), x[fl].mean()
            ss = ((x[fe] - me) ** 2).sum() + ((x[fl] - ml) ** 2).sum()
            rec.update({"d": float(ml - me), "s2": float(ss / (ne + nl - 2))})
        rows.append(rec)
    inc = [r for r in rows if r["included"]]
    good_phi = [r["phi_raw"] for r in rows if r["pairs"] >= MIN_AR_PAIRS
                and math.isfinite(r["phi_raw"])]
    phi_med = float(np.median(good_phi)) if good_phi else 0.0
    for r in rows:
        r["phi"] = (r["phi_raw"] if r["pairs"] >= MIN_AR_PAIRS and math.isfinite(r["phi_raw"])
                    else phi_med)
        if r["included"]:
            r["expected_var"] = r["s2"] * (var_factor(r["n_early"], r["phi"]) / r["n_early"]
                                           + var_factor(r["n_late"], r["phi"]) / r["n_late"])
    out = {"n_wells": int(idx.size), "n_included": len(inc),
           "share_included": len(inc) / idx.size if idx.size else 0.0,
           "phi_median": phi_med, "rows": rows}
    if not inc:
        out.update({"tau2_per_year2": float("nan"), "evaluable": False})
        return out
    dd = np.array([r["d"] for r in inc])
    ee = np.array([r["expected_var"] for r in inc])
    exc = float(np.mean(dd ** 2 - ee))
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, dd.size, size=(B, dd.size))
    boot = np.maximum(0.0, (dd[bi] ** 2 - ee[bi]).mean(axis=1)) / DELTA_YEARS ** 2
    out.update({
        "evaluable": out["share_included"] >= INCLUSION_MIN_SHARE,
        "mean_d2": float(np.mean(dd ** 2)), "mean_expected": float(np.mean(ee)),
        "ratio_d2_expected": float(np.mean(dd ** 2) / np.mean(ee)) if np.mean(ee) > 0 else
        float("nan"),
        "excess": exc, "tau2_per_year2": max(0.0, exc) / DELTA_YEARS ** 2,
        "tau_m_per_year": math.sqrt(max(0.0, exc)) / DELTA_YEARS,
        "tau2_boot90_report_only": [float(np.quantile(boot, 0.05)),
                                    float(np.quantile(boot, 0.95))],
        "label": "lower bound: theta and the datum were fitted on both halves"})
    return out


def drift_rule(tau2: float, sigma_rel: float, lead_years: float,
               fraction: float = DRIFT_RULE_FRACTION) -> dict:
    """The declared 10 % rule: include the drift term when sqrt(sigma^2 + tau^2 L^2) exceeds
    sigma by more than ``fraction``."""
    if not (math.isfinite(tau2) and sigma_rel > 0):
        return {"include_drift": False, "increase": float("nan")}
    inc = math.sqrt(sigma_rel ** 2 + tau2 * lead_years ** 2) / sigma_rel - 1.0
    return {"include_drift": bool(inc > fraction), "increase": float(inc),
            "sigma_rel_m": float(sigma_rel), "lead_years": float(lead_years),
            "tau2_per_year2": float(tau2)}


# ----------------------------------------------------------------------------- interval
def sigma_ens_rel(members: np.ndarray, groups, min_others: int = MIN_OTHERS) -> np.ndarray:
    """Relative member spread (W, T): root mean square over members m of
    a_m - LOOmedian_group(a_m), a_m = member m minus the member mean (population form, as
    forward.run's heads_std). ``members``: (M, W, T)."""
    M = np.asarray(members, dtype=float)
    a = M - M.mean(axis=0, keepdims=True)
    rel = relative(a, groups, min_others)
    return np.sqrt(np.nanmean(rel ** 2, axis=0))


def sigma_idio(sd_fit_rel: np.ndarray, ens_rel: np.ndarray, tau2: np.ndarray,
               dt_years: np.ndarray) -> np.ndarray:
    """sqrt(sd_fit_rel^2 + ens_rel^2 + tau2 dt^2), per well-month (W, T); ``tau2`` is 0 for
    wells whose county did not trigger the drift term."""
    s = np.asarray(sd_fit_rel, float)[:, None]
    tt = np.asarray(tau2, float)[:, None]
    return np.sqrt(s ** 2 + np.asarray(ens_rel, float) ** 2 + tt * np.asarray(dt_years) ** 2)


# ----------------------------------------------------------------------------- scoring
def gate_kwargs() -> dict:
    return {"method": GATE_METHOD, "target": TARGET, "level": GATE_LEVEL,
            "window": POINT_WINDOW, "n_eff_window_min": N_EFF_WINDOW_MIN,
            "conf": WINDOW_CONF}


def county_verdict(hits: np.ndarray, county_mask: np.ndarray, below=None) -> dict:
    """G-idio for one county from (W, T) hits (1/0, NaN = not scored): wells with >=
    MIN_MONTHS scored months; undecided below MIN_WELLS_COUNTY such wells; otherwise
    harness.coverage3 gate-3 (clusters = wells) AND point coverage within POINT_WINDOW."""
    from harness import coverage3

    H = np.asarray(hits, dtype=float)[np.asarray(county_mask, bool)]
    nm = np.isfinite(H).sum(axis=1)
    keep = nm >= MIN_MONTHS
    k = int(keep.sum())
    if k < MIN_WELLS_COUNTY:
        return {"verdict": "not_evaluable", "wells": k,
                "why": f"{k} wells with >= {MIN_MONTHS} scored months (< {MIN_WELLS_COUNTY})"}
    Hk = H[keep]
    ww, tt = np.nonzero(np.isfinite(Hk))
    flat = Hk[ww, tt]
    g = coverage3.coverage_gate(flat, ww, **gate_kwargs())
    in_win = POINT_WINDOW[0] <= g["coverage"] <= POINT_WINDOW[1]
    ok = bool(g["pass"] and in_win)
    rec = {"verdict": "pass" if ok else "fail", "wells": k, "rows": int(flat.size),
           "coverage": g["coverage"], "gate3_pass": bool(g["pass"]),
           "in_point_window": bool(in_win), "gate": g}
    if below is not None:
        B = np.asarray(below, dtype=float)[np.asarray(county_mask, bool)][keep]
        b = B[ww, tt]
        rec["miss_below"] = float(np.mean(b == 1))
        rec["miss_above"] = float(np.mean((flat == 0) & (b == 0)))
    return rec


def relative_hits(R: np.ndarray, sigma: np.ndarray, groups, min_others: int = MIN_OTHERS,
                  z: float = Z90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hits, below, e): e = relative(R); hit = |e| <= z sigma (NaN where e is undefined);
    below = e < -z sigma."""
    e = relative(R, groups, min_others)
    ok = np.isfinite(e) & np.isfinite(sigma)
    hw = z * np.asarray(sigma, float)
    hits = np.where(ok, (np.abs(e) <= hw).astype(float), np.nan)
    below = np.where(ok, (e < -hw).astype(float), np.nan)
    return hits, below, e


def frame_to_arrays(frame: pd.DataFrame, wells: list[str], months: pd.DatetimeIndex,
                    cols: tuple[str, ...]) -> dict:
    """Pivot scored rows (station_id, date, cols...) to (W, T) arrays on ``wells`` x
    ``months`` (NaN where absent)."""
    f = frame.copy()
    f["station_id"] = f["station_id"].astype(str)
    f["_m"] = pd.to_datetime(f["date"]).dt.to_period("M").dt.to_timestamp()
    if f.duplicated(["station_id", "_m"]).any():
        raise ValueError("duplicate station-months")
    wi = {w: i for i, w in enumerate(wells)}
    ti = {m: i for i, m in enumerate(months)}
    f = f[f["station_id"].isin(wi) & f["_m"].isin(ti)]
    out = {}
    r = f["station_id"].map(wi).to_numpy(int)
    c = f["_m"].map(ti).to_numpy(int)
    for col in cols:
        a = np.full((len(wells), len(months)), np.nan)
        a[r, c] = f[col].to_numpy(float)
        out[col] = a
    return out
