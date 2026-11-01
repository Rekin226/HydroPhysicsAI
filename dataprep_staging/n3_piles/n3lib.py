"""Shared constants and estimators of N3.piles_prospective.v1 and its after-the-fact companion.

One implementation, hashed into the registration, used by the synthetic decidability gate
(``g0_n3.py``), the 2028 scorer (``score_n3.py``) and the rank-12 after-the-fact diagnostic
(``posthoc_pile_conversion.py``). Pure functions on arrays and frames: nothing here opens a
product or a forward archive.

Sign convention (fixed from the product docs and the extraction code, no value read):
``value`` = compression (deep_benchmarks_printed MANIFEST note 'value = compression in mm';
twin.report_extract SYSTEM 'negative = rebound'); drawdown D = minus head, so a falling
head is a positive dD and a compressing pile a positive increment; the model's g_mod is
positive (compaction per metre of drawdown, cm/m).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E = "dataprep_staging/n3_piles"
# the decision rule is N2's, called from the one shared implementation (judge rank 6:
# 'the scorer code shared with N2's ratio_verdict'; user ruling 2026-11-01, rank 10)
SHARED_SCORER = "dataprep_staging/n2_policy/bootstrap_scorer.py"
if str(REPO / "dataprep_staging" / "n2_policy") not in sys.path:
    sys.path.insert(0, str(REPO / "dataprep_staging" / "n2_policy"))
import bootstrap_scorer as BS  # noqa: E402

# ----------------------------------------------------------------------------- identity
ID = "N3.piles_prospective.v1"
WP = "N3"
PH_ID = "N3.posthoc_pile_conversion.v1"           # rank 12, after-the-fact declaration
TEST_CONV = "N3P.conversion_12m"
TEST_LEVEL = "N3P.level_12m"
TEST_CONV24 = "N3P.conversion_24m"
TEST_LEVEL24 = "N3P.level_24m"
TEST_XZ = "N3P.xizhou_report_only"
UNIT = "N3:piles_printed_prospective_2026-12_2028-11"
UNIT_XZ = "N3:xizhou_pile_prospective_2026-12_2027-11"
WINDOW = ("2026-12", "2027-11")
WINDOW24 = ("2026-12", "2028-11")
REF_MONTH = "2026-11"                 # first head month (dD of 2026-12) and level reference
SCOPE12 = "deep_piles_printed:2026-12..2027-11"
SCOPE24 = "deep_piles_printed:2026-12..2028-11"
# the conversion tests also read nest heads (dD of 2026-12 back to lag 2 needs h(2026-09));
# the head months outside N2's / N2C's fresh units (2026-11..2027-10) are N3's own fresh units
SCOPE_CONV12 = "deep_piles_printed+heads:2026-09..2027-11"
SCOPE_CONV24 = "deep_piles_printed+heads:2026-09..2028-11"
UNIT_HEADS_YL = "N3:nest_heads_yunlin_outside_N2_2026-09_2028-11"
UNIT_HEADS_XZ = "N3:nest_heads_xizhou_outside_N2C_2027-11"
HEADS_WINDOW_YL = "2026-09..2026-10, 2027-11..2028-11"
HEADS_WINDOW_XZ = "2027-11"
UTC = dt.timezone.utc
DEADLINE = dt.datetime(2026, 11, 30, 16, 0, tzinfo=UTC)
NOT_BEFORE = dt.datetime(2028, 1, 15, 0, 0, tzinfo=UTC)
NOT_BEFORE24 = dt.datetime(2029, 1, 15, 0, 0, tzinfo=UTC)
NO_LATER_THAN = dt.datetime(2029, 6, 30, 23, 59, tzinfo=UTC)
NO_LATER_THAN24 = dt.datetime(2030, 6, 30, 23, 59, tzinfo=UTC)

# the tests this one is ordered after (registered ids; their score looks must be logged)
N2_ID = "N2.policy_response.prospective_2026-11.v1"
N2_TEST = "N2P.heads_yunlin_response"
N2C_ID = "N2C.changhua_response.prospective_2026-11.v1"
N2C_TEST = "N2CP.heads_changhua_response"

# ----------------------------------------------------------------------------- frozen inputs
FORWARD = "results/twin_forward/datum_gate_innov.npz"
FORWARD_SHA = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
COLUMN = "results/twin_runs/stage3_datum_gate/coupled_leveling/vep_zonal_leveling.json"
COLUMN_SHA = "d290c59209f758389b7133c3f5b9aadcc71b42ad55f9089d56645fafc547b162"
R1C_FORWARD = "results/twin_forward/datum_gate_ccz.npz"
R1C_COLUMN = "results/twin_runs/stage3_datum_gate_ccz/coupled_leveling/vep_zonal_leveling.json"
WELLS = "results/twin_runs/stage3_datum_gate/stage3_wells.csv"
PILE_SITES = "chou-shui-data/data/subsidence/deep_pile_sites.csv"
RHEOLOGY = "coupled_leveling"          # the page column (rheology axis 0)

# ----------------------------------------------------------------------------- piles
# names and depths from twin.report_extract PILE_ALIAS / FEED_ID and the WRA station list;
# role fixed by the judge's spec (2026-10-29) and the user's ruling (2026-11-01)
PILES = {
    "tuku": {"pile": "土庫國中", "depth_m": 400, "fallback_depth_m": 300,
             "site_id": "BM_TKJS(3)", "fallback_site_id": "BM_TKJS(1)", "role": "primary",
             "county": "Yunlin", "feed_preorigin": "TKJS(1)"},
    "huwei": {"pile": "中科虎尾", "depth_m": 300, "fallback_depth_m": None,
              "site_id": "BM_NTUH(1)", "fallback_site_id": None, "role": "primary",
              "county": "Yunlin", "feed_preorigin": "NTUH(1)"},
    "xizhou": {"pile": "溪州國小", "depth_m": 300, "fallback_depth_m": None,
               "site_id": "BM_SJES", "fallback_site_id": None, "role": "report_only",
               "county": "Changhua", "feed_preorigin": "SJES"},
}
EXCLUDED = {"秀潭國小": "70 m anchor sees only the top of the system (excluded)",
            "灣內國小": "off the model grid", "頂寮安檢所": "off the model grid"}
PRIMARY = ("tuku", "huwei")
NEST_RADIUS_M = {"tuku": 1500.0, "huwei": 1500.0, "xizhou": 3000.0}
# median screen depth per model aquifer, mid zone (twin.viewer_app LAYER_DEPTHS['mid'], m
# below ground, AMP_V2 station metadata); every one lies above both 300 m and 400 m anchors
LAYER_DEPTH_MID = (35.5, 119.8, 214.0, 289.0)
N_LAYERS = 4
# the column is driven by the unweighted mean of the four layer heads (twin.forward:
# drv = heads.mean(dim=0); vep_zonal_leveling.json driver 'flow-model heads')
LAYER_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

# ----------------------------------------------------------------------------- thresholds
RATIO_BAND = 1.5                       # N1/N2 ratio band
LEVEL = 0.90                           # interval level (N2)
BOOT_B = 2000
BOOT_BLOCK = 3                         # months, moving blocks within pile
BOOT_SEED = 20261130
MIN_RMS_DD_M = 0.30                    # model-side informative gate (computed now)
MIN_RMS_INC_CM = 0.10                  # = printed resolution
MIN_MONTHS = 8                         # observed: verified pile month AND nest dD, per pile
MIN_MONTHS24 = 16
PRINT_RES_CM = 0.1
SEASON = (6, 11)                       # Jun..Nov rebound season (溪州 arm)
MIN_SEASON_RISE_M = 0.10               # season head rise below this: season not informative
DISCONTINUITY_MIN_CM = 1.0             # reset rule floor (see discontinuity_threshold)
DISCONTINUITY_K = 5.0


# ----------------------------------------------------------------------------- small helpers
def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _now() -> dt.datetime:
    return dt.datetime.now(UTC)


def months(first: str, last: str) -> list[str]:
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def prev_month(m: str, k: int = 1) -> str:
    return str(pd.Period(m, freq="M") - k)


# ----------------------------------------------------------------------------- nest rule
def layer_fill(present: set[int], depths=LAYER_DEPTH_MID) -> dict[int, list[int]]:
    """For each model layer 1..4, the nest layers whose drawdown change it takes: itself when
    a nest well is screened there, else the screened layer nearest in median screen depth
    (a tie takes both). Empty dict if no layer is screened."""
    present = sorted(int(x) for x in present)
    if not present:
        return {}
    out = {}
    for L in range(1, N_LAYERS + 1):
        if L in present:
            out[L] = [L]
            continue
        d = {p: abs(depths[p - 1] - depths[L - 1]) for p in present}
        best = min(d.values())
        out[L] = [p for p in present if abs(d[p] - best) < 1e-9]
    return out


def nest_dd(heads: dict[str, pd.Series], layers: dict[str, int],
            month_list: list[str]) -> pd.Series:
    """Month-mean drawdown change dD(m) (m, positive = head fell) of a nest: per layer the
    mean over its wells with heads in both m and m-1 of -(h(m) - h(m-1)); layers without a
    nest well take the declared fill (:func:`layer_fill`); dD = the column's layer weights
    applied to the four filled layers. NaN in a month where a needed layer has no change.
    ``heads``: {well id: Series indexed by 'YYYY-MM'}; the same function serves observed
    heads and model heads sampled at the wells' cells and layers."""
    changes = {}
    for w, s in heads.items():
        s = pd.Series(s, dtype=float)
        changes[w] = pd.Series({m: float(s[m]) - float(s[prev_month(m)]) for m in month_list
                                if m in s.index and prev_month(m) in s.index}, dtype=float)
    return nest_dd_from_changes(changes, layers, month_list)


def nest_dd_from_changes(changes: dict[str, pd.Series], layers: dict[str, int],
                         month_list: list[str]) -> pd.Series:
    """:func:`nest_dd` on per-well monthly head CHANGES h(m) - h(m-1) (m), e.g. the printed
    table-3 month changes of the fallback."""
    fill = layer_fill({layers[w] for w in changes})
    out = {}
    for m in month_list:
        per_layer = {}
        for L in sorted({layers[w] for w in changes}):
            ch = [-float(s[m]) for w, s in changes.items()
                  if layers[w] == L and m in s.index and np.isfinite(s[m])]
            if ch:
                per_layer[L] = float(np.mean(ch))
        vals = []
        for L in range(1, N_LAYERS + 1):
            src = [per_layer[q] for q in fill.get(L, []) if q in per_layer]
            vals.append(float(np.mean(src)) if src and len(src) == len(fill[L]) else np.nan)
        v = np.asarray(vals)
        out[m] = float(np.dot(LAYER_WEIGHTS, v)) if np.isfinite(v).all() else np.nan
    return pd.Series(out, dtype=float)


# ----------------------------------------------------------------------------- estimators
def _demean(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=float).copy()
    for g in np.unique(groups):
        s = groups == g
        out[s] -= out[s].mean()
    return out


def fe_slope(y, x, groups) -> float:
    """OLS slope of y on x with a fixed effect per group (both demeaned within group)."""
    y, x, g = np.asarray(y, float), np.asarray(x, float), np.asarray(groups)
    xd, yd = _demean(x, g), _demean(y, g)
    den = float(xd @ xd)
    return float(xd @ yd / den) if den > 0 else float("nan")


def fe_multi(y, X, groups) -> np.ndarray:
    """OLS coefficients of y on the columns of X with a fixed effect per group."""
    y, X, g = np.asarray(y, float), np.asarray(X, float), np.asarray(groups)
    Xd = np.column_stack([_demean(X[:, j], g) for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(Xd, _demean(y, g), rcond=None)
    return beta


def block_indices(n: int, B: int, block: int, rng) -> np.ndarray:
    """(B, n) moving-block resample indices of one series of length n (blocks of ``block``
    consecutive positions, starts uniform on 0..n-block, truncated to n)."""
    if n <= block:
        return rng.integers(0, n, size=(B, n))
    k = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(B, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, k * block)
    return idx[:, :n]


def boot_fe_slopes(y, x, groups, B: int = BOOT_B, block: int = BOOT_BLOCK,
                   seed: int = BOOT_SEED, rng=None) -> np.ndarray:
    """Moving-block bootstrap (within group, in time order) of :func:`fe_slope`; (B,)."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    y, x, g = np.asarray(y, float), np.asarray(x, float), np.asarray(groups)
    num = np.zeros(B)
    den = np.zeros(B)
    for grp in pd.unique(g):
        s = np.flatnonzero(g == grp)
        idx = s[block_indices(len(s), B, block, rng)]
        xs, ys = x[idx], y[idx]
        xd = xs - xs.mean(axis=1, keepdims=True)
        yd = ys - ys.mean(axis=1, keepdims=True)
        num += (xd * yd).sum(axis=1)
        den += (xd * xd).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def conversion_estimate(y, x, groups, level: float = LEVEL, **kw) -> dict:
    """g_obs with its central ``level`` percentile block-bootstrap interval."""
    g = fe_slope(y, x, groups)
    bs = boot_fe_slopes(y, x, groups, **kw)
    bs = bs[np.isfinite(bs)]
    a = (1.0 - level) / 2.0
    lo, hi = (np.quantile(bs, [a, 1.0 - a]).tolist() if bs.size else [np.nan, np.nan])
    return {"g": g, "lo": float(lo), "hi": float(hi), "n": int(len(np.asarray(y))),
            "n_boot_finite": int(bs.size)}


RMSE_CLAUSE = ("not applicable: N3 has no PR/ZR pair, so N2's 'RMSE(PR) < RMSE(ZR)' clause is "
               "passed to the shared rule as satisfied (rmse_pr 0 < rmse_zr 1)")


def conversion_verdict(g_obs: float, lo: float, hi: float, g_mod: float,
                       band: float = RATIO_BAND) -> dict:
    """N2's decision rule on the ratio rho = g_obs / g_mod (g_mod a frozen number), called
    from the shared implementation (:func:`bootstrap_scorer.decide`, ratio branch; the RMSE
    clause is not applicable, :data:`RMSE_CLAUSE`): 'not_supported' if a quantity is not
    finite, the rho interval includes 0 or rho <= 0; 'conversion_supported' (N2's
    'response_supported') if rho lies in [1/band, band]; otherwise
    'direction_supported_size_not'. 'not_informative' only when g_mod itself is unusable
    (non-finite or 0: a model-side failure, before the shared rule)."""
    if not np.isfinite(g_mod) or g_mod == 0:
        return {"verdict": "not_informative", "rho": None, "rho_ci": None,
                "why": "g_mod non-finite or 0 (model side)"}
    rho = g_obs / g_mod if np.isfinite(g_obs) else float("nan")
    fin = bool(np.isfinite(lo) and np.isfinite(hi))
    r_lo, r_hi = sorted((lo / g_mod, hi / g_mod)) if fin else (float("nan"), float("nan"))
    v = BS.decide(True, rho, r_lo, r_hi, 0.0, 1.0, band=band)
    v = "conversion_supported" if v == "response_supported" else v
    return {"verdict": v, "rho": float(rho) if np.isfinite(rho) else None,
            "rho_ci": [float(r_lo), float(r_hi)] if fin else None,
            "rule": f"{SHARED_SCORER}:decide (ratio branch)"}


def level_verdict(obs: dict, model: dict, leveling: dict, piles=PRIMARY) -> dict:
    """Level rule: 'level_supported' only if at EVERY primary pile |M - obs| < |0 - obs| and
    |M - obs| < |L - obs| (L = the pre-origin leveling-cell rate times the window length);
    'not_evaluable' if a primary pile lacks its window value or its leveling competitor."""
    rows, missing = {}, []
    for p in piles:
        o, m, lv = obs.get(p), model.get(p), leveling.get(p)
        if o is None or m is None or lv is None or not all(np.isfinite([o, m, lv])):
            missing.append(p)
            continue
        e_m, e_0, e_l = abs(m - o), abs(o), abs(lv - o)
        rows[p] = {"obs_cm": o, "model_cm": m, "leveling_cm": lv, "abs_err_model": e_m,
                   "abs_err_zero": e_0, "abs_err_leveling": e_l,
                   "beats_both": bool(e_m < e_0 and e_m < e_l)}
    if missing:
        return {"verdict": "not_evaluable", "missing": missing, "piles": rows}
    ok = all(r["beats_both"] for r in rows.values())
    return {"verdict": "level_supported" if ok else "level_not_supported", "piles": rows}


def t_ppf(p: float, df: float) -> float:
    from scipy import stats

    return float(stats.t.ppf(p, df))


def discontinuity_threshold(model_inc_cm) -> float:
    """Reset rule threshold (cm): a printed monthly |value| above max(1.0 cm, 5 x the frozen
    model's largest |monthly increment| at the cell over the window) is a discontinuity.
    Model-only, fixed before any window value exists."""
    a = np.abs(np.asarray(model_inc_cm, float))
    a = a[np.isfinite(a)]
    return float(max(DISCONTINUITY_MIN_CM, DISCONTINUITY_K * (a.max() if a.size else 0.0)))


def season_coefficients(inc_cm: pd.Series, dd_m: pd.Series, years, *,
                        threshold_cm: float, depth_ok: pd.Series | None = None) -> dict:
    """溪州 arm (and rank 12): per Jun-Nov season, rebound (cm, positive = uplift = minus the
    summed compression of Jun..Nov) divided by the season head rise h(Nov) - h(May) (m) =
    minus the summed nest dD of Jun..Nov (:func:`nest_dd`, so a well missing a month cannot
    shift a level). A season is dropped (no bridging) if any Jun..Nov month lacks a verified
    value, a month exceeds ``threshold_cm`` in absolute value (reset/discontinuity rule), a
    printed depth differs from the declared one (``depth_ok`` False), a month's dD is
    missing, or the rise is below MIN_SEASON_RISE_M. Pooled: mean of the season
    coefficients with a t-interval, df = n - 1 (no interval for n < 2)."""
    out = []
    for y in years:
        ms = [f"{y}-{m:02d}" for m in range(SEASON[0], SEASON[1] + 1)]
        rec = {"season": int(y)}
        vals = [inc_cm.get(m, np.nan) for m in ms]
        if not np.isfinite(vals).all():
            rec["dropped"] = "a Jun-Nov month is missing or not verified"
        elif np.max(np.abs(vals)) > threshold_cm:
            rec["dropped"] = f"discontinuity (|monthly| > {threshold_cm:.2f} cm)"
        elif depth_ok is not None and not all(bool(depth_ok.get(m, False)) for m in ms):
            rec["dropped"] = "printed depth differs from the declared sensor"
        else:
            rise = -float(np.sum([dd_m.get(m, np.nan) for m in ms]))
            if not np.isfinite(rise):
                rec["dropped"] = "a season month has no nest dD"
            elif rise < MIN_SEASON_RISE_M:
                rec.update({"rise_m": rise, "dropped": "head rise below "
                                                       f"{MIN_SEASON_RISE_M} m"})
            else:
                reb = -float(np.sum(vals))
                rec.update({"rebound_cm": reb, "rise_m": rise, "k_cm_per_m": reb / rise})
        out.append(rec)
    ks = np.array([r["k_cm_per_m"] for r in out if "k_cm_per_m" in r])
    pooled = {"n": int(ks.size), "k": float(ks.mean()) if ks.size else None, "ci90": None}
    if ks.size >= 2:
        se = float(ks.std(ddof=1) / math.sqrt(ks.size))
        q = t_ppf(0.95, ks.size - 1)
        pooled["ci90"] = [pooled["k"] - q * se, pooled["k"] + q * se]
    return {"seasons": out, "pooled": pooled}


def report_only_lines(inc: np.ndarray, dd: np.ndarray, dd1: np.ndarray, dd2: np.ndarray,
                      groups: np.ndarray, model_inc: np.ndarray, g_mod: float) -> dict:
    """Declared report-only lines of the conversion test: distributed lag 0-2 (sum of the
    three FE coefficients, rows with all three lags), slope split by the sign of dD, and the
    seasonal compaction-calibration slope (per-pile-demeaned observed increments on the
    frozen ZR model increments: head path x conversion jointly)."""
    out = {}
    ok = np.isfinite(dd1) & np.isfinite(dd2) & np.isfinite(dd) & np.isfinite(inc)
    if ok.sum() >= 6:
        b = fe_multi(inc[ok], np.column_stack([dd[ok], dd1[ok], dd2[ok]]), groups[ok])
        out["lag0_2"] = {"coef": b.tolist(), "sum": float(b.sum()),
                         "sum_over_g_mod": float(b.sum() / g_mod), "n": int(ok.sum())}
    ok = np.isfinite(dd) & np.isfinite(inc)
    if ok.sum() >= 4:
        pos = np.where(dd > 0, dd, 0.0)
        neg = np.where(dd <= 0, dd, 0.0)
        b = fe_multi(inc[ok], np.column_stack([pos[ok], neg[ok]]), groups[ok])
        out["sign_split"] = {"falling_head": float(b[0]), "rising_head": float(b[1]),
                             "n_falling": int((dd[ok] > 0).sum()),
                             "n_rising": int((dd[ok] <= 0).sum())}
    ok = np.isfinite(model_inc) & np.isfinite(inc)
    if ok.sum() >= 4:
        out["calibration_slope_obs_on_zr"] = {
            "slope": fe_slope(inc[ok], model_inc[ok], groups[ok]), "n": int(ok.sum()),
            "label": "head path x conversion jointly (report-only; not the conversion)"}
    return out
