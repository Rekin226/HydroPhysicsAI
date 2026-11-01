"""Estimator of X5.gnss_seasonal_column.v1: the annual-harmonic subsidence amplitude of a
monthly series, the pooled ratio r_A = sum A_obs / sum A_mod, its station-block bootstrap
over years, and the declared readings. Pure numpy; no file is read here.

Series are monthly SUBSIDENCE (positive down: GNSS height is negated before it gets here),
keyed by month index m = 12 * year + month - 1.

Per station:

1. one joint least-squares fit over the station's window of a constant, a linear trend and
   annual plus semiannual harmonics; its trend is removed;
2. per calendar year with at least ``MIN_MONTHS`` months, a fit of a constant plus annual and
   semiannual harmonics to the detrended months gives the year's annual coefficients
   (a, b) (cos, sin of 2 pi (month - 0.5) / 12);
3. the station amplitude is the length of the MEAN (a, b) vector over its years (vector
   averaging before the modulus keeps the noise bias of a modulus low).

The model side uses exactly the same months (the observed station-years and months), so
A_mod is the same estimator run noise-free on the model's subsidence at the station cell.
The pooled ratio is r_A = sum_s A_obs(s) / sum_s A_mod(s). The 90 % interval comes from a
two-stage bootstrap: stations resampled with replacement within their family (NGL, WRA),
then years resampled with replacement within each drawn station, obs and model resampled
together; percentile interval (5 %, 95 %). Each replicate also perturbs every drawn
station's observed amplitude by one draw of the declared residual nuisance (systematic error
the station resampling cannot see, declared before any GNSS read; NUISANCE): a regional
annual loading vector of amplitude U(0, 6) mm at a random phase common to the replicate, of
which a station share U(0.3, 1.0) survives the common-mode removal, plus a monument thermal
vector phased to temperature (height maximum in July) of signed amplitude U(-2, 2) mm (the
off-fan reference monuments removed as common mode carry their own thermal term, so the
residual's sign is unknown). The perturbation is MEAN-NEUTRAL: it enters through its
projection on the station's drawn mean annual vector v, A* = max(|v| - u . v/|v|, 0) (the
first-order change of |v - u|), which has zero mean for a random-phase or signed u. The full
modulus |v - u| would inflate every replicate (E|v - u|^2 = |v|^2 + |u|^2) and shift the
interval upward away from the point estimate. The point estimate carries no correction.

Readings (fixed in the registration): CI inside [1/1.5, 1.5] -> 'consistent'; CI wholly
below 1/1.5 -> 'larger' (the model's elastic response is larger than observed); CI wholly
above 1.5 -> 'smaller'; otherwise 'inconclusive'. The reading must be the same at every
observation scale s in SCALES (the GNSS over-read 1.42x of fast sinking), else
'inconclusive'.
"""
from __future__ import annotations

import math

import numpy as np

MIN_MONTHS = 8
MIN_YEARS = 2
BAND = (1.0 / 1.5, 1.5)
SCALES = (1.0, 1.0 / 1.42)
LEVEL = 0.90
B_SCORE = 2000
SEED = 20261101
READINGS = ("consistent", "larger", "smaller", "inconclusive")
#: readings that clash: a stratum reading in OPPOSITE[primary reading] holds the label at N
OPPOSITE = {"consistent": frozenset({"larger", "smaller"}),
            "larger": frozenset({"consistent", "smaller"}),
            "smaller": frozenset({"consistent", "larger"}),
            "inconclusive": frozenset()}
#: declared residual-nuisance prior propagated into the interval (metres; see module doc)
NUISANCE = {"loading_max_m": 0.006, "survive": (0.3, 1.0), "thermal_max_m": 0.002,
            "thermal_signed": True, "thermal_peak_month": 6, "form": "projection"}


def harmonics(m: np.ndarray, semiannual: bool = True) -> np.ndarray:
    """Columns cos/sin of the annual (and semiannual) cycle at month index ``m``."""
    ang = 2.0 * math.pi * ((np.asarray(m) % 12) + 0.5) / 12.0
    cols = [np.cos(ang), np.sin(ang)]
    if semiannual:
        cols += [np.cos(2 * ang), np.sin(2 * ang)]
    return np.column_stack(cols)


def detrend(m: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y minus the linear trend of the joint fit [1, t, annual, semiannual] (t in years)."""
    m = np.asarray(m)
    t = (m - m.mean()) / 12.0
    X = np.column_stack([np.ones_like(t), t, harmonics(m)])
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - coef[1] * t


def year_coefficients(m: np.ndarray, y: np.ndarray, min_months: int = MIN_MONTHS) -> dict:
    """{year: (a, b)} annual coefficients of each year with >= min_months months, on the
    detrended series."""
    m = np.asarray(m)
    yd = detrend(m, y)
    out = {}
    for yr in np.unique(m // 12):
        sel = (m // 12) == yr
        if int(sel.sum()) < min_months:
            continue
        X = np.column_stack([np.ones(int(sel.sum())), harmonics(m[sel])])
        coef, *_ = np.linalg.lstsq(X, yd[sel], rcond=None)
        out[int(yr)] = (float(coef[1]), float(coef[2]))
    return out


def amplitude(coefs: dict | list) -> float:
    v = np.array(list(coefs.values()) if isinstance(coefs, dict) else coefs, float)
    if v.size == 0:
        return float("nan")
    return float(np.hypot(*v.mean(axis=0)))


def phase_months(coefs: dict) -> float:
    """Month of the annual subsidence maximum (0-12) of the mean coefficient vector."""
    a, b = np.array(list(coefs.values()), float).mean(axis=0)
    ang = math.atan2(b, a) % (2 * math.pi)
    return ang / (2 * math.pi) * 12.0


def station_pairs(obs: dict, mod: dict, min_months: int = MIN_MONTHS,
                  min_years: int = MIN_YEARS) -> dict:
    """obs/mod: {station: (m, y)} with identical months per station (the model series
    restricted to the observed months). Returns {station: {"years": [...], "obs": (Y, 2),
    "mod": (Y, 2)}} for stations with >= min_years usable years."""
    out = {}
    for s, (m, y) in obs.items():
        mm, ym = mod[s]
        if not np.array_equal(np.asarray(m), np.asarray(mm)):
            raise ValueError(f"{s}: model months differ from the observed months")
        co = year_coefficients(m, y, min_months)
        cm = year_coefficients(mm, ym, min_months)
        yrs = sorted(set(co) & set(cm))
        if len(yrs) < min_years:
            continue
        out[s] = {"years": yrs, "obs": np.array([co[k] for k in yrs]),
                  "mod": np.array([cm[k] for k in yrs])}
    return out


def pooled_ratio(pairs: dict) -> float:
    ao = sum(amplitude(list(p["obs"])) for p in pairs.values())
    am = sum(amplitude(list(p["mod"])) for p in pairs.values())
    return ao / am if am > 0 else float("nan")


def _draw(rng, pairs: dict, family: dict, B: int):
    """Bootstrap draws shared by obs and model: per family, station picks (B, n_f) and year
    picks (B, n_f, Ymax) with the picks beyond the drawn station's year count masked."""
    st = sorted(pairs)
    out = []
    for f in sorted({family[s] for s in st}):
        g = [s for s in st if family[s] == f]
        ny = np.array([len(pairs[s]["years"]) for s in g])
        ymax = int(ny.max())
        pick = rng.integers(0, len(g), size=(B, len(g)))
        u = rng.random(size=(B, len(g), ymax))
        n_drawn = ny[pick]                                    # (B, n_f)
        k = np.minimum((u * n_drawn[..., None]).astype(int), n_drawn[..., None] - 1)
        valid = np.arange(ymax)[None, None, :] < n_drawn[..., None]
        out.append((g, pick, k, valid, ymax))
    return out


def _resampled_sum(draws, coefs: dict, shifts=None) -> np.ndarray:
    """Sum over drawn stations of the amplitude of the mean of the drawn years' (a, b) ->
    (B,). ``shifts`` is one (B, n_f, 2) nuisance array per family, or None; a shift u moves
    the amplitude |v| by its projection only, max(|v| - u . v/|v|, 0) (mean-neutral)."""
    tot = 0.0
    for j, (g, pick, k, valid, ymax) in enumerate(draws):
        pad = np.zeros((len(g), ymax, 2))
        for i, s in enumerate(g):
            pad[i, :len(coefs[s])] = coefs[s]
        v = pad[pick[..., None], k]                           # (B, n_f, ymax, 2)
        v = np.where(valid[..., None], v, 0.0)
        mean = v.sum(axis=2) / valid.sum(axis=2)[..., None]
        amp = np.hypot(mean[..., 0], mean[..., 1])
        if shifts is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                proj = np.where(amp > 0, (shifts[j] * mean).sum(axis=-1) / amp, 0.0)
            amp = np.maximum(amp - proj, 0.0)
        tot = tot + amp.sum(axis=1)
    return tot


def nuisance_shifts(rng, draws, B: int, nuisance: dict | None = NUISANCE):
    """One (B, n_f, 2) nuisance vector array per family: common loading phase and amplitude
    per replicate, station survival share, temperature-phased thermal term of signed
    amplitude U(-thermal_max, thermal_max) (mean zero)."""
    if not nuisance:
        return None
    amp = rng.uniform(0.0, nuisance["loading_max_m"], size=B)
    ph = rng.uniform(0.0, 2 * math.pi, size=B)
    tamp = rng.uniform(-nuisance["thermal_max_m"], nuisance["thermal_max_m"], size=B)
    tpk = 2 * math.pi * (nuisance["thermal_peak_month"] + 0.5) / 12.0
    out = []
    for g, *_ in draws:
        surv = rng.uniform(*nuisance["survive"], size=(B, len(g)))
        lo = (amp[:, None] * surv)[..., None] * np.stack([np.cos(ph), np.sin(ph)],
                                                         axis=-1)[:, None, :]
        th = -tamp[:, None, None] * np.array([math.cos(tpk), math.sin(tpk)])[None, None, :]
        out.append(lo + th)
    return out


def bootstrap(pairs: dict, family: dict, B: int = B_SCORE, seed: int = SEED,
              level: float = LEVEL, nuisance: dict | None = NUISANCE) -> dict:
    """Two-stage station-block bootstrap over years (see module doc) of r_A, with the
    declared mean-neutral nuisance draw applied to the observed side of every replicate."""
    rng = np.random.default_rng(seed)
    draws = _draw(rng, pairs, family, B)
    sh = nuisance_shifts(rng, draws, B, nuisance)
    so = _resampled_sum(draws, {s: p["obs"] for s, p in pairs.items()}, sh)
    sm = _resampled_sum(draws, {s: p["mod"] for s, p in pairs.items()})
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(sm > 0, so / sm, np.nan)
    a = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(r, [a, 1.0 - a])
    return {"lo": float(lo), "hi": float(hi), "B": int(B), "seed": int(seed),
            "level": level, "nuisance": nuisance, "n_stations": len(pairs),
            "n_station_years": int(sum(len(p["years"]) for p in pairs.values()))}


def reading(lo: float, hi: float, band=BAND) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if lo >= band[0] and hi <= band[1]:
        return "consistent"
    if hi < band[0]:
        return "larger"
    if lo > band[1]:
        return "smaller"
    return "inconclusive"


def scaled_reading(lo: float, hi: float, scales=SCALES, band=BAND) -> dict:
    """The reading at each observation scale s (r and its CI scale by s); the verdict is the
    common reading, else 'inconclusive'."""
    per = {f"{s:.4f}": reading(s * lo, s * hi, band) for s in scales}
    vals = set(per.values())
    return {"per_scale": per, "verdict": vals.pop() if len(vals) == 1 else "inconclusive"}


def score_stratum(pairs: dict, family: dict, B: int = B_SCORE, seed: int = SEED,
                  nuisance: dict | None = NUISANCE) -> dict:
    if not pairs:
        return {"evaluable": False, "verdict": "inconclusive", "reason": "no station"}
    r = pooled_ratio(pairs)
    ci = bootstrap(pairs, family, B=B, seed=seed, nuisance=nuisance)
    sr = scaled_reading(ci["lo"], ci["hi"])
    return {"evaluable": True, "r_A": r, "ci90": [ci["lo"], ci["hi"]], **ci, **sr}


def slope_through_origin(obs: dict, mod: dict) -> float:
    """Report-only: obs-on-model slope of the detrended monthly anomalies (step C's form)."""
    num = den = 0.0
    for s, (m, y) in obs.items():
        yo = detrend(m, y)
        ym = detrend(*mod[s])
        yo, ym = yo - yo.mean(), ym - ym.mean()
        num += float(yo @ ym)
        den += float(ym @ ym)
    return num / den if den > 0 else float("nan")


# --------------------------------------------------------------------------- fast path
def prepare(m: np.ndarray, min_months: int = MIN_MONTHS) -> dict:
    """Linear projections of :func:`year_coefficients` for fixed months ``m``: applying
    :func:`coefficients` to any series on these months gives the same (a, b) per year (the
    estimator is linear in y). Used by the synthetic operating characteristics."""
    m = np.asarray(m)
    t = (m - m.mean()) / 12.0
    X = np.column_stack([np.ones_like(t), t, harmonics(m)])
    p_trend = np.linalg.pinv(X)[1]                            # trend coefficient row
    years, sels, projs = [], [], []
    for yr in np.unique(m // 12):
        sel = np.nonzero((m // 12) == yr)[0]
        if sel.size < min_months:
            continue
        Xy = np.column_stack([np.ones(sel.size), harmonics(m[sel])])
        years.append(int(yr))
        sels.append(sel)
        projs.append(np.linalg.pinv(Xy)[1:3])                 # (2, k)
    return {"m": m, "t": t, "p_trend": p_trend, "years": years, "sels": sels,
            "projs": projs}


def coefficients(prep: dict, Y: np.ndarray) -> np.ndarray:
    """(..., n) series on ``prep``'s months -> (..., n_years, 2) annual coefficients."""
    Y = np.asarray(Y, float)
    slope = Y @ prep["p_trend"]
    yd = Y - slope[..., None] * prep["t"]
    out = [yd[..., sel] @ P.T for sel, P in zip(prep["sels"], prep["projs"], strict=True)]
    return np.stack(out, axis=-2)
