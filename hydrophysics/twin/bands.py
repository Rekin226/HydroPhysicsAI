"""Signed Mondrian conformal bands from residual tables (uncertainty plan, step E1).

The page's ranges so far are member spreads, and the harness scorer
(``dataprep_staging/harness/scorers.conformal_split``) scores ``|err|``: a band symmetric
about the prediction. Neither can absorb a systematic bias such as the distal next-year
leveling offset, and neither says what happens when a stratum has too few exchangeable
units. This module builds the band from a residual table instead, with four choices made
explicit:

- **Signed.** The residual is ``observed - predicted`` (note: the repository's error terms,
  e.g. ``score_x5nh`` bias and ``twin/coverage.py``, are ``model - observed``; negate them
  before passing them here). The lower and upper offsets are
  separate finite-sample split-conformal quantiles at ``alpha/2`` each, so the band for a
  new prediction ``p`` is ``[p + lo, p + hi]`` and a constant bias moves both ends. Under
  exchangeability each tail misses with probability at most ``alpha/2``.
- **Mondrian.** Bands are calibrated per stratum (zone x lead bucket, county x layer group
  x lead bucket, ...): the guarantee then holds within each stratum, not only on average.
  Coarser strata may be declared as a fallback, tried in the declared order (never chosen
  after seeing the result).
- **Cluster-aware.** Residuals that share a cluster (a forecast origin, whose year shock is
  common to every township and well) are not exchangeable rows. With ``cluster`` given,
  each cluster carries total weight one (the pooled within-cluster CDFs of Dunn, Wasserman
  and Ramdas 2022), the finite-sample rank is taken on the number of clusters ``k``, and
  the band targets a row of a *new* cluster, which is what a forecast from a new origin
  needs. Without ``cluster``, rows are the units (only honest for exchangeable rows).
- **Not available.** A signed band at central level ``1 - alpha`` needs at least
  ``ceil(2 / alpha) - 1`` units (19 at 0.90): below that the upper rank exceeds the sample
  and the finite-sample quantile is infinite. Such a stratum gets a declared
  ``available=False`` band with NaN offsets and a reason, never a silently narrow one. With
  clusters = origins (at most six rolling origins), a 0.90 band is therefore not available
  by construction; that is the honest answer, not a defect to tune away.

The coverage evaluator (:func:`coverage_report`) mirrors the INTERVAL, not the gate, of the
registered cluster-adjusted Wilson interval of ``harness/coverage.py`` (ANOVA intra-cluster correlation, design effect,
Student-t on ``k - 1`` df; parity is tested) without importing it, because the harness is
hash-bound and not on the package path. It also reports the signed miss rates (below and
above the band), the share of rows with no band, and a verdict that is ``not_evaluable``
below ``min_clusters`` clusters. Leave-one-cluster-out hits (:func:`loco_hits`) give the
cross-conformal check: a band calibrated on the other origins scored on the held-out one.
With three to six origins and a shared year shock (tests), a row-calibrated band covers a
new origin at 0.5-0.8 while a random row split reports 0.90; the leave-one-origin-out point
coverage exposes the shortfall, the iid interval overstates its certainty (design effect
far above one), and the cluster-adjusted interval is too wide to confirm it: with that few
origins the honest verdict is ``consistent`` only in the sense of uninformative.

The library is model-agnostic: persistence or climatology baseline bands are built by
feeding their residual tables to the same functions. It reads no data; the tests use
synthetic draws only.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LEVEL = 0.90
TARGET = 0.90
GATE_LEVEL = 0.95
POINT_WINDOW = (0.85, 0.95)
#: fewer clusters than this and a coverage rate is "not evaluable" (uncertainty review,
#: must_fix 3: a gate with fewer than 3 origins in a stratum is not a pass)
MIN_CLUSTERS = 3
_EPS = 1e-9


def _rank(units: float, prob: float) -> int:
    """Finite-sample conformal rank ``ceil((units + 1) * prob)`` with a guard against
    float noise (``0.95 * 20`` must give 19, not 20)."""
    return math.ceil((units + 1) * prob - _EPS)


def min_units(level: float = LEVEL, signed: bool = True) -> int:
    """Smallest number of exchangeable units for which the finite-sample band at central
    ``level`` is finite: 19 for a signed 0.90 band (``alpha/2`` per tail), 9 for an
    absolute-score 0.90 band."""
    if not 0 < level < 1:
        raise ValueError(f"level must lie in (0, 1), not {level}")
    prob = 1 - (1 - level) / 2 if signed else level
    n = 1
    while _rank(n, prob) > n:
        n += 1
    return n


def _upper_value(x: np.ndarray, w: np.ndarray, rank: int) -> float:
    """Smallest ``x`` whose cumulative weight (ascending) reaches ``rank``; ``inf`` when
    the total weight falls short. With unit weights this is the ``rank``-th order
    statistic."""
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order])
    if cw[-1] < rank - _EPS:
        return float("inf")
    return float(x[order][np.searchsorted(cw, rank - _EPS)])


def signed_offsets(residuals, level: float = LEVEL, clusters=None) -> tuple[float, float, int]:
    """Signed split-conformal offsets ``(lo, hi, units)`` of ``residuals``
    (``observed - predicted``) at central ``level``.

    ``hi`` is the ``ceil((u + 1)(1 - alpha/2))``-th smallest residual and ``lo`` the
    ``floor((u + 1) alpha/2)``-th smallest, where ``u`` is the number of units: rows, or
    clusters when ``clusters`` is given (each cluster then weighs one, split equally over
    its rows). Non-finite residuals are dropped. Offsets are ``-inf``/``inf`` when ``u`` is
    below :func:`min_units`."""
    r = np.asarray(residuals, dtype=float).ravel()
    keep = np.isfinite(r)
    if clusters is None:
        r = r[keep]
        w = np.ones(r.size)
        units = int(r.size)
    else:
        c = np.asarray(clusters).ravel()
        if c.size != keep.size:
            raise ValueError("residuals and clusters differ in length")
        r, c = r[keep], c[keep]
        _no_missing(c, "cluster")
        codes, uniq = pd.factorize(pd.Series(c))
        m = np.bincount(codes, minlength=len(uniq)).astype(float)
        w = 1.0 / m[codes]
        units = len(uniq)
    if units == 0:
        return float("-inf"), float("inf"), 0
    rank = _rank(units, 1 - (1 - level) / 2)
    hi = _upper_value(r, w, rank)
    lo = -_upper_value(-r, w, rank)
    return lo, hi, units


@dataclass(frozen=True)
class Band:
    """One stratum's band. ``lo``/``hi`` are offsets added to the prediction (NaN when
    ``available`` is False); ``units`` are rows or clusters, ``n_min`` the units needed;
    ``calibrated_on`` names the (possibly coarser, declared) stratum the offsets came from;
    ``median`` is the stratum's median residual (its bias), reported, not used."""

    available: bool
    lo: float
    hi: float
    level: float
    mode: str
    n_rows: int
    units: int
    n_min: int
    median: float
    calibrated_on: str
    reason: str


def calibrate_band(residuals, clusters=None, level: float = LEVEL,
                   calibrated_on: str = "all") -> Band:
    """Signed band from one set of residuals, or a declared not-available band."""
    r = np.asarray(residuals, dtype=float).ravel()
    finite = np.isfinite(r)
    n_min = min_units(level, signed=True)
    mode = "row" if clusters is None else "cluster"
    lo, hi, units = signed_offsets(r, level, clusters)
    med = float(np.median(r[finite])) if finite.any() else float("nan")
    ok = units >= n_min and np.isfinite(lo) and np.isfinite(hi)
    reason = "" if ok else (f"{units} {'clusters' if clusters is not None else 'rows'} "
                            f"< {n_min} needed for a signed {level:.2f} band")
    return Band(available=bool(ok), lo=float(lo) if ok else float("nan"),
                hi=float(hi) if ok else float("nan"), level=level, mode=mode,
                n_rows=int(finite.sum()), units=int(units), n_min=n_min, median=med,
                calibrated_on=calibrated_on, reason=reason)


def _key(row: tuple, cols: Sequence[str]) -> str:
    return "all" if not cols else ",".join(f"{c}={v}" for c, v in zip(cols, row, strict=True))


def _check_columns(table: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise KeyError(f"residual table lacks columns {missing}")


def _no_missing(values, what: str) -> None:
    """A missing cluster or stratum label would silently drop or merge rows: refused."""
    if pd.isna(pd.Series(np.asarray(values, dtype=object))).any():
        raise ValueError(f"{what} labels must not be missing (NaN/None)")


def _strata_bands(table: pd.DataFrame, strata: Sequence[str], residual: str,
                  cluster: str | None, level: float,
                  fallback: Sequence[Sequence[str]]) -> dict[tuple, Band]:
    """Band per fine stratum, trying the declared ``fallback`` strata (coarser column
    lists, in order) when the fine one is not available."""
    hierarchy = [list(strata)] + [list(f) for f in fallback]
    for coarse in hierarchy[1:]:
        if not set(coarse) <= set(strata):
            raise ValueError(f"fallback {coarse} is not a subset of strata {list(strata)}")
    cache: dict[tuple, Band] = {}

    def band_for(cols: list[str], values: tuple) -> Band:
        key = (tuple(cols), values)
        if key not in cache:
            mask = np.ones(len(table), dtype=bool)
            for c, v in zip(cols, values, strict=True):
                mask &= (table[c] == v).to_numpy()
            sub = table.loc[mask]
            cl = sub[cluster].to_numpy() if cluster is not None else None
            cache[key] = calibrate_band(sub[residual].to_numpy(), cl, level,
                                        calibrated_on=_key(values, cols))
        return cache[key]

    fine = (table[list(strata)].drop_duplicates().itertuples(index=False, name=None)
            if strata else [()])
    out = {}
    for values in fine:
        values = tuple(values)
        first = band = band_for(list(strata), values)
        for coarse in hierarchy[1:]:
            if band.available:
                break
            band = band_for(coarse, tuple(values[list(strata).index(c)] for c in coarse))
        if not band.available:
            band = first           # report the fine stratum's own shortfall
        out[values] = band
    return out


def mondrian_bands(table: pd.DataFrame, strata: Sequence[str] = (), *,
                   residual: str = "residual", cluster: str | None = None,
                   level: float = LEVEL,
                   fallback: Sequence[Sequence[str]] = ()) -> pd.DataFrame:
    """Signed band per stratum of a residual table.

    ``table`` has a ``residual`` column (``observed - predicted``), the ``strata`` columns
    and, for cluster-aware calibration, a ``cluster`` column (the forecast origin).
    ``fallback`` lists coarser strata (subsets of ``strata``) to calibrate on, in the
    declared order, when a fine stratum is short of units. Returns one row per fine stratum
    with the strata values and the :class:`Band` fields."""
    _check_columns(table, [residual, *strata] + ([cluster] if cluster else []))
    for col in [*strata] + ([cluster] if cluster else []):
        _no_missing(table[col], col)
    bands = _strata_bands(table, strata, residual, cluster, level, fallback)
    rows = [{**dict(zip(strata, values, strict=True)), **asdict(b)}
            for values, b in bands.items()]
    return pd.DataFrame(rows)


def apply_bands(predictions: pd.DataFrame, bands: pd.DataFrame, strata: Sequence[str] = (),
                *, predicted: str = "predicted") -> pd.DataFrame:
    """Attach ``lower``/``upper``/``band_available`` to ``predictions`` by stratum. A
    stratum with no band, or a not-available band, gets NaN bounds."""
    _check_columns(predictions, [predicted, *strata])
    clash = sorted({"available", "lo", "hi"} & set(predictions.columns))
    if clash:
        raise ValueError(f"predictions already have column(s) {clash}; rename them first")
    cols = list(strata) + ["available", "lo", "hi"]
    if strata:
        merged = predictions.merge(bands[cols], on=list(strata), how="left",
                                   validate="many_to_one")
    else:
        if len(bands) != 1:
            raise ValueError("unstratified bands must have exactly one row")
        merged = predictions.assign(**{k: bands[k].iloc[0] for k in ("available", "lo", "hi")})
    avail = merged["available"].eq(True).to_numpy()   # no band row: NaN, not available
    p = merged[predicted].to_numpy(dtype=float)
    out = predictions.copy()
    out["lower"] = np.where(avail, p + merged["lo"].to_numpy(dtype=float), np.nan)
    out["upper"] = np.where(avail, p + merged["hi"].to_numpy(dtype=float), np.nan)
    out["band_available"] = avail
    return out


def loco_hits(table: pd.DataFrame, strata: Sequence[str] = (), *, cluster: str,
              residual: str = "residual", level: float = LEVEL,
              cluster_aware: bool = True,
              fallback: Sequence[Sequence[str]] = ()) -> pd.DataFrame:
    """Leave-one-cluster-out (leave-one-origin-out) cross-conformal hits.

    For each cluster, bands are calibrated on the rows of every *other* cluster (per
    stratum, with the declared ``fallback``) and the held-out rows are scored. With
    ``cluster_aware=False`` the band treats rows as units (the naive calibration a common
    year shock breaks); the held-out unit is still the cluster. Returns ``table`` with
    ``lo``, ``hi``, ``band_available``, ``calibrated_on`` (the stratum whose band served,
    fine or declared fallback), ``hit`` (1.0/0.0, NaN where no band) and ``side``
    (``"below"``, ``"inside"``, ``"above"`` or ``"none"``)."""
    _check_columns(table, [residual, cluster, *strata])
    for col in [cluster, *strata]:
        _no_missing(table[col], col)
    out = table.copy()
    lo = np.full(len(table), np.nan)
    hi = np.full(len(table), np.nan)
    avail = np.zeros(len(table), dtype=bool)
    on = np.full(len(table), "", dtype=object)
    cl = table[cluster].to_numpy()
    strata_vals = (table[list(strata)].itertuples(index=False, name=None) if strata
                   else [()] * len(table))
    strata_vals = [tuple(v) for v in strata_vals]
    for c in pd.unique(cl):
        held = cl == c
        cal = table.loc[~held]
        bands = _strata_bands(cal, strata, residual, cluster if cluster_aware else None,
                              level, fallback)
        for i in np.flatnonzero(held):
            b = bands.get(strata_vals[i])
            if b is None and fallback:
                # stratum unseen in calibration: only a declared coarser stratum can serve
                b = _fallback_unseen(cal, strata, strata_vals[i], residual,
                                     cluster if cluster_aware else None, level, fallback)
            if b is not None and b.available:
                lo[i], hi[i], avail[i], on[i] = b.lo, b.hi, True, b.calibrated_on
    r = table[residual].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        inside = (r >= lo) & (r <= hi)
        below, above = r < lo, r > hi
    out["lo"], out["hi"], out["band_available"] = lo, hi, avail
    out["calibrated_on"] = on        # the stratum (fine or declared fallback) that served
    out["hit"] = np.where(avail & np.isfinite(r), inside.astype(float), np.nan)
    side = np.full(len(table), "none", dtype=object)
    ok = avail & np.isfinite(r)
    side[ok & inside] = "inside"
    side[ok & below] = "below"
    side[ok & above] = "above"
    out["side"] = side
    return out


def _fallback_unseen(cal, strata, values, residual, cluster, level, fallback) -> Band | None:
    for coarse in fallback:
        mask = np.ones(len(cal), dtype=bool)
        for c in coarse:
            mask &= (cal[c] == values[list(strata).index(c)]).to_numpy()
        sub = cal.loc[mask]
        b = calibrate_band(sub[residual].to_numpy(),
                           sub[cluster].to_numpy() if cluster else None, level)
        if b.available:
            return b
    return None


# ----------------------------------------------------------------------------- coverage
def _t_or_z(level: float, k: int, small_k: str) -> float:
    from scipy import stats  # lazy: the band builders need numpy/pandas only

    if small_k == "t" and k >= 2:
        return float(stats.t.ppf(0.5 + level / 2, k - 1))
    if small_k in ("z", "t"):
        return float(stats.norm.ppf(0.5 + level / 2))
    raise ValueError(f"small_k must be 'z' or 't', not {small_k!r}")


def _wilson(p: float, n: float, q: float) -> tuple[float, float]:
    q2 = q ** 2
    den = 1 + q2 / n
    centre = (p + q2 / (2 * n)) / den
    half = q * math.sqrt(max(p * (1 - p) / n + q2 / (4 * n ** 2), 0.0)) / den
    return float(np.clip(centre - half, 0, 1)), float(np.clip(centre + half, 0, 1))


def _icc_anova(y: np.ndarray, m: np.ndarray) -> float:
    """ANOVA intra-cluster correlation of binary hits, as ``harness.coverage``."""
    k, n = m.size, m.sum()
    if k == 1:
        return 1.0
    if k == n:
        return 0.0
    tot = y.sum()
    s2 = (y ** 2 / m).sum()
    msb = (s2 - tot ** 2 / n) / (k - 1)
    msw = (tot - s2) / (n - k)
    m0 = (n - (m ** 2).sum() / n) / (k - 1)
    den = msb + (m0 - 1) * msw
    return float(np.clip((msb - msw) / den, 0.0, 1.0)) if den > 0 else 0.0


def coverage_report(hits, clusters, side=None, *, target: float = TARGET,
                    level: float = GATE_LEVEL, small_k: str = "t",
                    window=POINT_WINDOW, min_clusters: int = MIN_CLUSTERS) -> dict:
    """Coverage of a band with clustered misses, and a verdict.

    ``hits`` is 1/0 per row, NaN where the row had no band (counted in
    ``share_unavailable``, excluded from the rates). ``clusters`` are the exchangeable units
    (origins). Reports the row coverage, the equal-cluster coverage, the naive iid Wilson
    interval (what ignoring the clusters would claim) and the cluster-adjusted one (the
    registered harness rule), plus the signed miss rates when ``side`` is given.

    ``verdict``: ``not_evaluable`` with fewer than ``min_clusters`` clusters or no scored
    row; ``under`` when the cluster-adjusted upper bound is below ``target``; ``over`` when
    its lower bound is above; ``consistent`` otherwise (which, with few clusters, often
    means uninformative). ``point_flag`` compares the point coverage with ``window``, and
    ``iid_verdict`` is the verdict the iid interval would give."""
    h = np.asarray(hits, dtype=float).ravel()
    c = np.asarray(clusters).ravel()
    if h.size != c.size:
        raise ValueError("hits and clusters differ in length")
    scored = np.isfinite(h)
    out: dict = {"n_rows": int(h.size), "n_scored": int(scored.sum()),
                 "share_unavailable": float(1 - scored.mean()) if h.size else float("nan"),
                 "target": target, "level": level, "small_k": small_k}
    if side is not None:
        s = np.asarray(side, dtype=object).ravel()[scored]
        n_s = max(int(scored.sum()), 1)
        out["miss_below"] = float(np.sum(s == "below") / n_s)
        out["miss_above"] = float(np.sum(s == "above") / n_s)
    h, c = h[scored], c[scored]
    _no_missing(c, "cluster")
    codes, uniq = pd.factorize(pd.Series(c))
    k = len(uniq)
    out["k_clusters"] = k
    if k == 0:
        out.update({"coverage": float("nan"), "verdict": "not_evaluable",
                    "iid_verdict": "not_evaluable", "point_flag": "none"})
        return out
    y = np.bincount(codes, weights=h, minlength=k)
    m = np.bincount(codes, minlength=k).astype(float)
    n = m.sum()
    p = float(y.sum() / n)
    rho = _icc_anova(y, m)
    deff = 1 + ((m ** 2).sum() / n - 1) * rho
    n_eff = n / deff
    q_iid = _t_or_z(level, k, "z")
    lo_iid, hi_iid = _wilson(p, n, q_iid)
    lo, hi = _wilson(p, n_eff, _t_or_z(level, k, small_k))

    def verdict(a: float, b: float) -> str:
        if k < min_clusters:
            return "not_evaluable"
        return "under" if b < target else "over" if a > target else "consistent"

    out.update({"coverage": p, "coverage_cluster_mean": float(np.mean(y / m)),
                "rho": rho, "deff": float(deff), "n_eff": float(n_eff),
                "lo": lo, "hi": hi, "lo_iid": lo_iid, "hi_iid": hi_iid,
                "verdict": verdict(lo, hi), "iid_verdict": verdict(lo_iid, hi_iid),
                "point_flag": ("under" if p < window[0] else "over" if p > window[1]
                               else "ok")})
    return out


def interval_censored_hits(lower, upper, obs_lower, obs_upper, tol: float = 0.0) -> dict:
    """Hits of bands ``[lower, upper]`` on interval-censored observations
    ``[obs_lower, obs_upper]`` (for example a township rate printed only as "above the
    listing floor", or not listed so at most the floor): a hit when the two intervals,
    the observed one widened by ``tol``, intersect. Infinite observed ends are allowed.

    A band can barely miss an observation with no finite lower bound (a township not
    listed, so only "below the floor" is known), so ``n_informative`` counts the scored
    rows with a finite ``obs_lower`` (the listed ones). The uncertainty review asks that it
    be reported as the effective sample, and that a check made mostly of unlisted rows be
    described as detecting under-coverage only."""
    lo, hi = np.asarray(lower, float).ravel(), np.asarray(upper, float).ravel()
    ol, ou = np.asarray(obs_lower, float).ravel(), np.asarray(obs_upper, float).ravel()
    if not (lo.size == hi.size == ol.size == ou.size):
        raise ValueError("band and observation bounds differ in length")
    if np.isnan(ol).any() or np.isnan(ou).any():
        raise ValueError("observed bounds must be numbers or +-inf, not NaN")
    if (ol > ou).any():
        raise ValueError("obs_lower exceeds obs_upper")
    avail = np.isfinite(lo) & np.isfinite(hi)
    with np.errstate(invalid="ignore"):
        hit = (lo <= ou + tol) & (hi >= ol - tol)
    hits = np.where(avail, hit.astype(float), np.nan)
    return {"hits": hits, "n_informative": int(np.sum(avail & np.isfinite(ol))),
            "n_scored": int(avail.sum())}
