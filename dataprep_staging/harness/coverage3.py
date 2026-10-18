"""Coverage interval and gate, repaired again: ``harness.coverage3`` (rule ``gate-3``).

WP A1, second repair round (2026-10-12). Use this module for every coverage verdict from now
on. ``harness.coverage`` (frozen interval) and ``harness.coverage2`` (rule ``gate-2``) stay
importable and unchanged because published products and pre-registrations hash them.

Why (A1 verifier, second round): the frozen interval (cluster-adjusted Wilson on
``n_eff_hat = n / (1 + (m_w - 1) rho_hat)``, ANOVA ICC truncated to [0, 1], Student-t on
k - 1 df, level 0.95) met the A1 acceptance (at least 0.89 actual coverage of a true 0.90) in
the 20 pre-registered configurations, which cap the ICC at 0.3 for 10+ clusters (one at 0.4).
With few clusters and a higher ICC it under-covers: 14 sites x 13 months at ICC 0.5 0.876 and
at 0.7 0.834; 7 x 12 at ICC 0.6 0.742; 10 x 50 at ICC 0.5 0.844. Two mechanisms: the ANOVA ICC
is biased low with few clusters (and is 0 when every cluster is all hits, so ``n_eff_hat = n``
exactly when the data carry no information on clustering), and heavy-tailed cluster coverage
(most clusters all hits, a few poor) is often not sampled at all.

This module offers the candidate intervals compared by the design-time sweep
``harness.coverage3_design`` (seeds ``20291012 + ...``) and the rule chosen there
(``DEFAULT_METHOD``), which ``harness.coverage3_confirm`` re-tests on fresh seeds under the
pre-registration ``A1r2.coverage3_confirm.v1``. All candidates use the Wilson score interval on
an effective sample size with the Student-t quantile on k - 1 df, except ``kg_cp``:

``anova``      the frozen interval (``harness.coverage``), for comparison.
``anova_z0``   as ``anova``, but rho = 1 (full clustering) when there is no within-cluster
               variance (MSW = 0: every cluster all hits or all misses) and k >= 2, n > k.
``median``     rho = the F-based median-unbiased ICC: theta = F / F_{0.5}(k-1, n-k),
               rho = (theta - 1) / (theta + m0 - 1), clipped to [0, 1]; rho = 1 when MSW = 0.
``upper75``    rho = the one-sided 0.75 upper confidence bound of the ICC (same construction
               with F_{0.25}); rho = 1 when MSW = 0.
``upper90``    the same with the one-sided 0.90 bound.
``cr``         design-based (linearisation / CR1) variance of the ratio estimator,
               v = k / (k - 1) * sum((y_i - p m_i)^2) / n^2, n_eff = p (1 - p) / v capped at n;
               n_eff = n / m_w (full clustering) when p is 0 or 1.
``kg_cp``      Korn-Graubard: the ``cr`` effective size, degrees-of-freedom adjusted by
               (z / t_{k-1})^2, in a Clopper-Pearson interval.

Rule ``gate-3`` = rule ``gate-2`` (``harness.coverage2``: the 0.85-0.95 point window applies
only when the one-sided 0.95 lower bound of n_eff is at least 30) with the interval of
``DEFAULT_METHOD``. With every verdict report ``coverage``, ``n_eff``, ``n_eff_lower``,
``k_clusters``, ``rho`` and the interval, plus ``calibrated``: False when the stratum lies
outside the envelope in which the interval was shown to reach 0.89 (``ENVELOPE``).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from harness.coverage import (  # noqa: F401 - re-exported
    GATE_LEVEL,
    N_EFF_WINDOW_MIN,
    POINT_WINDOW,
    TARGET,
    cluster_totals,
    icc_anova_batch,
    wilson_batch,
)
from harness.coverage2 import WINDOW_CONF, icc_upper_batch

RULE = "gate-3"
METHODS = ("anova", "anova_z0", "median", "upper75", "upper90", "cr", "kg_cp")
#: chosen by the design-time sweep ``harness.coverage3_design`` (seeds 20291012+, 328
#: configurations) under the rule declared in that script before it ran: no candidate met 0.89
#: in every configuration with ICC <= 0.7 (frozen ``anova``: 46 below, minimum 0.538);
#: ``upper90`` met it in the most (4 below: k = 10 and 14 clusters of 50 rows at ICC 0.7,
#: 0.885 / 0.888, and the gap-fill pilot's two 9-gap N structures at ICC 0.7), so it is the
#: default, and the envelope is k >= 20 clusters (results/dataprep/reports/A1_evidence/
#: repair2/coverage3_design.json). Re-tested on fresh seeds by A1r2.coverage3_confirm.v1.
DEFAULT_METHOD = "upper90"
#: strata with fewer clusters than ``min_k`` are reported with ``calibrated = False``; the ICC
#: bound is the simulated range (the data cannot verify it)
ENVELOPE = {"min_k": 20, "max_rho_simulated": 0.7,
            "source": "harness.coverage3_design decision rule (design-time, seeds 20291012+)"}
_TINY = 1e-12


def _anova_parts(Y: np.ndarray, m: np.ndarray):
    k, N = m.size, m.sum()
    tot = Y.sum(1)
    s2 = (Y ** 2 / m).sum(1)
    msb = (s2 - tot ** 2 / N) / (k - 1)
    msw = (tot - s2) / (N - k)
    m0 = (N - (m ** 2).sum() / N) / (k - 1)
    return msb, msw, m0


def icc_quantile_batch(Y, m, q: float) -> np.ndarray:
    """ICC from the F statistic scaled by its ``q`` quantile under H0-free pivot:
    theta = F / F_q(k-1, n-k), rho = (theta - 1) / (theta + m0 - 1), clipped to [0, 1];
    rho = 1 without within-cluster variance (MSW = 0) or with one cluster; rho = 0 for
    singleton clusters. ``q = 0.5`` is the median-unbiased estimate, ``q = 1 - c`` the
    one-sided ``c`` upper bound (``harness.coverage2.icc_upper_batch`` with conf = c)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m = np.asarray(m, dtype=float)
    k, N = m.size, m.sum()
    if k == 1:
        return np.ones(Y.shape[0])
    if k == N:
        return np.zeros(Y.shape[0])
    msb, msw, m0 = _anova_parts(Y, m)
    fq = stats.f.ppf(q, k - 1, N - k)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.where(msw > _TINY, (msb / np.where(msw > _TINY, msw, 1.0)) / fq, np.inf)
        rho = np.where(np.isfinite(theta), (theta - 1.0) / (theta + m0 - 1.0), 1.0)
    return np.clip(rho, 0.0, 1.0)


def _t_quantile(k: int, level: float) -> float:
    if k >= 2:
        return float(stats.t.ppf(0.5 + level / 2, k - 1))
    return float(stats.norm.ppf(0.5 + level / 2))


def n_eff_batch(Y, m, method: str) -> tuple[np.ndarray, np.ndarray]:
    """(n_eff, rho) per draw for ``method`` (rho is NaN for the design-based methods)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m = np.asarray(m, dtype=float)
    k, N = m.size, m.sum()
    m_w = (m ** 2).sum() / N
    if method in ("anova", "anova_z0", "median", "upper75", "upper90"):
        if method == "anova":
            rho = icc_anova_batch(Y, m)
        elif method == "anova_z0":
            rho = icc_anova_batch(Y, m)
            if 2 <= k < N:
                _, msw, _ = _anova_parts(Y, m)
                rho = np.where(msw > _TINY, rho, 1.0)
        elif method == "median":
            rho = icc_quantile_batch(Y, m, 0.5)
        elif method == "upper75":
            rho = icc_upper_batch(Y, m, 0.75)
        else:
            rho = icc_upper_batch(Y, m, 0.90)
        return N / (1.0 + (m_w - 1.0) * rho), rho
    if method in ("cr", "kg_cp"):
        p = Y.sum(1) / N
        if k < 2:
            return np.full(Y.shape[0], N / m_w), np.full(Y.shape[0], np.nan)
        e = Y - p[:, None] * m[None, :]
        v = k / (k - 1.0) * (e ** 2).sum(1) / N ** 2
        pq = p * (1.0 - p)
        with np.errstate(invalid="ignore", divide="ignore"):
            ne = np.where(v > _TINY, pq / np.where(v > _TINY, v, 1.0), N)
        ne = np.where(pq > _TINY, np.minimum(ne, N), N / m_w)
        return ne, np.full(Y.shape[0], np.nan)
    raise ValueError(f"unknown method {method!r}; one of {METHODS}")


def interval_batch(Y, m, method: str | None = None, level: float = GATE_LEVEL) -> dict:
    """Vectorised interval of ``method`` (default ``DEFAULT_METHOD``) from cluster totals
    ``Y`` (D, k) and sizes ``m`` (k,). Returns p, n_eff, rho, lo, hi, n, k, m_w, quantile."""
    method = method or DEFAULT_METHOD
    if method is None:
        raise ValueError("no method given and DEFAULT_METHOD is not set")
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m = np.asarray(m, dtype=float)
    k, N = m.size, float(m.sum())
    p = Y.sum(1) / N
    ne, rho = n_eff_batch(Y, m, method)
    q = _t_quantile(k, level)
    if method == "kg_cp":
        z = float(stats.norm.ppf(0.5 + level / 2))
        ne_df = np.minimum(N, ne * (z / q) ** 2)
        x = p * ne_df
        a = (1.0 - level) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            lo = np.where(x > _TINY, stats.beta.ppf(a, np.maximum(x, _TINY),
                                                    ne_df - x + 1.0), 0.0)
            hi = np.where(ne_df - x > _TINY, stats.beta.ppf(1 - a, x + 1.0,
                                                            np.maximum(ne_df - x, _TINY)), 1.0)
        lo, hi = np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
    else:
        lo, hi = wilson_batch(p, ne, q)
    return {"p": p, "n_eff": ne, "rho": rho, "lo": lo, "hi": hi, "n": N, "k": k,
            "m_w": float((m ** 2).sum() / N), "quantile": q, "method": method}


def in_envelope(k: int, method: str | None = None) -> bool:
    """True when a stratum with ``k`` clusters lies inside the envelope where the default
    interval was shown to reach the criterion (``ENVELOPE``: k >= 20 clusters). Any other
    method is never calibrated by this module."""
    if (method or DEFAULT_METHOD) != DEFAULT_METHOD:
        return False
    return k >= ENVELOPE["min_k"]


def gate_from_totals_batch(Y, m, method: str | None = None, target: float = TARGET,
                           level: float = GATE_LEVEL, window=POINT_WINDOW,
                           n_eff_window_min: float = N_EFF_WINDOW_MIN,
                           conf: float = WINDOW_CONF) -> dict:
    """Vectorised rule ``gate-3``: the interval of ``method`` must contain ``target``; the
    point window applies when the one-sided ``conf`` lower bound of n_eff (gate-2) is at least
    ``n_eff_window_min``."""
    r = interval_batch(Y, m, method, level)
    m = np.asarray(m, dtype=float)
    rho_u = icc_upper_batch(Y, m, conf)
    n_eff_lo = r["n"] / (1.0 + (r["m_w"] - 1.0) * rho_u)
    in_ci = (r["lo"] <= target) & (r["hi"] >= target)
    applies = n_eff_lo >= n_eff_window_min
    in_win = (r["p"] >= window[0]) & (r["p"] <= window[1])
    r.update({"rho_upper": rho_u, "n_eff_lower": n_eff_lo, "in_ci": in_ci,
              "window_applies": applies, "in_window": in_win,
              "pass": in_ci & (in_win | ~applies)})
    return r


def coverage_gate(hits, clusters, method: str | None = None, target: float = TARGET,
                  level: float = GATE_LEVEL, window=POINT_WINDOW,
                  n_eff_window_min: float = N_EFF_WINDOW_MIN, conf: float = WINDOW_CONF
                  ) -> dict:
    """Plan 4.4 rule for one stratum under ``gate-3`` (see the module docstring)."""
    y, m, _ = cluster_totals(hits, clusters)
    g = gate_from_totals_batch(y[None, :], m, method, target, level, window,
                               n_eff_window_min, conf)
    rho_anova = float(icc_anova_batch(y[None, :], m)[0])
    k = int(m.size)
    out = {"coverage": float(g["p"][0]), "n": int(g["n"]), "k_clusters": k,
           "method": g["method"], "rule": RULE, "level": level, "quantile": g["quantile"],
           "n_eff": float(g["n_eff"][0]), "rho_method": float(g["rho"][0]),
           "rho_anova": rho_anova, "lo": float(g["lo"][0]), "hi": float(g["hi"][0]),
           "rho_upper": float(g["rho_upper"][0]), "n_eff_lower": float(g["n_eff_lower"][0]),
           "window_conf": conf, "target": target, "contains_target": bool(g["in_ci"][0]),
           "window_applies": bool(g["window_applies"][0]),
           "in_window": bool(g["in_window"][0]),
           "needs_pooling": not bool(g["window_applies"][0]),
           "pass": bool(g["pass"][0]),
           "calibrated": in_envelope(k, g["method"])}
    if not out["calibrated"]:
        out["calibration_note"] = (
            f"uncalibrated: {k} clusters < {ENVELOPE['min_k']} (A1 coverage3 envelope); pool "
            f"with neighbours first (plan 4.4) or report the verdict as uncalibrated")
    return out
