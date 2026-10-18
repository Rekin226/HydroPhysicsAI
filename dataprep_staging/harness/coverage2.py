"""Coverage gate, repaired rule ``gate-2`` (WP A1 repair round, 2026-10-12).

Use this module for every coverage verdict from now on. ``harness.coverage`` stays frozen
because five products and one pre-registration hash it (surface_water_v4 full/v1, full/v2 and
preorigin/v1; A1.coverage_confirm.v1); its interval is unchanged and is re-used here.

What changed and why (verifier finding, A1 repair round):

* The plan 4.4 rule applies the 0.85-0.95 point window "only for strata with an effective n of
  at least 30". The first harness decided that with the point estimate
  ``n_eff_hat = n / (1 + (m_w - 1) rho_hat)``. With few clusters the ANOVA ICC is truncated to
  0 in about half of the draws, so ``n_eff_hat`` reaches n and the window fired in strata whose
  true n_eff is 9-20 (5 regions x 6 rows at ICC 0.5: 47 % of draws), failing calibrated
  products by chance; plan 4.4 names exactly these strata (W1, S1, E6) as the ones the window
  must not decide.
* Rule ``gate-2`` applies the window only when the data show the stratum is large: when the
  one-sided ``WINDOW_CONF`` (0.95) lower confidence bound of n_eff,
  ``n_eff_lo = n / (1 + (m_w - 1) rho_U)``, is at least 30. ``rho_U`` is the F-based upper
  confidence bound of the one-way ANOVA ICC (Searle; unbalanced clusters through m0):
  ``theta_U = F / F_{1-conf}(k-1, n-k)``, ``rho_U = (theta_U - 1) / (theta_U + m0 - 1)``,
  ``F = MSB / MSW``, clipped to [0, 1]. Without within-cluster variance (MSW = 0: every
  cluster all hits or all misses, or no variance at all) there is no evidence against full
  clustering and ``rho_U = 1``. One cluster: ``rho_U = 1``. Singleton clusters: m_w = 1, so
  ``n_eff_lo = n``.
* Strata whose ``n_eff_lo`` is below 30 are the "smaller strata" of plan 4.4: they are pooled
  with their nearest neighbour first (:func:`pool_small_strata`), and the verdict of a stratum
  that stays small is the interval alone.
* :func:`pool_small_strata` no longer stops pooling when the smallest under-sized stratum has
  no listed neighbour (first harness: ``if target is None: break``); it skips that stratum
  and tries the next one, and reports the strata left under-sized.

The interval itself (cluster-adjusted Wilson on ``n_eff_hat``, ANOVA ICC, Student-t quantile on
k - 1 df, level 0.95) is the frozen ``harness.coverage`` interval, accepted by the
pre-registered A1.coverage_confirm.v1 run. Rule ``gate-2`` was evaluated by the pre-registered
run A1r.gate_window.v1 (``gate_window_confirm.py``).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from harness.coverage import (  # noqa: F401 - re-exported for callers of this module
    DEFAULT_SMALL_K,
    GATE_LEVEL,
    N_EFF_WINDOW_MIN,
    POINT_WINDOW,
    TARGET,
    cluster_coverage_interval,
    cluster_totals,
    icc_anova_batch,
    interval_from_totals_batch,
    wilson_batch,
)

RULE = "gate-2"
#: one-sided confidence of the lower bound of n_eff that decides whether the window applies
WINDOW_CONF = 0.95


def _anova_parts(Y: np.ndarray, m: np.ndarray):
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m = np.asarray(m, dtype=float)
    k, N = m.size, m.sum()
    tot = Y.sum(1)
    s2 = (Y ** 2 / m).sum(1)
    msb = (s2 - tot ** 2 / N) / (k - 1)
    msw = (tot - s2) / (N - k)
    m0 = (N - (m ** 2).sum() / N) / (k - 1)
    return msb, msw, m0, k, N


def icc_upper_batch(Y, m, conf: float = WINDOW_CONF) -> np.ndarray:
    """One-sided ``conf`` upper confidence bound of the ANOVA ICC for a batch of draws
    (``Y`` (D, k) hit counts, ``m`` (k,) sizes); see the module docstring."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m = np.asarray(m, dtype=float)
    k, N = m.size, m.sum()
    if k == 1:
        return np.ones(Y.shape[0])
    if k == N:
        return np.zeros(Y.shape[0])
    msb, msw, m0, k, N = _anova_parts(Y, m)
    f_lo = stats.f.ppf(1.0 - conf, k - 1, N - k)
    tiny = 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.where(msw > tiny, (msb / np.where(msw > tiny, msw, 1.0)) / f_lo, np.inf)
        rho = np.where(np.isfinite(theta), (theta - 1.0) / (theta + m0 - 1.0), 1.0)
    return np.clip(rho, 0.0, 1.0)


def gate_from_totals_batch(Y, m, target: float = TARGET, level: float = GATE_LEVEL,
                           small_k: str = DEFAULT_SMALL_K, window=POINT_WINDOW,
                           n_eff_window_min: float = N_EFF_WINDOW_MIN,
                           conf: float = WINDOW_CONF) -> dict:
    """Vectorised rule ``gate-2`` from cluster totals (simulation and :func:`coverage_gate`)."""
    r = interval_from_totals_batch(Y, m, level, small_k)
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


def coverage_gate(hits, clusters, target: float = TARGET, level: float = GATE_LEVEL,
                  small_k: str = DEFAULT_SMALL_K, window=POINT_WINDOW,
                  n_eff_window_min: float = N_EFF_WINDOW_MIN,
                  conf: float = WINDOW_CONF) -> dict:
    """Plan 4.4 rule for one stratum, rule ``gate-2``: pass if the cluster-adjusted interval
    contains ``target``; the point window applies only when the one-sided ``conf`` lower bound
    of n_eff is at least ``n_eff_window_min``. Returns the interval fields of
    ``harness.coverage.cluster_coverage_interval`` plus ``rho_upper``, ``n_eff_lower``,
    ``window_applies``, ``needs_pooling`` (= not window_applies), ``pass`` and ``rule``.
    Report ``n_eff``, ``n_eff_lower``, ``k_clusters`` and the interval with every verdict."""
    y, m, _ = cluster_totals(hits, clusters)
    r = cluster_coverage_interval(hits, clusters, level, small_k)
    g = gate_from_totals_batch(y[None, :], m, target, level, small_k, window,
                               n_eff_window_min, conf)
    applies = bool(g["window_applies"][0])
    in_win = bool(g["in_window"][0])
    in_ci = bool(g["in_ci"][0])
    r.update({"target": target, "rule": RULE, "window_conf": conf,
              "rho_upper": float(g["rho_upper"][0]), "n_eff_lower": float(g["n_eff_lower"][0]),
              "contains_target": in_ci, "window_applies": applies, "in_window": in_win,
              "needs_pooling": not applies, "pass": bool(in_ci and (in_win or not applies))})
    return r


def pool_small_strata(strata: dict, neighbours: dict, level: float = GATE_LEVEL,
                      small_k: str = DEFAULT_SMALL_K,
                      n_eff_min: float = N_EFF_WINDOW_MIN, conf: float = WINDOW_CONF) -> dict:
    """Pool strata whose ``n_eff_lower`` is below ``n_eff_min`` with their nearest neighbour.

    ``strata``: {name: (hits, clusters)}; ``neighbours``: {name: [names in order of
    nearness]} (gap type or region). Repeatedly merges the smallest under-sized stratum that
    has a listed neighbour still present into that neighbour's group; strata without one are
    skipped, not a reason to stop. Ends when no under-sized stratum can be merged. Cluster ids
    are used as given (make them unique per stratum unless a cluster genuinely spans strata).
    Returns ``{pooled_name: {"members", "gate", "under_sized"}}``; ``under_sized`` marks a
    group that stayed below ``n_eff_min`` (its verdict is the interval alone)."""
    groups = {k: [k] for k in strata}
    data = {k: (np.asarray(h, bool).ravel(), np.asarray(c).ravel())
            for k, (h, c) in strata.items()}

    def size(name):
        h, c = data[name]
        return coverage_gate(h, c, level=level, small_k=small_k, n_eff_window_min=n_eff_min,
                             conf=conf)["n_eff_lower"]

    while len(groups) > 1:
        small = sorted((size(g), g) for g in groups if size(g) < n_eff_min)
        merged = False
        for _, g in small:
            target = None
            for member in groups[g]:
                for nb in neighbours.get(member, []):
                    owner = next((k for k, v in groups.items() if nb in v and k != g), None)
                    if owner is not None:
                        target = owner
                        break
                if target is not None:
                    break
            if target is None:
                continue
            new = f"{target}+{g}"
            data[new] = (np.concatenate([data[target][0], data[g][0]]),
                         np.concatenate([data[target][1], data[g][1]]))
            groups[new] = groups.pop(target) + groups.pop(g)
            merged = True
            break
        if not merged:
            break
    out = {}
    for k, v in groups.items():
        gate = coverage_gate(data[k][0], data[k][1], level=level, small_k=small_k,
                             n_eff_window_min=n_eff_min, conf=conf)
        out[k] = {"members": v, "gate": gate, "under_sized": bool(gate["n_eff_lower"] < n_eff_min)}
    return out
