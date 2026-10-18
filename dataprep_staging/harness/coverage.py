"""Cluster-adjusted interval for a coverage rate, and the coverage gate (plan 4.4, A1).

A product's 90 % intervals are checked on held-out data whose misses are clustered (months of
one gap, wells of one region, years of one district). The binomial interval for the coverage
rate must use the effective sample size

    n_eff = n / deff,   deff = 1 + (m_w - 1) * rho,   m_w = sum(m_i^2) / n

where m_i are the cluster sizes and rho is the intra-cluster correlation of the hit/miss
indicators, estimated by the one-way ANOVA estimator for binary data (Fleiss; Ridout,
Demetrio and Firth 1999), truncated to [0, 1]. The interval is the Wilson score interval on
n_eff (level 0.95 by default). With ``small_k="t"`` the normal quantile is replaced by the
Student-t quantile on (k - 1) degrees of freedom (k = number of clusters), a small-sample
correction for few clusters.

Gate rule (plan 4.4, one target across the plan, 0.90 nominal): a stratum passes if the
cluster-adjusted interval contains 0.90; strata with ``n_eff >= 30`` must in addition have the
point coverage inside 0.85-0.95. Smaller strata are pooled with their nearest neighbour in gap
type or region before testing (:func:`pool_small_strata`).

The default quantile rule (``DEFAULT_SMALL_K``) and interval level (``GATE_LEVEL``) were fixed by
the A1 coverage simulation (``coverage_sim.py``; acceptance: at least 0.89 actual coverage of a
true 0.90 in 10,000 simulated clustered binomial draws per configuration), under the decision
rule written in that script before it ran.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TARGET = 0.90
GATE_LEVEL = 0.95
POINT_WINDOW = (0.85, 0.95)
N_EFF_WINDOW_MIN = 30
#: "z" (plan method) or "t" (Student-t on k-1 df); set by the A1 simulation's declared rule:
#: "z" fell below 0.89 in one declared configuration (5 clusters x 200, rho 0.05: 0.8735),
#: "t" met it in all 15 (minimum 0.918), so "t" is the default
#: (results/dataprep/reports/A1_evidence/coverage_sim.json).
DEFAULT_SMALL_K = "t"


def cluster_totals(hits, clusters) -> tuple[np.ndarray, np.ndarray, list]:
    """Per-cluster hit counts y_i and sizes m_i (clusters in order of first appearance)."""
    h = np.asarray(hits, dtype=float).ravel()
    c = pd.Series(np.asarray(clusters).ravel())
    if h.size != c.size:
        raise ValueError("hits and clusters differ in length")
    if np.isnan(h).any():
        raise ValueError("hits contain NaN")
    codes, uniq = pd.factorize(c)
    y = np.bincount(codes, weights=h, minlength=len(uniq))
    m = np.bincount(codes, minlength=len(uniq)).astype(float)
    return y, m, list(uniq)


def icc_anova_batch(Y: np.ndarray, m: np.ndarray) -> np.ndarray:
    """ANOVA intra-cluster correlation of binary data for a batch of draws.

    ``Y`` (D, k) hit counts per cluster, ``m`` (k,) cluster sizes. Returns rho (D,) truncated
    to [0, 1]; rho = 1 when k = 1 (no between-cluster information: assume full clustering);
    rho = 0 when every cluster has size 1 or the data carry no variance (all hits or all
    misses)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m = np.asarray(m, dtype=float)
    k = m.size
    N = m.sum()
    if k == 1:
        return np.ones(Y.shape[0])
    if k == N:
        return np.zeros(Y.shape[0])
    tot = Y.sum(1)
    s2 = (Y ** 2 / m).sum(1)
    msb = (s2 - tot ** 2 / N) / (k - 1)
    msw = (tot - s2) / (N - k)
    m0 = (N - (m ** 2).sum() / N) / (k - 1)
    den = msb + (m0 - 1) * msw
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, (msb - msw) / den, 0.0)
    return np.clip(rho, 0.0, 1.0)


def wilson_batch(p, n, q):
    """Wilson score interval for proportion(s) ``p`` on (effective) size(s) ``n`` with
    quantile ``q``."""
    p, n, q = np.asarray(p, float), np.asarray(n, float), np.asarray(q, float)
    q2 = q ** 2
    den = 1 + q2 / n
    centre = (p + q2 / (2 * n)) / den
    half = q * np.sqrt(np.maximum(p * (1 - p) / n + q2 / (4 * n ** 2), 0.0)) / den
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def interval_from_totals_batch(Y, m, level: float = GATE_LEVEL,
                               small_k: str = DEFAULT_SMALL_K) -> dict:
    """Vectorised cluster-adjusted Wilson interval from cluster totals (see module doc)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m = np.asarray(m, dtype=float)
    k, N = m.size, m.sum()
    p = Y.sum(1) / N
    rho = icc_anova_batch(Y, m)
    m_w = (m ** 2).sum() / N
    deff = 1 + (m_w - 1) * rho
    n_eff = N / deff
    if small_k == "t" and k >= 2:
        q = stats.t.ppf(0.5 + level / 2, k - 1)
    elif small_k in ("z", "t"):
        q = stats.norm.ppf(0.5 + level / 2)
    else:
        raise ValueError(f"small_k must be 'z' or 't', not {small_k!r}")
    lo, hi = wilson_batch(p, n_eff, q)
    return {"p": p, "rho": rho, "deff": deff, "n_eff": n_eff, "lo": lo, "hi": hi,
            "n": N, "k": k, "m_w": m_w, "quantile": float(q)}


def cluster_coverage_interval(hits, clusters, level: float = GATE_LEVEL,
                              small_k: str = DEFAULT_SMALL_K) -> dict:
    """Cluster-adjusted Wilson interval for the coverage rate of per-observation ``hits``
    (bool) grouped by ``clusters``."""
    y, m, _ = cluster_totals(hits, clusters)
    r = interval_from_totals_batch(y[None, :], m, level, small_k)
    return {"coverage": float(r["p"][0]), "n": int(r["n"]), "k_clusters": int(r["k"]),
            "rho": float(r["rho"][0]), "deff": float(r["deff"][0]),
            "n_eff": float(r["n_eff"][0]), "lo": float(r["lo"][0]), "hi": float(r["hi"][0]),
            "level": level, "small_k": small_k, "quantile": r["quantile"]}


def coverage_gate(hits, clusters, target: float = TARGET, level: float = GATE_LEVEL,
                  small_k: str = DEFAULT_SMALL_K, window=POINT_WINDOW,
                  n_eff_window_min: float = N_EFF_WINDOW_MIN) -> dict:
    """Plan 4.4 rule for one stratum: pass if the cluster-adjusted interval contains
    ``target``; if ``n_eff >= n_eff_window_min`` the point coverage must also lie in
    ``window``. ``needs_pooling`` flags a stratum below that size (pool it first)."""
    r = cluster_coverage_interval(hits, clusters, level, small_k)
    in_ci = r["lo"] <= target <= r["hi"]
    big = r["n_eff"] >= n_eff_window_min
    in_win = window[0] <= r["coverage"] <= window[1]
    r.update({"target": target, "contains_target": bool(in_ci), "window_applies": bool(big),
              "in_window": bool(in_win), "needs_pooling": bool(not big),
              "pass": bool(in_ci and (in_win or not big))})
    return r


def pool_small_strata(strata: dict, neighbours: dict, level: float = GATE_LEVEL,
                      small_k: str = DEFAULT_SMALL_K,
                      n_eff_min: float = N_EFF_WINDOW_MIN) -> dict:
    """Pool strata whose n_eff is below ``n_eff_min`` with their nearest neighbour.

    ``strata``: {name: (hits, clusters)}; ``neighbours``: {name: [names in order of
    nearness]} (gap type or region). Repeatedly merges the smallest under-sized stratum into
    its nearest still-existing neighbour, until every pooled stratum reaches ``n_eff_min`` or
    one stratum is left. Cluster ids are used as given (make them unique per stratum unless
    a cluster genuinely spans strata). Returns ``{pooled_name: {"members", "gate"}}``."""
    groups = {k: [k] for k in strata}
    data = {k: (np.asarray(h, bool).ravel(), np.asarray(c).ravel())
            for k, (h, c) in strata.items()}

    def neff(name):
        h, c = data[name]
        return cluster_coverage_interval(h, c, level, small_k)["n_eff"]

    while len(groups) > 1:
        small = sorted((neff(g), g) for g in groups if neff(g) < n_eff_min)
        if not small:
            break
        _, g = small[0]
        target = None
        for member in groups[g]:
            for nb in neighbours.get(member, []):
                owner = next((k for k, v in groups.items() if nb in v and k != g), None)
                if owner is not None:
                    target = owner
                    break
            if target:
                break
        if target is None:
            break
        new = f"{target}+{g}"
        h = np.concatenate([data[target][0], data[g][0]])
        c = np.concatenate([data[target][1], data[g][1]])
        data[new] = (h, c)
        groups[new] = groups.pop(target) + groups.pop(g)
    return {k: {"members": v, "gate": coverage_gate(data[k][0], data[k][1], level=level,
                                                    small_k=small_k)}
            for k, v in groups.items()}
