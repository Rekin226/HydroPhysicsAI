"""Shared code of N1.policy_response.v1: model runs, exposures, regressions, bootstraps.

Model runs repeat ``calibrate_coupled``'s own rollout (initial heads of month 0, the
recipe's energy, recharge, canal and extra terms), so every column is driven by exactly the
head path it was fitted on (memory: the column pairs with its head field). Run with the
N1 policy worktree on PYTHONPATH (``results/twin_runs/par24/py.sh``): its twin.forward reads
the ``column_zones_collapsed`` columns of batch 24.

Pure functions (exposure, panel regressions, bootstraps, partition errors) take arrays and
are tested on synthetic data in ``tests/``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = pd.period_range("2012-01", "2022-12", freq="M")     # heads[..., k] <-> MONTHS[k]
B_BOOT = 2000
LEVEL = 0.90
MIN_CLUSTERS = 5
BLOCK_M = 5000.0                                              # bootstrap clusters: 5 km


# ----------------------------------------------------------------------------- the model
class TwinModel:
    """A calibrated flow model with its leveling column, on the CPU."""

    def __init__(self, theta: str, column_json: str, device: str = "cpu"):
        import torch

        from hydrophysics.twin.forward import (
            attach_sw_recharge,
            build_model,
            extra_hist,
            load_members,
            load_or_fit_vep,
            sw_hist,
        )
        from hydrophysics.twin.inputs import input_options, load_twin_inputs
        from hydrophysics.twin.zone_county import column_zone_map

        self.torch = torch
        self.member = m = load_members([str(theta)])[0]
        self.meta = m.meta
        inp = load_twin_inputs(dx=1000.0, meter_filter=m.meta.get("meter_filter", "none"),
                               cap_duty=float(m.meta.get("cap_duty", 1.0)),
                               **input_options(m.meta))
        attach_sw_recharge(inp, m.meta)
        self.inp = inp
        self.model, self.scalars, _ = build_model(inp.grid, m, device)
        ec = m.meta.get("eta_classes")
        if ec:
            self.E = torch.tensor(np.stack([inp.E_by_class[c] for c in ec]),
                                  dtype=torch.float64)[..., 1:]
        else:
            self.E = torch.as_tensor(inp.E_total, dtype=torch.float64)[:, 1:]
        self._sw, self._extra = sw_hist(inp), extra_hist(inp)
        zoc = column_zone_map(inp.grid, m.meta)
        self.col, self.col_meta = load_or_fit_vep(
            str(column_json), None, None, device, zone_of_cell=zoc,
            county_sha256=m.meta.get("zone_county_sha256"))
        self.cent = inp.grid.centroids()
        self.dx = float(inp.grid.dx)

    def energy(self) -> np.ndarray:
        """The rollout's energy, (A, T-1) or (K, A, T-1); step j is month MONTHS[j + 1]."""
        return self.E.numpy().copy()

    def heads(self, E: np.ndarray | None = None) -> np.ndarray:
        from hydrophysics.twin.forward import rollout

        E_t = self.E if E is None else self.torch.as_tensor(E, dtype=self.torch.float64)
        with self.torch.no_grad():
            H = rollout(self.model, self.scalars, self.inp.initial_heads(0), E_t,
                        self.inp.recharge_field[:, 1:], self.inp.ground_elev,
                        sw_field=self._sw, q_extra=self._extra)
        return H.cpu().numpy()                                    # (L, A, T)

    def subsidence(self, H: np.ndarray) -> np.ndarray:
        from hydrophysics.twin.forward import compaction

        return compaction(self.col, self.torch.as_tensor(H))      # (A, T) m, + = sinking

    def cell_of(self, xy: np.ndarray) -> np.ndarray:
        """Nearest active cell of each point, -1 beyond one cell spacing."""
        return nearest_cell(self.cent, xy, self.dx)


def nearest_cell(cent: np.ndarray, xy: np.ndarray, dx: float) -> np.ndarray:
    d2 = ((np.asarray(xy, float)[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    c = d2.argmin(axis=1)
    return np.where(np.sqrt(d2[np.arange(len(c)), c]) <= dx, c, -1)


def replace_months(E: np.ndarray, target: np.ndarray, ref: np.ndarray,
                   month_of_step: np.ndarray) -> np.ndarray:
    """ZR energy: the steps in ``target`` replaced, cell by cell, by the mean of the
    ``ref`` steps of the same calendar month (works on (A, S) and (K, A, S))."""
    Z = np.array(E, copy=True)
    for mo in range(1, 13):
        t = target & (month_of_step == mo)
        r = ref & (month_of_step == mo)
        if t.any():
            if not r.any():
                raise ValueError(f"no reference step for calendar month {mo}")
            Z[..., t] = E[..., r].mean(axis=-1, keepdims=True)
    return Z


# ----------------------------------------------------------------------------- exposure
def ring_weights(cent: np.ndarray, cells: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """(n, A) indicator of active cells whose centroid lies in [r_in, r_out) metres of each
    benchmark cell's centroid (r_in = 0 includes the cell itself)."""
    d = np.sqrt(((cent[cells][:, None, :] - cent[None, :, :]) ** 2).sum(-1))
    return ((d >= r_in) & (d < r_out)).astype(float)


def interval_exposure(E_cell_month: np.ndarray, W: np.ndarray, m0: np.ndarray, m1: np.ndarray,
                      years: np.ndarray) -> np.ndarray:
    """Annualised energy around each row: sum of ``E_cell_month`` (A, M) over months
    m0 < k <= m1 weighted by the row's ring ``W`` (row, A), divided by the interval years."""
    cs = np.concatenate([np.zeros((E_cell_month.shape[0], 1)),
                         np.cumsum(E_cell_month, axis=1)], axis=1)       # (A, M+1)
    tot = cs[:, m1 + 1] - cs[:, m0 + 1]                                   # (A, n)
    return (W * tot.T).sum(axis=1) / years


# ----------------------------------------------------------------------------- regressions
def _demean(v: np.ndarray, groups: list[np.ndarray], tol: float = 1e-10,
            max_iter: int = 500) -> np.ndarray:
    """Project ``v`` (n, k) off every fixed effect by alternating group demeaning (the
    Frisch-Waugh-Lovell within transform for several, possibly unbalanced, effects)."""
    v = np.array(v, float, copy=True)
    scale = np.maximum(np.abs(v).max(axis=0), 1e-300)            # per column
    for _ in range(max_iter):
        delta = 0.0
        for inv, cnt in groups:
            m = np.stack([np.bincount(inv, weights=v[:, j], minlength=cnt.size)
                          for j in range(v.shape[1])], axis=1) / cnt[:, None]
            v -= m[inv]
            delta = max(delta, float((np.abs(m) / scale).max()))
        if delta < tol:
            break
    return v


def fe_ols(y: np.ndarray, X: np.ndarray, fe: list[np.ndarray]) -> np.ndarray:
    """OLS of y on X with fixed effects for each array in ``fe``; returns the coefficients
    of X (identical to the dummy-variable regression, by Frisch-Waugh-Lovell)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    groups = []
    for g in fe:
        _, inv = np.unique(g, return_inverse=True)
        groups.append((inv, np.bincount(inv).astype(float)))
    Z = _demean(np.column_stack([np.asarray(y, float), X]), groups)
    beta, *_ = np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)
    return beta


def blocks_of(xy: np.ndarray, size: float = BLOCK_M) -> np.ndarray:
    xy = np.asarray(xy, float)
    return np.array([f"{int(np.floor(x / size))}_{int(np.floor(y / size))}" for x, y in xy])


def cluster_bootstrap(stat, clusters: np.ndarray, *, B: int = B_BOOT, seed: int,
                      level: float = LEVEL) -> tuple[np.ndarray, list[np.ndarray]]:
    """``stat(rows) -> array``; returns (stat on all rows, [lo, hi] percentile interval)
    from resampling whole clusters with replacement."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    full = np.asarray(stat(np.arange(len(clusters))), float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        rows = np.concatenate([idx[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        try:
            draws.append(np.asarray(stat(rows), float))
        except np.linalg.LinAlgError:
            continue
    d = np.array(draws)
    q = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return full, [q[0], q[1]]


def ratio_verdict(beta_model: float, obs_ci: list[float], band: float = 1.5) -> dict:
    """Step B rule: with the observed coefficient's interval [lo, hi] excluding 0, the
    ratio interval is beta_model / [hi, lo]; 'consistent' if it lies inside
    [1/band, band], 'inconsistent' if it excludes 1, else 'inconclusive'. An observed
    interval that includes 0 gives 'not_informative'."""
    lo, hi = float(obs_ci[0]), float(obs_ci[1])
    if lo <= 0.0 <= hi or not np.isfinite(beta_model):
        return {"verdict": "not_informative", "ratio_ci": [float("nan")] * 2}
    r = sorted([beta_model / hi, beta_model / lo])
    if r[0] >= 1.0 / band and r[1] <= band:
        v = "consistent"
    elif r[0] > 1.0 or r[1] < 1.0:
        v = "inconsistent"
    else:
        v = "inconclusive"
    return {"verdict": v, "ratio_ci": [float(r[0]), float(r[1])]}


# ----------------------------------------------------------------------------- step C
def partition_deviation(cum: np.ndarray) -> np.ndarray:
    """(n, Y+1) cumulative subsidence at consecutive surveys -> (n, Y) yearly increments
    minus each row's mean increment (the split of the total across the years)."""
    inc = np.diff(np.asarray(cum, float), axis=1)
    return inc - inc.mean(axis=1, keepdims=True)


def rms_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=1))


def pooled_slope(x: np.ndarray, y: np.ndarray, clusters: np.ndarray, *, seed: int) -> dict:
    """Slope through the origin of y on x pooled over all entries of each row, with a
    cluster-bootstrap interval (reported, never gating)."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def stat(rows):
        xx, yy = x[rows].ravel(), y[rows].ravel()
        ok = np.isfinite(xx) & np.isfinite(yy)
        den = float(np.sum(xx[ok] ** 2))
        return np.array([np.sum(xx[ok] * yy[ok]) / den if den > 0 else np.nan])

    full, ci = cluster_bootstrap(stat, clusters, seed=seed)
    return {"slope": float(full[0]), "ci": [float(ci[0][0]), float(ci[1][0])]}
