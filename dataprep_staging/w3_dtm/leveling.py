"""Leveling-based epoch correction for the DTM (WP W3).

The WRA leveling panel (``ls-wra-lsp-obs__choushui_panel.parquet``; one row per benchmark and
survey month, TWVD2001 heights, last survey 2021-09) gives benchmark elevations at annual
campaigns. The epoch correction moves a DTM value from its sheet's acquisition epoch t_s to a
target epoch t by the interpolated cumulative ground displacement u(x, t) - u(x, t_s).

Displacement model (fixed before any DTM-benchmark comparison):

* main campaign = survey months April-July; one elevation per benchmark and campaign year;
* campaign epoch T_Y = median decimal date of the year's main-campaign observations;
* annual increments d_b(Y) = e_b(Y+1) - e_b(Y) for benchmarks observed in both campaigns;
  increments with |d| > 0.30 m are treated as benchmark resets and dropped;
* d(x; Y) by inverse-distance weighting (power 2) of the k = 8 nearest training increments;
* u(x, T_Y) = sum of d(x; y) for y < Y from the first campaign year (u = 0 there), linear in
  time between campaigns; before the first campaign the first increment's rate is used, after
  the last campaign the mean rate of the last four increments (extrapolated: flag 6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MAIN_MONTHS = (4, 5, 6, 7)
RESET_THRESHOLD_M = 0.30
IDW_K = 8
IDW_POWER = 2.0
N_LAST_RATE = 4


def decimal_date(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts)
    y = ts.dt.year.to_numpy()
    start = pd.to_datetime(pd.DataFrame({"year": y, "month": 1, "day": 1}))
    end = pd.to_datetime(pd.DataFrame({"year": y + 1, "month": 1, "day": 1}))
    return y + ((ts - start) / (end - start)).to_numpy()


def load_panel(path: str) -> pd.DataFrame:
    """Benchmark observations: sid, x, y, t (decimal year), year, month, elev (finite only)."""
    lev = pd.read_parquet(path, columns=["datetime", "elev_m", "sid", "x_3826", "y_3826"])
    lev = lev[np.isfinite(lev["elev_m"])].copy()
    out = pd.DataFrame({
        "sid": lev["sid"].astype(str).to_numpy(),
        "x": lev["x_3826"].to_numpy(dtype="float64"),
        "y": lev["y_3826"].to_numpy(dtype="float64"),
        "t": decimal_date(lev["datetime"]),
        "year": lev["datetime"].dt.year.to_numpy(),
        "month": lev["datetime"].dt.month.to_numpy(),
        "elev": lev["elev_m"].to_numpy(dtype="float64"),
    })
    return out


def campaigns(obs: pd.DataFrame, years: range) -> pd.DataFrame:
    """Main-campaign elevation per benchmark and year (one observation per sid-month in the
    panel; the median is taken if a benchmark has several main-campaign months)."""
    m = obs[obs["month"].isin(MAIN_MONTHS) & obs["year"].isin(list(years))]
    c = m.groupby(["sid", "year"], as_index=False).agg(
        x=("x", "first"), y=("y", "first"), t=("t", "median"), elev=("elev", "median"))
    return c


class DisplacementModel:
    """u(x, t) from annual leveling increments of a training set of benchmarks."""

    def __init__(self, camp: pd.DataFrame, years: range, train_sids=None,
                 k: int = IDW_K, power: float = IDW_POWER,
                 reset_threshold: float = RESET_THRESHOLD_M):
        c = camp if train_sids is None else camp[camp["sid"].isin(set(train_sids))]
        self.years = list(years)
        self.k, self.power = k, power
        # campaign epochs from the whole panel (not only training): a timing convention
        self.T = np.array([float(camp.loc[camp["year"] == y, "t"].median())
                           for y in self.years])
        piv = c.pivot_table(index="sid", columns="year", values="elev")
        xy = c.groupby("sid")[["x", "y"]].first()
        self.inc = []          # per pair: (xy array, d array)
        self.n_dropped = 0
        for y0, y1 in zip(self.years[:-1], self.years[1:], strict=True):
            if y0 not in piv.columns or y1 not in piv.columns:
                self.inc.append((np.zeros((0, 2)), np.zeros(0)))
                continue
            d = (piv[y1] - piv[y0]).dropna()
            bad = d.abs() > reset_threshold
            self.n_dropped += int(bad.sum())
            d = d[~bad]
            self.inc.append((xy.loc[d.index].to_numpy(), d.to_numpy()))
        self.trees = [cKDTree(p) if len(p) else None for p, _ in self.inc]

    @property
    def n_increments(self) -> list[int]:
        return [len(d) for _, d in self.inc]

    def increments_at(self, pts: np.ndarray) -> np.ndarray:
        """(n_pts, n_pairs) IDW increments at points."""
        pts = np.atleast_2d(np.asarray(pts, dtype="float64"))
        out = np.full((len(pts), len(self.inc)), np.nan)
        for j, ((_p, d), tr) in enumerate(zip(self.inc, self.trees, strict=True)):
            if tr is None or len(d) == 0:
                continue
            kk = min(self.k, len(d))
            dist, idx = tr.query(pts, k=kk)
            dist = np.atleast_2d(dist).reshape(len(pts), kk)
            idx = np.atleast_2d(idx).reshape(len(pts), kk)
            w = 1.0 / np.maximum(dist, 1.0) ** self.power
            out[:, j] = (w * d[idx]).sum(1) / w.sum(1)
        return out

    def cumulative(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(U (n, Y) cumulative displacement at the campaign epochs, first-segment rate,
        mean rate of the last N_LAST_RATE segments) at points."""
        pts = np.atleast_2d(np.asarray(pts, dtype="float64"))
        inc = self.increments_at(pts)                      # (n, P)
        inc = np.where(np.isfinite(inc), inc, 0.0)
        U = np.concatenate([np.zeros((len(pts), 1)), np.cumsum(inc, 1)], 1)   # (n, Y)
        T = self.T
        r_first = inc[:, 0] / (T[1] - T[0])
        nl = min(N_LAST_RATE, inc.shape[1])
        r_last = inc[:, -nl:].sum(1) / (T[-1] - T[-1 - nl])
        return U, r_first, r_last

    def interp(self, U_i: np.ndarray, r_first_i: float, r_last_i: float,
               t: np.ndarray) -> np.ndarray:
        """Displacement of one point at epochs ``t`` from its cumulative series."""
        T = self.T
        t = np.atleast_1d(np.asarray(t, dtype="float64"))
        out = np.interp(t, T, U_i)
        lo, hi = t < T[0], t > T[-1]
        out[lo] = U_i[0] + r_first_i * (t[lo] - T[0])
        out[hi] = U_i[-1] + r_last_i * (t[hi] - T[-1])
        return out

    def u(self, pts: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
        """Displacement (m, relative to the first campaign) at points for epochs ``t``
        (scalar, or one epoch per point, or (n_pts, n_t)); returns (u, extrapolated)."""
        pts = np.atleast_2d(np.asarray(pts, dtype="float64"))
        U, r_first, r_last = self.cumulative(pts)
        T = self.T
        t = np.asarray(t, dtype="float64")
        if t.ndim == 0:
            t = np.full((len(pts), 1), float(t))
        elif t.ndim == 1:
            t = t.reshape(-1, 1) if len(t) == len(pts) else np.broadcast_to(t, (len(pts),
                                                                               len(t)))
        t = np.broadcast_to(t, (len(pts), t.shape[1]))
        out = np.empty(t.shape)
        for i in range(len(pts)):
            out[i] = self.interp(U[i], r_first[i], r_last[i], t[i])
        ext = (t < T[0]) | (t > T[-1])
        return out, ext

    def correction(self, pts: np.ndarray, t: np.ndarray, epochs: list,
                   shares: list) -> np.ndarray:
        """Epoch correction u(p, t_p) - sum_k w_k u(p, t_k) for points p with their own target
        epoch t_p and sheet-epoch mix (epochs[p], shares[p])."""
        pts = np.atleast_2d(np.asarray(pts, dtype="float64"))
        U, r_first, r_last = self.cumulative(pts)
        t = np.broadcast_to(np.asarray(t, dtype="float64"), (len(pts),))
        out = np.empty(len(pts))
        for i in range(len(pts)):
            ut = self.interp(U[i], r_first[i], r_last[i], t[i:i + 1])[0]
            us = self.interp(U[i], r_first[i], r_last[i], np.asarray(epochs[i]))
            out[i] = ut - float(np.dot(np.asarray(shares[i], dtype="float64"), us))
        return out

    def rate(self, pts: np.ndarray, t0: float, t1: float) -> np.ndarray:
        """Mean displacement rate (m/yr) between t0 and t1."""
        u, _ = self.u(pts, np.array([t0, t1]))
        return (u[:, 1] - u[:, 0]) / (t1 - t0)


def benchmark_elev_at(camp: pd.DataFrame, sid: str, t: float) -> float:
    """A benchmark's own elevation at epoch t, linearly interpolated between its main
    campaigns (NaN if t is not bracketed by two of its campaigns)."""
    c = camp[camp["sid"] == sid].sort_values("t")
    if len(c) < 2 or not (c["t"].iloc[0] <= t <= c["t"].iloc[-1]):
        return float("nan")
    return float(np.interp(t, c["t"].to_numpy(), c["elev"].to_numpy()))
