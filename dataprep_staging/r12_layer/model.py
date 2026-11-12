"""SL.subsidence_layer.v1: the frozen head-driven statistical subsidence layer.

Rate field  : mean May-to-May leveling increment of 2019-2021 at each benchmark (at least two
              of the three), Gaussian-kernel average over benchmarks (3 km, own weight 4),
              mapped to the registered 1 km cells (Gaussian 3 km over the 40 nearest
              benchmarks; cells farther than 6 km from a benchmark carry no prediction).
Head driver : year-on-year change of each well's hydrological-year mean head (June-May, at
              least 9 months), inverse-distance weighted to benchmarks (6 wells, all layers),
              smoothed over 10 km, minus its mean over the reference years 2019-2021.
Prediction  : rate x (1 + gamma x driver) for survey year y (arm HL); gamma = 0 is the
              constant-rate arm (C0).
Heads       : ``heads/L1_yearbook`` (2001-2011) joined to ``heads/L1_clean`` v3 (2012 on).
Cells       : the N3S-L freeze's ``cells.csv`` (cell, county, township, centre).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataprep_staging.r12_leveling.increments import hydro_year, site_table
from dataprep_staging.r12_leveling.layer import kernel_rate
from dataprep_staging.r12_leveling.paths import DATAPREP

GAMMA_YEARS = (2007, 2021)           # leveling years the coefficient is fitted on
REF_YEARS = (2019, 2020, 2021)       # rate window and driver reference
THRESH = 0.03                        # m/yr, the official "significant subsidence" rate
YEARS = list(range(2002, 2031))
YEARBOOK = os.path.join(DATAPREP, "heads", "L1_yearbook", "v1",
                        "heads_monthly_yearbook.parquet")
CLEAN = os.path.join(DATAPREP, "heads", "L1_clean", "v3", "")
CELLS = os.path.join(DATAPREP, "reports", "N3S-L", "freeze", "cells.csv")


def monthly_heads(last_month: str | None = None) -> pd.DataFrame:
    """dates x wells monthly means; ``last_month`` (YYYY-MM) truncates the held record."""
    y = pd.read_parquet(YEARBOOK)
    y = y[(y.year <= 2011) & y.parse_flag.isin((0, 1, 2)) & y.value_m.notna()]
    a = pd.DataFrame({"sid": y.sid, "value": y.value_m,
                      "date": pd.to_datetime(dict(year=y.year, month=y.month, day=1))})
    h = pd.read_parquet(CLEAN + "heads_monthly.parquet")
    h = h[h.flag <= 1][["sid", "date", "value"]]
    W = pd.concat([a, h]).pivot_table(index="date", columns="sid", values="value")
    return W if last_month is None else W.loc[:pd.Timestamp(last_month + "-01")]


def well_dmean(W: pd.DataFrame, min_months: int = 9) -> pd.DataFrame:
    g = W.groupby(hydro_year(W.index))
    m = g.mean().where(g.count() >= min_months)
    return m.diff().where(m.notna() & m.shift(1).notna()).reindex(YEARS).T


def driver_at(xy: np.ndarray, dmean: pd.DataFrame, L: float = 10_000.0) -> np.ndarray:
    """(n_benchmarks, n_years) smoothed head-change field; NaN where under 20 wells report."""
    meta = pd.read_parquet(CLEAN + "station_meta_clean.parquet").set_index("sid")
    meta = meta.reindex(dmean.index)
    wxy = meta[["x_twd97", "y_twd97"]].to_numpy("float64")
    V = dmean.to_numpy("float64")
    d2, i2 = cKDTree(xy).query(xy, k=min(200, len(xy)))
    w2 = np.exp(-0.5 * (d2 / L) ** 2)
    out = np.full((len(xy), V.shape[1]), np.nan)
    for j in range(V.shape[1]):
        ok = np.isfinite(V[:, j]) & np.isfinite(wxy).all(1)
        if ok.sum() < 20:
            continue
        d, ix = cKDTree(wxy[ok]).query(xy, k=6)
        w = 1.0 / np.maximum(d, 500.0) ** 2
        f = (w * V[ok, j][ix]).sum(1) / w.sum(1)
        out[:, j] = (w2 * f[i2]).sum(1) / w2.sum(1)
    return out


def rate_at(D: np.ndarray, xy: np.ndarray, years=REF_YEARS) -> np.ndarray:
    cols = [YEARS.index(y) for y in years]
    w = D[:, cols]
    tr = np.nanmean(np.where(np.isfinite(w), w, np.nan), axis=1)
    tr[np.isfinite(w).sum(1) < 2] = np.nan
    return kernel_rate(tr, xy, None, 3.0, 4.0)


def fit_gamma(D, F, xy, n: int = 3, years=GAMMA_YEARS) -> tuple[float, dict]:
    """Least squares of (increment - rate) on rate x centred driver, pooled over years."""
    num = den = 0.0
    per_year = {}
    for y in range(years[0], years[1] + 1):
        j = YEARS.index(y)
        w = D[:, j - n:j]
        tr = np.nanmean(np.where(np.isfinite(w), w, np.nan), axis=1)
        tr[np.isfinite(w).sum(1) < 2] = np.nan
        if np.isfinite(tr).sum() < 50:
            continue
        base = kernel_rate(tr, xy, None, 3.0, 4.0)
        x = (F[:, j] - np.nanmean(F[:, j - n:j], axis=1)) * np.clip(base, 0.002, None)
        ok = np.isfinite(x) & np.isfinite(D[:, j]) & np.isfinite(base)
        if ok.sum() < 50:
            continue
        a, b = float((x[ok] * (D[ok, j] - base[ok])).sum()), float((x[ok] ** 2).sum())
        per_year[y] = (a, b)
        num, den = num + a, den + b
    return num / den, per_year


def to_cells(v: np.ndarray, xy: np.ndarray, cxy: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    d, ix = cKDTree(xy[ok]).query(cxy, k=40)
    w = np.exp(-0.5 * (d / 3000.0) ** 2) + 1e-12
    out = (w * v[ok][ix]).sum(1) / w.sum(1)
    out[d[:, 0] > 6000.0] = np.nan
    return out


class Layer:
    """Everything the frozen model needs, built once from the held inputs."""

    def __init__(self, last_month: str | None = None):
        d, self.xy, self.sids, _ = site_table()
        self.D = d.reindex(columns=YEARS).to_numpy("float64")
        self.F = driver_at(self.xy, well_dmean(monthly_heads(last_month)))
        self.cells = pd.read_csv(CELLS)
        self.cxy = self.cells[["cx", "cy"]].to_numpy("float64")
        self.rate = rate_at(self.D, self.xy)
        self.ref = np.nanmean(self.F[:, [YEARS.index(y) for y in REF_YEARS]], axis=1)

    def driver(self, year: int) -> np.ndarray:
        return self.F[:, YEARS.index(year)] - self.ref

    def cell_rates(self, year: int, gamma: float) -> np.ndarray:
        """m/yr per registered cell for survey year ``year`` (May year-1 to May year)."""
        r = self.rate if gamma == 0.0 else \
            self.rate + gamma * self.driver(year) * np.clip(self.rate, 0.002, None)
        return to_cells(r, self.xy, self.cxy)

    def area(self, rates: np.ndarray, county: str) -> int:
        return int(np.nansum((rates > THRESH) & (self.cells.county.to_numpy() == county)))

    def county_driver(self, year: int, county: str) -> float:
        c = to_cells(self.driver(year), self.xy, self.cxy)
        return float(np.nanmean(c[self.cells.county.to_numpy() == county]))

    def township_peaks(self, rates: np.ndarray) -> pd.DataFrame:
        t = self.cells.assign(rate_cm_yr=100 * rates)
        return t.groupby(["county", "township"]).rate_cm_yr.agg(
            peak_cm_yr="max", mean_cm_yr="mean").reset_index()
