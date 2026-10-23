"""Supply-limited diversion indicator and its benchmarking (plan W1 steps 3-4).

    D(day) = max(0, min(beta * I(day) - R(day), C_chc(day) + C_yl(day)))

I is the inflow composite (cms), R the industrial plus public supply (the weir's annual
table spread evenly over the year's days), C_d = a_d * w_d(month) the district demand
capacity with crop-1 share s_d and winter (Dec-Jan) weight w. The diversion is split pro
rata to C_d. beta, a_chc, a_yl are fitted by least squares to the annual district totals
of the variant's fit years; s_chc, s_yl, w are chosen by leave-one-year-out on the Jan-May
benchmark (validate.py). The monthly indicator is benchmarked to the annual totals with
proportional first-difference Denton, keeping zero months zero.

Adapted from the 2026-10-12 scout's pilot2.py (results/dataprep/_pilots/
surface_water_monthly/); differences: R uses the actual days of each year (pilot 365.25),
the fit years are a parameter (pre-origin 2012-2022), and the Denton solver also returns the
benchmark-to-indicator ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from w1_canal.config import CAL_DAYS, CALENDAR, JIJI_COL

SEC_DAY = 86400.0


def wvec(share1: float, winter: float) -> np.ndarray:
    """Daily-rate weight per calendar month (index 1..12), each crop's weights summing (over
    days) to its share; Dec and Jan get ``winter`` times the smallest positive weight."""
    tot = {1: 0.0, 2: 0.0}
    for m, (c, w) in CALENDAR.items():
        tot[c] += w * CAL_DAYS[m]
    out = np.zeros(13)
    for m, (c, wt) in CALENDAR.items():
        out[m] = (share1 if c == 1 else 1 - share1) * wt / tot[c]
    out[1] = out[12] = winter * out[out > 0].min()
    return out


@dataclass
class Daily:
    """Daily drivers of one variant."""
    dates: pd.DatetimeIndex
    inflow: np.ndarray       # cms (the composite I)
    R: np.ndarray            # cms
    years: list
    mon: np.ndarray          # calendar month 1..12 per day
    mi: np.ndarray           # month index 0..NM-1 per day
    nm: int

    @property
    def months(self) -> pd.DatetimeIndex:
        return pd.date_range(f"{self.years[0]}-01-01", periods=self.nm, freq="MS")


def make_daily(I_cms: pd.Series, jiji: pd.DataFrame, years) -> Daily:
    years = list(years)
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    inflow = I_cms.reindex(idx).to_numpy(dtype=float)
    if np.isnan(inflow).any():
        raise ValueError("inflow composite has gaps")
    ndays = pd.Series(idx.year).value_counts().sort_index()
    rr = (jiji["industrial_1e8_m3"] + jiji["public_1e8_m3"]) * 1e8 / (ndays * SEC_DAY)
    R = pd.Series(idx.year).map(rr).to_numpy(dtype=float)
    mon = idx.month.to_numpy()
    mi = (idx.year.to_numpy() - years[0]) * 12 + mon - 1
    return Daily(idx, inflow, R, years, mon, mi, len(years) * 12)


def simulate(dd: Daily, beta, a_c, a_y, s_c, s_y, winter, cap=True):
    """Monthly district volumes (10^8 m3) of the indicator."""
    Cc = a_c * wvec(s_c, winter)[dd.mon]
    Cy = a_y * wvec(s_y, winter)[dd.mon]
    dem = Cc + Cy
    tot = np.minimum(np.clip(beta * dd.inflow - dd.R, 0, None), dem) if cap else dem
    fc = np.divide(Cc, dem, out=np.full_like(dem, 0.5), where=dem > 0)
    mc = np.zeros(dd.nm)
    my = np.zeros(dd.nm)
    np.add.at(mc, dd.mi, tot * fc * SEC_DAY / 1e8)
    np.add.at(my, dd.mi, tot * (1 - fc) * SEC_DAY / 1e8)
    return mc, my


def annual_targets(jiji: pd.DataFrame, years) -> dict[str, np.ndarray]:
    return {d: jiji[c].reindex(list(years)).to_numpy(dtype=float) for d, c in JIJI_COL.items()}


def fit(dd: Daily, targets: dict, s_c, s_y, winter, cap=True, fit_years=None):
    """(beta, a_c, a_y) by least squares on the annual totals of ``fit_years``."""
    ny = len(dd.years)
    fy = dd.years if fit_years is None else list(fit_years)
    idx = np.array([dd.years.index(y) for y in fy])

    def res(p):
        b = np.exp(p[0]) if cap else 1.0
        mc, my = simulate(dd, b, np.exp(p[-2]), np.exp(p[-1]), s_c, s_y, winter, cap)
        ac = mc.reshape(ny, 12).sum(1)
        ay = my.reshape(ny, 12).sum(1)
        return np.r_[ac[idx] - targets["chc"][idx], ay[idx] - targets["yl"][idx]]

    p0 = np.log([0.9, 30.0, 30.0]) if cap else np.log([30.0, 30.0])
    r = least_squares(res, p0, method="trf")
    p = np.exp(r.x)
    if not cap:
        p = np.r_[1.0, p]
    return p, float(np.sqrt(np.mean(r.fun ** 2)))


def denton_pfd(ind: np.ndarray, totals: np.ndarray, per: int = 12) -> np.ndarray:
    """Proportional first-difference Denton on the months with a positive indicator
    (structural zeros stay zero): minimise sum((x_t/i_t - x_s/i_s)^2) over consecutive
    positive months, subject to the annual sums. ``ind`` covers exactly len(totals) years."""
    n = len(ind)
    if n != per * len(totals):
        raise ValueError("indicator length != 12 x number of annual totals")
    pos = np.where(ind > 1e-12)[0]
    m = len(pos)
    Dm = np.diff(np.eye(m), axis=0) @ np.diag(1 / ind[pos])
    A = Dm.T @ Dm
    C = np.kron(np.eye(len(totals)), np.ones(per))[:, pos]
    K = np.block([[A, C.T], [C, np.zeros((len(totals), len(totals)))]])
    sol = np.linalg.solve(K, np.r_[np.zeros(m), totals])
    x = np.zeros(n)
    x[pos] = sol[:m]
    return x


def janmay(x: np.ndarray, years, bench_years) -> np.ndarray:
    years = list(years)
    j = x.reshape(len(years), 12)[:, :5].sum(1)
    return j[[years.index(y) for y in bench_years]]
