"""WP C0 repair round, v3 products: one rule changed from v2 (``repair_lib``).

v2's registered check T5 exposed a product defect: a Dec-Jan-paired meter whose census install
date lies after October 2022 but which was read before that date (positive readings, method
``observed_before_install_date``) got no December-2022 fill (v2 required the census install by
October 2022) and was set to a structural zero by date, although its pre-origin readings show
it operating. v3 codes the Oct-Nov 2022 fill base the way the product codes those months:

* a pre-install month (by the census date) whose reading is NaN or 0 is a structural zero (0);
* a positive reading before the install date is an observation (kept);
* an installed month keeps its reading (NaN stays missing).

A paired kept meter gets the fill ``rho_bar(county, class) x base`` whenever both base months are
coded finite, except a meter not yet installed in 2022-12 whose base is 0, which stays a
structural zero by date; a meter with a missing (NaN) installed base month stays missing
(flag 8). The ratio fit (2012-2021, installed meters with four finite readings) and every other
rule are v2's. Decision recorded in the look log ([C0-repair-v3-decision]) before any v2 R3,
R7, R8 or R10 result was read.
"""
from __future__ import annotations

import lib
import numpy as np


def coded_base(X: np.ndarray, inst: np.ndarray, y: int = 2022) -> np.ndarray:
    """(Oct_y + Nov_y)/2 with pre-install NaN/0 months coded 0 (structural zeros)."""
    t_o = lib.tidx(f"{y}-10")
    out = []
    for t in (t_o, t_o + 1):
        x = X[:, t]
        out.append(np.where((inst > t) & (np.isnan(x) | (x == 0)), 0.0, x))
    return (out[0] + out[1]) / 2.0


def meter_fill(X: np.ndarray, inst: np.ndarray, cls: np.ndarray, sel: np.ndarray,
               params: dict, y: int = 2022) -> dict:
    """v3 per-meter fill of December y (see module docstring); same outputs as v2's."""
    j = lib.tidx(f"{y}-12")
    base = coded_base(X, inst, y)
    ok = sel & np.isfinite(base) & ~((inst > j) & (base == 0))
    rho = np.array([params["classes"][c]["rho_bar"] if c in params["classes"] else np.nan
                    for c in cls])
    s = np.array([params["classes"][c]["s"] if c in params["classes"] else np.nan for c in cls])
    nan = np.full(len(base), np.nan)
    return {"ok": ok, "base": np.where(ok, base, nan), "fill": np.where(ok, rho * base, nan),
            "s": np.where(ok, s, nan), "sigma": params["sigma"]}
