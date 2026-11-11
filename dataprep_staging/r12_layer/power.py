"""SL1 synthetic power study (user ruling R12.3). Settings are fixed here before the run.

SL1 rule: over survey years 2026-2028, the head-driven arm (HL) passes if its absolute error
on the official Yunlin area sinking faster than 3 cm/yr is smaller than the constant-rate
arm's (C0) in at least two of the three years and its mean absolute error is smaller.

Worlds: the true rate field of year t is  rate x (1 + k x gamma x driver_t) + u_t, with
  k        the true head sensitivity relative to the frozen gamma (0 = no response),
  driver_t the frozen 2026 field for the first year; for the two later years a field drawn
           from the 2008-2026 driver fields (centred like the model's),
  u_t      a fan-wide year shock, N(0, s_u), the part of a year's sinking that heads do not
           explain; central s_u = 0.5 cm/yr (the 2013-2021 forward test left year-mean
           residuals with sd 0.75 cm/yr over all years and 0.3 cm/yr over 2017-2021).
The official figure is the area of the true field plus N(0, 20 km2) (the spread between
the official area and the area of the observed-increment field, 2016-2021), floored at 0.

Decision (fixed before the run): SL1 is registered as gating if, at s_u = 0.5, power at
k = 1 is at least 0.80 and the pass rate at k = 0 is at most 0.25; otherwise report-only.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from dataprep_staging.r12_layer.model import YEARS, Layer, fit_gamma, to_cells
from dataprep_staging.r12_leveling.paths import DATAPREP

SEED = 20261112
N_WORLDS = 4000
K_GRID = (0.0, 0.5, 1.0, 2.0)
S_U_CM = (0.3, 0.5, 0.75)
SIGMA_AREA = 20.0
COUNTY = "雲林縣"
POOL = list(range(2008, 2027))
CENTRAL, POWER_MIN, FALSE_MAX = 0.5, 0.80, 0.25


def run() -> dict:
    L = Layer()
    gamma, _ = fit_gamma(L.D, L.F, L.xy)
    yl = L.cells.county.to_numpy() == COUNTY
    rate_c = to_cells(L.rate, L.xy, L.cxy)
    clip_c = to_cells(np.clip(L.rate, 0.002, None), L.xy, L.cxy)
    drv = {}
    for y in POOL:
        f = L.F[:, YEARS.index(y)]
        if np.isfinite(f).sum() > 100:
            drv[y] = to_cells(f - L.ref, L.xy, L.cxy)
    pool = sorted(drv)
    c0 = float(np.nansum((rate_c > 0.03) & yl))

    def area(field):
        return float(np.nansum((field > 0.03) & yl))

    rng = np.random.default_rng(SEED)
    out = {"gamma": gamma, "c0_area_km2": c0, "driver_years_pool": pool,
           "n_worlds": N_WORLDS, "seed": SEED, "cells": {}}
    for s_u in S_U_CM:
        for k in K_GRID:
            passes = wins = 0
            for _ in range(N_WORLDS):
                ys = [2026] + list(rng.choice(pool, 2))
                e_hl, e_c0 = [], []
                for y in ys:
                    d = drv[y]
                    truth = rate_c + k * gamma * d * clip_c + rng.normal(0, s_u / 100)
                    official = max(0.0, area(truth) + rng.normal(0, SIGMA_AREA))
                    hl = area(rate_c + gamma * d * clip_c)
                    e_hl.append(abs(hl - official))
                    e_c0.append(abs(c0 - official))
                w = sum(a < b for a, b in zip(e_hl, e_c0, strict=True))
                wins += w
                passes += (w >= 2) and (np.mean(e_hl) < np.mean(e_c0))
            out["cells"][f"s_u={s_u},k={k}"] = {"pass_rate": passes / N_WORLDS,
                                                "mean_years_won": wins / N_WORLDS}
    power = out["cells"][f"s_u={CENTRAL},k=1.0"]["pass_rate"]
    false = out["cells"][f"s_u={CENTRAL},k=0.0"]["pass_rate"]
    out["decision"] = {"power_k1": power, "false_pass_k0": false,
                       "rule": f"gating iff power >= {POWER_MIN} and false pass <= {FALSE_MAX} "
                               f"at s_u = {CENTRAL} cm/yr",
                       "sl1_role": "gating" if power >= POWER_MIN and false <= FALSE_MAX
                       else "report_only"}
    return out


if __name__ == "__main__":
    res = run()
    d = os.path.join(DATAPREP, "reports", "SL", "power")
    path = os.path.join(d, "result.json")
    if os.path.exists(path) and "--force" not in sys.argv:
        raise SystemExit(f"{path} exists: the power study is run once")
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
