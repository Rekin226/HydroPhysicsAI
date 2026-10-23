"""S1a r3: calibration constants of the length-aware b_deep error model (M1, M5).

Rules (declared in S1a.rings_bridged_rules.v3):

* ``sigma_x`` (m): the deep section's excess-shortening noise over a ~13-month gap. Solved so
  that the mean over the calibration site-windows of
  ``e^2 / (max(sd_jk, 10 ppm)^2 + (sigma_x / L_deep)^2)`` equals 1, where ``e`` is b_deep of
  the UNINJECTED surveys (true scale 0), ``sd_jk`` its jackknife sd and ``L_deep`` the deep
  length. Calibration site-windows: every (site, window) entering the registered (e) geometry
  (13-month masks 2015-01..2021-12, >= 20 / >= 18 surveys, >= 10 sites), i.e. ring surveys
  2014-05..2021-12 only, with a finite b_deep (``repair_r3/diag/truth_bdeep.csv``).
* ``sigma_c`` (m): the common-mode part, the sd over those windows of the window-mean excess
  (b_deep x L_deep).

The calibration uses uninjected surveys and never the (e) statistic; its data are the spent
(design-exposed) 2014-2021 surveys, which is why the re-test uses the pre-2012 record.

    python -m s1_rings.r3_calibrate         (from dataprep_staging/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

DIAG = K.EVIDENCE / "repair_r3" / "diag"
OUT = K.EVIDENCE / "repair_r3" / "calibration.json"


def solve_sigma_x(e, sdjk, L) -> float:
    """sigma_x (m) with mean(e^2 / (sdjk^2 + (sigma_x/L)^2)) = 1 (bisection; 0 if the
    jackknife alone already gives a mean <= 1)."""
    s0 = np.maximum(sdjk, K.SD_FLOOR)

    def f(sx):
        return float(np.mean(e ** 2 / (s0 ** 2 + (sx / L) ** 2))) - 1.0

    if f(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 0.1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    T = pd.read_csv(DIAG / "truth_bdeep.csv").dropna(subset=["b_deep_truth_ppm"])
    e = T.b_deep_truth_ppm.to_numpy() * 1e-6
    sd = T.sd_deep_ppm.to_numpy() * 1e-6
    ld = T.deep_len_m.to_numpy()
    sx = solve_sigma_x(e, sd, ld)
    sig = np.sqrt(np.maximum(sd, K.SD_FLOOR) ** 2 + (sx / ld) ** 2)
    z = e / sig
    ex = e * ld
    wm = pd.Series(ex).groupby(T.start.to_numpy()).mean()
    out = {
        "rule_sigma_x": "mean over calibration site-windows of e^2/(max(sd_jk,10ppm)^2 + "
                        "(sigma_x/L_deep)^2) = 1; e = b_deep of the uninjected surveys",
        "rule_sigma_c": "sd over calibration windows of the window-mean excess b_deep x L_deep",
        "calibration_data": "repair_r3/diag/truth_bdeep.csv: registered (e) geometry, ring "
                            "surveys 2014-05..2021-12, uninjected",
        "n_site_windows": int(len(T)), "n_windows": int(T.start.nunique()),
        "sigma_x_m": float(sx), "sigma_x_mm": round(1e3 * sx, 4),
        "sigma_c_m": float(wm.std()), "sigma_c_mm": round(1e3 * float(wm.std()), 4),
        "check_z_rms": round(float(np.sqrt(np.mean(z ** 2))), 4),
        "check_z_mean": round(float(np.mean(z)), 4), "check_z_sd": round(float(np.std(z)), 4),
        "check_z_rms_by_deep_length": {
            f"{lo}-{hi}": round(float(np.sqrt(np.mean(z[(ld > lo) & (ld <= hi)] ** 2))), 3)
            for lo, hi in ((0, 30), (30, 50), (50, 80), (80, 400))},
        "jackknife_only_z_rms": round(float(np.sqrt(np.mean((e / np.maximum(sd, K.SD_FLOOR))
                                                              ** 2))), 4),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
