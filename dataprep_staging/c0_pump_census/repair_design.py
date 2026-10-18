"""WP C0 repair round: design-time look on PRE-ORIGIN data only (L0 kWh truncated to <= 2022-12
right after reading), logged in the A1 look log before it runs.

It answers three design questions for pump_electricity/preorigin/v2 without comparing anything
with a reference and without reading any month after 2022-12:

1. Billing code x detected phase (2012-2022) cross-tab of every meter: which meters have a
   December value that is half of a December-January bill (the "Dec-Jan-paired" set)?
2. How many meters change kept status when the December-2022 value of the Dec-Jan-paired
   meters is left out of the duty statistic (two candidate rules)?
3. How many Dec-Jan-paired kept meters lack an Oct-Nov 2022 reading (the base of the fill)?

No ratio, fill value or back-test number is computed here (the fill and its sd model are fixed
blind and back-tested under the repair pre-registration).

Usage: python -I repair_design.py     (inside tools/cpu_slot.sh)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

OUT = lib.EVIDENCE / "repair" / "design_look.json"
T_PRE = lib.tidx("2022-12") + 1
J_DEC22 = lib.tidx("2022-12")


def county(tag: str, grid) -> dict:
    df = lib.harmonise(tag, lib.read_census_l0(tag), grid)
    _, L = lib.read_kwh_l0(tag, df["sid"].to_numpy(), len(df))
    L = L[:, :T_PRE].copy()                       # pre-origin only, from here on
    first = df["meter_first_pump"].to_numpy()
    midx = df.loc[first, "meter_idx"].to_numpy()
    X = L[first]
    hp = lib.hp_numeric(df["PUMP_HP"])
    hp_m = pd.Series(hp).groupby(df["meter_idx"].to_numpy()).sum().loc[midx].to_numpy()
    code = lib.phase_from_code(df.loc[first, "billing_code"].to_numpy())
    det = lib.detect_phase(X, lib.tidx("2012-01"), T_PRE)["phase_detected"].to_numpy()
    duty0 = lib.meter_duty(X, hp_m)
    kept0 = ~(duty0 > lib.CAP_DUTY)
    rules = {
        "R_code": (code == "B") | ((code == "unknown") & (det == "B")),
        "R_det": (det == "B") | ((code == "B") & np.isin(det, ["mixed", "insufficient",
                                                               "both-flat"])),
    }
    out = {"meters": int(len(midx)), "meters_kept_v1_rule": int(kept0.sum()),
           "crosstab_all_meters": pd.crosstab(code, det).to_dict(),
           "crosstab_kept_meters": pd.crosstab(code[kept0], det[kept0]).to_dict()}
    for name, sel in rules.items():
        Xr = X.copy()
        Xr[sel, J_DEC22] = np.nan
        d = lib.meter_duty(Xr, hp_m)
        kept = ~(d > lib.CAP_DUTY)
        base_ok = np.isfinite(X[:, J_DEC22 - 2]) & np.isfinite(X[:, J_DEC22 - 1])
        out[name] = {
            "meters_in_rule": int(sel.sum()), "kept_meters_in_rule": int((sel & kept0).sum()),
            "kept_flips_dropped_to_kept": int((kept & ~kept0).sum()),
            "kept_flips_kept_to_dropped": int((~kept & kept0).sum()),
            "kept_rule_meters_without_octnov_2022_base": int((sel & kept0 & ~base_ok).sum()),
            "kept_rule_meters_dec2022_nan_in_L0": int((sel & kept0 & np.isnan(X[:, J_DEC22]))
                                                      .sum()),
        }
    out["rule_difference_kept"] = {
        "in_R_code_not_R_det": int((rules["R_code"] & ~rules["R_det"] & kept0).sum()),
        "in_R_det_not_R_code": int((rules["R_det"] & ~rules["R_code"] & kept0).sum())}
    return out


def main() -> int:
    grid = lib.load_grid()
    res = {"note": "design-time look, pre-origin data only (L0 truncated to <= 2022-12); "
                   "no fill, ratio or reference comparison computed",
           "counties": {t: county(t, grid) for t in lib.COUNTIES}}
    lib.save_json(res, OUT)
    for t, r in res["counties"].items():
        print(t, {k: v for k, v in r.items() if not k.startswith("crosstab")})
        print(" crosstab kept:", r["crosstab_kept_meters"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
