"""Fit the X5 timing model on the calibration years 2015-2021 only (design-time).

Reads W1 pre-origin/v2 monthly district values for the calibration years only (rows of other
years are dropped on read), fits the Yunlin calendar and b (T3 primary, T2 alternative), the
Changhua calendar (W1 Changhua shares, same years), the leave-one-year-out share sd, and
writes results/dataprep/x5_changhua_water/fit/fit.json once.

Usage: python -I fit_x5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_changhua_water import pipeline as P  # noqa: E402
from x5_changhua_water import x5const as K  # noqa: E402
from x5_changhua_water import x5lib as L  # noqa: E402

DISTRICT = {"chc": "changhua", "yl": "yunlin"}


def load_w1(path: Path, district: str, years) -> pd.Series:
    """W1 monthly district values (10^8 m3) for ``years`` only."""
    d = pd.read_csv(path / "district_monthly.csv", usecols=["date", "district", "value"])
    d = d[d["district"] == DISTRICT[district]]
    per = pd.PeriodIndex(pd.to_datetime(d["date"]), freq="M")
    keep = np.isin(per.year, list(years))
    return pd.Series(d["value"].to_numpy()[keep], index=per[keep]).sort_index()


def main() -> int:
    out = K.FIT_DIR / "fit.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the fit runs once")
    rec = {"schema": "x5-fit/1", "prereg": K.PREREG_ID, "calib_years": K.CALIB_YEARS,
           "label": "design-time fit on the calibration years; W1 reference is inferred "
                    "(flag 5)", "models": {}, "standardisation": {}}
    z = {}
    for d in ("yl", "chc"):
        z[d], rec["standardisation"][d] = L.standardise(L.indicators(d), K.CALIB_YEARS)
    ref_yl = load_w1(K.W1_PRE, "yl", K.CALIB_YEARS)
    ref_chc = load_w1(K.W1_PRE, "chc", K.CALIB_YEARS)
    if sorted(set(ref_yl.index.year)) != K.CALIB_YEARS:
        raise SystemExit("calibration years missing from W1")
    for name, keys in (("T3", K.INDICATORS_T3), ("T2", K.INDICATORS_T2)):
        f = P.fit(ref_yl, z["yl"], keys)
        rec["models"][name] = {"keys": list(keys), "b": f["b"],
                               "loyo_share_sd_by_month": P.loyo_share_sd(
                                   ref_yl, z["yl"], keys).tolist()}
    rec["calendar"] = {"yl": L.calendar(L.ref_shares(ref_yl), K.CALIB_YEARS).tolist(),
                       "chc": L.calendar(L.ref_shares(ref_chc), K.CALIB_YEARS).tolist()}
    K.FIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({"b": {k: v["b"] for k, v in rec["models"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
