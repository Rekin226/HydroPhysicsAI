"""Check station_et0.fao56 on FAO-56 Example 18 (Allen et al. 1998, Brussels, 6 July).

Inputs as published: latitude 50 deg 48' N, altitude 100 m, Tmax 21.5 C, Tmin 12.3 C,
ea 1.409 kPa (given here as the equivalent dew point), wind 10 km/h at 10 m (u2 2.078 m/s),
Rs 22.07 MJ m-2 d-1, P 100.1 kPa. Published ET0: 3.9 mm/d.

Usage: python fao56_check.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import pandas as pd
from station_et0 import fao56
from wcommon import write_json_new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ea = 1.409
    x = math.log(ea / 0.6108)
    td = 237.3 * x / (17.27 - x)
    df = pd.DataFrame({"tmax": [21.5], "tmin": [12.3], "td": [td], "u": [10 / 3.6],
                       "p_hpa": [1001.0], "rs": [22.07]},
                      index=pd.DatetimeIndex(["2026-07-06"]))   # day of year 187
    et0 = float(fao56(df, 50 + 48 / 60, 100.0).iloc[0])
    out = {"case": "FAO-56 Example 18 (Brussels, 6 July)", "published_mm_d": 3.9,
           "computed_mm_d": et0, "dew_point_for_ea_1.409_C": td,
           "abs_diff_mm_d": abs(et0 - 3.9), "pass_within_0.05": abs(et0 - 3.9) < 0.05}
    write_json_new(a.out, out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
