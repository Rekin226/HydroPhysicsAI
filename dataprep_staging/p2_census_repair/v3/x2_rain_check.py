"""Design-time corroboration (look [P2-r3-explore-rain], logged before this runs): monthly
fan-mean rain June-September 2012-2025 on each side of the Choushui (y = 2,640,000 m TWD97),
from the CWA CODIS rebuild of the 26 model gauges. No product value depends on this script.

Usage: python -I x2_rain_check.py <out_json>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from p3lib import L  # noqa: E402

RAIN = L.REPO / "data_fetch_api" / "new_measurements" / "cwa_codis_rain" / "rf_timeseries_codis.csv"
ST = L.REPO / "chou-shui-data" / "data" / "rf_stations.csv"
Y_SPLIT = 2_640_000.0


def main(out_json: str) -> None:
    r = pd.read_csv(RAIN, parse_dates=["date time"]).set_index("date time")
    st = pd.read_csv(ST, encoding="utf-8-sig").set_index("rf_id")
    side = {g: ("changhua" if st.loc[g, "TM_Y97"] >= Y_SPLIT else "yunlin") for g in r.columns
            if g in st.index}
    m = r.resample("MS").sum(min_count=20)                  # month needs >= 20 valid days
    out = {"n_gauges": {s: sum(1 for v in side.values() if v == s) for s in ("changhua",
                                                                            "yunlin")}}
    for s in ("changhua", "yunlin"):
        cols = [g for g, v in side.items() if v == s]
        fm = m[cols].mean(axis=1)
        tab = {}
        for y in range(2012, 2026):
            tab[str(y)] = {mo: round(float(fm.loc[f"{y}-{mo}-01"]), 1)
                           for mo in ("06", "07", "08", "09")}
            tab[str(y)]["jul_aug"] = round(tab[str(y)]["07"] + tab[str(y)]["08"], 1)
        ja = pd.Series({y: v["jul_aug"] for y, v in tab.items()})
        out[s] = {"monthly_mm": tab, "jul_aug_rank_desc": ja.rank(ascending=False)
                  .astype(int).to_dict(), "jul_aug_median_2012_2024": float(ja.loc[:"2024"]
                                                                             .median())}
    L.save_json(out, out_json)
    for s in ("changhua", "yunlin"):
        print(s, {y: (v["jul_aug"], out[s]["jul_aug_rank_desc"][y])
                  for y, v in out[s]["monthly_mm"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
