"""Metered pumping response to the spring shortfalls: Jan-May in-grid census kWh per county.

Reads ``pump_electricity/preorigin/v3`` (WP C0, gate passed) through the loader guard as
``design_time_evidence`` (a pre-origin use: the numbers describe the interventions the N1
tests are designed around). Values are model units (kWh of the cleaned census); nothing is
filled here. Event years 2015, 2020, 2021 are compared with the mean of 2016-2019 (the W1
benchmark's reference years) and with the mean of the N1 reference years.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402

from n1_natural_experiments import n1const as K  # noqa: E402

USE = "design_time_evidence"


def janmay_kwh() -> tuple[pd.DataFrame, list[str]]:
    p = guard.open_product(K.C0_PRE, USE)
    df = p.read_table("county_class_monthly.parquet")
    df = df[(df.scope == "in_grid") & df.county.isin(["changhua", "yunlin"])
            & df["class"].isin(["irrigation", "all", "all_nodomestic"])].copy()
    df["year"] = df.date.str[:4].astype(int)
    df["month"] = df.date.str[5:7].astype(int)
    jm = df[df.month.le(5) & df.year.between(2012, 2022)]
    flags = jm.groupby(["county", "class", "year"]).flag.max().rename("max_flag")
    out = jm.groupby(["county", "class", "year"]).value.sum().rename("janmay_kwh").to_frame()
    out = out.join(flags).reset_index()
    return out, list(p.notes)


def event_summary(jm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cty, cls), g in jm.groupby(["county", "class"]):
        s = g.set_index("year").janmay_kwh
        ref_w1 = s.loc[[2016, 2017, 2018, 2019]].mean()
        ref_n1 = s.loc[K.REFERENCE_YEARS].mean()
        for ev, e in K.EVENTS.items():
            y = e["year"]
            rows.append({"county": cty, "class": cls, "event": ev, "year": y,
                         "janmay_kwh": float(s.loc[y]), "ref_2016_2019_kwh": float(ref_w1),
                         "ratio_to_2016_2019": float(s.loc[y] / ref_w1),
                         "ref_n1_years_kwh": float(ref_n1),
                         "ratio_to_n1_ref": float(s.loc[y] / ref_n1),
                         "max_flag": int(g.set_index("year").max_flag.loc[y])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    jm, notes = janmay_kwh()
    pd.set_option("display.width", 200)
    print(event_summary(jm).round(3).to_string())
    print(np.round(jm.pivot_table(index="year", columns=["county", "class"],
                                  values="janmay_kwh") / 1e6, 2))
