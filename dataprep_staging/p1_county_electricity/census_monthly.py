"""WP P1 step 0: Yunlin census monthly electricity by purpose class, in model units.

Model units (plan header): kWh of the de-duplicated, duty-capped Yunlin census
(``hydrophysics.twin.pumping.clean_census``, cap 1.0, duty statistic over 2012-01..2022-12,
the twin's calibration window), then the full record allocated over the kept pumps (as the
forcing experiment does). The ``dedup`` series (cap None) is kept for the coverage tables.

Two variants:

* ``full``      every census month (2007-01..2025-07; July 2025 is partial).
* ``preorigin`` the kWh table is truncated to months <= 2022-12 *before* cleaning, so no
                post-origin census value can touch it (U3). The duty window is 2012-2022 in
                both variants, so the two must agree on every month <= 2022-12; the script
                checks that and records the result.

Privacy: the census columns 戶名1 (name) and 用電地址2 (address) are never read; columns are
selected explicitly. Outputs are county x class x month totals only (no pump rows,
coordinates or meter ids).

Usage: python census_monthly.py <out_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hydrophysics.twin.pumping import clean_census  # noqa: E402
from hydrophysics.twin.scenario import purpose_class  # noqa: E402

PUMPS = REPO / "AMP_V2/data/tpc_pumps.parquet"
KWH = REPO / "AMP_V2/data/pump_kwh_all.parquet"
#: explicit column list; never 戶名1 or 用電地址2 (plan 2.6)
PUMP_COLS = ["sid", "COUNTRYCITY", "TOWN", "PUMP_HP", "PURPOSE", "電號1", "每期用電度數1"]
KWH_COLS = ["datetime", "electricity_kwh", "pump"]
DUTY_T0, DUTY_T1 = "2012-01-01", "2023-01-01"   # twin calibration window 2012-01..2022-12
ORIGIN_END = pd.Timestamp("2023-01-01")


def monthly_by_class(pumps: pd.DataFrame, kwh: pd.DataFrame) -> pd.DataFrame:
    """GWh per month per class for the dedup and the model-cleaned census."""
    _, kd, rep_d = clean_census(pumps, kwh, cap_duty=None)
    kept, _, rep_c = clean_census(pumps, kwh, cap_duty=1.0, t0=DUTY_T0, t1=DUTY_T1)
    _, kc, _ = clean_census(kept, kwh, cap_duty=None)
    cls = pumps.set_index("sid")["PURPOSE"].map(purpose_class)
    bill = pumps.set_index("sid")["每期用電度數1"].astype(str)
    parts = []
    for name, k in (("dedup", kd), ("clean", kc)):
        k = k.copy()
        k["cls"] = k["pump"].map(cls)
        k["bill"] = k["pump"].map(bill)
        p = k.pivot_table(index="datetime", columns="cls", values="electricity_kwh",
                          aggfunc="sum") / 1e6
        p.columns = [f"{name}_{c}" for c in p.columns]
        parts.append(p)
        b = k.pivot_table(index="datetime", columns="bill", values="electricity_kwh",
                          aggfunc="sum") / 1e6
        b.columns = [f"{name}_bill_{c}" for c in b.columns]
        parts.append(b)
    out = pd.concat(parts, axis=1).sort_index()
    out.index = pd.DatetimeIndex(out.index, name="date")
    out = out.fillna(0.0)
    classes = sorted({c.split("_", 1)[1] for c in out.columns
                      if c.startswith("clean_") and "bill" not in c})
    out["clean_total"] = out[[f"clean_{c}" for c in classes]].sum(axis=1)
    out["dedup_total"] = out[[f"dedup_{c}" for c in classes if f"dedup_{c}" in out]].sum(axis=1)
    out["dedup_agri"] = out[[f"dedup_{c}" for c in ("irrigation", "aquaculture", "livestock")]
                            ].sum(axis=1)
    out["clean_agri"] = out[[f"clean_{c}" for c in ("irrigation", "aquaculture", "livestock")]
                            ].sum(axis=1)
    out["clean_other_classes"] = out["clean_total"] - out["clean_irrigation"]
    return out, rep_c


def main(out_dir: str) -> int:
    t0 = time.time()
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    pumps = pd.read_parquet(PUMPS, columns=PUMP_COLS)
    pumps["sid"] = pumps["sid"].astype(str)
    counties = pumps["COUNTRYCITY"].value_counts().to_dict()
    kwh = pd.read_parquet(KWH, columns=KWH_COLS)
    kwh["pump"] = kwh["pump"].astype(str)
    kwh["datetime"] = pd.to_datetime(kwh["datetime"])
    info = {"pumps": int(len(pumps)), "counties": counties, "kwh_rows": int(len(kwh)),
            "kwh_first": str(kwh["datetime"].min().date()),
            "kwh_last": str(kwh["datetime"].max().date())}
    res = {}
    for variant, k in (("full", kwh), ("preorigin", kwh[kwh["datetime"] < ORIGIN_END])):
        m, rep = monthly_by_class(pumps, k)
        m.to_parquet(od / f"census_class_month_{variant}.parquet")
        rep.to_csv(od / f"clean_report_{variant}.csv")
        res[variant] = m
        info[f"{variant}_months"] = [str(m.index.min().date()), str(m.index.max().date()),
                                     int(len(m))]
    # the two variants must agree on every month <= 2022-12 (cleaning is pre-origin in both)
    a = res["full"].loc[: "2022-12-01"]
    b = res["preorigin"].reindex(columns=a.columns).loc[a.index]
    diff = (a - b).abs().to_numpy()
    rel = diff / np.maximum(a.abs().to_numpy(), 1e-12)
    info["preorigin_vs_full_max_abs_GWh"] = float(np.nanmax(diff))
    info["preorigin_vs_full_max_rel"] = float(np.nanmax(rel[a.abs().to_numpy() > 1e-9]))
    info["columns_only_in_full"] = sorted(set(res["full"].columns) - set(res["preorigin"].columns))
    info["seconds"] = round(time.time() - t0, 1)
    (od / "census_monthly_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
