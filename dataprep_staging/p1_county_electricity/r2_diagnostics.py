"""WP P1 round 2: POST HOC diagnostics (designed after seeing the round-2 results; nothing
here is a pre-registered test, a gate or a threshold).

1. Why T10 failed: the census-to-TPC ratio over July-June years (cleaned and dedup census
   against TPC d007012 農林漁牧, both counties) and the year-on-year ratio of meters with a
   non-zero reading (is the census complete in the new months?).
2. Why C9 is not identical: pump-month rows present in the new Yunlin sheet but absent from
   the AMP_V2 cache (and the reverse), and value differences on common rows, 2007-01..2025-06.
   Pump ids are counted, never written out.

Usage: python -I r2_diagnostics.py <truth_dir> <out_json>   (inside tools/cpu_slot.sh)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib  # noqa: E402

REPO = HERE.parents[1]
L0N = REPO / "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/parquet"
AMP_KWH = REPO / "AMP_V2/data/pump_kwh_all.parquet"
TPC = REPO / "data_fetch_api/new_measurements/tpc_d007012/2026-10-12/d007012.csv"


def ratio_table(tm: pd.DataFrame) -> dict:
    t = lib.read_tpc_d007012(TPC)
    ag = t[t.sector == lib.AG].pivot(index="date", columns="county", values="kwh") / 1e6
    out = {}
    for c, cn in (("changhua", lib.CH), ("yunlin", lib.YL)):
        x = tm[(tm.county == c) & (tm.variant == "full") & (tm.scope == "county")]
        cl = x[x.series == "clean"].groupby("date")["gwh"].sum()
        de = x[x.series == "dedup"].groupby("date")["gwh"].sum()
        rows = {}
        for y in range(2016, 2026):
            w = (f"{y}-07-01", f"{y + 1}-06-01")
            tp = float(ag[cn].loc[w[0]:w[1]].sum())
            rows[f"{y}/{y + 1}"] = {"census_clean_GWh": float(cl.loc[w[0]:w[1]].sum()),
                                   "census_dedup_GWh": float(de.loc[w[0]:w[1]].sum()),
                                   "tpc_ag_GWh": tp,
                                   "clean_over_tpc": float(cl.loc[w[0]:w[1]].sum()) / tp}
        out[c] = rows
    return out


def meter_yoy(mc: pd.DataFrame) -> dict:
    out = {}
    for c in ("changhua", "yunlin"):
        x = mc[mc.county == c].set_index("date")["meters_nonzero"].sort_index()
        r = (x / x.shift(12)).dropna()
        out[c] = {str(k): float(v) for k, v in r.groupby(r.index.year).median().items()}
    return out


def c9_rows() -> dict:
    a = pd.read_parquet(AMP_KWH, columns=["datetime", "electricity_kwh", "pump"])
    a["datetime"] = pd.to_datetime(a["datetime"])
    a = a[a["datetime"] <= "2025-06-01"]
    a["pump"] = a["pump"].astype(str)
    c = pd.read_parquet(L0N / "yunlin_census.parquet", columns=["row_idx", "WELL_NO"])
    sid = c["WELL_NO"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).to_numpy()
    b = pd.read_parquet(L0N / "yunlin_kwh_long.parquet")
    b["datetime"] = pd.to_datetime(b["datetime"])
    b = b[b["datetime"] <= "2025-06-01"]
    b["pump"] = sid[b["row_idx"].to_numpy()]
    m = a.merge(b[["datetime", "pump", "electricity_kwh"]], on=["datetime", "pump"],
                how="outer", suffixes=("_amp", "_new"), indicator=True)
    both = m[m["_merge"] == "both"]
    va, vb = both["electricity_kwh_amp"], both["electricity_kwh_new"]
    diff = (va - vb).abs()
    only_new = m[m["_merge"] == "right_only"]
    only_amp = m[m["_merge"] == "left_only"]
    nan_mis = int((va.isna() != vb.isna()).sum())
    return {"rows_amp": int(len(a)), "rows_new": int(len(b)), "rows_common": int(len(both)),
            "rows_only_in_new": int(len(only_new)),
            "rows_only_in_new_nonzero": int((only_new["electricity_kwh_new"].fillna(0) != 0)
                                            .sum()),
            "kwh_only_in_new_GWh": float(only_new["electricity_kwh_new"].sum() / 1e6),
            "months_with_rows_only_in_new": int(only_new["datetime"].nunique()),
            "rows_only_in_amp": int(len(only_amp)),
            "common_rows_value_diff_gt0": int((diff > 0).sum()),
            "common_rows_value_max_abs_kwh": float(diff.max()) if len(diff) else 0.0,
            "common_rows_nan_mismatch": nan_mis}


def main(truth_dir: str, out_json: str) -> int:
    td = Path(truth_dir)
    tm = pd.read_parquet(td / "truth_monthly.parquet")
    mc = pd.read_parquet(td / "meter_counts.parquet")
    out = {"status": "post hoc diagnostics (not pre-registered; no gate, threshold or "
                     "selection depends on them)",
           "census_over_tpc_jul_jun": ratio_table(tm),
           "meters_nonzero_yoy_median_by_year": meter_yoy(mc),
           "c9_amp_vs_new_yunlin_rows_le_2025_06": c9_rows()}
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["c9_amp_vs_new_yunlin_rows_le_2025_06"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
