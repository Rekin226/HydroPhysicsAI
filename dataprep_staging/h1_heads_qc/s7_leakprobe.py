"""H1a leakage probe (acceptance test 6): the product's pre-origin rows must not depend on any
post-origin record. The chain is rerun with ``--pre-only`` (stage 1 never opens the 2023-26
network files; stage 2 and 3 see no post-origin sample), and its pre-origin rows are compared
with the full build's: heads_daily and heads_monthly (period 'pre'), events dated <= 2022-12
and the barometric table. Criterion: 0 differing rows.

Usage: s7_leakprobe.py BUILD_FULL BUILD_PREONLY OUT_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _eq(a: pd.Series, b: pd.Series) -> np.ndarray:
    if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
        av, bv = a.astype(float).to_numpy(), b.astype(float).to_numpy()
        return (np.isnan(av) & np.isnan(bv)) | (np.abs(av - bv) <= 1e-9)
    return (a.astype(str).to_numpy() == b.astype(str).to_numpy())


def compare(A: pd.DataFrame, B: pd.DataFrame, keys: list[str], cols: list[str]) -> dict:
    m = A.merge(B, on=keys, how="outer", suffixes=("_full", "_pre"), indicator=True)
    only = m[m["_merge"] != "both"]
    both = m[m["_merge"] == "both"]
    diff = np.zeros(len(both), bool)
    per_col = {}
    for c in cols:
        e = _eq(both[f"{c}_full"], both[f"{c}_pre"])
        per_col[c] = int((~e).sum())
        diff |= ~e
    ex = both[diff].head(10)
    return {"n_full": int(len(A)), "n_pre": int(len(B)), "n_only_one_side": int(len(only)),
            "n_differing": int(diff.sum()), "per_column": per_col,
            "examples": ex.astype(str).to_dict("records"),
            "only_examples": only.head(10).astype(str).to_dict("records")}


def main():
    full, pre, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    res = {}
    D1 = pd.read_parquet(full / "heads_daily.parquet")
    D2 = pd.read_parquet(pre / "heads_daily.parquet")
    res["heads_daily"] = compare(D1[D1.period == "pre"], D2[D2.period == "pre"], ["sid", "date"],
                                 ["value", "flag", "day_valid", "n_samples", "n_hours",
                                  "value_raw", "baro_m", "corr_m", "kinds"])
    M1 = pd.read_parquet(full / "heads_monthly.parquet")
    M2 = pd.read_parquet(pre / "heads_monthly.parquet")
    # wells without any pre-origin record have all-8 pre rows in the full build only
    rec2 = set(M2.sid)
    res["heads_monthly"] = compare(M1[(M1.period == "pre") & M1.sid.isin(rec2)],
                                   M2[M2.period == "pre"], ["sid", "date"],
                                   ["value", "flag", "c20", "n_clean_days", "sigma_obs",
                                    "value_raw", "baro_m", "corr_m"])
    E1 = pd.read_csv(full / "events.csv", dtype={"sid": str})
    E2 = pd.read_csv(pre / "events.csv", dtype={"sid": str})
    E1 = E1[E1.date <= "2022-12-31"]
    E2 = E2[E2.date <= "2022-12-31"]
    k = ["sid", "kind", "t_start", "t_end"]
    E1 = E1.assign(_n=E1.groupby(k).cumcount())
    E2 = E2.assign(_n=E2.groupby(k).cumcount())
    res["events_preorigin"] = compare(E1, E2, k + ["_n"], ["flag", "j1", "applied", "backing",
                                                          "neighbour_verdict"])
    B1 = pd.read_csv(full / "barometric_efficiency.csv", dtype={"sid": str})
    B2 = pd.read_csv(pre / "barometric_efficiency.csv", dtype={"sid": str})
    B1 = B1[B1.date <= "2022-12-31"]
    B2 = B2[B2.date <= "2022-12-31"]
    res["barometric_efficiency"] = compare(B1, B2, ["sid", "epoch"], ["status", "A_K", "se_A_K",
                                                             "p_ref_hpa", "med_ref", "mad_ref"])
    n_rows = sum(v["n_full"] for v in res.values())
    n_diff = sum(v["n_differing"] + v["n_only_one_side"] for v in res.values())
    out = {"test": "H1a.leakprobe", "n_rows_compared": n_rows, "n_diff_total": n_diff,
           "tables": res}
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("examples",
                                                                        "only_examples")}
                      for k, v in res.items()}, indent=1))
    print("n_diff_total", n_diff)


if __name__ == "__main__":
    main()
