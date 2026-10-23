"""WP S3 fix round of the S3r repair (2026-10-18): A3 and the after-the-fact twin scoring of the
gnss_clean full variant on MATCHED station sets across v2, v3 and v4 (verifier's major issue:
v3's A3 and twin numbers compared different site sets).

    python -I -B dataprep_staging/s3_gnss/v4/matched_sets.py V4_VALIDATION_FULL_JSON OUT_JSON

Reads the per-site tables already computed by each version's validation run (v2 and v3 from
their product ``validation.json``, v4 from its validation run): A3 (excess-rate change at
co-located rings, report only, no threshold) and the twin scores (frozen forward archive
against GNSS rates 2023-01..2025-03, R2 / RMSE / bias, same definition as ``validate.r2_rmse``).
For each version it reports the statistics on its own site set, on the sites common to all
three versions, and on the sites common to v2 and v4; and for every site its window
membership and flag-0/1 days in 2023-25. No new data are read except the product daily
tables (flags).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate as V  # noqa: E402  (byte-identical v3 copy; r2_rmse)

REPO = HERE.parents[2]
PROD = REPO / "results/dataprep/gnss_clean/full"


def load(v4_json: Path) -> dict:
    out = {}
    for v in ("v2", "v3"):
        out[v] = json.loads((PROD / v / "validation.json").read_text())["this_variant"]
    out["v4"] = json.loads(Path(v4_json).read_text())
    return out


def a3(vals: dict) -> dict:
    T = {v: pd.DataFrame(d["A3_excess_stability"]["table"]).set_index("gnss")
         for v, d in vals.items()}
    both = {v: set(t.index[t.excess_change_mm_yr.notna()]) for v, t in T.items()}
    sets = {"own": None, "common_v2_v3_v4": both["v2"] & both["v3"] & both["v4"],
            "common_v2_v4": both["v2"] & both["v4"]}
    res = {"sites_with_both_windows": {v: sorted(s) for v, s in both.items()}}
    for name, s in sets.items():
        res[name] = {}
        for v, t in T.items():
            if name == "common_v2_v4" and v == "v3":
                continue
            sub = t[t.excess_change_mm_yr.notna()]
            if s is not None:
                sub = sub.loc[sorted(s)]
            ch = sub.excess_change_mm_yr
            res[name][v] = {"n": int(len(sub)), "sites": sorted(sub.index),
                            "median_change_mm_yr": float(ch.median()) if len(ch) else None,
                            "median_abs_change_mm_yr": float(ch.abs().median()) if len(ch)
                            else None,
                            "median_pre_excess_mm_yr": float(sub.pre_excess_mm_yr.median())
                            if len(sub) else None,
                            "median_post_excess_mm_yr": float(sub.post_excess_mm_yr.median())
                            if len(sub) else None}
    res["per_site"] = {v: t[["pre_excess_mm_yr", "post_excess_mm_yr", "pre_n_months",
                             "post_n_months", "excess_change_mm_yr"]].round(2)
                       .reset_index().to_dict("records") for v, t in T.items()}
    res["note"] = ("A3 is report only (plan: no threshold); median_abs_change_mm_yr is the "
                   "product's own A3 statistic (validate.excess_stability)")
    return res


def twin(vals: dict, label: str = "raw_cleaned") -> dict:
    T = {v: pd.DataFrame(d["twin_scores_wra"][label]["table"]).set_index("station")
         for v, d in vals.items()}
    st = {v: set(t.index) for v, t in T.items()}
    sets = {"own": None, "common_v2_v3_v4": st["v2"] & st["v3"] & st["v4"],
            "common_v2_v4": st["v2"] & st["v4"]}
    res = {"stations": {v: sorted(s) for v, s in st.items()}}
    for name, s in sets.items():
        res[name] = {}
        for v, t in T.items():
            if name == "common_v2_v4" and v == "v3":
                continue
            sub = t if s is None else t.loc[sorted(s)]
            o = sub.obs_rate_cm_yr.to_numpy(float)
            res[name][v] = {"n": int(len(sub)),
                            "model_archive": V.r2_rmse(sub.model_rate_cm_yr.to_numpy(float), o),
                            "base_continue_2020_22": V.r2_rmse(
                                sub.base_continue_cm_yr.to_numpy(float), o)}
    # per-station obs rates of the common set: how much the cleaning itself moved them
    common = sorted(st["v2"] & st["v3"] & st["v4"])
    res["obs_rate_cm_yr_common"] = {k: {v: round(float(T[v].loc[k, "obs_rate_cm_yr"]), 3)
                                        for v in T} for k in common}
    return res


def flags(versions=("v2", "v3", "v4")) -> dict:
    out = {}
    for v in versions:
        d = pd.read_parquet(PROD / v / "wra" / "daily.parquet", columns=["station", "date", "flag"])
        d = d[(d.date >= "2023-01-01") & (d.date <= "2025-03-31")]
        out[v] = d.groupby("station").flag.apply(lambda f: int(f.isin([0, 1]).sum())).to_dict()
    return out


def main(argv: list[str]) -> int:
    vals = load(Path(argv[0]))
    res = {"A3": a3(vals), "twin_raw_cleaned": twin(vals, "raw_cleaned"),
           "twin_cm_removed_wra_hf": twin(vals, "cm_removed_wra_hf"),
           "twin_cm_removed_ngl_offfan": twin(vals, "cm_removed_ngl_offfan")}
    if len(argv) > 2 and argv[2] == "--with-v4-product":
        res["flag01_days_2023_25"] = flags()
    Path(argv[1]).write_text(json.dumps(res, indent=1, default=str))
    for k in ("own", "common_v2_v3_v4", "common_v2_v4"):
        print("A3", k, {v: (x["n"], None if x["median_change_mm_yr"] is None else
                            round(x["median_change_mm_yr"], 2),
                            None if x["median_abs_change_mm_yr"] is None else
                            round(x["median_abs_change_mm_yr"], 2))
                        for v, x in res["A3"][k].items()})
    for k in ("own", "common_v2_v3_v4", "common_v2_v4"):
        print("twin", k, {v: (x["n"], round(x["model_archive"]["r2"], 3),
                              round(x["model_archive"]["rmse"], 3),
                              round(x["model_archive"]["bias"], 3))
                          for v, x in res["twin_raw_cleaned"][k].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
