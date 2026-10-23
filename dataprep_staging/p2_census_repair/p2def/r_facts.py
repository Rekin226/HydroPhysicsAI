"""Descriptive facts for the P2def report, computed after every registered test was scored and
the products were published (logged as an after-the-fact look). No decision depends on them.

Usage: python -I r_facts.py <out_json>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
import looks_def as LK  # noqa: E402
import r_ledger as RL  # noqa: E402
from common import guard  # noqa: E402


def main(out: str) -> None:
    LK.record(
        "P2def-r-facts",
        units={"P2def:r_products_facts": "published P2def products and the C0/P2abc census "
                                          "totals they are compared with (2016-2026)"},
        scope="c0_pump_electricity_v3:2016-01..2026-08", purpose="after_the_fact",
        note="Report facts after every P2def registered test was scored and the products were "
             "published: layer sizes by 2022/2025 against the census, the 2026-07 census-based "
             "value of P2abc full v2 against the extension. No decision depends on them.")
    res = {}
    # P2d sizes
    for v in ("preorigin", "full"):
        pv = RL.latest(f"census_post2016/{v}")
        p = guard.open_product(D.DATAPREP / pv, "after_the_fact_analysis")
        t = pd.read_parquet(p.path("county_class_monthly.parquet"))
        t["year"] = t["date"].str[:4].astype(int)
        c0 = guard.open_product(D.C0[v], "after_the_fact_analysis")
        cm = pd.read_parquet(c0.path("county_class_monthly.parquet"))
        cm["year"] = cm["date"].str[:4].astype(int)
        sizes = {}
        for k in D.COUNTIES:
            for m in ("low", "high"):
                s = t[(t["county"] == k) & (t["member"] == m)]
                row = {}
                for y in ((2022,) if v == "preorigin" else (2022, 2025)):
                    sy = s[s["year"] == y]
                    dec = s[s["date"] == f"{y}-12"]
                    cen = cm[(cm["county"] == k) & (cm["scope"] == "in_grid") &
                             (cm["class"] == "all") & (cm["year"] == y)]["value"].sum()
                    row[str(y)] = {
                        "n_county_dec": float(dec["n_county"].sum()),
                        "n_in_grid_dec": float(dec["n_in_grid"].sum()),
                        "GWh_county": float(sy["value_county_kwh"].sum() / 1e6),
                        "GWh_in_grid": float(sy["value_in_grid_kwh"].sum() / 1e6),
                        "census_in_grid_GWh": float(cen / 1e6),
                        "share_of_census_in_grid": float(sy["value_in_grid_kwh"].sum() / cen)
                        if cen > 0 else None}
                sizes[f"{k}.{m}"] = row
        res[f"p2d_sizes.{v}"] = sizes
    # P2e: 2026-07 census-based value (P2abc full v2, all arms, any flag) vs the extension
    pv = RL.latest("census_extension/full")
    ext = pd.read_csv(D.DATAPREP / pv / "county_extension.csv")
    p2 = guard.open_product(D.P2ABC["full"], "after_the_fact_analysis")
    cc = pd.read_parquet(p2.path("county_class_monthly.parquet"))
    cmp = {}
    for k in D.COUNTIES:
        for sc in ("in_grid", "county"):
            r = cc[(cc["county"] == k) & (cc["scope"] == sc) & (cc["class"] == "all") &
                   (cc["date"].isin(["2025-07", "2026-06", "2026-07"]))]
            cmp[f"{k}.{sc}"] = r[["date", "arm", "value", "flag"]].assign(
                GWh=lambda x: x["value"] / 1e6).drop(columns="value").round(4).to_dict(
                orient="records")
        e = ext[ext["county"] == k]
        cmp[f"{k}.extension_in_grid_GWh"] = {m: float(x / 1e6) for m, x in
                                             zip(e["month"], e["in_grid_extension_kwh"],
                                                 strict=True)}
    res["p2e_2026_07_comparison"] = cmp
    # the census 2025-07..2026-06 (county scope) for scale
    c0f = guard.open_product(D.C0["full"], "after_the_fact_analysis")
    cm = pd.read_parquet(c0f.path("county_class_monthly.parquet"))
    for k in ("yunlin", "changhua"):
        s = cm[(cm["county"] == k) & (cm["scope"] == "county") & (cm["class"] == "all")]
        res[f"census_county_GWh.{k}"] = {d: float(v / 1e6) for d, v in zip(
            s["date"], s["value"], strict=True) if d >= "2025-06"}
    # P2f extra: sites with non-negative layer-2 slope (primary)
    lr = D.DATAPREP / RL.latest("layer_response")
    st = pd.read_parquet(lr / "site_ratios.parquet")
    res["p2f_sites_primary"] = {f"{s}.{h}": int(len(st[(st["scope"] == s) & (st["heads"] == h)]))
                                for s in ("yunlin", "changhua", "pooled")
                                for h in ("obs", "cwa_forward")}
    o = st[(st["scope"] == "yunlin") & (st["heads"] == "obs")]
    res["p2f_yunlin_obs_b2_nonneg_sites"] = int((o["b_l2"] >= 0).sum())
    res["p2f_yunlin_obs_ratio_quartiles"] = [float(x) for x in
                                             np.quantile(o["ratio"], [0.25, 0.5, 0.75])]
    D.save_json(res, out)
    print({k: v for k, v in res.items() if k.startswith("p2d")})
    print(res["p2e_2026_07_comparison"])


if __name__ == "__main__":
    main(sys.argv[1])
