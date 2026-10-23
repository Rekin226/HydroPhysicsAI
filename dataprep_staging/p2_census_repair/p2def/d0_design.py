"""Design-time looks of WP P2def, run before the P2def registration (each logged first):

* ``census``: census_L1 v3 attributes of kept pumps per county - install-month validity,
  townships, install-year counts for years <= 2012 and from 2016-07 on only (the 2013 surge
  and the back-test truth window 2014-01..2016-06 are not counted here);
* ``billing``: billing-group completeness of C0 full v3 2016-01..2026-07 per county and
  billing code (missing / zero / positive readings of installed kept meters and their kWh),
  the check WP P1 asked for before the T10 error is attributed to TPC drift;
* ``tpc``: months present in the L0 TPC d007012 farm-sector rows of Yunlin and Changhua.

Usage: python -I d0_design.py census|billing|tpc <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
import looks_def as LK  # noqa: E402


def census(out: Path) -> None:
    LK.record("P2def-d0-census-design")
    l1 = D.l1_table(use="prior")
    res = {}
    for k in D.COUNTIES:
        s = l1[(l1["county"] == k) & l1["kept"].astype(bool)]
        inst = np.where(s["install_valid"].fillna(False).astype(bool),
                        D.month_index(s["install_month"]), -10**6)
        iy = D.install_year(inst)
        mo = np.where(inst > -10**5, inst % 12 + 1, 0)
        keep_y = (iy <= 2012) | (iy >= 2017) | ((iy == 2016) & (mo >= 7))
        cnt = pd.Series(iy[keep_y & (iy > 0)]).value_counts().sort_index()
        h2_2016 = int(((iy == 2016) & (mo >= 7)).sum())
        win = (iy >= 2007) & (iy <= 2012)
        res[k] = {
            "kept_pumps": int(len(s)),
            "kept_in_grid": int((s["cell"] >= 0).sum()),
            "install_valid_share": float(s["install_valid"].fillna(False).mean()),
            "town_missing": int(s["town"].isna().sum()),
            "n_towns": int(s["town"].nunique()),
            "n_towns_with_2007_2012_installs": int(s.loc[win, "town"].nunique()),
            "installs_by_year_le2012_ge2017": {int(y): int(n) for y, n in cnt.items()
                                               if y <= 2012 or y >= 2017},
            "installs_2016_h2": h2_2016,
            "installs_2007_2012_by_class": s.loc[win, "purpose_class"].value_counts().to_dict(),
            "classes_kept": s["purpose_class"].value_counts().to_dict(),
        }
    D.save_json(res, out / "d0_census_design.json")
    for k, v in res.items():
        print(k, {a: b for a, b in v.items() if a != "installs_by_year_le2012_ge2017"})
        print("   by year:", v["installs_by_year_le2012_ge2017"])


def billing(out: Path) -> None:
    LK.record("P2def-d0-billing-completeness")
    prod = D.open_c0("full", "after_the_fact_analysis")
    rows = []
    t0 = D.tidx("2016-01")
    for k in ("yunlin", "changhua"):
        z = np.load(prod.path(f"pumps_{D.FILETAG[k]}.npz"), allow_pickle=False)
        mt = pd.read_parquet(prod.path(f"meters_{D.FILETAG[k]}.parquet"))
        share = z["hp_share"]
        midx = z["meter_idx"]
        # one pump per meter (largest share) -> meter series = value / share
        order = np.lexsort((-share, midx))
        first = pd.Series(order).groupby(midx[order]).first()
        M = z["value"][first.to_numpy()] / share[first.to_numpy(), None]
        mi = first.index.to_numpy()
        mt = mt.set_index("meter_idx").reindex(mi)
        inst = np.where(mt["install_month"].notna(), D.month_index(mt["install_month"].fillna("")),
                        -10**6)
        code = mt["billing_code"].astype(str).to_numpy()
        for c in ("M0", "M1", "M2", "M9"):
            sel = code == c
            for t in range(t0, D.T_FULL):
                inst_ok = sel & (inst <= t)
                v = M[inst_ok, t]
                rows.append({"county": k, "code": c, "month": D.ym(t),
                             "n_installed": int(inst_ok.sum()),
                             "n_nan": int(np.isnan(v).sum()), "n_zero": int((v == 0).sum()),
                             "n_pos": int((v > 0).sum()), "kwh": float(np.nansum(v))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "d0_billing_completeness.csv", index=False)
    # summaries: share missing and zero by year (Jul-Jun years) and code
    df["yr"] = df["month"].str[:4].astype(int) - (df["month"].str[5:].astype(int) < 7)
    g = df.groupby(["county", "code", "yr"]).agg(
        n_inst=("n_installed", "mean"), nan=("n_nan", "sum"), zero=("n_zero", "sum"),
        pos=("n_pos", "sum"), kwh=("kwh", "sum"))
    g["nan_share"] = g["nan"] / (g["nan"] + g["zero"] + g["pos"])
    g["zero_share"] = g["zero"] / (g["nan"] + g["zero"] + g["pos"])
    g["pos_per_month"] = g["pos"] / 12.0
    g["GWh"] = g["kwh"] / 1e6
    pd.set_option("display.width", 200)
    print(g[["n_inst", "nan_share", "zero_share", "pos_per_month", "GWh"]].round(4).to_string())
    g.reset_index().to_csv(out / "d0_billing_completeness_by_year.csv", index=False)


def tpc(out: Path) -> None:
    LK.record("P2def-d0-tpc-parse")
    t = D.read_tpc()
    res = {"sha256": D.sha256(D.TPC_CSV), "n_rows": int(len(t)),
           "sectors": sorted(t["sector"].unique().tolist())}
    for k in ("yunlin", "changhua"):
        s = t[(t["county_zh"] == D.COUNTY_ZH[k]) & (t["sector"] == "農林漁牧")]
        months = sorted(s["month"].unique())
        full = pd.date_range(months[0] + "-01", months[-1] + "-01", freq="MS").strftime("%Y-%m")
        res[k] = {"first": months[0], "last": months[-1], "n": len(months),
                  "gaps": sorted(set(full) - set(months)),
                  "duplicates": int(s["month"].duplicated().sum())}
    D.save_json(res, out / "d0_tpc_parse.json")
    print(res)


if __name__ == "__main__":
    what, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    {"census": census, "billing": billing, "tpc": tpc}[what](out)
