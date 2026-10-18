"""WP P3 after-the-fact diagnostics (logged as an after_the_fact look; never fed back):

1. irrigation kWh per planted crop-season hectare, Changhua (census, whole townships) against
   Yunlin, per year 2012-2022: why the crop-statistics level transfers at about half;
2. the primary map's battery means (corr, AA) against its Changhua scores (optimism check);
3. per-township ratios of the primary map and of the crop model (largest misses).

Usage: python -I after_fact.py      (writes P3_evidence/after_fact.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crop_model as CM  # noqa: E402
import frame  # noqa: E402
import lib  # noqa: E402
from harness import prereg3  # noqa: E402

UNIT = "P3:after_fact_changhua_2012_2022"


def main() -> None:
    members = {UNIT: {"description": "after-the-fact diagnostics on the Changhua census "
                                     "(whole-township irrigation per planted ha; township "
                                     "ratios), after P3.changhua_transfer.v1 was scored",
                      "none": "TPC census electricity aggregated to townships; no registered "
                              "well or leveling benchmark"}}
    prereg3.record_look([UNIT], "c0_pump_electricity_v3_changhua:2012-01..2022-12", "P3",
                        "after_the_fact", members=members,
                        note="[P3-after-fact] after_fact.py: kWh per planted ha by county and "
                             "year, battery vs Changhua for the primary map, township ratios")
    t = lib.crop_table()
    out = {"kwh_per_planted_ha": []}
    tk_c = CM.town_kwh("changhua", "validator")
    tk_y = CM.town_kwh("yunlin", "prior")
    for y in CM.YEARS:
        row = {"year": y}
        for cz, tk, nm in (("彰化縣", tk_c, "changhua"), ("雲林縣", tk_y, "yunlin")):
            X = CM.design(t, y, cz)
            e = tk[(tk.cls == "irrigation") & (tk.year == y)]["kwh"].sum()
            row[f"{nm}_irr_GWh"] = float(e / 1e6)
            row[f"{nm}_planted_kha"] = float(X.to_numpy().sum() / 1e3)
            row[f"{nm}_kwh_per_ha"] = float(e / X.to_numpy().sum())
            for c in ("rice1", "rice2", "veg", "upland", "fruit"):
                row[f"{nm}_{c}_kha"] = float(X[c].sum() / 1e3)
        row["ratio_changhua_to_yunlin"] = row["changhua_kwh_per_ha"] / row["yunlin_kwh_per_ha"]
        out["kwh_per_planted_ha"].append(row)
    bs = pd.read_csv(lib.EVID / "battery_scores.csv")
    b = bs[(bs.model == "rf_LU+GEO") & (bs.split != "LOTO")]
    lo = bs[(bs.model == "rf_LU+GEO") & (bs.split == "LOTO")].iloc[0]
    sc = pd.read_csv(lib.EVID / "scores_maps.csv").set_index("map").loc["rf_LU+GEO"]
    out["primary_battery_vs_changhua"] = {
        "battery_8_mean_corr_5km": float(b["corr_5km_density"].mean()),
        "battery_8_mean_AA_5km": float(b["AA_5km"].mean()),
        "battery_8_ratio_range": [float(b["ratio"].min()), float(b["ratio"].max())],
        "loto_corr_5km": float(lo["corr_5km_density"]), "loto_AA_5km": float(lo["AA_5km"]),
        "changhua_corr_5km": float(sc["corr_5km_density|D"]),
        "changhua_AA_5km": float(sc["AA_5km|D"]), "changhua_ratio": float(sc["ratio_all"])}
    # township ratios: primary map (in-fan) and crop model (whole-township irrigation)
    _, aux = frame.load("prior")
    tz = aux["tz"]
    names = [str(n) for n in tz["names"]]
    S = tz["share"]
    z = np.load(lib.WORK / "predictions.npz", allow_pickle=True)
    k = [str(x) for x in z["map_names"]].index("rf_LU+GEO")
    m = z["maps"][k]
    ing = tk_c.groupby("town")["kwh_ingrid"].sum() / len(CM.YEARS)
    rows = []
    cp = pd.read_csv(lib.WORK / "crop_model_changhua_pred.csv").groupby("town")[
        "kwh_irr_whole_pred"].mean()
    irr = tk_c[tk_c.cls == "irrigation"].groupby("town")["kwh"].sum() / len(CM.YEARS)
    for j, n in enumerate(names):
        if not n.startswith("彰化縣"):
            continue
        rows.append({"town": n, "infan_km2": float(S[j].sum()),
                     "true_ingrid_GWh": float(ing.get(n, 0.0) / 1e6),
                     "rf_pred_GWh": float((m * S[j]).sum() / 1e6),
                     "true_irr_whole_GWh": float(irr.get(n, 0.0) / 1e6),
                     "crop_pred_irr_whole_GWh": float(cp.get(n, 0.0) / 1e6)})
    tw = pd.DataFrame(rows)
    tw["rf_ratio"] = tw.rf_pred_GWh / tw.true_ingrid_GWh.replace(0, np.nan)
    tw["crop_ratio"] = tw.crop_pred_irr_whole_GWh / tw.true_irr_whole_GWh.replace(0, np.nan)
    tw.to_csv(lib.EVID / "after_fact_townships.csv", index=False)
    out["township_ratio_quantiles"] = {
        c: [float(x) for x in tw[c].quantile([0.1, 0.5, 0.9])] for c in ("rf_ratio",
                                                                         "crop_ratio")}
    lib.save_json(lib.EVID / "after_fact.json", out)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=lib._jdefault)[:5000])
    print(tw.round(2).to_string())


if __name__ == "__main__":
    main()
