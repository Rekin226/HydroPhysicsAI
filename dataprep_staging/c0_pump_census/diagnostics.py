"""WP C0: descriptive statistics of the delivered products and post-registration diagnostics
(after every registered check had been scored). Logged in the A1 look log as diagnoses; none of
these numbers enters a gate.

Writes results/dataprep/reports/C0_evidence/diagnostics.json.

Usage: python -I diagnostics.py      (inside tools/cpu_slot.sh)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from harness import prereg2  # noqa: E402

CLS = list(lib.CLASSES)


def seasonal_cv(x: np.ndarray, dates: list[str]) -> float:
    s = pd.Series(x, index=[d[5:7] for d in dates])
    moy = s.groupby(level=0).mean()
    return float(moy.std() / moy.mean())


def t2a_differ_months() -> dict:
    df = lib.harmonise("yunlin", lib.read_census_l0("yunlin"), lib.load_grid())
    _, L = lib.read_kwh_l0("yunlin", df["sid"].to_numpy(), len(df))
    _, La, info = lib.read_amp_kwh(df["sid"].to_numpy())
    both = np.isfinite(L) & np.isfinite(La)
    differ = both & (La != L)
    by_m = differ.sum(0)
    months = {lib.ym(j): int(by_m[j]) for j in np.nonzero(by_m)[0]}
    return {"differ_pump_months_by_month": months, "amp_last_month": info["last_month"],
            "note": "the AMP_V2 cache ends with a partial 2025-07 (phase-A bill missing); the "
                    "new sheet holds the complete 2025-07"}


def product_stats(d: Path, variant: str) -> dict:
    z = np.load(d / "cells_by_class.npz")
    zt = np.load(d / "cells_total.npz")
    dates = list(z["dates"])
    counties = list(z["counties"])
    vbc = z["value_by_county"]                              # (K, C, A, T)
    fbc = z["flag_by_county"]
    out = {}
    yrs = {y: [i for i, dd in enumerate(dates) if dd.startswith(str(y))]
           for y in range(2007, 2027)}
    w = [i for i, dd in enumerate(dates) if "2012-01" <= dd <= "2022-12"]
    for k, tag in enumerate(counties):
        ing = np.nansum(vbc[k], 1)                          # (C, T) in grid
        r = {"in_grid_GWh_per_yr_2012_2022_by_class":
             {c: float(ing[i, w].sum() / 1e6 / 11) for i, c in enumerate(CLS)}}
        tot = ing.sum(0)
        r["in_grid_GWh_per_yr_2012_2022_all"] = float(tot[w].sum() / 1e6 / 11)
        dom = CLS.index("domestic")
        r["domestic_share_2012_2022"] = float(ing[dom, w].sum() / tot[w].sum())
        r["seasonal_cv_2012_2022"] = {
            "all": seasonal_cv(tot[w], [dates[i] for i in w]),
            "irrigation": seasonal_cv(ing[CLS.index("irrigation"), w], [dates[i] for i in w]),
            "domestic": seasonal_cv(ing[dom, w], [dates[i] for i in w])}
        moy = pd.Series(ing[dom, w], index=[dates[i][5:7] for i in w]).groupby(level=0).mean()
        r["domestic_month_of_year_GWh_2012_2022"] = {m: float(v / 1e6) for m, v in moy.items()}
        r["in_grid_GWh_by_year"] = {
            str(y): float(np.nansum(tot[idx]) / 1e6) for y, idx in yrs.items()
            if idx and not np.isnan(vbc[k][:, :, idx]).all()}
        fk = fbc[k]                                          # worst flag over classes,
        f = np.where((fk == 8).any(0), 8, np.where((fk == 9).any(0), 9, 0))  # 8 over 9
        has = (np.nan_to_num(vbc[k]).sum(0) > 0) | (f > 0)
        r["cell_months_with_pumps_2012_2022"] = int(has[:, w].sum())
        r["cell_months_flag9_2012_2022"] = int(((f == 9) & has)[:, w].sum())
        e9 = np.where(fbc[k] == 9, np.nan_to_num(vbc[k]), 0.0)
        r["energy_share_in_flag9_class_cells_2012_2022"] = float(
            e9[..., w].sum() / max(np.nansum(vbc[k][..., w]), 1.0))
        if variant == "full":
            j = len(dates) - 1
            r["cells_with_pumps_flag8_last_month"] = int(((f[:, j] == 8) & has[:, j]).sum())
            r["cells_with_pumps_last_month"] = int(has[:, j].sum())
        out[tag] = r
    fa = zt["flag_all"]
    out["total_cells_flag_counts"] = {str(int(v)): int((fa == v).sum())
                                      for v in np.unique(fa)}
    if variant == "full":
        ev = zt["value_all_evensplit"]
        va = zt["value_all"]
        last12 = [i for i, dd in enumerate(dates) if "2025-07" <= dd <= "2026-06"]
        prev = [i for i, dd in enumerate(dates) if "2012-01" <= dd <= "2025-06"]
        dd_ = np.abs(np.nansum(ev, 0) - np.nansum(va, 0))
        out["evensplit_vs_primary"] = {
            "fan_total_equal_over_record_rel": float(abs(np.nansum(ev[:, :-1])
                                                         / np.nansum(va[:, :-1]) - 1)),
            "mean_abs_monthly_fan_delta_rel_2025_07_2026_06": float(
                dd_[last12].sum() / np.nansum(va[:, last12])),
            "mean_abs_monthly_fan_delta_rel_2012_2025_06": float(
                dd_[prev].sum() / np.nansum(va[:, prev])),
            "share_of_cell_months_resplit_2025_07_2026_06": float(
                (zt["flag_all_evensplit"][:, last12] == 1).mean())}
    return out


def meter_stats(d: Path) -> dict:
    out = {}
    for tag, suf in (("yunlin", ""), ("changhua", ""), ("chiayi", "_ingrid")):
        mt = pd.read_parquet(d / f"meters_{tag}{suf}.parquet")
        g = mt[mt["in_grid_any"]]
        out[tag] = {
            "meters": int(len(mt)), "meters_in_grid": int(len(g)),
            "phase_code_counts": mt["phase_code"].value_counts().to_dict(),
            "phase_agree_bimonthly": float(mt.loc[mt.phase_code.isin(["A", "B"]),
                                                  "phase_agree"].mean()),
            "phase_agree_monthly": float(mt.loc[mt.phase_code == "monthly",
                                                "phase_agree"].mean()),
            "terminated": int(mt["terminated"].sum()),
            "terminated_in_grid": int(g["terminated"].sum()),
            "terminated_in_grid_last12_kwh_GWh": float(
                g.loc[g["terminated"], "kwh_last12_before_termination"].sum() / 1e6),
            "terminated_from_by_year_in_grid": g.loc[g["terminated"], "terminated_from"]
            .str[:4].value_counts().sort_index().to_dict(),
            "pump_months_missing": int(mt["n_months_missing"].sum()),
        }
    return out


def main() -> int:
    prods = {k: lib.REPO / v for k, v in lib.load_json(lib.WORK / "products.json").items()}
    prereg2.record_look(["C0:c0_pipeline_newL0_masked_vs_cached_forcing_2012_2022"],
                        "tpc_census_yunlin_vs_amp:2007-01..2025-07", lib.WP, "diagnose",
                        test_id="C0.v1.T2a_new_l0_common_rows", prereg_id=lib.PREREG_ID,
                        note="[C0-diag] post-registration diagnostic: months of the 35,789 "
                             "pump-months where the new sheet and AMP_V2 are both finite but "
                             "differ (after scoring)")
    prereg2.record_look(["C0:product_descriptive_stats"], "c0_products:2007-01..2026-07",
                        lib.WP, "diagnose",
                        note="[C0-diag] descriptive statistics of the delivered products (rerun after fixing the flag precedence of the per-county cell counts) "
                             "(energy by county/class/year, seasonal CV, flags, terminated "
                             "meters, billing phases, even-split effect), after scoring")
    res = {"t2a_differ": t2a_differ_months()}
    for v in ("preorigin", "full"):
        res[f"stats_{v}"] = product_stats(prods[v], v)
        res[f"meters_{v}"] = meter_stats(prods[v])
    res["coding_sensitivity_2012_2022"] = {
        t: lib.load_json(lib.WORK / f"summary_{t}.json")["coding_sensitivity_2012_2022"]
        for t in lib.COUNTIES}
    lib.save_json(res, lib.EVIDENCE / "diagnostics.json")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
