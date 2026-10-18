"""Design-time look (logged as [P2-d0] before it ran): structure and unit counts of the C0 v3
inputs, needed to fix the round-2 design before registration. No Chow-Lin estimate, profile,
skill, coverage, regression or backtest is computed here.

Counts per county and variant: pump-vs-meter consistency of the C0 arrays; kept meters by
billing code x detected phase; Chow-Lin set and profile populations; candidate validation
meters (kept irrigation meters, detected monthly, flag 0 and finite in every month
2012-01..2022-12, positive total, coordinates); pairs with exactly one missing month; the
open-month populations (pre-origin December 2022 phase B, full July 2026 phase A); the share
of phase-A (M2) meters reading exactly 0 in July of 2022-2025 (is July 2025 complete in the
new delivery?); fan-station counts by county.

Usage: python -I d0_design_counts.py <out.json>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402


def main(out: str) -> None:
    l1 = L.l1_pumps()
    res = {}
    for variant in ("preorigin", "full"):
        prod = L.open_c0(variant)
        for county in L.COUNTIES:
            cd = L.CountyData(variant, county, prod=prod, l1=l1)
            T = cd.T
            ph, code, cls = cd.phase, cd.code, cd.mclass
            r = {"check": cd.check, "n_pumps": len(cd.uid), "n_meters": len(cd.meters),
                 "n_meters_first_pump_in_grid": int(cd.m_first_in_grid.sum()),
                 "code_x_phase": pd.crosstab(code, ph).to_dict(),
                 "irrigation_meters": int((cls == "irrigation").sum())}
            rows = L.chowlin_rows(cd)
            r["chowlin_set"] = {p: int(len(v)) for p, v in rows.items()}
            irr = cls == "irrigation"
            ing = cd.m_first_in_grid
            r["profile_pop"] = {"A": int((irr & ing & (code == "M2") & (ph == "A")).sum()),
                                "B": int((irr & ing & (code == "M1") & (ph == "B")).sum())}
            w0, w1 = L.tidx("2012-01"), L.tidx("2022-12")
            Xw, Fw = cd.M[:, w0:w1 + 1], cd.F[:, w0:w1 + 1]
            pre = (L.CountyData("preorigin", county, l1=l1) if variant == "full" else cd)
            ph_pre = pre.phase
            val = (irr & (ph_pre == "monthly") & np.isfinite(Xw).all(1) & (Fw == 0).all(1)
                   & (np.nansum(Xw, 1) > 0) & np.isfinite(cd.m_x) & np.isfinite(cd.m_y))
            r["validation_candidates"] = int(val.sum())
            r["validation_candidates_by_code"] = pd.Series(code[val]).value_counts().to_dict()
            # pairs with exactly one missing month (detected A/B meters)
            inc = 0
            for p_ in ("A", "B"):
                rr = np.where(ph == p_)[0]
                st = L.pair_starts(p_, 0, T, T)
                a, b = np.isfinite(cd.M[np.ix_(rr, st)]), np.isfinite(cd.M[np.ix_(rr, st + 1)])
                inc += int((a ^ b).sum())
            r["pairs_one_month_missing"] = inc
            r["flag_counts_meter_months"] = {int(k): int(v) for k, v in zip(
                *np.unique(cd.F, return_counts=True), strict=True)}
            if variant == "full":
                t26 = L.tidx("2026-07")
                isA = (code == "M2")
                r["july2026_phaseA_code_nan"] = int(np.isnan(cd.M[isA, t26]).sum())
                r["july_zero_share_M2"] = {
                    str(y): float(np.mean(cd.M[isA, L.tidx(f"{y}-07")] == 0))
                    for y in (2022, 2023, 2024, 2025)}
                r["july_nan_share_M2"] = {
                    str(y): float(np.mean(np.isnan(cd.M[isA, L.tidx(f"{y}-07")])))
                    for y in (2022, 2023, 2024, 2025, 2026)}
            else:
                t22 = L.tidx("2022-12")
                r["dec2022_flag4_meters"] = int((cd.F[:, t22] == 4).sum())
            res[f"{variant}/{county}"] = r
            print(variant, county, {k: r[k] for k in ("n_meters", "chowlin_set", "profile_pop",
                                                      "validation_candidates")}, flush=True)
    st = pd.read_parquet(L.REPO / "AMP_V2" / "data" / "fan_stations.parquet",
                         columns=["sid", "CountyName", "GroundwaterLayerCode"])
    res["fan_stations_by_county"] = st["CountyName"].value_counts().to_dict()
    L.save_json(res, out)


if __name__ == "__main__":
    main(sys.argv[1])
