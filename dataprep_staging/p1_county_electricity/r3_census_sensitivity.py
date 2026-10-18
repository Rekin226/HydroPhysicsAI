"""WP P1 repair round (r3): census-side checks the verifier asked for. After-the-fact analysis
only; nothing here is a pre-registered test, a gate or a threshold.

1. **Pre-install coding (verifier minor issue 5).** The Changhua sheet codes months before a
   pump's installation as NaN, the Yunlin sheet (and the AMP_V2 cache the twin was calibrated
   on) as 0. ``clean_census`` drops NaN rows before it takes each meter's mean duty over
   2012-2022, so Changhua's duty mean excludes those months and Yunlin's includes them. Two
   sensitivity codings of the Changhua kWh table inside the duty window are run through the
   unchanged model cleaning:

   * ``nan_to_zero_all``: every NaN month in 2012-01..2022-12 becomes 0 (the verifier's run);
   * ``nan_to_zero_leading``: only NaN months before the pump's first non-missing reading
     (pre-install months in the Yunlin sense) become 0.

   The kept-pump set changes; the energy of the kept pumps is then summed from the original
   table (NaN and 0 add the same). The baseline coding is re-run first and must reproduce
   ``census_check/v1`` exactly. Reported: kept pumps, cleaned energy, the census ratio a*
   (Yunlin unchanged, from ``census_check/v1``), phi* and the in-grid level.
2. **C9 (verifier minor issue 6).** Pump-month rows of the new Yunlin sheet absent from the
   AMP_V2 cache (months <= 2025-06): how many distinct pumps carry them, and whether those
   pumps have any kWh row, or any pump record, in AMP_V2. Pump ids are counted, never written.
3. **T10 wording (verifier minor issue 4).** Cleaned census energy per meter with a non-zero
   reading, July-June years, from the ``census_check/v1`` tables (truth only; no candidate).

The census columns are selected explicitly (no 戶名1, 用電地址2 or postal code is read).

Usage: python -I r3_census_sensitivity.py <out_json>   (inside tools/cpu_slot.sh; ~5 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO))
from hydrophysics.twin.pumping import _cell_lookup, clean_census  # noqa: E402
from hydrophysics.twin.scenario import purpose_class  # noqa: E402

import census_truth as ct  # noqa: E402  (read-only reuse of the round-2 loaders)

CC1 = REPO / "results/dataprep/county_electricity/census_check/v1"
AMP_KWH = REPO / "AMP_V2/data/pump_kwh_all.parquet"
AMP_PUMPS = REPO / "AMP_V2/data/tpc_pumps.parquet"
WIN = {"preorigin_2016_2022": ("2016-01-01", "2022-12-01"),
       "full_2016_01_2025_06": ("2016-01-01", "2025-06-01"),
       "long_2012_2022": ("2012-01-01", "2022-12-01")}
DUTY0, DUTY1 = pd.Timestamp(ct.DUTY_T0), pd.Timestamp(ct.DUTY_T1)


def wsum(s: pd.Series, w) -> float:
    return float(s.loc[w[0]:w[1]].sum())


def v1_truth() -> pd.DataFrame:
    t = pd.read_parquet(CC1 / "census_truth_monthly.parquet")
    return t[(t.variant == "full") & (t.series == "clean")]


def series(t: pd.DataFrame, county: str, scope: str = "county") -> pd.Series:
    x = t[(t.county == county) & (t.scope == scope)].groupby("date")["value"].sum()
    x.index = pd.DatetimeIndex(x.index)
    return x.sort_index()


def coded(k: pd.DataFrame, how: str) -> tuple[pd.DataFrame, int]:
    """The kWh table with NaN months inside the duty window recoded to 0 (``how``)."""
    if how == "baseline":
        return k, 0
    inwin = (k["datetime"] >= DUTY0) & (k["datetime"] < DUTY1)
    nan = k["electricity_kwh"].isna()
    if how == "nan_to_zero_all":
        sel = inwin & nan
    elif how == "nan_to_zero_leading":
        first = k.loc[~nan].groupby("pump")["datetime"].min()
        f = k["pump"].map(first)
        sel = inwin & nan & (f.isna() | (k["datetime"] < f))
    else:
        raise ValueError(how)
    k2 = k.copy()
    k2.loc[sel, "electricity_kwh"] = 0.0
    return k2, int(sel.sum())


def changhua_sensitivity(yl: dict) -> dict:
    grid = ct.load_grid()
    pumps, k = ct.load_county("changhua")
    cls_all = pumps.set_index("sid")["PURPOSE"].map(purpose_class)
    inwin = (k["datetime"] >= DUTY0) & (k["datetime"] < DUTY1)
    out = {"duty_window": [ct.DUTY_T0[:7], "2022-12"],
           "nan_rows_in_duty_window": int((inwin & k["electricity_kwh"].isna()).sum()),
           "rows_in_duty_window": int(inwin.sum()), "codings": {}}
    for how in ("baseline", "nan_to_zero_all", "nan_to_zero_leading"):
        t0 = time.time()
        k2, nrec = coded(k, how)
        kept, _, _ = clean_census(pumps, k2, cap_duty=1.0, t0=ct.DUTY_T0, t1=ct.DUTY_T1)
        del k2
        _, kc, _ = clean_census(kept, k, cap_duty=None)
        cell_kept = _cell_lookup(kept, grid)
        a = ct.aggregate(kc, cls_all, cell_kept)
        a["date"] = pd.to_datetime(a["date"])
        ch = a[a.scope == "county"].groupby("date")["gwh"].sum().sort_index()
        chi = a[a.scope == "in_grid"].groupby("date")["gwh"].sum().sort_index()
        r = {"rows_recoded": nrec, "pumps_kept": int(len(kept)),
             "pumps_kept_in_grid": int(len(cell_kept)),
             "kept_class_counts": kept["PURPOSE"].map(purpose_class).value_counts().to_dict(),
             "seconds": round(time.time() - t0, 1)}
        for wn, w in WIN.items():
            e = wsum(ch, w)
            r[wn] = {"changhua_clean_GWh": e, "changhua_in_grid_GWh": wsum(chi, w),
                     "a_star": e / yl[wn], "phi_star": wsum(chi, w) / e}
        r["in_grid_GWh_per_yr_2012_2022"] = wsum(chi, WIN["long_2012_2022"]) / 11.0
        out["codings"][how] = r
        del kc, kept, a
        print(how, json.dumps({x: r[x] for x in ("rows_recoded", "pumps_kept", "seconds")}),
              flush=True)
    b = out["codings"]["baseline"]
    for how in ("nan_to_zero_all", "nan_to_zero_leading"):
        r = out["codings"][how]
        r["relative_to_baseline"] = {
            wn: {"energy": r[wn]["changhua_clean_GWh"] / b[wn]["changhua_clean_GWh"] - 1.0,
                 "a_star": r[wn]["a_star"] / b[wn]["a_star"] - 1.0}
            for wn in WIN}
    return out


def c9_single_pump() -> dict:
    a = pd.read_parquet(AMP_KWH, columns=["datetime", "electricity_kwh", "pump"])
    a["datetime"] = pd.to_datetime(a["datetime"])
    a["pump"] = a["pump"].astype(str)
    a_le = a[a["datetime"] <= "2025-06-01"]
    c = pd.read_parquet(ct.L0 / "yunlin_census.parquet", columns=["row_idx", "WELL_NO",
                                                                    "PURPOSE", "TWD97_X",
                                                                    "TWD97_Y"])
    sid = c["WELL_NO"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).to_numpy()
    b = pd.read_parquet(ct.L0 / "yunlin_kwh_long.parquet", columns=ct.KWH_COLS)
    b["datetime"] = pd.to_datetime(b["datetime"])
    b = b[b["datetime"] <= "2025-06-01"]
    b["pump"] = sid[b["row_idx"].to_numpy()]
    m = a_le[["datetime", "pump"]].merge(b[["datetime", "pump", "electricity_kwh"]],
                                         on=["datetime", "pump"], how="right", indicator=True)
    only = m[m["_merge"] == "right_only"]
    pumps_only = set(only["pump"])
    amp_pumps = pd.read_parquet(AMP_PUMPS, columns=["sid"])["sid"].astype(str)
    amp_pumps = set(amp_pumps.str.strip().str.replace(r"\.0$", "", regex=True))
    rec = c.assign(sid=sid)
    rec = rec[rec["sid"].isin(pumps_only)]
    grid = ct.load_grid()
    cells = _cell_lookup(rec.drop(columns=["row_idx", "WELL_NO"]), grid) if len(rec) else {}
    nz = only[only["electricity_kwh"].fillna(0) != 0]
    return {"rows_only_in_new": int(len(only)),
            "distinct_pumps": int(len(pumps_only)),
            "these_pumps_kwh_rows_in_amp_any_month": int(a["pump"].isin(pumps_only).sum()),
            "these_pumps_in_amp_pump_table": int(sum(p in amp_pumps for p in pumps_only)),
            "months_first_last": [str(only["datetime"].min().date())[:7],
                                  str(only["datetime"].max().date())[:7]],
            "months": int(only["datetime"].nunique()),
            "nonzero_months": int(len(nz)), "kwh_total": float(only["electricity_kwh"].sum()),
            "purpose_class": sorted(rec["PURPOSE"].map(purpose_class).astype(str).unique()
                                    .tolist()),
            "in_active_grid": int(len(cells)),
            "note": "pump ids counted, never written out"}


def energy_per_meter() -> dict:
    t = v1_truth()
    mc = pd.read_parquet(CC1 / "meter_counts.parquet")
    out = {}
    for cty in ("changhua", "yunlin"):
        e = series(t, cty)
        n = mc[mc.county == cty].set_index("date")["meters_nonzero"]
        n.index = pd.DatetimeIndex(n.index)
        rows = {}
        for y in range(2016, 2026):
            w = (f"{y}-07-01", f"{y + 1}-06-01")
            en, mm = wsum(e, w), float(n.loc[w[0]:w[1]].mean())
            rows[f"{y}/{y + 1}"] = {"clean_GWh": en, "mean_meters_nonzero": mm,
                                   "MWh_per_nonzero_meter": 1e3 * en / mm}
        out[cty] = rows
    return out


def main(out_json: str) -> int:
    t_all = time.time()
    t = v1_truth()
    yl_s = series(t, "yunlin")
    yl = {wn: wsum(yl_s, w) for wn, w in WIN.items()}
    ch_v1 = series(t, "changhua")
    res = {"status": "after-the-fact analysis (repair round r3); not pre-registered; no gate, "
                     "threshold or selection depends on it",
           "yunlin_clean_GWh_census_check_v1": yl,
           "changhua_clean_GWh_census_check_v1": {wn: wsum(ch_v1, w) for wn, w in WIN.items()}}
    res["changhua_preinstall_coding"] = changhua_sensitivity(yl)
    b = res["changhua_preinstall_coding"]["codings"]["baseline"]
    res["baseline_reproduces_census_check_v1"] = {
        wn: abs(b[wn]["changhua_clean_GWh"] - res["changhua_clean_GWh_census_check_v1"][wn])
        for wn in WIN}
    res["c9_rows_only_in_new_sheet"] = c9_single_pump()
    res["energy_per_nonzero_meter_jul_jun"] = energy_per_meter()
    res["seconds"] = round(time.time() - t_all, 1)
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                              encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("baseline_reproduces_census_check_v1",
                                          "c9_rows_only_in_new_sheet", "seconds")},
                     ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
