"""P5 stage 2: assemble the well-month table and the derived well-level tables.

Usage: s2_assemble.py WORKDIR
Reads WORKDIR/wells/*.parquet and WORKDIR/diag/*.json (stage 1). Writes to WORKDIR:

* ``well_month_features.parquet``: one row per well-month (2012-01..2026-09), every pilot
  index on the barometrically corrected samples plus the ``*_raw`` (uncorrected) variants,
  ``flag`` (1: barometric loading removed in that month; 0: no accepted response), ``method``,
  ``county``, ``layer``, ``product_version`` and ``source_ids``;
* ``well_year_features.parquet``: spring (MAM) recession slope and seasonal drawdown depth per
  well-year (pilot ``seasonal_nested.py`` definitions: months with >= 15 days, years with
  >= 10 such months; depth = max - min of monthly means; spring slope = mean of the MAM
  monthly OLS slopes);
* ``nested_pair_months.parquet``: per nested site (station code prefix, pilot rule) the
  head difference shallowest-minus-deeper layer per month (both monthly means present);
* ``tidal_response.csv``: per well and period, M2 and O1 amplitude, circular-mean phase
  relative to the astronomical argument and phase coherence across months (corrected
  samples; no ocean-loading correction, so not yet usable for T and S, plan WP P5 step 4);
* ``extraction_summary.json``: counts and the day-by-day construction check against L1 v3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p5_head_signatures import config as C  # noqa: E402
from p5_head_signatures import inputs as I  # noqa: E402

PRODUCT_VERSION = "head_signatures/v1"
SOURCE_IDS = ("heads/L1_clean/v3;AMP_V2/data/wells;"
              "data_fetch_api/new_measurements/heads_202210_202610")


def circ(g: pd.Series) -> tuple[float, float]:
    z = np.exp(1j * g.dropna().to_numpy())
    if not len(z):
        return np.nan, np.nan
    m = z.mean()
    return float(np.angle(m)), float(np.abs(m))


def main():
    work = Path(sys.argv[1])
    files = sorted((work / "wells").glob("*.parquet"))
    F = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    meta = I.station_meta()
    F = F.merge(meta[["sid", "county", "layer"]], on="sid", how="left")
    F["product_version"] = PRODUCT_VERSION
    F["source_ids"] = SOURCE_IDS
    F = F.sort_values(["sid", "month"]).reset_index(drop=True)
    lead = ["sid", "month", "period", "county", "layer", "flag", "method"]
    F = F[lead + [c for c in F.columns if c not in lead]]
    F.to_parquet(work / "well_month_features.parquet", index=False)

    # ---------------------------------------------------------------- well-year
    G = F[F["n_days"] >= 15].copy()
    G["year"] = G["month"].str[:4].astype(int)
    G["mon"] = G["month"].str[5:7].astype(int)
    g = G.groupby(["sid", "year"])
    Y = pd.DataFrame({"n_months": g.size(), "hmax": g["mean"].max(), "hmin": g["mean"].min()})
    sp = G[G["mon"].isin([3, 4, 5])].groupby(["sid", "year"])["slope"].mean()
    Y["spring_slope_m_per_day"] = sp
    Y = Y.reset_index()
    Y = Y[Y["n_months"] >= 10].copy()
    Y["seasonal_depth_m"] = Y["hmax"] - Y["hmin"]
    Y["period"] = np.where(Y["year"] <= 2022, "pre", "post")
    Y = Y.merge(meta[["sid", "county", "layer"]], on="sid", how="left")
    Y.to_parquet(work / "well_year_features.parquet", index=False)

    # ---------------------------------------------------------------- nested pairs
    m = meta[meta["layer"].isin(["1", "2", "3", "4"])].copy()
    m["site"] = m["sid"].str[:6]
    m["depth"] = pd.to_numeric(m["WellDepth"], errors="coerce")
    pairs = []
    for site, gs in m.groupby("site"):
        gs = gs.sort_values(["depth", "layer"])
        if gs["layer"].nunique() < 2:
            continue
        sh = gs.iloc[0]
        for _, dp in gs.iloc[1:].iterrows():
            if dp["layer"] == sh["layer"]:
                continue
            pairs.append((site, sh["sid"], dp["sid"], sh["layer"], dp["layer"], sh["county"]))
    piv = F.pivot_table(index="month", columns="sid", values="mean")
    rows = []
    for site, s1, s2, l1, l2, cty in pairs:
        if s1 not in piv or s2 not in piv:
            continue
        dh = (piv[s1] - piv[s2]).dropna()
        for mth, v in dh.items():
            rows.append({"site": site, "shallow": s1, "deep": s2, "layer_shallow": l1,
                         "layer_deep": l2, "county": cty, "month": mth, "dh_m": float(v),
                         "period": "pre" if mth <= C.ORIGIN else "post"})
    NP = pd.DataFrame(rows)
    NP.to_parquet(work / "nested_pair_months.parquet", index=False)

    # ---------------------------------------------------------------- tidal response
    tr = []
    for (sid, per), gg in F.dropna(subset=["gM2"]).groupby(["sid", "period"]):
        phM2, cohM2 = circ(gg["gM2"])
        phO1, cohO1 = circ(gg["gO1"])
        phO1r, cohO1r = circ(gg["gO1_raw"]) if "gO1_raw" in gg else (np.nan, np.nan)
        tr.append({"sid": sid, "period": per, "n_months": int(len(gg)),
                   "A_M2_med_m": float(gg["A_M2"].median()),
                   "gM2_circmean_deg": float(np.degrees(phM2)), "M2_phase_coherence": cohM2,
                   "A_O1_med_m": float(gg["A_O1"].median()),
                   "gO1_circmean_deg": float(np.degrees(phO1)), "O1_phase_coherence": cohO1,
                   "O1_phase_coherence_raw": cohO1r,
                   "baro_corrected_months": int((gg["flag"] == 1).sum()),
                   "note": "phase = astronomical argument (Greenwich, V+u) minus fitted phase; "
                           "no ocean-loading or longitude correction; not usable for T/S "
                           "until an ocean-loading model is added (plan WP P5 step 4, E8)"})
    TR = pd.DataFrame(tr).merge(meta[["sid", "county", "layer"]], on="sid", how="left")
    TR.to_csv(work / "tidal_response.csv", index=False)

    # ---------------------------------------------------------------- summary
    diags = [json.loads(f.read_text()) for f in sorted((work / "diag").glob("*.json"))]
    dc = [d.get("daily_check", {}) for d in diags]
    summ = {
        "n_wells_attempted": len(diags),
        "n_wells_with_rows": int(F["sid"].nunique()),
        "n_rows": int(len(F)),
        "rows_by_period": F.groupby("period").size().to_dict(),
        "wells_by_period": F.groupby("period")["sid"].nunique().to_dict(),
        "rows_by_county": F.groupby("county").size().to_dict(),
        "rows_flag1": int((F["flag"] == 1).sum()),
        "wells_with_accepted_baro": int(sum(bool(d.get("baro_accepted")) for d in diags)),
        "daily_check": {
            "days_compared": int(sum(x.get("n_days_compared", 0) for x in dc)),
            "max_abs_diff_m": float(max(x.get("max_abs_diff_m", 0.0) for x in dc)),
            "days_diff_gt_1mm": int(sum(x.get("n_days_diff_gt_1mm", 0) for x in dc)),
            "max_abs_diff_baro_m": float(max(x.get("max_abs_diff_baro_m", 0.0) for x in dc)),
            "l1_days_not_covered": int(sum(x.get("n_l1_days_not_covered", 0) for x in dc))},
        "n_well_years": int(len(Y)), "n_nested_pairs": int(NP[["shallow", "deep"]]
                                                           .drop_duplicates().shape[0])
        if len(NP) else 0,
        "n_tidal_rows": int(len(TR)),
    }
    (work / "extraction_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
