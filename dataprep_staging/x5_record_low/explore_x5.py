"""X5 exploration (U7: 2023+ data never gate; labelled exploration, cannot promote).

Does the preconsolidation model explain the post-2021 slowdown (rings 0.48x, GNSS 0.51x of
2020-21 in the 2026-10-25 audit)? Per ring site-band (primary input), with heads read to
2025-03 under ``after_the_fact_analysis``:

* observed OLS band rate in 2020-01..2021-12 and 2023-01..2025-03 (rates are unaffected by
  the 2022 depth-reference change), their ratio;
* the same two rates predicted by (a) the scored TF fit (FIT only), (b) the TF refitted on
  2014-01..2021-12 (exploration fit), (c) the linear-head model refitted on 2014-2021;
* the head statistics behind them: mean-head change and the new-low depth gained in
  2023-25 (x_i growth after 2021-12);
* the twin's page column (``vep_zonal_leveling.json``, the zone of the site's easting)
  driven by the site's observed nest-mean head, against the ring ``total`` interval.

Writes results/dataprep/x5_record_low/exploration/ once (refuses to overwrite).

Usage: python explore_x5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402

OUT = K.OUT / "exploration"
T_END = "2025-03"
LABEL = "post-2022 exploration (U7); never validation; cannot promote"


def ols_rate(y: np.ndarray, months, win) -> float:
    m = L.in_window(months, win) & np.isfinite(y)
    if m.sum() < 12:
        return float("nan")
    return float(np.polyfit(L.t_years(months)[m], y[m], 1)[0])


def rings_full() -> dict[str, pd.DataFrame]:
    from hydrophysics.twin.mechanism import band_compaction, ring_positions

    out = {}
    end = pd.Period(T_END, "M").to_timestamp(how="end")
    for site, p in ring_positions(str(K.RING_DDIR)).items():
        b = band_compaction(p.loc[:end])
        b.index = b.index.to_period("M")
        out[site] = b
    return out


def page_column(h: np.ndarray, zone: int) -> np.ndarray:
    import torch

    from hydrophysics.twin.compaction import column_compaction
    from hydrophysics.twin.mechanism import column_params

    p = column_params(str(K.PAGE_COLUMN))[zone]
    ht = torch.tensor(h[None, :], dtype=torch.float64)
    with torch.no_grad():
        return column_compaction(ht, p, K.DT_DAYS).numpy()[0]


def main() -> int:
    from hydrophysics.twin.zones import fan_zones

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "explore.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the exploration runs once")
    months = L.month_axis(K.T0, T_END)
    sites = L.ring_sites()
    all_w = sorted({w for ws in sites.wells for w in ws})
    W = L.load_heads(all_w, use=K.HEAD_USE_EXPLORE, window_end=T_END)
    depths = L.well_depths(all_w)
    rings = rings_full()
    w1, w2 = K.EXPLORE_RATE
    fit = L.in_window(months, K.FIT)
    fit21 = L.in_window(months, ("2014-01", "2021-12"))
    late = L.in_window(months, w2)
    rows, trows = [], []
    for _, s in sites.iterrows():
        b = rings.get(s.site)
        if b is None:
            continue
        zone = int(fan_zones(np.array([[s.x, s.y]]))[0])
        hn = L.nest_mean_head(W, s.wells, months)
        tot = b["total"].reindex(months).to_numpy(dtype=float)
        tr = {"site": s.site, "zone": zone, "obs_r1_mm_yr": ols_rate(tot, months, w1) * 1000,
              "obs_r2_mm_yr": ols_rate(tot, months, w2) * 1000}
        if hn is not None:
            pc = page_column(hn, zone)
            tr.update({"page_r1_mm_yr": ols_rate(pc, months, w1) * 1000,
                       "page_r2_mm_yr": ols_rate(pc, months, w2) * 1000,
                       "nest_mean_head_change_m": float(
                           hn[late].mean() - hn[L.in_window(months, w1)].mean())})
        trows.append(tr)
        for lab in [c for c in b.columns if c != "total"]:
            obs = b[lab].reindex(months).to_numpy(dtype=float)
            well, h = L.choose_well(s.wells, depths, W, months, L.band_mid_depth(lab))
            row = {"site": s.site, "band": lab, "well": well,
                   "obs_r1_mm_yr": ols_rate(obs, months, w1) * 1000,
                   "obs_r2_mm_yr": ols_rate(obs, months, w2) * 1000}
            if h is not None and np.isfinite(obs[fit]).sum() >= K.MIN_FIT_OBS:
                inp = L.precon_inputs(h, months)
                i21 = int(np.flatnonzero(L.in_window(months, ("2021-12", "2021-12")))[0])
                row.update({
                    "head_change_m": float(h[late].mean() - h[L.in_window(months, w1)].mean()),
                    "new_low_2020_21_m": float(inp["x_i"][i21]
                                               - inp["x_i"][L.in_window(months, K.FIT)][-1]),
                    "new_low_after_2021_m": float(inp["x_i"][late].max() - inp["x_i"][i21])})
                for tag, msk in (("tf_fit19", fit), ("tf_fit21", fit21)):
                    obs_fit = np.where(msk, obs, np.nan)
                    f = L.fit_tf(h, obs_fit, months, msk)
                    row[f"{tag}_r1_mm_yr"] = ols_rate(f["pred"], months, w1) * 1000
                    row[f"{tag}_r2_mm_yr"] = ols_rate(f["pred"], months, w2) * 1000
                lf = L.fit_linear_head(h, np.where(fit21, obs, np.nan), months, fit21)
                row["lin_fit21_r1_mm_yr"] = ols_rate(lf["pred"], months, w1) * 1000
                row["lin_fit21_r2_mm_yr"] = ols_rate(lf["pred"], months, w2) * 1000
            rows.append(row)
    bands = pd.DataFrame(rows)
    tots = pd.DataFrame(trows)
    bands.to_csv(OUT / "bands_rates.csv", index=False)
    tots.to_csv(OUT / "total_rates.csv", index=False)

    def ratio_summary(df: pd.DataFrame, a: str, b: str) -> dict:
        r = df[b] / df[a]
        ok = np.isfinite(r) & (df[a] > 0.5)   # rate in 2020-21 > 0.5 mm/yr
        return {"n": int(ok.sum()), "median_ratio": float(r[ok].median()) if ok.any() else None}

    res = {"label": LABEL, "declaration": K.PREREG_ID, "windows": [w1, w2],
           "bands": {
               "observed": ratio_summary(bands, "obs_r1_mm_yr", "obs_r2_mm_yr"),
               **{m: ratio_summary(bands, f"{m}_r1_mm_yr", f"{m}_r2_mm_yr")
                  for m in ("tf_fit19", "tf_fit21", "lin_fit21") if f"{m}_r1_mm_yr" in bands}},
           "total_interval": {
               "observed": ratio_summary(tots, "obs_r1_mm_yr", "obs_r2_mm_yr"),
               "page_column_on_observed_nest_heads": ratio_summary(
                   tots, "page_r1_mm_yr", "page_r2_mm_yr") if "page_r1_mm_yr" in tots else None},
           "heads": {
               "median_head_change_m": float(bands["head_change_m"].median())
               if "head_change_m" in bands else None,
               "median_new_low_2020_21_m": float(bands["new_low_2020_21_m"].median())
               if "new_low_2020_21_m" in bands else None,
               "median_new_low_after_2021_m": float(bands["new_low_after_2021_m"].median())
               if "new_low_after_2021_m" in bands else None,
               "n_bands_new_low_after_2021_ge_0.1m": int(
                   (bands.get("new_low_after_2021_m", pd.Series(dtype=float)) >= 0.1).sum())},
           "files": {"bands": str(OUT / "bands_rates.csv"), "total": str(OUT / "total_rates.csv")}}
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n",
                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
