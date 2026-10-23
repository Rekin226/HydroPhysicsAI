"""Power check for the DRAFT N2 prospective policy-response registration (no unit read).

The draft scores the frozen page model's head response to recorded Yunlin pumping
anomalies on the prospective unit (heads at R0's 158 wells, 2026-11..2027-10). This
check sizes the model contrast it would score WITHOUT reading any head of the unit (none
exists yet) and without reading any electricity month after 2026-08 (none exists):

- sensitivity: the page basis (``response_basis_innov.npz``) layer-2 head change at each
  well's cell for retiring each class at the horizon, summed over classes = the head rise
  if all modelled (Yunlin) pumping stopped; a fractional county anomaly ``a`` gives about
  ``-a x`` that (the basis is linear to 3-9 %, page build log);
- anomaly sizes: the X1 Yunlin in-grid county series (recorded census months and TPC
  downscaled months, 2012-01..2026-08) as ratios to its 2012-2022 month-of-year mean, over
  the 12-month Nov-Oct windows that are complete, and the last 12 recorded months;
- detectability: if the model's response were right and residuals independent of it, the
  RMSE ratio of the policy arm to the zero-response arm is sqrt(1 - c^2 / s^2) for a
  contrast c and zero-response RMSE s.

Usage: python -I dataprep_staging/n1_gaps/n2_power.py
Output: results/dataprep/reports/N1_gaps_2026-10-24/science-15/n2_power_check.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from n1_gaps import gaps_lib as G  # noqa: E402

OUT = G.OUT_ROOT / "science-15"
X1 = G.REPO / "results/dataprep/electricity_x1_tpc/full/v1/county_monthly.parquet"
BASIS = G.FWD / "response_basis_innov.npz"
WELLS = G.RUNS / "stage3_datum_gate" / "stage3_wells.csv"
LAST_ALLOWED = pd.Period("2026-08", "M")          # last month that exists; nothing later read
SIGMAS = (1.0, 1.5, 2.0)


def main() -> int:
    os.chdir(G.REPO)
    from n1_policy.policy_lib import nearest_cell

    from hydrophysics.twin.app.prep import basis_deltas, year_ends

    gg = G.grid_groups()
    ch = gg["changhua"]
    wells = pd.read_csv(WELLS, usecols=["sid", "layer", "x", "y"], dtype={"sid": str})
    wc = nearest_cell(gg["grid"].centroids(), wells[["x", "y"]].to_numpy(float), 1000.0)
    keep = wc >= 0
    wells, wc = wells[keep], wc[keep]
    w_ch = ch[wc]
    z = np.load(BASIS, allow_pickle=False)
    names = [str(n) for n in z["scenario_names"]]
    ye, years = year_ends(pd.DatetimeIndex(z["dates"]))
    b = basis_deltas(z["subs_mean"] * 100.0, z["heads_mean"], names, ye)
    K = len(G.CLASSES)
    dh2 = b["dheadL2"][:K, :, -1]                      # 2026 start, Dec 2032, (K, A) m
    dsub = -b["dsub"][:K, :, -1]                        # cm avoided
    sens_h = dh2.sum(axis=0)[wc]                        # m head rise if all pumping stopped
    sens_s = dsub.sum(axis=0)                           # cm avoided, all cells

    x = pd.read_parquet(X1)
    x = x[(x.county == "yunlin")].copy()
    x["p"] = pd.PeriodIndex(x.date, freq="M")
    x = x[x.p <= LAST_ALLOWED].set_index("p").sort_index()
    clim = x[(x.index.year >= 2012) & (x.index.year <= 2022)].groupby(
        x[(x.index.year >= 2012) & (x.index.year <= 2022)].index.month)["value"].mean()
    x["ratio"] = x["value"] / x.index.month.map(clim).to_numpy()
    win = []
    for y0 in range(2012, 2026):
        p0, p1 = pd.Period(f"{y0}-11", "M"), pd.Period(f"{y0 + 1}-10", "M")
        sel = x.loc[p0:p1]
        if len(sel) == 12:
            a = float(sel["value"].sum() / sum(clim[m] for m in range(1, 13)) - 1.0)
            win.append({"window": f"{p0}..{p1}", "annual_anomaly": a,
                        "flags": sorted({int(f) for f in sel["flag"]})})
    last12 = x.iloc[-12:]
    a_last = float(last12["value"].sum() / sum(clim[m] for m in range(1, 13)) - 1.0)
    an = np.array([w["annual_anomaly"] for w in win])
    post = np.array([w["annual_anomaly"] for w in win if int(w["window"][:4]) >= 2022])
    mon = x["ratio"].to_numpy() - 1.0

    def contrast(a: float) -> dict:
        c_y = np.abs(a) * sens_h[~w_ch]
        c_c = np.abs(a) * sens_h[w_ch]
        rms = float(np.sqrt(np.mean(c_y ** 2)))
        return {"annual_anomaly": a,
                "yunlin_wells_rms_contrast_m": rms,
                "yunlin_wells_mean_contrast_m": float(c_y.mean()),
                "changhua_wells_rms_contrast_m": float(np.sqrt(np.mean(c_c ** 2))),
                "fan_subsidence_contrast_cm": float(np.abs(a) * sens_s.mean()),
                "rmse_ratio_if_model_right": {f"sigma_{s}m": float(np.sqrt(max(
                    1 - rms ** 2 / s ** 2, 0.0))) for s in SIGMAS}}

    res = {
        "what": "model-only power check for the draft N2 policy-response test; no head of "
                "the unit and no electricity month after 2026-08 is read",
        "n_wells": int(len(wc)), "n_yunlin_wells": int((~w_ch).sum()),
        "n_changhua_wells": int(w_ch.sum()),
        "sensitivity_all_yunlin_pumping_off": {
            "layer2_head_rise_at_yunlin_wells_m_mean_p10_p90": [
                float(sens_h[~w_ch].mean()), float(np.percentile(sens_h[~w_ch], 10)),
                float(np.percentile(sens_h[~w_ch], 90))],
            "layer2_head_rise_at_changhua_wells_m_mean": float(sens_h[w_ch].mean()),
            "fan_subsidence_avoided_cm": float(sens_s.mean())},
        "x1_yunlin_annual_anomalies_nov_oct": win,
        "annual_anomaly_abs_p50_p90": [float(np.median(np.abs(an))),
                                       float(np.percentile(np.abs(an), 90))],
        "annual_anomaly_since_2022_values": [float(v) for v in post],
        "last_12_recorded_months": {"window": f"{last12.index[0]}..{last12.index[-1]}",
                                    "annual_anomaly": a_last,
                                    "flags": sorted({int(f) for f in last12["flag"]})},
        "monthly_ratio_anomaly_sd": float(np.std(mon)),
        "contrast_at": {"p50_abs_annual_anomaly": contrast(float(np.median(np.abs(an)))),
                        "p90_abs_annual_anomaly": contrast(float(np.percentile(np.abs(an), 90))),
                        "last_12_months_anomaly": contrast(a_last)},
        "sigma_note": "zero-response RMSE s at the wells is unknown for the window; 1.0-2.0 m "
                      "brackets the page's published post-origin head RMSEs (hybrid layer "
                      "1.51 m, physics 2.05 m; page modelcard), which are not gating here",
        "reading": "a test is informative only if the realised contrast exceeds the "
                   "registered informative_head_m (0.05 m) AND its RMSE ratio could move "
                   "beyond sampling noise; with s >= 1.5 m even the p90 anomaly moves the ratio "
                   "by about 1-c^2/(2 s^2), so the primary decision should rest on the "
                   "obs-on-contrast slope with a cluster bootstrap, not on the RMSE ratio",
        "provenance": {"code": "dataprep_staging/n1_gaps/n2_power.py", "git_head": G.git_head(),
                       "inputs": [G.source(X1), G.source(BASIS), G.source(WELLS)],
                       "status": "model-only design aid; no unit read; nothing registered"}}
    G.log(f"sensitivity {res['sensitivity_all_yunlin_pumping_off']}; anomalies p50/p90 "
          f"{res['annual_anomaly_abs_p50_p90']}; last12 {a_last:+.3f}; contrast p50 "
          f"{res['contrast_at']['p50_abs_annual_anomaly']['yunlin_wells_rms_contrast_m']:.3f} m")
    G.write_new(OUT / "n2_power_check.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
