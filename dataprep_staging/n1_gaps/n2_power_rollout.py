"""Power check 2 for the DRAFT N2 registration: the page model's own dynamics (no unit read).

``n2_power.py`` sized the contrast from the sustained-cut basis, which over-states the
response to month-to-month anomalies. Here the frozen page model (stage3_datum_gate theta,
coupled_leveling column; CPU rollouts with the frozen code via ``n1_policy.policy_lib``)
is run twice over the record months 2012-01..2022-12 from the same initial field:

- ZR arm: pumping energy = the record's month-of-year climatology (2012-2022) in every
  cell, the rule the frozen forecast uses after the origin;
- PR arm: the same climatology times the X1 Yunlin county ratio of each month to its
  2012-2022 month-of-year mean (the draft's forcing rule; R0/innov carry Yunlin pumping
  only, so the ratio scales every cell).

Rain recharge and the other forcings are the record's in both arms, so PR - ZR is the
model's response to the pumping anomaly alone. Per Nov-Oct window it reports the contrast
at the 158 well cells (each well's own layer) split into the window mean (a level shift,
confounded with any level bias of the zero-response arm) and the within-window part
(well-demeaned; what the draft's primary slope would use), and the fan subsidence
contrast. The model's own inputs (record forcing and the month-0 head field) are read; no
head or leveling observation of any window and no electricity month after 2026-08 is.

Usage: CUDA_VISIBLE_DEVICES="" python -I dataprep_staging/n1_gaps/n2_power_rollout.py
Output: results/dataprep/reports/N1_gaps_2026-10-24/science-15/n2_power_rollout.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from n1_gaps import gaps_lib as G  # noqa: E402
from n1_gaps.n2_power import LAST_ALLOWED, WELLS, X1  # noqa: E402

OUT = G.OUT_ROOT / "science-15"
THETA = G.RUNS / "stage3_datum_gate" / "stage3_theta.json"
COLUMN = G.RUNS / "stage3_datum_gate" / "coupled_leveling" / "vep_zonal_leveling.json"


def main() -> int:
    os.chdir(G.REPO)
    from n1_policy import policy_lib as PL

    x = pd.read_parquet(X1)
    x = x[x.county == "yunlin"].copy()
    x["p"] = pd.PeriodIndex(x.date, freq="M")
    x = x[x.p <= LAST_ALLOWED].set_index("p").sort_index()
    rec = x[(x.index.year >= 2012) & (x.index.year <= 2022)]
    clim = rec.groupby(rec.index.month)["value"].mean()
    ratio = (x["value"] / x.index.month.map(clim).to_numpy()).reindex(PL.MONTHS)
    if ratio.isna().any():
        raise SystemExit("X1 ratio missing inside 2012-2022")

    G.log("building the page model on CPU")
    tm = PL.TwinModel(str(THETA), str(COLUMN))
    if tm.meta.get("eta_classes"):
        raise SystemExit("per-class energy model not expected for the page theta")
    E = tm.energy()                                        # (A, T-1); step j = MONTHS[j+1]
    step_m = np.array([p.month for p in PL.MONTHS[1:E.shape[-1] + 1]])
    E_clim = np.zeros_like(E)
    for m in range(1, 13):
        sel = step_m == m
        E_clim[:, sel] = E[:, sel].mean(axis=1, keepdims=True)
    r = ratio.to_numpy()[1:E.shape[-1] + 1]
    G.log("rollouts")
    H_zr = tm.heads(E_clim)
    H_pr = tm.heads(E_clim * r[None, :])
    S_zr, S_pr = tm.subsidence(H_zr), tm.subsidence(H_pr)
    G.log("rollouts done")

    wells = pd.read_csv(WELLS, usecols=["sid", "layer", "x", "y"], dtype={"sid": str})
    wc = tm.cell_of(wells[["x", "y"]].to_numpy(float))
    ok = wc >= 0
    wells, wc = wells[ok], wc[ok]
    lay = np.clip(wells["layer"].to_numpy(int) - 1, 0, H_zr.shape[0] - 1)
    from hydrophysics.twin.zone_county import DEFAULT_MAP, county_flags_for_grid
    ch = np.asarray(county_flags_for_grid(tm.inp.grid, DEFAULT_MAP)[0], bool)[wc]
    C = H_pr[lay, wc, :] - H_zr[lay, wc, :]                # (n_wells, T) m, + = higher head
    dS = (S_pr - S_zr).mean(axis=0) * 100.0                # fan cm, + = more sinking
    rows = []
    for y0 in range(2012, 2022):
        sel = np.array([(p >= pd.Period(f"{y0}-11", "M")) and (p <= pd.Period(f"{y0 + 1}-10", "M"))
                        for p in PL.MONTHS])
        c = C[:, sel]
        cm = c.mean(axis=1, keepdims=True)
        within = c - cm
        ann = float(x.loc[pd.Period(f"{y0}-11", "M"):pd.Period(f"{y0 + 1}-10", "M"), "value"].sum()
                    / clim.sum() - 1.0)
        rows.append({
            "window": f"{y0}-11..{y0 + 1}-10", "annual_anomaly": ann,
            "yunlin_mean_contrast_m": float(cm[~ch].mean()),
            "yunlin_within_window_rms_m": float(np.sqrt((within[~ch] ** 2).mean())),
            "changhua_mean_contrast_m": float(cm[ch].mean()),
            "changhua_within_window_rms_m": float(np.sqrt((within[ch] ** 2).mean())),
            "fan_subsidence_contrast_change_cm": float(dS[sel][-1] - dS[sel][0])})
    df = pd.DataFrame(rows)
    res = {
        "what": "page model's head contrast between recorded-anomaly pumping (PR) and "
                "climatology pumping (ZR), replayed over the record; model output only",
        "n_wells": int(len(wc)), "n_yunlin_wells": int((~ch).sum()),
        "windows": rows,
        "summary": {
            "yunlin_abs_mean_contrast_m_p50_p90": [
                float(df.yunlin_mean_contrast_m.abs().median()),
                float(df.yunlin_mean_contrast_m.abs().quantile(0.9))],
            "yunlin_within_window_rms_m_p50_p90": [
                float(df.yunlin_within_window_rms_m.median()),
                float(df.yunlin_within_window_rms_m.quantile(0.9))],
            "slope_mean_contrast_per_unit_annual_anomaly_m": float(np.polyfit(
                df.annual_anomaly, df.yunlin_mean_contrast_m, 1)[0]),
            "fan_subsidence_contrast_per_year_cm_p50_abs": float(
                df.fan_subsidence_contrast_change_cm.abs().median())},
        "reading": "the level part of the contrast is about the annual anomaly times the "
                   "slope above; the within-window part is what survives per-well "
                   "demeaning. The registered informative_head_m is 0.05 m.",
        "provenance": {"code": "dataprep_staging/n1_gaps/n2_power_rollout.py",
                       "git_head": G.git_head(),
                       "inputs": [G.source(THETA), G.source(COLUMN), G.source(X1),
                                  G.source(WELLS)],
                       "status": "model-only design aid; no unit read; nothing registered"}}
    G.log(f"summary {res['summary']}")
    G.write_new(OUT / "n2_power_rollout.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
