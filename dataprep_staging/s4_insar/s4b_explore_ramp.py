"""WP S4b, exploratory and unregistered (after the gate failed): does the stable-area plane
bias the fan? Per WRA station, the 2020-01..2025-03 rate of the run-F vertical-equivalent
(3 x 3 pixels) minus the GNSS rate, regressed on station easting, for the registered plane
reference and for the offset + height reference; plus the untied G1 bias for both.

    python -I s4b_explore_ramp.py    (hydro env) -> reports/S4b_evidence/explore_ramp.json
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_evaluate as Ev  # noqa: E402
import s4b_lib as Lb  # noqa: E402

#: catalogued steps inside 2020-2025 and the plan's dropped / flagged stations (LNJS rate
#: +53.6 cm/yr, YSLL 14 steps, ANES flagged); after the fact, so post-origin evidence is allowed
EXCL = {"TKJS", "SJES", "GFES", "LNJS", "YSLL", "ANES"}


def main() -> int:
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    meta = Lb.wra_meta()
    st = [s for s in inv.index if inv.loc[s, "on_fan"] and s not in EXCL
          and s in meta.index]
    out = {}
    for nm, terms in (("plane", C.REF_TERMS), ("offset_hgt", ("offset", "hgt_km"))):
        run, g, v, _ = Ev.processed("F", terms)
        ins, gn = Ev.gnss_tables(run, g, v, inv, st, "exclude")
        sel = run["dates"] <= pd.Timestamp("2025-03-31")
        t = Lb.decimal_years(run["dates"][sel])
        rows = []
        for s in st:
            a, b = ins[s].to_numpy()[sel], gn[s].to_numpy()[sel]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 60 or (t[ok].max() - t[ok].min()) < 4:
                continue
            ri = np.polyfit(t[ok], a[ok], 1)[0]
            rg = np.polyfit(t[ok], b[ok], 1)[0]
            rows.append({"station": s, "x_km": float(meta.loc[s, "X_3826"]) / 1000.0,
                         "y_km": float(meta.loc[s, "Y_3826"]) / 1000.0, "insar_rate": ri,
                         "gnss_rate": rg, "diff_rate": ri - rg, "n_epochs": int(ok.sum())})
        df = pd.DataFrame(rows)
        X = np.column_stack([np.ones(len(df)), df.x_km - df.x_km.mean(),
                             df.y_km - df.y_km.mean()])
        beta, *_ = np.linalg.lstsq(X, df.diff_rate.to_numpy(), rcond=None)
        res = df.diff_rate.to_numpy() - X @ beta
        g1 = Ev.loso_eval(ins, gn, C.GNSS_T0, C.GNSS_T1, "none")
        out[nm] = {"n_stations": len(df), "mean_diff_rate_mm_yr": float(df.diff_rate.mean()),
                   "slope_east_mm_yr_per_km": float(beta[1]),
                   "slope_north_mm_yr_per_km": float(beta[2]),
                   "resid_sd_mm_yr": float(res.std(ddof=3)),
                   "corr_insar_gnss_rate": float(np.corrcoef(df.insar_rate, df.gnss_rate)[0, 1]),
                   "rms_diff_rate_mm_yr": float(np.sqrt(np.mean(df.diff_rate ** 2))),
                   "g1_window_untied_mean_mm": g1["mean_mm"], "g1_window_untied_rms_mm":
                   g1["rms_mm"], "rows": df.round(3).to_dict("records")}
    Lb.jdump(out, C.EVID / "explore_ramp.json")
    for k, v in out.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                  if kk != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
