"""AMP-G across the full fan network."""
import glob
import os

import numpy as np
import pandas as pd
from ampg import ampg

FS = 24.0
rows, meta = [], []
files = sorted(glob.glob("data/wells/*.parquet"))
for i, f in enumerate(files):
    sid = os.path.basename(f)[:-8]
    s = pd.read_parquet(f)["value"].dropna().sort_index()
    if len(s) < 24 * 365 * 3:
        continue
    med, mad = s.median(), (s - s.median()).abs().median()
    if mad == 0:
        continue
    s = s[(s - med).abs() < 15 * mad].resample("1h").mean().interpolate(limit=6).dropna()
    if len(s) < 24 * 365 * 3:
        continue
    x = s.to_numpy()
    x = x - pd.Series(x).rolling(24 * 15, center=True, min_periods=1).mean().to_numpy()
    if not np.isfinite(x).all() or np.std(x) == 0:
        continue
    try:
        r = ampg(x, FS)
    except Exception:
        continue
    if not r["found"]:
        continue
    m = (pd.DataFrame({"datetime": s.index, "amp_fund": r["amp_fund"],
                       "amp_total": r["amp_total"], "amp_rot": r["amp_rot"]})
         .set_index("datetime").resample("MS").mean().reset_index())
    m["well"] = sid
    rows.append(m)
    meta.append({"well": sid, "f0": r["f0"], "n_lines": len(r["lines"]),
                 "n_harm": r["n_harm"], "n_rot": r["n_rot"], "n_tidal": r["n_tidal"],
                 "duty_h": r["duty_fit"] * 24, "alpha": r["alpha"],
                 "amp_fund_mean": float(np.mean(r["amp_fund"])),
                 "amp_total_mean": float(np.mean(r["amp_total"])), "n_hours": len(s)})
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(files)} processed, {len(meta)} usable", flush=True)
pd.concat(rows, ignore_index=True).to_parquet("data/ampg_all_monthly.parquet")
M = pd.DataFrame(meta)
M.to_csv("data/ampg_all_meta.csv", index=False)
print(f"\n{len(M)} wells with an AMP-G signature")
print(f"lines/well median {M.n_lines.median():.1f} | harmonics {M.n_harm.median():.0f} | "
      f"duty {M.duty_h.median():.1f} h/d (n={M.duty_h.notna().sum()}) | alpha {M.alpha.median():.2f}")
