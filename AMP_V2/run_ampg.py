"""AMP-G on the real network, then the nested comparison against irrigation electricity."""
import glob

import pandas as pd
from ampg import ampg

RAW = "/home/rekin226/Desktop/code_space/HydroPhysicsAI/chou-shui-data/chou-shui-data/data/ls_cache"
FS = 24.0
rows, meta = [], []
for f in sorted(glob.glob(f"{RAW}/gw__*__raw.parquet")):
    sid = f.split("gw__")[1].split("__")[0]
    s = pd.read_parquet(f)["value"].dropna().sort_index()
    med, mad = s.median(), (s - s.median()).abs().median()
    if mad == 0 or len(s) < 20000:
        continue
    s = s[(s - med).abs() < 15 * mad].resample("1h").mean().interpolate(limit=6)
    s = s.loc["2012-01-01":"2022-12-31"].dropna()
    if len(s) < 24 * 365 * 3:
        continue
    x = s.to_numpy()
    x = x - pd.Series(x).rolling(24 * 15, center=True, min_periods=1).mean().to_numpy()
    r = ampg(x, FS)
    if not r["found"]:
        continue
    df = pd.DataFrame({"datetime": s.index, "amp_fund": r["amp_fund"],
                       "amp_total": r["amp_total"], "amp_harm": r["amp_harm"],
                       "amp_rot": r["amp_rot"], "volume": r["volume"]})
    m = df.set_index("datetime").resample("MS").mean().reset_index()
    m["well"] = sid
    rows.append(m)
    meta.append({"well": sid, "f0": r["f0"], "n_harm": r["n_harm"], "n_rot": r["n_rot"],
                 "n_tidal": r["n_tidal"], "duty_h": r["duty_fit"] * 24, "alpha": r["alpha"],
                 "n_lines": len(r["lines"])})
    print(f"{sid}: f0={r['f0']:.4f} lines={len(r['lines'])} harm={r['n_harm']} rot={r['n_rot']} "
          f"tide={r['n_tidal']} duty={r['duty_fit']*24:.1f}h alpha={r['alpha']:.2f}", flush=True)

pd.concat(rows, ignore_index=True).to_parquet("data/ampg_monthly.parquet")
M = pd.DataFrame(meta)
M.to_csv("data/ampg_meta.csv", index=False)
print(f"\n{len(M)} wells | median lines/well {M.n_lines.median():.1f} | "
      f"harmonics {M.n_harm.median():.0f} | rotation lines {M.n_rot.median():.0f}")
print(f"apparent duty: median {M.duty_h.median():.1f} h/day  (IQR {M.duty_h.quantile(.25):.1f}-{M.duty_h.quantile(.75):.1f})")
