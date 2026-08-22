"""Compute AMP v1 and v2 monthly series for every monitoring well with a raw record."""
import glob, os
import numpy as np, pandas as pd
from amp import amp_v1, amp_v2

RAW = "/home/rekin226/Desktop/code_space/HydroPhysicsAI/chou-shui-data/chou-shui-data/data/ls_cache"
OUT = "/home/rekin226/Desktop/code_space/HydroPhysicsAI/AMP_V2/data"
FS = 24.0                       # analyse at hourly resolution


def clean(s: pd.Series) -> pd.Series:
    """Robust sentinel/spike removal, then hourly regularisation."""
    med, mad = s.median(), (s - s.median()).abs().median()
    if mad == 0:
        mad = s.std() or 1.0
    s = s[(s - med).abs() < 15 * mad]
    return s.resample("1h").mean().interpolate(limit=6)


rows = []
for f in sorted(glob.glob(f"{RAW}/gw__*__raw.parquet")):
    sid = f.split("gw__")[1].split("__")[0]
    s = pd.read_parquet(f)["value"].dropna().sort_index()
    if len(s) < 20000:
        continue
    s = clean(s).loc["2012-01-01":"2022-12-31"].dropna()
    if len(s) < 24 * 365 * 3:
        continue
    x = s.to_numpy()
    r1 = amp_v1(x, FS, f_pump=1.0, bandwidth=0.05)
    r2 = amp_v2(x, FS, mode="band")
    df = pd.DataFrame({"datetime": s.index, "amp_v1": r1["amp"],
                       "amp_v2": r2["amp"] if r2["found"] else np.nan,
                       "f_v2": r2["freq"] if r2["found"] else np.nan})
    m = df.set_index("datetime").resample("MS").mean()
    m["well"] = sid
    m["f_v1"] = r1["f_pump"]
    m["found_v2"] = r2["found"]
    rows.append(m.reset_index())
    print(f"{sid}: n={len(s)}  v1 f={r1['f_pump']:.3f} AMPmed={np.median(r1['amp'])*100:5.2f}cm | "
          f"v2 found={r2['found']} f={r2.get('f_pump', np.nan):.3f} "
          f"AMPmed={np.nanmedian(r2['amp'])*100 if r2['found'] else np.nan:5.2f}cm "
          f"nIMF={r2['n_imf']} nCand={r2.get('n_cand','-')}", flush=True)

out = pd.concat(rows, ignore_index=True)
os.makedirs(OUT, exist_ok=True)
out.to_parquet(f"{OUT}/amp_monthly.parquet")
print(f"\nwrote {len(out)} well-months for {out.well.nunique()} wells; "
      f"v2 found at {out.groupby('well').found_v2.first().sum()} wells")
