"""Decisive test: does AMP track independently measured pump electricity?

Your 2023 paper could not do this -- there was no ground truth for pumping. The
116,769-pump census with monthly kWh provides one. For each monitoring well we build a
distance-weighted electricity index from the registered pumps within 1 km and correlate it
with the monthly AMP series from each estimator.

Weighting: drawdown from a pumping well falls off roughly as -ln(r) (Theis), so influence
is weighted 1/(1+r/r0) as a cheap monotone proxy. Both estimators see the identical index,
so the comparison is fair whatever the weighting.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

D = "/home/rekin226/Desktop/code_space/HydroPhysicsAI/AMP_V2/data"
amp = pd.read_parquet(f"{D}/amp_monthly.parquet")
link = pd.read_parquet(f"{D}/well_pump_link.parquet")
kwh = pd.read_parquet(f"{D}/pump_kwh_monthly.parquet")
kwh["datetime"] = pd.to_datetime(kwh.datetime)

R0 = 300.0
link["w"] = 1.0 / (1.0 + link.dist_m / R0)
j = kwh.merge(link[["well", "pump", "w"]], on="pump")
j["wkwh"] = j.electricity_kwh * j.w
idx = (j.groupby(["well", "datetime"])
         .agg(elec=("wkwh", "sum"), npump=("pump", "nunique")).reset_index())

df = amp.merge(idx, left_on=["well", "datetime"], right_on=["well", "datetime"], how="inner")
df = df.dropna(subset=["amp_v1", "amp_v2", "elec"])
print(f"matched {len(df)} well-months across {df.well.nunique()} wells "
      f"(median {int(df.groupby('well').npump.median().median())} pumps/well within 1 km)\n")

rows = []
for well, g in df.groupby("well"):
    if len(g) < 48 or g.elec.std() == 0:
        continue
    g = g.sort_values("datetime")
    de = g.elec.to_numpy()
    r = {"well": well, "n": len(g), "npump": int(g.npump.median())}
    for k in ("amp_v1", "amp_v2"):
        a = g[k].to_numpy()
        r[f"pearson_{k[-2:]}"] = np.corrcoef(a, de)[0, 1]
        r[f"spearman_{k[-2:]}"] = spearmanr(a, de).statistic
    rows.append(r)
res = pd.DataFrame(rows)

print(res.sort_values("spearman_v2", ascending=False).head(15).to_string(
    index=False, float_format=lambda v: f"{v:.3f}"))
print("\n" + "=" * 72)
print(f"wells scored: {len(res)}")
for m in ("pearson", "spearman"):
    v1, v2 = res[f"{m}_v1"], res[f"{m}_v2"]
    print(f"{m:9s}  v1 median {v1.median():+.3f}  mean {v1.mean():+.3f}  >0 at {(v1>0).sum():2d}/{len(res)}"
          f"   |  v2 median {v2.median():+.3f}  mean {v2.mean():+.3f}  >0 at {(v2>0).sum():2d}/{len(res)}")
win = (res.spearman_v2 > res.spearman_v1).sum()
print(f"\nv2 beats v1 (Spearman) at {win}/{len(res)} wells")
if len(res) > 5:
    w = wilcoxon(res.spearman_v2, res.spearman_v1)
    print(f"Wilcoxon signed-rank on paired Spearman: statistic={w.statistic:.0f}, p={w.pvalue:.4f}")
res.to_csv(f"{D}/electricity_validation.csv", index=False)
