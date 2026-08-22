"""Nested comparison: does the multi-band signature beat single-band AMP?

v1 is the special case where every band but the fundamental is zeroed, so this is a strict
generalisation and the comparison cannot be rigged by construction. Ground truth is
rice + dry-crop pump electricity within 1 km (nearest 100 pumps, 1/(1+r/300) weighted).
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

S = "/tmp/claude-1000/-home-rekin226-Desktop-code-space-HydroPhysicsAI/55ec15e7-c585-41a8-a463-e69e4ca3c0cf/scratchpad"
g = pd.read_parquet("data/ampg_monthly.parquet"); g["datetime"] = pd.to_datetime(g.datetime)
v1 = pd.read_parquet("data/amp_monthly.parquet"); v1["datetime"] = pd.to_datetime(v1.datetime)
g = g.merge(v1[["well", "datetime", "amp_v1", "amp_v2"]], on=["well", "datetime"], how="left")

p = pd.read_parquet(f"{S}/tpc_pumps.parquet")[["sid", "PURPOSE"]].rename(columns={"sid": "pump"})
k = pd.read_parquet("data/pump_kwh_monthly.parquet"); k["datetime"] = pd.to_datetime(k.datetime)
k = k.merge(p, on="pump", how="left")
IRR = ["農業用水(灌溉-一期水稻)", "農業用水(灌溉-二期水稻)", "農業用水(灌溉-旱作)"]
link = pd.read_parquet("data/well_pump_link.parquet").merge(p, on="pump", how="left")
L = link[link.PURPOSE.isin(IRR)].sort_values("dist_m").groupby("well").head(100).copy()
L["w"] = 1 / (1 + L.dist_m / 300)
j = k[k.PURPOSE.isin(IRR)].merge(L[["well", "pump", "w"]], on="pump")
j["x"] = j.electricity_kwh * j.w
idx = j.groupby(["well", "datetime"]).x.sum().rename("elec").reset_index()

d = g.merge(idx, on=["well", "datetime"]).dropna(subset=["elec"])
MODELS = {"v1 fundamental only": "amp_v1", "AMP-G amp_fund": "amp_fund",
          "AMP-G fund+harm": "amp_total", "AMP-G volume": "volume",
          "AMP-G volume+rot": None}
rows = []
for well, gg in d.groupby("well"):
    gg = gg.dropna(subset=["amp_v1", "amp_fund", "amp_total", "volume"])
    if len(gg) < 48 or gg.elec.std() == 0:
        continue
    r = {"well": well, "n": len(gg)}
    for name, col in MODELS.items():
        v = (gg.volume + gg.amp_rot) if col is None else gg[col]
        r[name] = spearmanr(v, gg.elec).statistic
    rows.append(r)
res = pd.DataFrame(rows)
print(f"wells scored: {len(res)}\n")
print(f"{'model':<24}{'median rho':>12}{'mean':>8}{'>0':>7}{'vs v1 p':>10}")
print("-" * 62)
base = res["v1 fundamental only"]
for name in MODELS:
    v = res[name]
    p_ = "-" if name == "v1 fundamental only" else f"{wilcoxon(v, base).pvalue:.4f}"
    print(f"{name:<24}{v.median():>12.3f}{v.mean():>8.3f}{(v>0).sum():>4d}/{len(v)}{p_:>10}")
res.to_csv("data/ampg_validation.csv", index=False)
best = max(MODELS, key=lambda n: res[n].median())
print(f"\nbest: {best}  (median rho {res[best].median():+.3f} vs v1 {base.median():+.3f}, "
      f"improvement {100*(res[best].median()-base.median())/abs(base.median()):+.0f}%)")
print(f"beats v1 at {(res[best] > base).sum()}/{len(res)} wells")
