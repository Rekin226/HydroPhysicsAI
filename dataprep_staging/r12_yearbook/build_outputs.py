"""Assemble yearbook monthly heads for the target wells, validate against held data, report coverage."""
from __future__ import annotations

import os  # noqa
# ruff: noqa
import glob, json
import numpy as np
import pandas as pd

ROOT = os.environ.get("HYDRO_YEARBOOK_ROOT", "data_fetch_api/new_measurements/wra_hydro_yearbook_gw")
D = os.path.join(os.environ.get("HYDRO_RESULTS", "results"), "dataprep/heads/L1_clean/v3/")
meta = pd.read_parquet(D + "station_meta_clean.parquet")[["sid", "WellName", "CountyName", "TownName", "layer_code"]]
held = pd.read_parquet(D + "heads_monthly.parquet")
held = held[held.flag <= 1].copy()
held["year"] = held.date.dt.year; held["month"] = held.date.dt.month

p = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/tmp/parsed/pages_*.parquet"))], ignore_index=True)
p["source_pdf"] = "gw_" + p.year.astype(str) + ".pdf"
diff = (p.value_m - p.daily_mean_m).abs()
flag = np.full(len(p), 0)
flag[(p.yb_mark == "*").values] = 1            # yearbook: month incomplete, statistic for reference only
flag[p.yb_mark.str.contains("@").values] = 2   # yearbook: contains gap-filled days
flag[(p.value_m.notna() & ((diff > 0.02) | (p.n_days == 0))).values] = 3  # printed mean inconsistent with printed daily values
flag[p.value_m.isna().values] = 9              # '***' or blank: no monthly mean printed (value NaN, nothing filled)
p["parse_flag"] = flag
p["sid"] = p.yb_station_id
p["well_name"] = p.yb_name
cols = ["sid", "well_name", "year", "month", "value_m", "source_pdf", "page", "parse_flag",
        "yb_mark", "n_days", "daily_mean_m", "n_days_filled", "well_top_m", "well_depth_m", "basin"]
p[cols].to_parquet(f"{ROOT}/out/heads_monthly_yearbook_all_taiwan.parquet", index=False)
t = p[p.sid.isin(meta.sid)].merge(meta, on="sid", how="left")
t[cols + ["CountyName", "TownName", "layer_code"]].sort_values(["sid", "year", "month"]).to_parquet(
    f"{ROOT}/out/heads_monthly_yearbook.parquet", index=False)

# ---- match table: every yearbook station of the Choshui fan basin + every matched sid
st = p.groupby("yb_station_id").agg(yb_name=("yb_name", "last"), basin=("basin", "last"),
                                    years=("year", lambda s: " ".join(map(str, sorted(set(s))))),
                                    well_top_min=("well_top_m", "min"), well_top_max=("well_top_m", "max")).reset_index()
st = st[st.basin.str.contains("濁水") | st.yb_station_id.isin(meta.sid)]
st = st.merge(meta.rename(columns={"sid": "yb_station_id"}), on="yb_station_id", how="left")
st["sid"] = np.where(st.WellName.notna(), st.yb_station_id, "")
st["match_method"] = np.where(st.WellName.notna(), "station_number_exact", "unmatched_not_in_our_344")
norm = lambda s: str(s).replace("（", "(").replace("）", ")").replace(" ", "")
st["name_agrees"] = [norm(a) == norm(b) if isinstance(b, str) else None for a, b in zip(st.yb_name, st.WellName)]
st.to_csv(f"{ROOT}/out/match_table.csv", index=False)
missing = meta[~meta.sid.isin(p.yb_station_id)]
missing.to_csv(f"{ROOT}/out/our_wells_never_in_parsed_yearbooks.csv", index=False)

# ---- validation on overlap years
ok = t[t.parse_flag.isin([0, 2])]
v = ok.merge(held[["sid", "year", "month", "value", "value_raw", "flag"]], on=["sid", "year", "month"])
v["d"] = v.value_m - v.value; v["d_raw"] = v.value_m - v.value_raw
rows = []
for sid, g in v.groupby("sid"):
    if len(g) < 6: continue
    rows.append(dict(sid=sid, n=len(g), bias=g.d.mean(), median_diff=g.d.median(), sd_diff=g.d.std(),
                     rmse=float(np.sqrt((g.d ** 2).mean())), bias_raw=g.d_raw.mean(),
                     rmse_raw=float(np.sqrt((g.d_raw ** 2).mean())),
                     corr=g.value_m.corr(g.value) if g.value.std() > 0 else np.nan,
                     years=" ".join(map(str, sorted(set(g.year))))))
w = pd.DataFrame(rows)
w["const_offset"] = (w.median_diff.abs() > 0.05) & (w.sd_diff < 0.03)
w.to_csv(f"{ROOT}/out/validation_per_well.csv", index=False)
peryear = {}
for y, g in v.groupby("year"):
    b = g.groupby("sid").d.mean()
    peryear[int(y)] = dict(n_wells=int(b.size), n_months=int(len(g)), median_abs_bias=float(b.abs().median()),
                           share_wells_within_0p05=float((b.abs() <= 0.05).mean()),
                           share_months_within_0p05=float((g.d.abs() <= 0.05).mean()),
                           share_months_within_0p01=float((g.d.abs() <= 0.01).mean()))
val = dict(
    overlap_years=sorted(map(int, v.year.unique())), yearbook_flags_used=[0, 2], held_filter="flag<=1",
    n_wells=int(len(w)), n_well_months=int(len(v)),
    median_abs_bias_m=float(w.bias.abs().median()), p90_abs_bias_m=float(w.bias.abs().quantile(0.9)),
    median_rmse_m=float(w.rmse.median()), median_rmse_vs_value_raw_m=float(w.rmse_raw.median()),
    median_abs_bias_vs_value_raw_m=float(w.bias_raw.abs().median()),
    median_corr=float(w["corr"].median()), share_wells_corr_ge_0p99=float((w["corr"] >= 0.99).mean()),
    share_wells_abs_bias_le_0p05=float((w.bias.abs() <= 0.05).mean()),
    share_wells_abs_bias_le_0p02=float((w.bias.abs() <= 0.02).mean()),
    share_months_abs_diff_le_0p05=float((v.d.abs() <= 0.05).mean()),
    share_months_abs_diff_le_0p01=float((v.d.abs() <= 0.01).mean()),
    wells_constant_offset=w[w.const_offset][["sid", "n", "median_diff", "sd_diff", "years"]].round(3).to_dict("records"),
    wells_abs_bias_gt_0p05=w[w.bias.abs() > 0.05][["sid", "n", "bias", "sd_diff", "rmse", "corr", "years"]].round(3).to_dict("records"),
    per_year=peryear)
json.dump(val, open(f"{ROOT}/out/validation.json", "w"), indent=1, ensure_ascii=False)

# ---- coverage
h2 = held[(held.year >= 2012) & (held.year <= 2022)]
long_sids = set(h2.groupby("sid").size().loc[lambda s: s >= 100].index)
have = t[t.value_m.notna()].groupby(["sid", "year"]).size().rename("n_months").reset_index().merge(meta, on="sid")
cov = []
for y, g in have.groupby("year"):
    r = dict(year=int(y), wells_any=len(g), wells_12_months=int((g.n_months == 12).sum()),
             long_record_wells_any=int(g.sid.isin(long_sids).sum()))
    for c, gg in g.groupby("CountyName"): r[f"county_{c}"] = len(gg)
    for l, gg in g.groupby("layer_code"): r[f"layer_{l}"] = len(gg)
    cov.append(r)
cov = pd.DataFrame(cov).fillna(0)
cov.to_csv(f"{ROOT}/out/coverage_by_year.csv", index=False)
print(cov.to_string()); print("long-record wells (>=100 of 132 months 2012-2022, flag<=1):", len(long_sids))
pre = have[have.year <= 2011]
print("wells with any pre-2012 yearbook data:", pre.sid.nunique(), "long:", pre[pre.sid.isin(long_sids)].sid.nunique())
print("flag counts target wells:", t.parse_flag.value_counts().to_dict())
print("unmatched Choshui yb stations:", st[st.sid == ""].yb_station_id.tolist(), "name disagreements:", int((st.name_agrees == False).sum()))
print(json.dumps({k: val[k] for k in val if not k.startswith("wells_")}, indent=1))
print("const offset:", val["wells_constant_offset"]); print("bias>0.05:", len(val["wells_abs_bias_gt_0p05"]))
print(pd.DataFrame(val["wells_abs_bias_gt_0p05"]).to_string())
