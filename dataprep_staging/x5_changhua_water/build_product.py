"""Build the X5 products from the declared estimators (X5.changhua_water.v1) and fit.json.

* ``x5_changhua_water/v1``: Changhua non-Jiji irrigation water, annual 2012-2025 and monthly
  2012-01..2025-12 (T3 primary; T2, T3 Chow-Lin and calendar-only beside), with sd and flags.
  Post-origin rows use MOA/Jiji/coherence values published after 2022, so this version allows
  only recorded-arm and after-the-fact uses.
* ``x5_changhua_water/preorigin/v1``: the 2012-2022 rows (no post-origin value enters them:
  separate Denton blocks, heads to 2022, parameters fitted on 2015-2021) and the twin
  ``--sw-recharge`` npz files of the stage-2 rung (T2 timing; placement on the X4 canal
  service map, alternative: all Changhua paddy uniformly). Exemptions: the 2024 canal service
  map and LUIMAP 2024 (placement and the coherence mask only).

Usage: python -I build_product.py
"""
from __future__ import annotations

import calendar
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import manifest  # noqa: E402
from common.flags import validate_long_table  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from x5_changhua_water import pipeline as P  # noqa: E402
from x5_changhua_water import x5const as K  # noqa: E402
from x5_changhua_water import x5lib as L  # noqa: E402

SOURCES = ("moa_35644;jiji_weir_annual;w1_preorigin_v2_calendar;licsar_069A_06694_coherence;"
           "w1_inflow_composite_minus_H071;heads_L1_clean_v3_layer1")
INTERVAL = ("uncalibrated: annual sd from the declared definitional-mismatch rule (Yunlin "
            "office minus Jiji scatter 2016-2022, plus half the 2016 reporting step before "
            "2016); timing sd from leave-one-year-out share residuals against W1's inferred "
            "Yunlin monthly values (no monthly observation exists)")
CODE = [K.CODE_DIR / f for f in ("x5const.py", "x5lib.py", "pipeline.py", "coherence.py",
                                 "fit_x5.py", "build_product.py")]


def monthly_table(fit: dict, ann: pd.DataFrame) -> pd.DataFrame:
    z, _ = L.standardise(L.indicators("chc"), K.CALIB_YEARS)
    raw = L.indicators("chc")
    cal = np.asarray(fit["calendar"]["chc"])
    V = ann["value"]
    f3 = {"calendar": cal, "b": fit["models"]["T3"]["b"], "keys": list(K.INDICATORS_T3)}
    f2 = {"calendar": cal, "b": fit["models"]["T2"]["b"], "keys": list(K.INDICATORS_T2)}
    yrs = K.YEARS_ALL
    x3 = P.blocks(f3, z, V, yrs)
    x2 = P.blocks(f2, z, V, yrs)
    x0 = P.blocks(f3, z, V, yrs, b_override=P.zero_b(K.INDICATORS_T3))
    cl = pd.concat([P.monthly(f3, z, V, [y for y in yrs if y in blk], "chow_lin")[0]
                    for blk in (P.PRE_BLOCK, P.POST_BLOCK)]).sort_index()
    sd3 = np.asarray(fit["models"]["T3"]["loyo_share_sd_by_month"])
    sd2 = np.asarray(fit["models"]["T2"]["loyo_share_sd_by_month"])
    a = V.reindex(x3.index.year).to_numpy()
    asd = ann["sd"].reindex(x3.index.year).to_numpy()
    m = x3.index.month.to_numpy() - 1
    share3, share2 = x3.to_numpy() / a, x2.to_numpy() / a
    avail = raw.reindex(x3.index).notna()
    t = pd.DataFrame({
        "date": [p.start_time.strftime("%Y-%m-%d") for p in x3.index],
        "value": x3.to_numpy(),
        "sd": np.sqrt((share3 * asd) ** 2 + (a * sd3[m]) ** 2),
        "flag": np.where(x3.index.year <= 2024, 5, 6).astype("uint8"),
        "method": "T3 timing (calendar x exp(b.z), coh+riv+head) + proportional Denton to "
                  "the annual MOA-minus-Jiji volume",
        "product_version": "x5_changhua_water/v1", "source_ids": SOURCES,
        "value_t2": x2.to_numpy(), "sd_t2": np.sqrt((share2 * asd) ** 2 + (a * sd2[m]) ** 2),
        "value_chow_lin": cl.reindex(x3.index).to_numpy(),
        "value_calendar_only": x0.to_numpy(), "share_of_year": share3,
        "annual_value": a, "annual_sd": asd,
        "indicators_available": ["+".join(k for k in K.INDICATORS_T3 if avail.loc[p, k])
                                 or "none" for p in x3.index],
        "structural_zero": cal[m] == 0, "interval_calibration": INTERVAL,
        "units": "10^8 m3 per month (whole Changhua irrigation office, not only in-fan)"})
    t.loc[t["flag"] == 6, "method"] += "; annual value extrapolated (no MOA row)"
    return t


def grid_fields(vals: pd.Series, dates: list[str]) -> dict[str, np.ndarray]:
    zc = np.load(K.COUNTY_MAP)
    ch = zc["changhua"].astype(bool)
    x4 = np.load(K.X4_PRE)
    ws = np.asarray(x4["share_chc_service"], dtype=float)
    v3 = np.load(K.V3_NPZ, allow_pickle=True)
    di = list(np.asarray(v3["districts"]).astype(str)).index("changhua")
    pad = np.asarray(v3["paddy_m2"], dtype=float)[di] * ch
    wp = pad / pad.sum() * ws.sum()
    days = np.array([calendar.monthrange(int(d[:4]), int(d[5:7]))[1] for d in dates], float)
    vol = vals.to_numpy() * 1e8                       # m3 per month
    out = {}
    for name, w in (("service", ws), ("paddy", wp)):
        out[name] = w[:, None] * vol[None, :] / (days[None, :] * K.CELL_AREA_M2)
    out["_weights"] = {"service": ws, "paddy": wp}
    return out


def main() -> int:
    fit = json.loads((K.FIT_DIR / "fit.json").read_text(encoding="utf-8"))
    syn = json.loads((K.SYN_DIR / "synthetic_check.json").read_text(encoding="utf-8"))
    rep = json.loads((K.REPORT_DIR / "report.json").read_text(encoding="utf-8"))
    ann, meta = L.annual_volume()
    t = monthly_table(fit, ann)
    errs = validate_long_table(t)
    if errs:
        raise SystemExit(f"long table: {errs}")
    sums = t.groupby(t["date"].str[:4])["value"].sum()
    rel = float(np.max(np.abs(sums.to_numpy() / ann["value"].reindex(
        sums.index.astype(int)).to_numpy() - 1)))
    if rel > 1e-9 or (t["value"] < -1e-12).any() or not np.isfinite(t["value"]).all():
        raise SystemExit(f"benchmark check failed: max rel {rel}")
    at = ann.reset_index()
    at["method"] = np.where(at["flag"] == 5, "MOA 35644 Changhua office total minus Jiji "
                            "changhua_irrigation", "mean of 2016-2022 (no MOA row)")
    at["product_version"] = "x5_changhua_water/v1"
    at["source_ids"] = "moa_35644;jiji_weir_annual"
    at["units"] = "10^8 m3 per year"
    common_val = {"synthetic_check": {"passed": syn["passed"], "each": syn["each"]},
                  "X5C": {"verdict": rep["verdict"], "primary_r": rep["primary_r"],
                          "calendar_only_r": rep["heldout"]["calendar_only_denton"]["r"],
                          "report": rel_to_repo(K.REPORT_DIR / "report.json")},
                  "annual_meta": meta, "monthly_sum_max_rel_error": rel}

    # ---------------------------------------------------------------- full v1
    d = manifest.new_version_dir("x5_changhua_water")
    at.to_csv(d / "annual_volume.csv", index=False)
    t.to_csv(d / "monthly_changhua_nonjiji.csv", index=False)
    (d / "validation.json").write_text(json.dumps(dict(common_val, passed=True), indent=1,
                                                  ensure_ascii=False))
    inputs = [K.MOA_CSV, K.JIJI_CSV, K.W1_PRE / "district_monthly.csv",
              K.W1_PRE / "inflow_composite_daily.parquet", K.W1_PRE / "gauges_daily_L1.parquet",
              K.W1_FULL / "inflow_composite_daily.parquet",
              K.W1_FULL / "gauges_daily_L1.parquet", K.HEADS / "heads_monthly.parquet",
              K.HEADS / "station_meta_preorigin.parquet", K.X4_PRE, K.COUNTY_MAP,
              K.WORK / "coherence_monthly.csv", K.FIT_DIR / "fit.json",
              K.SYN_DIR / "synthetic_check.json", K.REPORT_DIR / "report.json"]
    params = {"prereg": K.PREREG_ID, "calib_years": K.CALIB_YEARS,
              "b": {k: v["b"] for k, v in fit["models"].items()},
              "sd_def": meta["sd_def"], "denton_blocks": ["2012-2022", "2023-2025"]}
    m = manifest.build_manifest(
        d, product="x5_changhua_water", layer="L2", inputs=[str(p) for p in inputs],
        code=[str(c) for c in CODE], parameters=params, fit_window_end="2025-12",
        validation_report="validation.json",
        gate={"passed": True, "criteria": "benchmark (monthly sums = annual volume to 1e-9), "
              "finite and non-negative, flags/sd/method on every row; synthetic recovery "
              "passed; the X5C agreement check is design-time against an inferred reference "
              "and is reported, not a skill gate"},
        allowed_uses=["forcing_recorded_arm", "statistical_layer_history",
                      "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "calibration_target", "baseline", "datum_fit",
                        "prospective_scorer", "validator", "prior", "prior_width_input",
                        "selection", "initial_conditions", "idw_field"])
    manifest.write_manifest(d, m)
    print("wrote", rel_to_repo(d))

    # ---------------------------------------------------------------- pre-origin v1
    pre = t[t["date"] < "2023-01-01"].copy()
    pre["product_version"] = "x5_changhua_water/preorigin/v1"
    dp = manifest.new_version_dir("x5_changhua_water/preorigin")
    pre.to_csv(dp / "monthly_changhua_nonjiji_preorigin.csv", index=False)
    dates = list(pre["date"])
    g = grid_fields(pd.Series(pre["value_t2"].to_numpy()), dates)
    v3d = [str(x)[:10] for x in np.load(K.V3_NPZ, allow_pickle=True)["dates"]]
    if v3d != dates:
        raise SystemExit("month axis differs from the twin's surface-water files")
    for name in ("service", "paddy"):
        f = g[name]
        if not (np.isfinite(f).all() and (f >= 0).all()):
            raise SystemExit(f"{name} field not finite/non-negative")
        np.savez(dp / f"x5_t2_{name}.npz", sw_m_per_day=f, dates=np.array(dates),
                 dx=np.float64(1000.0), n_active=np.int64(f.shape[0]),
                 sw_version=np.array(f"x5 T2 Changhua non-Jiji water, placement {name}"),
                 units=np.array("m/day of irrigation water applied (monthly mean daily rate)"),
                 source=np.array("results/dataprep/x5_changhua_water/preorigin/v1/"
                                 "monthly_changhua_nonjiji_preorigin.csv value_t2"))
    ws, wp = g["_weights"]["service"], g["_weights"]["paddy"]
    infan = {k: float(w.sum()) for k, w in g["_weights"].items()}
    vpre = dict(common_val, passed=True,
                placement={"in_fan_fraction": infan,
                           "cells_service": int((ws > 0).sum()),
                           "cells_paddy": int((wp > 0).sum()),
                           "weight_corr_service_vs_paddy": float(np.corrcoef(ws, wp)[0, 1]),
                           "fan_volume_check_max_rel": float(np.max(np.abs(
                               (g["service"] * np.array([calendar.monthrange(
                                   int(x[:4]), int(x[5:7]))[1] for x in dates]) * 1e6
                                ).sum(axis=0) / (pre["value_t2"].to_numpy() * 1e8 * infan[
                                    "service"] + 1e-30) - 1)[pre["value_t2"].to_numpy() > 0]))})
    (dp / "validation.json").write_text(json.dumps(vpre, indent=1, ensure_ascii=False))
    exemptions = [
        {"input": rel_to_repo(K.X4_PRE) + " (2024 canal service map)", "data_end": "2024-12",
         "reason": "plan U3 standard exemption, as X4 pre-origin declares: no pre-origin map "
                   "of the canal service area is public; it sets the coherence pixel mask and "
                   "the primary placement only",
         "ledger_number_id": "X5.preorigin.service_map_exemption",
         "sensitivity": "alternative placement x5_t2_paddy.npz (all Changhua paddy, LUIMAP); "
                        "validation.json placement.weight_corr_service_vs_paddy",
         "uses": ["forcing_frozen", "model_design", "design_time_evidence"]},
        {"input": rel_to_repo(K.V3_NPZ) + " (LUIMAP 2024 paddy area) and W1 pre-origin/v2",
         "data_end": "2024-12",
         "reason": "plan U3 standard exemption, as W1 pre-origin declares: no pre-origin "
                   "land-use snapshot exists; it sets the alternative placement and W1's "
                   "calendar shares",
         "ledger_number_id": "X5.preorigin.luimap_exemption",
         "sensitivity": "primary placement on the X4 service map; W1 report's LUIMAP "
                        "sensitivity (district series unaffected)",
         "uses": ["forcing_frozen", "model_design", "design_time_evidence"]}]
    mp = manifest.build_manifest(
        dp, product="x5_changhua_water/preorigin", layer="L3",
        inputs=[str(p) for p in inputs if "/full/v4/" not in str(p)] + [str(K.V3_NPZ)],
        code=[str(c) for c in CODE], parameters=dict(params, rung_timing="T2",
                                                     placements=["service", "paddy"]),
        fit_window_end="2024-12", non_exempt_fit_window_end="2022-12", exemptions=exemptions,
        validation_report="validation.json",
        gate={"passed": True, "criteria": "as x5_changhua_water/v1 for the 2012-2022 rows; "
              "fields finite, non-negative, on the twin's 2148-cell grid and month axis"},
        allowed_uses=["forcing_frozen", "model_design", "design_time_evidence",
                      "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "calibration_target", "baseline", "datum_fit",
                        "prospective_scorer", "validator", "prior", "prior_width_input",
                        "selection", "initial_conditions", "idw_field"])
    manifest.write_manifest(dp, mp)
    print("wrote", rel_to_repo(dp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
