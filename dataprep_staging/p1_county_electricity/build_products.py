"""WP P1: build ``county_electricity/{preorigin,full}/v<N>/`` with manifests and ledger rows.

Inputs are L0 files only (the census build, phi_in and the validation results are rebuilt
from L0 by this WP's scripts and shipped as outputs). Run after census_monthly.py, phi_in.py
and validate.py (their outputs in results/dataprep/county_electricity/_build/).

Usage: python -I build_products.py [--dry]   (--dry stops before creating version dirs)
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import data as dmod  # noqa: E402
import lib  # noqa: E402
import phi_in as phimod  # noqa: E402
import production as prod  # noqa: E402
from common import ledger, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, REPO, rel_to_repo  # noqa: E402

BUILD = dmod.BUILD
VAL = BUILD / "validation"
PREREG = REPO / "results/dataprep/preregistration/P1_county_electricity_2026-10-12.json"
CODE = [HERE / n for n in ("lib.py", "data.py", "production.py", "census_monthly.py",
                           "phi_in.py", "validate.py", "prereg.py", "fetch_check.py",
                           "build_products.py")] + [
    REPO / "hydrophysics/twin/pumping.py", REPO / "hydrophysics/twin/scenario.py",
    HERE.parent / "common/manifest.py", HERE.parent / "common/ledger.py",
    HERE.parent / "common/hashing.py"]
INPUTS = dmod.L0_INPUTS + [p for p in phimod.INPUTS]
FWE = {"preorigin": "2022-12", "full": "2026-08"}


def next_version(product: str) -> str:
    base = DATAPREP / product
    ns = [int(p.name[1:]) for p in base.iterdir()
          if p.is_dir() and re.match(r"^v[1-9]\d*$", p.name)] if base.exists() else []
    return f"v{max(ns, default=0) + 1}"


def source_ids(*keys: str) -> str:
    m = {"tpc": "tpc_d007012/2026-10-12", "nat": "moeaea_agri_electricity/2026-10-12",
         "census": "AMP_V2/data/tpc_pumps.parquet;AMP_V2/data/pump_kwh_all.parquet",
         "sub": "tpc_d003004/2026-10-12", "phi": "phi_in.json"}
    return ";".join(m[k] for k in keys)


def tpc_table(D, F, S, pv, V) -> pd.DataFrame:
    t = D["tpc"].copy()
    t["value"] = t["kwh"] / 1e6
    t = t.rename(columns={"pct": "pct_of_county"})
    t["sd"], t["flag"], t["method"] = np.nan, 0, "tpc_d007012_observed"
    t["source_ids"] = source_ids("tpc")
    rows = [t[["date", "county", "sector", "value", "sd", "flag", "method", "pct_of_county",
               "source_ids"]]]
    ag = S["ag"]
    bc = ag[S["is_backcast"]]
    for c, cty in (("CH", lib.CH), ("YL", lib.YL)):
        cv = V[c]
        rows.append(pd.DataFrame({"date": bc.index, "county": cty, "sector": lib.AG,
                                  "value": bc[f"{c}_bc"].to_numpy(),
                                  "sd": bc[f"{c}_bc"].to_numpy() * cv, "flag": 5,
                                  "method": "national_share_backcast_trend",
                                  "pct_of_county": np.nan,
                                  "source_ids": source_ids("nat", "tpc")}))
    nat = D["d"]["nat_ag"].dropna()
    rows.append(pd.DataFrame({"date": nat.index, "county": "全國", "sector": "農業部門(能源署)",
                              "value": nat.to_numpy(), "sd": np.nan, "flag": 0,
                              "method": "moeaea_national_observed", "pct_of_county": np.nan,
                              "source_ids": source_ids("nat")}))
    out = pd.concat(rows, ignore_index=True)
    out["product_version"] = pv
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["county", "sector", "date"]).reset_index(drop=True)


def candidates(S, pv) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, wide = [], {}
    for name, sc in S["estimators"].items():
        for scope, df in sc.items():
            x = df.copy()
            x["date"] = x.index
            x["estimator"], x["scope"] = name, scope
            x["is_backcast"] = S["is_backcast"].to_numpy()
            x["is_extension"] = S["is_extension"].to_numpy()
            rows.append(x.reset_index(drop=True))
            wide[f"{name}__{scope}"] = df["value"]
            wide[f"{name}__{scope}__sd"] = df["sd"]
    long = pd.concat(rows, ignore_index=True)
    long = long[long["flag"] != 8].copy()
    long["unit"] = "GWh/month, model units (Yunlin cleaned-census kWh currency)"
    long["product_version"] = pv
    long["source_ids"] = np.where(long["estimator"] == "national_subindustry_weak",
                                  source_ids("tpc", "sub"),
                                  np.where(long["scope"] == "in_grid",
                                           source_ids("tpc", "nat", "census", "phi"),
                                           source_ids("tpc", "nat", "census")))
    cols = ["date", "estimator", "scope", "value", "sd", "flag", "method", "is_backcast",
            "is_extension", "unit", "product_version", "source_ids"]
    w = pd.DataFrame(wide)
    w["is_backcast"] = S["is_backcast"]
    w["is_extension"] = S["is_extension"]
    w.index.name = "date"
    return long[cols].sort_values(["estimator", "scope", "date"]).reset_index(drop=True), \
        w.reset_index()


def census_table(D, pv) -> pd.DataFrame:
    c = D["census"]
    keep = [x for x in c.columns if (x.startswith("clean_") and "bill" not in x)
            or x in ("dedup_total", "dedup_agri")]
    m = c[keep].reset_index().melt(id_vars="date", var_name="series", value_name="value")
    m["sd"], m["flag"], m["method"] = np.nan, 0, "census_clean_dedupe_cap1"
    m.loc[m.series.str.startswith("dedup"), "method"] = "census_dedupe_nocap"
    m["unit"] = "GWh/month (Yunlin county census totals)"
    m["product_version"] = pv
    m["source_ids"] = source_ids("census")
    return m


def annual_table(D, S) -> dict:
    d = D["d"]
    idx = S["ag"].index
    t = pd.DataFrame(index=idx)
    t["CH_ag"] = S["ag"]["CH_ag_x"]
    t["YL_ag"] = S["ag"]["YL_ag_x"]
    t["YL_clean"] = S["YL_clean"]["value"]
    for name, sc in S["estimators"].items():
        t[f"{name}_county"] = sc["county"]["value"]
    t["scaled_in_grid"] = S["estimators"]["scaled"]["in_grid"]["value"]
    n = t.groupby(t.index.year).size()
    y = t.groupby(t.index.year).sum(min_count=1)
    y["n_months"] = n
    del d
    return {str(k): {c: (None if pd.isna(v) else round(float(v), 4)) for c, v in row.items()}
            for k, row in y.iterrows()}


def cv_j(D) -> dict:
    c = D["census"].loc["2012-02-01":"2022-12-01"]
    out = {}
    for j, col in (("irrigation", "clean_irrigation"), ("other", "clean_other_classes")):
        v = c[col]
        out[j] = float(v.std(ddof=0) / v.mean())
    return {"status": "provisional: even-split recorded census (P2a Chow-Lin supersedes, plan "
                      "P1 step 7)", "window": ["2012-02", "2022-12"], "cv": out,
            "superseded_values_physics_scout": {"irrigation": 0.264, "other": 0.197}}


def decomposition(D, F, S, V, Vall, variant) -> dict:
    sep = F["sep"]
    syn = Vall["T8_SYN_coverage"][variant]
    k = syn["remedy_reported_only"]["k_halfwidth_factor"]
    a, lo, hi = sep["a"], sep["a_lo"], sep["a_hi"]
    mis = Vall["T9_MIS"][variant]
    phi = F["phi"]
    ratios = prod.estimator_ratios(D, F)
    a_pre = Vall["production_fits"]["preorigin"]["sep"]["a"]
    sig_mu = mis["sigma_mu_ln"] if variant == "preorigin" else \
        Vall["T9_MIS"]["preorigin"]["sigma_mu_ln"]
    yl = S["YL_clean"]["value"]
    ing = {}
    for t0, t1 in (("2012-01", "2022-12"), ("2013-01", "2022-12")) + (
            (("2016-01", "2025-06"),) if variant == "full" else ()):
        v = yl.loc[t0:t1]
        yrs = len(v) / 12
        ing[f"{t0}..{t1}"] = {"YL_clean_GWh_yr": float(v.sum() / yrs),
                              "mu_elec_county_GWh_yr": float(a_pre * v.sum() / yrs),
                              "mu_elec_in_grid_GWh_yr": float(a_pre * phi["phi_in"]
                                                              * v.sum() / yrs)}
    members = {kk: vv for kk, vv in ratios.items() if isinstance(vv, float)
               and kk not in ("CH_ag_over_YL_ag",)}
    mspread = [min(members.values()), max(members.values())]
    spread = [min(mspread[0], 0.5), max(mspread[1], 1.27)]
    out = {
        "variant": variant,
        "source_separation": {
            "model": "CH_ag(t) = a YL_clean(t) + b (YL_ag(t) - YL_clean(t)), NNLS",
            **{kk: sep[kk] for kk in ("a", "b", "r2", "a_lo", "a_hi", "b_lo", "b_hi",
                                      "a_boot_sd", "b_boot_sd", "n_months", "n_years",
                                      "resid_sd", "resid_ac1", "n_boot", "seed")},
            "window": F["window"],
            "a_interval_status": ("uncalibrated: the year-block percentile interval covered "
                                  f"{syn['point']:.3f} of synthetic truths (Wilson 95 % "
                                  f"{syn['wilson95'][0]:.3f}-{syn['wilson95'][1]:.3f}), "
                                  "failing the pre-registered 0.90 test (T8)"),
            "a_interval_remedy_reported_only": {
                "k_halfwidth_factor": k, "a_lo": a - k * (a - lo), "a_hi": a + k * (hi - a),
                "coverage_fresh_synthetic": syn["remedy_reported_only"]["coverage_fresh_set"],
                "status": "synthetic-calibrated remedy; did NOT pass the pre-registered gate"},
        },
        "irrigation_only": {kk: F["irr"][kk] for kk in ("a", "b", "r2", "a_lo", "a_hi")},
        "backcast": {c: {"s_moy": [float(x) for x in F["bc"][c]["s_moy"]],
                         "beta_share_per_yr": F["bc"][c]["beta"], "t_ref": F["bc"][c]["t_ref"],
                         "moy_window": F["bc"][c]["moy_window"],
                         "trend_window": F["bc"][c]["trend_window"]}
                     for c in ("CH", "YL")},
        "transfer_ratio_r_moy": {"r": [float(x) for x in F["r"]],
                                 "cv_by_month": [float(x) for x in F["r_cv"]],
                                 "window": F["window"]},
        "national_subindustry_weak": {kk: vv for kk, vv in F["sub"].items()
                                      if kk != "series_CH"},
        "estimator_ratios": ratios,
        "crop_statistics_prior_width_input": {
            "value_x_yunlin": 0.5,
            "status": "design-time value quoted from the plan (landuse_crop_priors pilot via "
                      "plan P1 step 4); P3's pre-origin crop-statistics total supersedes it; "
                      "ledger role prior_width_input; never the level, never validates mu"},
        "phi_in": phi | {"towns": "see phi_in.json"},
        "in_grid_anchor": {
            "rule": "mu_elec_in_grid(t) = a_pre x YL_clean(t) x phi_in",
            "a_pre": a_pre, "phi_in": phi["phi_in"], "sigma_ln_phi": phi["sigma_ln_phi"],
            "time_means": ing,
            "monthly_series": "changhua_candidates.parquet, estimator 'scaled' (preorigin) or "
                              "'scaled_apre' (full), scope 'in_grid'"},
        "mu_prior": {
            "log_sd": sig_mu,
            "components": {"base": 0.5, "sigma_ln_phi": phi["sigma_ln_phi"],
                           "b_mis_ln": Vall["T9_MIS"]["preorigin"]["b_mis_ln"]},
            "rule": "sigma_mu = sqrt(0.5^2 + sigma_ln_phi^2 + b_mis^2) (pre-registered; "
                    "preorigin window decides)",
            "estimator_members_x_yunlin": members,
            "estimator_member_spread": mspread,
            "plan_range_x_yunlin": [0.5, 1.27],
            "union_spread_x_yunlin": spread,
            "spread_in_sd_units_around_a_pre": [float(np.log(spread[0] / a_pre) / sig_mu),
                                                float(np.log(spread[1] / a_pre) / sig_mu)],
            "calibration": "uncalibrated (no public data can test the pump-like share; plan "
                           "P1 risks, R-a/R-b)"},
        "cv_j": cv_j(D),
        "annual_GWh": annual_table(D, S),
        "units": "GWh; model units = Yunlin cleaned-census kWh (plan header)",
    }
    if variant == "full":
        est = S["estimators"]
        tr, sc = est["transfer"]["county"]["value"], est["scaled"]["county"]["value"]
        out["no_splice_check"] = {
            "transfer_2024_GWh": float(tr.loc["2024"].sum()),
            "scaled_afull_2024_GWh": float(sc.loc["2024"].sum()),
            "transfer_2025_06_GWh": float(tr.loc["2025-06-01"]),
            "scaled_afull_2025_06_GWh": float(sc.loc["2025-06-01"]),
            "ratio_2024": float(tr.loc["2024"].sum() / sc.loc["2024"].sum()),
            "note": "splicing the transfer estimator on at 2025-07 would put this step into "
                    "the recorded-forcing arm; the extension uses a x YL_clean_hat instead"}
    return out


def build_variant(variant: str, Vall: dict, dry: bool = False) -> dict:
    D = dmod.load(variant)
    F = prod.fits(D)
    product = f"county_electricity/{variant}"
    ver = next_version(product)
    pv = f"{product}/{ver}"
    bv = Vall["T1_BV1"] if variant == "preorigin" else Vall["T2_BV2"]
    V = {"backcast_monthly_rel_rms": {c: bv[c]["trend"]["monthly_rel_rms"] for c in ("CH", "YL")},
         "extension_monthly_rel_rms": Vall["T4_EXT_LOYO"]["monthly_rel_rms"]}
    if variant == "full":
        sp = Vall["production_fits"]["preorigin"]["sep"]
        V["a_pre"], V["a_pre_cv"] = sp["a"], sp["a_boot_sd"] / sp["a"]
    # consistency: the production fits must equal those the validation used
    vf = Vall["production_fits"][variant]["sep"]
    assert abs(vf["a"] - F["sep"]["a"]) < 1e-12 and abs(vf["b"] - F["sep"]["b"]) < 1e-12
    S = prod.series(D, F, V)
    stage = BUILD / f"stage_{variant}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    tpc_table(D, F, S, pv, V["backcast_monthly_rel_rms"]).to_parquet(
        stage / "tpc_county_sector_monthly.parquet", index=False)
    long, wide = candidates(S, pv)
    long.to_parquet(stage / "changhua_candidates.parquet", index=False)
    wide.to_parquet(stage / "changhua_candidates_wide.parquet", index=False)
    census_table(D, pv).to_parquet(stage / "yunlin_census_class_monthly.parquet", index=False)
    if variant == "full":
        y = S["YL_clean"].copy()
        y["date"] = y.index
        y["unit"] = "GWh/month, model units"
        y["product_version"] = pv
        y["source_ids"] = np.where(y["is_extension"], source_ids("tpc", "census"),
                                   source_ids("census"))
        y = y[y["flag"] != 8]
        y[["date", "value", "sd", "flag", "method", "is_extension", "unit", "product_version",
           "source_ids"]].reset_index(drop=True).to_parquet(stage / "yunlin_extension.parquet",
                                                             index=False)
    dec = decomposition(D, F, S, V, Vall, variant)
    (stage / "decomposition.json").write_text(json.dumps(dec, ensure_ascii=False, indent=1,
                                                         default=float))
    mis = json.loads((VAL / "misspecification.json").read_text(encoding="utf-8"))[variant]
    (stage / "misspecification.json").write_text(json.dumps(mis, ensure_ascii=False, indent=1))
    shutil.copy2(dmod.PHI, stage / "phi_in.json")
    shutil.copy2(BUILD / "census/census_monthly_info.json", stage / "census_monthly_info.json")
    gate = Vall["gates"][variant]
    keys = (["T1_BV1", "T3_BV_LOYO", "T8_SYN_coverage", "T9_MIS"] if variant == "preorigin"
            else ["T1_BV1", "T2_BV2", "T3_BV_LOYO", "T4_EXT_LOYO", "T5_EXT_season",
                  "T6_EXT_2023_25", "T7_EXT_rolling",
                  "T4b_preorigin_transfer_LOYO_not_preregistered", "T8_SYN_coverage",
                  "T9_MIS"])
    val = {"product": pv, "prereg": rel_to_repo(PREREG), "prereg_sha256": sha256_file(PREREG),
           "gate": gate, "tests": {k: Vall[k] for k in keys},
           "note": ("preorigin gate uses data <= 2022-12 only (T1, T8 preorigin)"
                    if variant == "preorigin" else
                    "full gate uses 2023-26 TPC and census data (T2, T4); it gates only "
                    "the recorded-forcing arm and after-the-fact uses")}
    (stage / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                      default=float))
    failed = [k for k, v in gate.items() if k != "passed" and not v]
    crit = ("plan 3 WP P1 acceptance, pre-registration P1_county_electricity_2026-10-12: "
            + ("; ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in gate.items()
                         if k != "passed")))
    if variant == "preorigin":
        allowed = ["prior", "prior_width_input", "model_design", "forcing_frozen",
                   "design_time_evidence", "validator", "after_the_fact_analysis"]
        forbidden = ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
                     "calibration_target", "selection", "forcing_recorded_arm"]
    else:
        allowed = ["forcing_recorded_arm", "statistical_layer_history", "validator",
                   "after_the_fact_analysis"]
        forbidden = ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
                     "calibration_target", "prior", "prior_width_input", "model_design",
                     "selection", "forcing_frozen", "initial_conditions", "idw_field",
                     "design_time_evidence"]
    params = {
        "prereg": rel_to_repo(PREREG), "prereg_sha256": sha256_file(PREREG),
        "data_end_read": str(D["end"].date())[:7],
        "inputs_truncated_at_load": variant == "preorigin",
        "nnls_window": list(prod.WIN[variant]), "nnls_bootstrap": {"n": 2000, "seed": 0},
        "backcast": {"months": list(prod.BC), "moy_window": list(prod.MOY_BC),
                     "trend_window": list(prod.TREND[variant])},
        "transfer_window": list(prod.WIN[variant]),
        "extension": list(prod.EXT) if variant == "full" else None,
        "census": "clean_census cap 1.0, duty window 2012-01..2022-12; months from 2025-07 "
                  "masked",
        "phi_in": {"value": F["phi"]["phi_in"], "sigma_ln": F["phi"]["sigma_ln_phi"],
                   "status": F["phi"]["status"]},
        "synthetic": {"R": 1000, "B": 1000, "seeds": "default_rng(7000 + r)"},
        "grid": "results/twin_forward/datum_gate_innov.npz (mask and geometry only)",
        "gate_failed_criteria": failed,
    }
    m = manifest.build_manifest(
        stage, product=product, layer="L2", inputs=INPUTS, code=CODE, parameters=params,
        fit_window_end=FWE[variant], validation_report="validation.json",
        gate={"passed": bool(gate["passed"]), "criteria": crit}, allowed_uses=allowed,
        forbidden_uses=forbidden, version=ver,
        personal_data="none (county and county-class monthly totals only; no pump rows, "
                      "coordinates, meter ids, names or addresses)")
    errs = [e for e in manifest.validate_manifest(m) if "directory name" not in e]
    if errs:
        raise SystemExit(f"manifest invalid: {errs}")
    if dry:
        return {"dir": stage, "pv": pv, "F": F, "D": D, "S": S, "dec": dec, "gate": gate}
    d = manifest.new_version_dir(product)
    if d.name != ver:
        raise SystemExit(f"version race: expected {ver}, got {d.name}")
    for p in stage.iterdir():
        shutil.move(str(p), d / p.name)
    stage.rmdir()
    manifest.write_manifest(d, m)
    return {"dir": d, "pv": pv, "F": F, "D": D, "S": S, "dec": dec, "gate": gate}


def ledger_rows(res: dict, Vall: dict) -> list[dict]:
    tpc = rel_to_repo(dmod.TPC)
    ch, yl = f"{tpc}#彰化縣農林漁牧", f"{tpc}#雲林縣農林漁牧"
    nat = [rel_to_repo(dmod.NAT236), rel_to_repo(dmod.NAT256)]
    cen = [rel_to_repo(REPO / "AMP_V2/data/tpc_pumps.parquet"),
           rel_to_repo(REPO / "AMP_V2/data/pump_kwh_all.parquet")]
    sub = rel_to_repo(dmod.SUBIND)
    geo = [rel_to_repo(p) for p in phimod.INPUTS]
    pre, full = res["preorigin"]["pv"], res["full"]["pv"]
    R = []

    def add(num, pv, pairs, note=""):
        for s, role in pairs:
            R.append({"number_id": num, "source": s, "role": role, "product_version": pv,
                      "note": note})

    sep_pairs = [(ch, "fit_target"), (yl, "input")] + [(c, "input") for c in cen]
    for v, pv in (("preorigin", pre), ("full", full)):
        for k in ("a", "b", "r2", "a_ci90"):
            add(f"P1.sep_{k}_{v}", pv, sep_pairs)
        add(f"P1.irr_only_a_{v}", pv, sep_pairs)
        add(f"P1.ratio_shape_{v}", pv, sep_pairs)
        add(f"P1.ratio_transfer_{v}", pv, [(ch, "input"), (yl, "input")]
            + [(c, "input") for c in cen])
        add(f"P1.ratio_subind_{v}", pv, [(ch, "fit_target"), (yl, "fit_target"),
                                         (sub, "input")] + [(c, "input") for c in cen])
        add(f"P1.syn_coverage_{v}", pv, [(ch, "input"), (yl, "input")]
            + [(c, "input") for c in cen], "synthetic truth from the production fit")
        add(f"P1.syn_remedy_k_{v}", pv, [(ch, "input"), (yl, "input")]
            + [(c, "input") for c in cen], "reported only")
        for mcase in ("M1", "M2", "M3", "M4", "M3b"):
            add(f"P1.mis_bias_{mcase}_{v}", pv, [(ch, "input"), (yl, "input")]
                + [(c, "input") for c in cen])
    add("P1.sigma_mu_ln", pre, [(ch, "input"), (yl, "input")] + [(c, "input") for c in cen]
        + [(g, "input") for g in geo], "prior log-sd of mu (0.5 base + phi + misspec)")
    add("P1.mu_elec_in_grid_GWh_yr", pre, [(yl, "input"), (ch, "input")]
        + [(c, "input") for c in cen] + [(g, "input") for g in geo],
        "a_pre x YL_clean x phi_in time means")
    add("P1.phi_in", pre, [(g, "input") for g in geo], "provisional; P3 supersedes")
    add("P1.crop_stats_prior_width_0.5", pre,
        [("results/dataprep/_pilots/landuse_crop_priors", "prior_width_input")],
        "design-time value quoted from the plan; P3 supersedes")
    for c, src in (("CH", ch), ("YL", yl)):
        add(f"P1.T1_BV1_{c}", pre, [(src + "@2019-01..2022-12", "input"),
                                    (src + "@2016-01..2018-12", "validator")]
            + [(n, "input") for n in nat], "matched lead, data <= 2022-12")
        add(f"P1.T2_BV2_{c}", full, [(src + "@2020-01..2026-08", "input"),
                                     (src + "@2016-01..2019-12", "validator")]
            + [(n, "input") for n in nat], "plan's matched lead; uses 2023-26 TPC")
        add(f"P1.T3_BV_LOYO_{c}", full, [(src, "validator")] + [(n, "input") for n in nat],
            "leave one year out: each held-out year validates; the other years are inputs")
    for num in ("P1.T4_EXT_LOYO", "P1.T5_EXT_season", "P1.T6_EXT_2023_25", "P1.T7_EXT_rolling",
                "P1.T4b_pre_transfer_LOYO", "P1.ext_monthly_rel_rms"):
        add(num, full, [(yl, "input")] + [(c, "validator") for c in cen],
            "held-out census months validate; the remaining census months fit r(moy)")
    add("P1.no_splice_ratio_2024", full, [(ch, "input"), (yl, "input")]
        + [(c, "input") for c in cen])
    add("P1.cv_j_provisional", pre, [(c, "input") for c in cen])
    add("P1.census_preorigin_equals_full", pre, [(c, "input") for c in cen])
    add("P1.refetch_identical", full, [(tpc, "input")] + [(n, "input") for n in nat])
    return R


def main() -> int:
    dry = "--dry" in sys.argv[1:]
    Vall = json.loads((VAL / "validation.json").read_text(encoding="utf-8"))
    res = {v: build_variant(v, Vall, dry) for v in ("preorigin", "full")}
    if dry:
        rows = ledger_rows(res, Vall)
        print("dry run ok; stages:", [str(r["dir"]) for r in res.values()], "rows", len(rows))
        return 0
    rows = ledger_rows(res, Vall)
    ledger.append_rows("P1", rows)
    summ = {v: {"dir": rel_to_repo(r["dir"]), "gate": r["gate"]} for v, r in res.items()}
    (BUILD / "build_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print("ledger rows:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
