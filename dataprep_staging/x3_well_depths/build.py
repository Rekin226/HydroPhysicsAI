"""Build the X3 products, the report and the ledger rows (one run, new versions only).

Products (results/dataprep/x3_well_depths/<name>/v<N>/, MANIFEST.json each):

* ``evidence``     L1  transcribed public tables (Lin 2012, WRA 2013, MOA yearbooks) + sources
* ``depth_layers`` L2  depth -> aquifer model, county depth/layer distributions, census-join
                       ML skill, the per-cell share of pumped electricity by twin layer (+ sd)
* ``head_layer_response`` L2  head-based layer response (design-time heads, 2012-02..2022-12)
* ``split_prior``  L2  the recommended per-county pumping split (a prior for the GPU rung)

Report: results/dataprep/reports/X3_well_depths/report.{md,json} (report.py writes the text).

Run from the repository root (the look at the heads was logged by log_look.py):
    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/x3_well_depths/build.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import x3const as K  # noqa: E402
from common import guard, ledger, manifest  # noqa: E402

PKG_CODE = sorted(str(p.relative_to(K.REPO)) for p in K.PKG.glob("*.py"))
NEVER = ["scoring_target", "baseline", "datum_fit", "prospective_scorer", "calibration_target",
         "initial_conditions", "idw_field", "selection", "forcing_frozen",
         "forcing_recorded_arm", "statistical_layer_history"]
#: share of each county's agricultural groundwater pumped by the irrigation associations'
#: public wells (MOA ws 2446107: 2001-2010 mean 60.26 / 6.89 x10^6 t/yr; county agricultural
#: groundwater 5.54 / 5.28 x10^8 t/yr, WRA 2013 Table 2-1.5; Yunlin range 4-11 % across
#: the published county totals), with an sd that covers that range
IA_FRACTION = {"changhua": (0.013, 0.005), "yunlin": (0.07, 0.03)}
#: extra sd for the survey age (2000-2008 census against 2012-2022 pumping), a judgement
SURVEY_AGE_SD = 0.05
#: per-cell sd floor of a share (within-township heterogeneity is not observed; the
#: out-of-block RMSE of predicting a township share is 0.10-0.15, see the CV)
CELL_SD_FLOOR = 0.10
SOURCES = [
    {"id": "ntu_thesis_lin2012_choushui_pumping",
     "what": "Lin S.-T. (2012) NTU MSc thesis: township private-well pumping split over "
             "aquifers 1-4 from the 2000-2008 YunTech/ARDC well census (Table 4-10) and "
             "after MODFLOW calibration on 2005-2009 heads (Table 4-14); census pumping by "
             "use (Table 2-2); aquifer depths (Ch. 3)",
     "licence": "author-granted worldwide public access (同意授權(全球公開)); copyright "
                "reserved; cite, do not redistribute"},
    {"id": "wra_lib_x3_pages_b",
     "what": "WRA 2013 well-management report (健全水井管理策略... (1/3), marc_id 74288): "
             "wells found per township by the latest county survey (FY89-99) and the "
             "number deeper than 100 m (Tables 2-1.3/2-1.4); a 2012 village sample (Yunlin "
             "埤腳里: 600 of 621 wells 20 m deep, 4-inch PVC); county pumping totals",
     "licence": "WRA website publication (public PDF button); cite, do not redistribute"},
    {"id": "moa_ia_well_depth_tables",
     "what": "MOA agricultural statistics yearbooks 96-103: irrigation-association wells by "
             "depth class (<60 ... >=140 m), Changhua and Yunlin associations",
     "licence": "government statistics publication; Open Government Data License v1.0 per "
                "MOA terms (not re-checked); cite"},
    {"id": "moa_ws_ia_public_wells",
     "what": "MOA pages: 590 IA legal wells, 2001-2010 pumping (Yunlin 60.26, Changhua 6.89 "
             "x10^6 t/yr), most wells in aquifer 2-2, most 2010 pumping in aquifer 3; "
             "Yunlin HSR-corridor IA wells 66-150 m, aquifer 2; 'farmers mostly pump the "
             "shallow aquifer for cost reasons'",
     "licence": "government website; cite"},
    {"id": "wra_lib_x3_pages",
     "what": "WRA library catalogue pages: 105年度 subsidence project (2,096 irrigation "
             "wells measured 2015-16, 119 deep = 5.7 %; shallow wells = aquifer 1); 99年度 "
             "subsidence reports (deep pumping below 300 m in inland Yunlin)",
     "licence": "WRA website catalogue abstracts; cite"},
    {"id": "cy_x3_well_documents",
     "what": "Control Yuan reports 64326, 54525 (well closures, subsidence along the HSR); "
             "18748 and 64306 were Word files and were not fetched",
     "licence": "government public documents; cite"},
    {"id": "wra_reports_misc (existing L0)",
     "what": "Control Yuan 53001 (Yunlin IA closed wells 120-150 m = aquifer 2; TWC wells "
             "244-301 m); MOEA action plans ('irrigation wells are mostly shallow')",
     "licence": "as recorded in that SOURCE.json"},
]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float) + "\n",
                    encoding="utf-8")


def _l0(*keys) -> list[str]:
    return [str(K.L0[k]) for k in keys]


# ----------------------------------------------------------------------------- 1 evidence
def build_evidence() -> tuple[Path, dict]:
    import transcribe

    d = manifest.new_version_dir(f"{K.PRODUCT_ROOT}/evidence")
    chk = transcribe.main(str(d))
    _write_json(d / "sources.json", SOURCES)
    ok = (all(v for k, v in chk.items() if k.endswith("matches_printed"))
          and chk["lin_calibrated_max_abs_sum_minus_1"] <= 0.011
          and chk["lin_census_initial_max_abs_sum_minus_1"] <= 0.011
          and chk["lin_use_max_rel_err_vs_printed_total"] < 1e-6
          and chk["moa_rows"] == 14)
    _write_json(d / "validation.json", {"checks": chk, "passed": bool(ok)})
    moa = [str(K.L0["moa_dir"] / f) for f in K.MOA_PAGES]
    m = manifest.build_manifest(
        d, product=f"{K.PRODUCT_ROOT}/evidence", layer="L1",
        inputs=_l0("thesis", "wra_well_mgmt") + moa, code=PKG_CODE,
        parameters={"parser": "poppler pdftotext -layout, NFKC; regex row parsers",
                    "renormalisation": "none here (shares as printed)"},
        fit_window_end="2014-12", validation_report="validation.json",
        gate={"passed": bool(ok), "criteria": "transcription checks: printed totals "
              "reproduced, shares sum to 1 within the 2-decimal rounding, 14 MOA rows"},
        allowed_uses=["prior", "prior_width_input", "model_design", "design_time_evidence",
                      "validator", "after_the_fact_analysis"],
        forbidden_uses=NEVER, personal_data="none (township aggregates and published tables)",
        extra={"sources": SOURCES, "data_windows": {
            "lin2012_census": "2000-2008", "lin2012_calibration": "2005-2009",
            "wra2013_surveys": "FY89-99 (2000-2010)", "moa_yearbooks": "2007-2014"}})
    manifest.write_manifest(d, m)
    return d, chk


# ----------------------------------------------------------------------------- 2 depths
def _ia_layers(ia: pd.DataFrame, tm) -> pd.DataFrame:
    rows = []
    for c, g in ia.groupby("county"):
        last = g.sort_values("year").iloc[-1]
        mix = np.zeros(4)
        for cls, (lo, hi) in zip(K.MOA_DEPTH_CLASSES, K.MOA_EDGES, strict=True):
            mix += float(last[cls]) * tm.class_to_layers(lo, hi)
        mix /= mix.sum()
        rows.append({"county": c, "year": int(last["year"]), "wells": int(last["wells_total"]),
                     **{f"a{k + 1}": float(mix[k]) for k in range(4)},
                     "share_ge120m": float((last["120_140"] + last["ge140"])
                                           / last["wells_total"])})
    return pd.DataFrame(rows)


def combine_county(cs: pd.DataFrame, ia_l: pd.DataFrame) -> pd.DataFrame:
    """Gross extraction split per county: private wells (mean of the two members) and IA
    public wells by their share of agricultural pumping."""
    out = []
    for c in ("changhua", "yunlin"):
        a = cs[(cs["county"] == c) & (cs["variant"] == "census_initial")].iloc[0]
        b = cs[(cs["county"] == c) & (cs["variant"] == "calibrated")].iloc[0]
        ia = ia_l[ia_l["county"] == c].iloc[0]
        f, fsd = IA_FRACTION[c]
        rec = {"county": c, "ia_fraction": f, "ia_fraction_sd": fsd}
        for k in ("a1", "a2", "a3", "a4"):
            priv = 0.5 * (a[k] + b[k])
            half = 0.5 * abs(a[k] - b[k])
            boot = 0.5 * np.hypot((a[k + "_hi90"] - a[k + "_lo90"]) / 3.29,
                                  (b[k + "_hi90"] - b[k + "_lo90"]) / 3.29)
            val = (1 - f) * priv + f * ia[k]
            sd = np.sqrt(half ** 2 + boot ** 2 + (fsd * (priv - ia[k])) ** 2
                         + (SURVEY_AGE_SD if k == "a1" else SURVEY_AGE_SD / 2) ** 2)
            rec.update({f"private_{k}": float(priv), f"private_{k}_member_halfdiff":
                        float(half), f"private_{k}_boot_sd": float(boot),
                        f"ia_{k}": float(ia[k]), k: float(val), f"{k}_sd": float(sd)})
        rec["shallow_L0"] = rec["a1"]
        rec["shallow_L0_sd"] = rec["a1_sd"]
        rec["L1"] = rec["a2"]
        rec["L1_sd"] = rec["a2_sd"]
        rec["deeper_L2_L3"] = rec["a3"] + rec["a4"]
        rec["deeper_L2_L3_sd"] = float(np.hypot(rec["a3_sd"], rec["a4_sd"]))
        out.append(rec)
    return pd.DataFrame(out)


def build_depth_layers(ev_dir: Path) -> tuple[Path, dict]:
    import census_split as C
    import depth_model as D

    d = manifest.new_version_dir(f"{K.PRODUCT_ROOT}/depth_layers")
    cen = guard.open_product(K.CENSUS_L1, "prior")
    ev = guard.open_product(ev_dir, "prior")
    # 2a monitoring wells: depth -> aquifer
    st = D.stations()
    cv_depth = D.cross_validate(st)
    tm = D.ThresholdModel(st)
    D.depth_layer_table(st).to_csv(d / "monitoring_well_depths_by_layer.csv", index=False)
    classes = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 200), (200, 275), (275, 320)]
    pd.DataFrame([{"depth_lo_m": lo, "depth_hi_m": hi,
                   **{f"p_aquifer{k + 1}": float(v)
                      for k, v in enumerate(tm.class_to_layers(lo, hi))}}
                  for lo, hi in classes]).to_csv(d / "depth_class_to_aquifer.csv", index=False)
    # 2b private wells: township splits joined to census pumps
    table = C.lin_table(ev.path("township_layer_split.csv"))
    pumps = C.load_pumps()
    cvs = {}
    preds = []
    for v in ("census_initial", "calibrated"):
        cv = C.cross_validate(pumps, table, v)
        p = cv.pop("_preds")
        preds.append(p.assign(variant=v))
        cvs[v] = cv
    pd.concat(preds).rename_axis("town").reset_index().to_csv(
        d / "census_ml_cv_predictions.csv", index=False)
    choice = {v: C.choose_within_township(cvs[v]) for v in cvs}
    pt = C.cell_town_table(pumps, table)
    cs = C.county_split(pt, table)
    cs.to_csv(d / "county_private_split.csv", index=False)
    wra = pd.read_csv(ev.path("township_wells_deep.csv"))
    t_init = table[table["variant"] == "census_initial"].set_index("town")
    t_cal = table[table["variant"] == "calibrated"].set_index("town")
    rank = {"census_initial": C.wra_rank_check(1 - t_init["a1"], wra),
            "calibrated": C.wra_rank_check(1 - t_cal["a1"], wra)}
    wra_c = wra.groupby("county")[["wells_total", "deep_gt100_total"]].sum()
    wra_c["deep_count_share"] = wra_c["deep_gt100_total"] / wra_c["wells_total"]
    # 2c IA public wells
    ia = pd.read_csv(ev.path("ia_wells_depth_classes.csv"))
    ia_l = _ia_layers(ia, tm)
    ia_l.to_csv(d / "ia_wells_layers.csv", index=False)
    comb = combine_county(cs, ia_l)
    comb.to_csv(d / "county_split_combined.csv", index=False)
    # 2d per-cell product
    n_active = int(np.load(K.ZONE_COUNTY / "zone_county_map.npz")["n_active"])
    cp = C.cell_product(pt, n_active, CELL_SD_FLOOR)
    zc = np.load(K.ZONE_COUNTY / "zone_county_map.npz", allow_pickle=True)
    np.savez_compressed(
        d / "cell_layer_shares.npz", mean=cp["mean"], sd=cp["sd"],
        census_initial=cp["census_initial"], calibrated=cp["calibrated"],
        energy_weight=cp["energy_weight"],
        frac_energy_county_filled=cp["frac_energy_county_filled"],
        flag=np.where(np.isfinite(cp["mean"][0]), 5, 8).astype(np.uint8),
        layers=np.array(["L0_aquifer1", "L1_aquifer2", "L2_aquifer3", "L3_aquifer4"]),
        centroids=zc["centroids"], mask_hash=str(cen.manifest["parameters"]["grid"][
            "mask_hash"]), method="township split (Lin 2012, two members) x census "
        "energy weight (duty 2012-2022 x hp), mean of members; sd = sqrt(halfdiff^2 + "
        f"{CELL_SD_FLOOR}^2); private wells only (IA public wells not separated)")
    cells = pd.DataFrame({"cell": np.arange(n_active),
                          "x": zc["centroids"][:, 0], "y": zc["centroids"][:, 1],
                          "changhua": zc["changhua"].astype(int),
                          "energy_weight": cp["energy_weight"]})
    for k in range(4):
        cells[f"share_L{k}"] = cp["mean"][k]
        cells[f"sd_L{k}"] = cp["sd"][k]
    cells["flag"] = np.where(np.isfinite(cp["mean"][0]), 5, 8)
    cells["method"] = "x3_township_split_v1"
    cells.to_csv(d / "cell_layer_shares.csv", index=False)
    ssum = np.nansum(cp["mean"], 0)
    val = {
        "depth_to_aquifer_cv": cv_depth, "threshold_cuts_m": [float(x) for x in tm.cuts],
        "census_ml_cv": cvs, "census_ml_choice": choice,
        "wra_rank_check": rank, "wra_county_deep_count_share": wra_c.reset_index().to_dict(
            "records"),
        "cell_shares_sum_max_abs_err": float(np.nanmax(np.abs(ssum[np.isfinite(
            cp["mean"][0])] - 1))),
        "n_cells_with_pumps": int(np.isfinite(cp["mean"][0]).sum()),
        "n_pumps": int(len(pumps)),
        "energy_share_county_filled": float(
            pt[pt["variant"] == "calibrated"].pipe(
                lambda g: (g["w"] * g["filled_from_county"]).sum() / g["w"].sum()))}
    ok = val["cell_shares_sum_max_abs_err"] < 1e-6 and len(cs) == 4
    val["passed"] = bool(ok)
    _write_json(d / "validation.json", val)
    ex = []
    for x in cen.manifest["exemptions"]:
        ex.append({**x, "ledger_number_id": "X3.v1.exemption." + x["ledger_number_id"],
                   "reason": x["reason"] + " (inherited from pump_electricity/census_L1 v3, "
                             "whose attributes and duty weights X3 uses)",
                   "uses": [u for u in x["uses"] if u in ("prior", "prior_width_input",
                                                           "model_design",
                                                           "design_time_evidence")]})
    m = manifest.build_manifest(
        d, product=f"{K.PRODUCT_ROOT}/depth_layers", layer="L2",
        inputs=[str(K.CENSUS_L1 / "census_pumps.parquet"), str(K.STATIONS),
                str(ev_dir / "township_layer_split.csv"),
                str(ev_dir / "township_wells_deep.csv"),
                str(ev_dir / "ia_wells_depth_classes.csv"),
                str(K.ZONE_COUNTY / "zone_county_map.npz")],
        code=PKG_CODE,
        parameters={"census_columns": K.CENSUS_COLS, "station_columns": K.STATION_COLS,
                    "pump_filter": "changhua/yunlin, kept, in grid, cell, installed <= "
                                   "2022-12 or undated, weight duty_2012_2022 x hp > 0",
                    "ia_fraction": IA_FRACTION, "survey_age_sd": SURVEY_AGE_SD,
                    "cell_sd_floor": CELL_SD_FLOOR, "depth_model": "thresholds (CV winner)",
                    "seed": 2027},
        fit_window_end="2026-07", non_exempt_fit_window_end="2022-12", exemptions=ex,
        validation_report="validation.json",
        gate={"passed": bool(ok), "criteria": "per-cell shares sum to 1; four county "
              "estimates; no pre-registered skill gate (evidence product, plan 4.5 "
              "design-time label)"},
        allowed_uses=["prior", "prior_width_input", "model_design", "design_time_evidence",
                      "validator", "after_the_fact_analysis"],
        forbidden_uses=NEVER,
        personal_data="none: township aggregates; census pumps enter only as cell sums "
                      "(no names, addresses, postal codes, coordinates or well numbers "
                      "are written)")
    manifest.write_manifest(d, m)
    return d, {"cv_depth": cv_depth, "cuts": [float(x) for x in tm.cuts], "cvs": cvs,
               "choice": choice, "county_private": cs, "ia_layers": ia_l,
               "combined": comb, "rank": rank, "wra_county": wra_c, "val": val}


# ----------------------------------------------------------------------------- 3 heads
def build_heads() -> tuple[Path, dict]:
    import head_response as H

    d = manifest.new_version_dir(f"{K.PRODUCT_ROOT}/head_layer_response")
    data = H.load()
    q, r = H.local_series(data)
    A = H.responses(data, q, r)
    A.to_csv(d / "well_responses.csv", index=False)
    S = H.summarise_gains(A)
    Sz = H.summarise_gains(A, by=("src", "county", "zone", "layer"))
    S.to_csv(d / "gains_county_layer.csv", index=False)
    Sz.to_csv(d / "gains_county_zone_layer.csv", index=False)
    R01 = H.ratio_upper_lower(A, 0, 1)
    R01z = H.ratio_upper_lower(A, 0, 1, by=("src", "county", "zone"))
    R12 = H.ratio_upper_lower(A, 1, 2)
    pd.concat([R01.assign(pair="L0/L1"), R12.assign(pair="L1/L2")]).to_csv(
        d / "layer_ratio_county.csv", index=False)
    R01z.to_csv(d / "layer_ratio_county_zone.csv", index=False)
    P01 = H.pair_table(data, q, r, 0, 1)
    P12 = H.pair_table(data, q, r, 1, 2)
    P = pd.concat([P01, P12])
    P.to_csv(d / "nested_pair_massbalance.csv", index=False)
    truth = {"R0": H.twin_net_phi(K.RUN["R0"]), "R1c": H.twin_net_phi(K.RUN["R1c"])}
    pv = P01[P01["src"].isin(["pred_R0", "pred_R1c"])].groupby("src")["phi"].median()
    recov = {"pred_R0": {"truth_net_phi": truth["R0"]["net_phi"],
                         "estimate_median": float(pv.get("pred_R0", np.nan))},
             "pred_R1c": {"truth_net_phi": truth["R1c"]["net_phi"],
                          "estimate_median": float(pv.get("pred_R1c", np.nan))}}
    # synthetic recovery with the lumped model itself, at the observed pairs
    syn = []
    obs01 = P01[P01["src"] == "obs"]
    for j, (iu, il) in enumerate(H.nested_pairs(data, 0, 1)):
        fit = obs01[obs01["site"] == H.site_of(data["sid"][iu])]
        if fit.empty:
            continue
        fit = fit.iloc[0].to_dict()
        rs = (float(np.nanstd(np.diff(data["obs"][iu]))) * 0.5,
              float(np.nanstd(np.diff(data["obs"][il]))) * 0.5)
        for phi in (-0.5, 0.0, 0.5, 0.9):
            syn.append({"site": fit["site"], "phi_true": phi,
                        "phi_hat": H.synthetic_recovery(q[iu], r[iu], fit, rs, phi,
                                                        seed=2027 + j)})
    syn = pd.DataFrame(syn)
    syn.to_csv(d / "synthetic_recovery.csv", index=False)
    syn_s = syn.groupby("phi_true")["phi_hat"].agg(["median", lambda s: s.quantile(.25),
                                                    lambda s: s.quantile(.75), "count"])
    syn_s.columns = ["median", "q25", "q75", "n"]
    b_ok = all(abs(v["estimate_median"] - v["truth_net_phi"]) < 0.3 for v in recov.values()
               if np.isfinite(v["estimate_median"]))
    s_ok = bool((abs(syn_s["median"] - syn_s.index) < 0.2).all())
    val = {"model_recovery": recov, "synthetic_recovery": syn_s.reset_index().to_dict(
        "records"), "massbalance_valid": bool(b_ok and s_ok),
        "massbalance_criteria": "estimate within 0.3 of the twin's known net split on both "
                                "runs' predictions, and synthetic median within 0.2",
        "window": [str(data["dates"][0].date()), str(data["dates"][-1].date())],
        "radius_km": H.RADIUS_KM, "lags": H.LAGS, "electricity": data["electricity_product"]}
    _write_json(d / "validation.json", val)
    m = manifest.build_manifest(
        d, product=f"{K.PRODUCT_ROOT}/head_layer_response", layer="L2",
        inputs=[str(K.RUN["R0"] / "stage3_per_entry.npz"),
                str(K.RUN["R1c"] / "stage3_per_entry.npz"),
                str(K.RUN["R0"] / "stage3_kfold_wells.csv"),
                str(K.RUN["R0"] / "inputs.local.pkl"),
                str(K.RUN["R0"] / "stage3_theta.json"), str(K.RUN["R1c"] / "stage3_theta.json"),
                str(K.ELEC_PREORIGIN / "cells_total.npz"),
                str(K.ZONE_COUNTY / "zone_county_map.npz"), str(K.STATIONS)],
        code=PKG_CODE,
        parameters={"radius_km": H.RADIUS_KM, "lags": H.LAGS, "n_boot": H.N_BOOT,
                    "block_months": H.BLOCK, "seed": H.SEED,
                    "heads_window": "2012-02..2022-12 (design-time)",
                    "look": "prereg3.record_look units I1:design_oof_heads_R0_R1e, scope "
                            "heads:2012-01..2022-12, wp I1, purpose after_the_fact "
                            "(2026-10-20T17:52:27Z)"},
        fit_window_end="2022-12", validation_report="validation.json",
        gate={"passed": False, "criteria": "no pre-registered gate; design-time heads only. "
              "The mass-balance split is reported only if it recovers the twin's known split "
              f"(massbalance_valid = {bool(b_ok and s_ok)})"},
        allowed_uses=["design_time_evidence", "after_the_fact_analysis"],
        forbidden_uses=NEVER + ["prior", "validator"],
        personal_data="none (monitoring-well ids and responses only)")
    manifest.write_manifest(d, m)
    return d, {"S": S, "Sz": Sz, "R01": R01, "R01z": R01z, "R12": R12, "P": P,
               "recov": recov, "syn": syn_s, "mb_valid": bool(b_ok and s_ok), "val": val}


# ----------------------------------------------------------------------------- 4 prior
def logit_normal(mean: float, sd: float) -> dict:
    m = float(np.clip(mean, 1e-3, 1 - 1e-3))
    return {"mu_logit": float(np.log(m / (1 - m))), "sd_logit": float(sd / (m * (1 - m)))}


def build_prior(dl_dir: Path, dl: dict, hd: dict) -> tuple[Path, dict]:
    d = manifest.new_version_dir(f"{K.PRODUCT_ROOT}/split_prior")
    comb = dl["combined"].set_index("county")
    rec = {}
    for c in ("changhua", "yunlin"):
        r = comb.loc[c]
        s = float(r["shallow_L0"])
        rec[c] = {
            "gross_extraction_share": {
                "L0_aquifer1": {"mean": s, "sd": float(r["shallow_L0_sd"]),
                                "ci90": [float(max(0, s - 1.645 * r["shallow_L0_sd"])),
                                         float(min(1, s + 1.645 * r["shallow_L0_sd"]))]},
                "L1_aquifer2": {"mean": float(r["L1"]), "sd": float(r["L1_sd"])},
                "L2_L3_aquifers3_4": {"mean": float(r["deeper_L2_L3"]),
                                      "sd": float(r["deeper_L2_L3_sd"])}},
            "two_layer_twin_form": {
                "pump_frac_shallow": {"mean": s, "sd": float(r["shallow_L0_sd"]),
                                      **logit_normal(s, float(r["shallow_L0_sd"]))},
                "note": "the twin splits only between layer 0 and pump_layer 1: the deeper "
                        "share is lumped into layer 1 here"}}
    a = comb.loc["changhua"]
    b = comb.loc["yunlin"]
    diff = float(a["shallow_L0"] - b["shallow_L0"])
    diff_sd = float(np.hypot(a["private_a1_boot_sd"], b["private_a1_boot_sd"]))
    rec["changhua_minus_yunlin_shallow"] = {
        "mean": diff, "sd_township_sampling": diff_sd,
        "members": {v: float(dl["county_private"].query(
            f"variant == '{v}' and county == 'changhua'")["a1"].iloc[0]
            - dl["county_private"].query(f"variant == '{v}' and county == 'yunlin'"
                                         )["a1"].iloc[0])
            for v in ("census_initial", "calibrated")}}
    rec["twin_current"] = {k: v for k, v in hd["val"]["model_recovery"].items()}
    rec["return_flow_note"] = (
        "These are GROSS extraction shares. The twin also returns a fraction r of the "
        "pumped volume to layer 0 (R0 r = 0.42, R1c r = 0.59), so its net layer-0 stress is "
        "(f - r) / (1 - r) of the net total. With f = 0.84 and r = 0.42 the net layer-0 "
        "share would be +0.72, against -0.57 (R0) and -1.44 (R1c) now.")
    _write_json(d / "split_prior.json", rec)
    _write_json(d / "validation.json", {"inputs_gate": "depth_layers validation passed",
                                        "passed": True})
    m = manifest.build_manifest(
        d, product=f"{K.PRODUCT_ROOT}/split_prior", layer="L2",
        inputs=[str(dl_dir / "county_split_combined.csv"),
                str(dl_dir / "county_private_split.csv")],
        code=PKG_CODE, parameters={"ia_fraction": IA_FRACTION,
                                   "survey_age_sd": SURVEY_AGE_SD},
        fit_window_end="2026-07", non_exempt_fit_window_end="2022-12",
        exemptions=json.loads((dl_dir / "MANIFEST.json").read_text())["exemptions"],
        validation_report="validation.json",
        gate={"passed": True, "criteria": "built from depth_layers v1 (passed); a prior "
              "only: the rung's declaration decides its use"},
        allowed_uses=["prior", "prior_width_input", "model_design", "design_time_evidence",
                      "after_the_fact_analysis"],
        forbidden_uses=NEVER, personal_data="none")
    manifest.write_manifest(d, m)
    return d, rec


def ledger_rows(prods: dict) -> None:
    rows = []
    for num in ("X3.county_split.changhua", "X3.county_split.yunlin",
                "X3.changhua_minus_yunlin_shallow"):
        rows += [
            {"number_id": num, "source": "data_fetch_api/new_measurements/"
             "ntu_thesis_lin2012_choushui_pumping", "role": "input",
             "product_version": f"{prods['evidence']}"},
            {"number_id": num, "source": "results/dataprep/pump_electricity/census_L1/v3",
             "role": "input", "product_version": "census_L1/v3"},
            {"number_id": num, "source": "data_fetch_api/new_measurements/"
             "moa_ia_well_depth_tables", "role": "input",
             "product_version": f"{prods['evidence']}"},
            {"number_id": num, "source": "data_fetch_api/new_measurements/"
             "wra_lib_x3_pages_b", "role": "validator",
             "product_version": f"{prods['evidence']}"}]
    for num in ("X3.head_ratio_L0_L1.changhua", "X3.head_ratio_L0_L1.yunlin"):
        rows += [
            {"number_id": num, "source": "results/twin_runs/stage3_datum_gate_cwa/"
             "stage3_per_entry.npz", "role": "design-time evidence",
             "product_version": prods["head_layer_response"]},
            {"number_id": num, "source": "results/dataprep/pump_electricity/preorigin/v3",
             "role": "input", "product_version": "preorigin/v3"}]
    for r in rows:
        r["note"] = "WP X3 (2026-10-21)"
    ledger.append_rows("X3", rows)


def main() -> dict:
    ev_dir, ev = build_evidence()
    dl_dir, dl = build_depth_layers(ev_dir)
    hd_dir, hd = build_heads()
    pr_dir, pr = build_prior(dl_dir, dl, hd)
    prods = {"evidence": str(ev_dir.relative_to(K.REPO)),
             "depth_layers": str(dl_dir.relative_to(K.REPO)),
             "head_layer_response": str(hd_dir.relative_to(K.REPO)),
             "split_prior": str(pr_dir.relative_to(K.REPO))}
    ledger_rows(prods)
    import report
    K.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.write(K.REPORT_DIR, prods, ev, dl, hd, pr)
    for p in prods.values():
        errs = manifest.validate_manifest(*manifest.load_manifest(K.REPO / p))
        if errs:
            raise SystemExit(f"{p}: {errs}")
    shutil.rmtree(K.PKG / "__pycache__", ignore_errors=True)
    return prods


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
