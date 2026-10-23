"""WP P3 report: ledger rows (U6) for every reported number, then
results/dataprep/reports/P3.md and P3.json, from the evidence files of P3_evidence/.

Usage: python -I report_p3.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frame  # noqa: E402
import lib  # noqa: E402
from common import ledger  # noqa: E402

E = lib.EVID
REP = lib.REPO / "results/dataprep/reports"
REL = "dataprep_staging/p3_changhua_prior"
EVR = "results/dataprep/reports/P3_evidence"
C0P = "results/dataprep/pump_electricity/preorigin/v3"
C0L = "results/dataprep/pump_electricity/census_L1/v3"
PV_C0 = "pump_electricity/preorigin/v3"


def jl(name):
    with open(E / name, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- sources
S_LUIMAP = ["chou-shui-data/data/water/irrigation/luimap_z14.png",
            "chou-shui-data/data/water/irrigation/luimap_z14.box.npy"]
S_GEO = ["chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg",
         "chou-shui-data/data/water/sea_TWD97.shp", "results/twin/basemap.npz",
         "results/twin_forward/datum_gate_innov.npz",
         "data_fetch_api/new_measurements/taiwan_atlas/towns-10t_2021.json"]
S_MOA = ["data_fetch_api/new_measurements/moa_crop_038_p3",
         "data_fetch_api/new_measurements/moa_rice_J51_p3"]
Y_CELLS = f"{C0P}/cells_by_class.npz:yunlin"
Y_CELLS_HO = f"{C0P}/cells_by_class.npz:yunlin_heldout_splits"
C_CELLS = f"{C0P}/cells_by_class.npz:changhua"
Y_PUMPS = f"{C0P}/pumps_yunlin.npz"
C_PUMPS = f"{C0P}/pumps_changhua.npz"
L1 = f"{C0L}/census_pumps.parquet"
PIL_LU = "results/dataprep/_pilots/landuse_crop_priors"
PIL_ML = "results/dataprep/_pilots/design_ml_transfer"
PIL_INV = "results/dataprep/_pilots/physics_inversion_hooks"


def rows_for(nid, spec, note=""):
    out = []
    for role, srcs in spec.items():
        for s in srcs:
            out.append({"number_id": nid, "source": s, "role": role,
                        "product_version": PV_C0 if C0P in s or C0L in s else "P3 evidence",
                        "note": note})
    return out


MAPFIT = {"fit_target": [Y_CELLS], "validator": [C_CELLS],
          "input": S_LUIMAP + S_GEO + [f"{EVR}/work/predictions.npz"]}
CROPFIT = {"fit_target": [Y_PUMPS], "validator": [C_PUMPS],
           "input": S_MOA + [L1, f"{EVR}/work/crop_model_changhua_pred.csv"]}
BATTERY = {"fit_target": [Y_CELLS], "validator": [Y_CELLS_HO],
           "input": S_LUIMAP + S_GEO + S_MOA + [Y_PUMPS, L1]}


def main() -> None:
    vr = jl("verify_rebuild.json")
    rp = jl("repro_landuse_cv.json")
    bat = jl("battery.json")
    cm = jl("crop_model.json")
    pr = jl("predictions.json")
    sh = jl("shift.json")
    sc = jl("scores.json")
    af = jl("after_fact.json")
    maps = pd.read_csv(E / "scores_maps.csv").set_index("map")
    cls = pd.read_csv(E / "scores_class.csv").set_index("map")
    bvp = pd.read_csv(E / "battery_vs_pilot.csv")
    # Yunlin target against the pilot's (design-time reproduction; Yunlin only)
    df, _ = frame.load("prior")
    t = np.load(lib.PILOT_LU / "target.npz", allow_pickle=True)
    etot = t["E"].sum(0).sum(1) / 11
    tr = frame.training_cells(df).to_numpy()
    d = df["E_tot"].to_numpy()[tr] - etot[tr]
    tgt = {"train_GWh_new": float(df["E_tot"].to_numpy()[tr].sum() / 1e6),
           "train_GWh_pilot": float(etot[tr].sum() / 1e6),
           "rel_diff": float(d.sum() / etot[tr].sum()),
           "max_abs_cell_kwh_yr": float(np.abs(d).max())}
    lib.save_json(E / "target_vs_pilot.json", tgt)

    truth = sc["truth"]
    T = truth["in_grid_GWh_all_classes"]
    TND = truth["in_grid_GWh_no_domestic"]
    prim = sc["T3_shape_5km"]["primary"]
    pm = maps.loc[prim]
    g = bat["gate"]
    plan_tab = {"rf_LU+GEO": (0.047, -0.007, 6), "compA_rf_per_class": (0.055, -0.015, 5),
                "compB_rf_irr+rates": (0.000, -0.148, 8), "hgbmono_LU+GEO": (0.026, -0.033, 5),
                "nnls_LU": (-0.079, -0.263, 6)}
    plan_dev = max(max(abs(g[m]["aa_mean_delta"] - v[0]), abs(g[m]["aa_worst_delta"] - v[1]))
                   for m, v in plan_tab.items())
    plan_nc = all(g[m]["corr_ge_0.5_splits"] == v[2] for m, v in plan_tab.items())
    plan_verdicts = (g["rf_LU+GEO"]["passes"] and not any(
        g[m]["passes"] for m in plan_tab if m != "rf_LU+GEO"))
    bvp_ok = bvp[~bvp.model.str.startswith("mlp_log")]
    crop_mean = sc["T4_township"]["crop_model"][-1]
    t1 = sc["T1_level_scouts"]
    est = t1["estimates"]
    lu_keys = [k for k in est if k.startswith("landuse_pilot|") and "crop" not in k and
               "bootstrap" not in k]
    cd_keys = [k for k in est if "crop_dasymetric" in k]
    heads = sc["T6_heads_only"]
    phi = sc["T7_phi_in"]
    t8 = sc["T8_battery_predicts"]
    pyl = cm["per_year"]
    loo_shr = [v["loo_r2_shrunk"] for v in pyl.values()]
    loo_pil = [v["loo_r2"] for v in cm["pilot_variant"].values()]
    kpha = pd.DataFrame(af["kwh_per_planted_ha"])
    one = pd.DataFrame(sc["T4_township"]["one_rate"])
    cs_ok = cls[cls["pass"]]
    cs_nd = cls[cls["tv_nodom"] <= 0.10]

    N = {}   # number_id -> (value text, ledger spec)

    def num(nid, val, spec, note=""):
        N[nid] = (val, spec, note)

    num("P3.repro.features_identical",
        f"classification sha256 equal {vr['features']['classified_w7_sha256_equal']}; "
        "features.npz and township.npz max |diff| 0",
        {"input": S_LUIMAP + S_GEO, "design_time_evidence": [PIL_LU]})
    num("P3.repro.moa_overlap_identical",
        "038 and J51 rows of ROC 103-105, 110 identical to the pilot's; crop table max |diff| "
        "0 ha", {"input": S_MOA + ["data_fetch_api/new_measurements/moa_crop_038",
                                   "data_fetch_api/new_measurements/moa_rice_J51"]})
    num("P3.repro.yunlin_target_rel_diff",
        f"{tgt['rel_diff']:+.1e} ({tgt['train_GWh_new']:.2f} vs {tgt['train_GWh_pilot']:.2f} "
        "GWh/yr on 1,100 training cells)",
        {"input": [Y_CELLS], "design_time_evidence": [f"{PIL_LU}/target.npz"]})
    num("P3.repro.landuse_cv_max_diff",
        json.dumps({k: round(v, 3) for k, v in rp["max_abs_diff_vs_pilot"].items()}),
        {"fit_target": [Y_CELLS], "input": S_LUIMAP + S_GEO,
         "design_time_evidence": [f"{PIL_LU}/cv_results.csv"]})
    num("P3.repro.battery_max_diff",
        f"AA_5km {bvp_ok['d_AA_5km'].abs().max():.3f}, corr_5km "
        f"{bvp_ok['d_corr_5km_density'].abs().max():.3f} (MLP-log ratios explode in both)",
        {**BATTERY, "design_time_evidence": [PIL_ML]})
    num("P3.repro.plan_gate_table",
        f"verdicts reproduced {plan_verdicts}; corr counts equal {plan_nc}; max |diff| of mean "
        f"and worst dAA {plan_dev:.4f}", {**BATTERY, "design_time_evidence": [PIL_ML]})
    num("P3.crop.yunlin_loo_r2_plan",
        f"{min(loo_shr):.3f}-{max(loo_shr):.3f} per year 2012-2022 (lambda {cm['lambda']})",
        {"fit_target": [Y_PUMPS], "input": S_MOA + [L1]})
    num("P3.crop.yunlin_loo_r2_pilot_variant",
        f"{min(loo_pil):.3f}-{max(loo_pil):.3f} (2014/15/16/21, 19 townships)",
        {"fit_target": [Y_PUMPS], "input": S_MOA + [L1]})
    num("P3.battery.primary",
        f"{prim}: mean dAA +{g[prim]['aa_mean_delta']:.3f}, worst "
        f"{g[prim]['aa_worst_delta']:+.3f}, {g[prim]['corr_ge_0.5_splits']}/8; cropdas_plan "
        f"{g['cropdas_plan']['aa_mean_delta']:+.3f} / {g['cropdas_plan']['aa_worst_delta']:+.3f}"
        f" / {g['cropdas_plan']['corr_ge_0.5_splits']}/8 (fails)", BATTERY)
    num("P3.shift.adversarial",
        f"AUC {sh['adv_A_landuse_geo']['auc_block_cv']:.3f} (no x "
        f"{sh['adv_A_minus_x']['auc_block_cv']:.3f}); Kish ESS "
        f"{sh['adv_A_landuse_geo']['kish_ess_fraction']:.3f}; canal water per farm ha outside "
        f"Yunlin 1-99 %: {sh['feature_shift']['sw_per_farm']['chg_frac_outside_yun_p1_p99']:.3f};"
        f" upland share {sh['upland_share_cell_median']['changhua']:.3f} vs "
        f"{sh['upland_share_cell_median']['yunlin_train']:.3f}",
        {"input": S_LUIMAP + S_GEO + S_MOA + ["results/twin/surface_water_v3.npz"]})
    num("P3.truth.in_grid_GWh",
        f"{T:.2f} all classes; {TND:.2f} without domestic; county {truth['county_from_pumps']['all_GWh']:.2f}",
        {"input": [C_CELLS, C_PUMPS]})
    num("P3.truth.class_GWh", json.dumps({k: round(v, 2) for k, v in truth["class_GWh"].items()}),
        {"input": [C_CELLS]})
    num("P3.T3.primary_shape",
        f"dAA_5km {pm['dAA_5km|D']:+.3f}, corr_5km {pm['corr_5km_density|D']:.3f} (pass "
        f"{bool(pm['shape_pass'])})", MAPFIT)
    num("P3.T3.n_maps_pass", f"{int(maps['shape_pass'].sum())} of {len(maps)}", MAPFIT)
    num("P3.T3.cropdas_shape",
        f"cropdas_plan dAA {maps.loc['cropdas_plan', 'dAA_5km|D']:+.3f}, corr "
        f"{maps.loc['cropdas_plan', 'corr_5km_density|D']:.3f}",
        {**MAPFIT, "input": MAPFIT["input"] + S_MOA + [Y_PUMPS, L1]})
    num("P3.T4.primary_town_r2_share",
        f"{pm['town_R2_share']:.3f} (level {pm['town_R2_level']:.3f})",
        {"fit_target": [Y_CELLS], "validator": [C_PUMPS],
         "input": S_LUIMAP + S_GEO + [L1, f"{EVR}/work/predictions.npz"]})
    num("P3.T4.n_maps_town_pass", f"{int(maps['town_pass'].sum())} of {len(maps)}",
        {"fit_target": [Y_CELLS], "validator": [C_PUMPS],
         "input": S_LUIMAP + S_GEO + [L1, f"{EVR}/work/predictions.npz"]})
    num("P3.T4.crop_model_r2",
        f"R2 level {crop_mean['R2_level']:.3f}, share {crop_mean['R2_share']:.3f}, ratio "
        f"{crop_mean['ratio']:.3f} (2012-22 mean, 26 townships)", CROPFIT)
    num("P3.T4.one_rate",
        f"R2 level {one['R2_level'].min():.2f}-{one['R2_level'].max():.2f}, ratio "
        f"{one['ratio'].min():.2f}-{one['ratio'].max():.2f}", CROPFIT)
    num("P3.T5.class_tv",
        f"{len(cs_ok)} of {len(cls)} maps TV <= 0.10 (range {cls['tv'].min():.3f}-"
        f"{cls['tv'].max():.3f}); without domestic {len(cs_nd)} of {len(cls)}",
        {"fit_target": [Y_CELLS], "validator": [C_CELLS],
         "input": S_LUIMAP + S_GEO + S_MOA + [Y_PUMPS, L1, f"{EVR}/work/predictions.npz"]})
    num("P3.T8.battery_spearman", f"{t8['spearman']:.3f} over {len(t8['maps'])} maps",
        {"design_time_evidence": [f"{EVR}/battery.json"], "validator": [C_CELLS],
         "input": S_LUIMAP + S_GEO + [f"{EVR}/work/predictions.npz"]})
    num("P3.T6.heads_blocks",
        f"Spearman {heads['spearman_q_vs_truth']:.3f}; AA {heads['AA_blocks']:.3f} vs uniform "
        f"{heads['AA_blocks_uniform_by_cells']:.3f}; zero blocks hold "
        f"{heads['truth_share_in_zero_blocks']:.3f} of the energy",
        {"design_time_evidence": [PIL_INV], "validator": [C_CELLS]})
    num("P3.T6.heads_level", f"{est['heads_only|I2_smooth_noprior']['GWh']:.2f} GWh/yr = "
        f"{heads['ratio_all']:.3f} x truth",
        {"design_time_evidence": [PIL_INV], "validator": [C_CELLS]})
    num("P3.L.published_range",
        f"[{t1['published_range_GWh'][0]:.1f}, {t1['published_range_GWh'][1]:.1f}] covers "
        f"{T:.2f}: {t1['truth_in_range']}",
        {"design_time_evidence": [PIL_LU], "validator": [C_CELLS]})
    num("P3.L.luimap_ratios",
        ", ".join(f"{k.split('|')[1]} {est[k]['ratio_all']:.2f}" for k in lu_keys),
        {"design_time_evidence": [PIL_LU], "validator": [C_CELLS]})
    num("P3.L.cropdas_ratios",
        ", ".join(f"{k.split('|')[1]} {est[k]['ratio_all']:.2f}" for k in cd_keys),
        {"design_time_evidence": [PIL_LU], "validator": [C_CELLS]})
    num("P3.L.n_published_pass", f"{t1['n_pass_level']} of {t1['n_estimates']}",
        {"design_time_evidence": [PIL_LU, PIL_ML, PIL_INV], "validator": [C_CELLS]})
    num("P3.L.n_rebuilt_pass", f"{sc['T2_level_rebuilt']['n_pass']} of "
        f"{sc['T2_level_rebuilt']['n_maps']}; primary ratio {pm['ratio_all']:.3f} "
        f"(no domestic {pm['ratio_nodom']:.3f})", MAPFIT)
    num("P3.L.phi_in",
        f"census {phi['census_energy_phi']:.3f} vs area {phi['phi_area']:.3f}, LUIMAP farmland "
        f"{phi['phi_farm_luimap']:.3f}, MOA planted {phi['phi_planted_moa']:.3f}",
        {"validator": [C_PUMPS], "input": S_LUIMAP + S_GEO + S_MOA})
    num("P3.after.kwh_per_planted_ha_ratio",
        f"Changhua / Yunlin {kpha['ratio_changhua_to_yunlin'].min():.2f}-"
        f"{kpha['ratio_changhua_to_yunlin'].max():.2f} (2012-2022)",
        {"input": [C_PUMPS, Y_PUMPS, L1] + S_MOA})
    pbc = af["primary_battery_vs_changhua"]
    num("P3.after.primary_battery_vs_real",
        f"corr_5km: 8-split mean {pbc['battery_8_mean_corr_5km']:.3f}, LOTO "
        f"{pbc['loto_corr_5km']:.3f}, Changhua {pbc['changhua_corr_5km']:.3f}",
        {"design_time_evidence": [f"{EVR}/battery_scores.csv"], "validator": [C_CELLS]})
    rows = []
    for nid, (_, spec, note) in N.items():
        rows += rows_for(nid, spec, note)
    path = ledger.append_rows("P3", rows)
    allrows = ledger.read_all()
    p3 = [r for r in allrows if r.get("wp") == "P3"]
    conf = ledger.conflicts(p3)
    unl = ledger.unledgered(list(N), rows=p3)

    # ------------------------------------------------------------------------- acceptance
    acc = [
        {"criterion": "L0: MOA 038 (Changhua, Yunlin) and J51 township surveys for the pre-origin "
                      "years 2012-2022 fetched with SOURCE.json (sha256, licence); overlap years "
                      "identical to the pilot's rows",
         "threshold": "0 re-hash problems; identical rows for ROC 103-105, 110",
         "measured": "35 files (23 + 12), re-hash problems 0; rows identical in 8/8 "
                     "county-years (038) and 4/4 years (J51)",
         "passed": True},
        {"criterion": "Features rebuilt from L0 (LUIMAP classification, cell class areas, county "
                      "and MOI township shares) equal the pilot's",
         "threshold": "identical", "measured": N["P3.repro.features_identical"][0],
         "passed": bool(vr["features"]["classified_w7_sha256_equal"])},
        {"criterion": "Yunlin training target from C0 preorigin/v3 (loader guard, use 'prior') "
                      "against the pilot's AMP_V2-based target (reported; reproduction)",
         "threshold": "reported (no threshold declared)",
         "measured": N["P3.repro.yunlin_target_rel_diff"][0], "passed": True},
        {"criterion": "Plan P3 gate table (pilot a5/a4 battery) reproduced on the rebuilt "
                      "frame: verdicts (rf_total passes; compA, compB, HGB mono, NNLS, MLP fail) "
                      "and the table's mean/worst dAA and corr counts",
         "threshold": "same verdicts and corr counts; mean/worst dAA within 0.005",
         "measured": N["P3.repro.plan_gate_table"][0],
         "passed": bool(plan_verdicts and plan_nc and plan_dev <= 0.005)},
        {"criterion": "Crop-statistics township model in Yunlin (plan P3 step 2): pilot variant "
                      "reproduces the plan's LOO R2 0.80-0.86",
         "threshold": "0.80-0.86 (plan)",
         "measured": N["P3.crop.yunlin_loo_r2_pilot_variant"][0],
         "passed": bool(min(loo_pil) >= 0.795 and max(loo_pil) <= 0.865)},
        {"criterion": "Every map and prediction frozen before the Changhua census was read at "
                      "cell, block or township scale; test pre-registered (prereg3) and scored "
                      "through assert_can_score",
         "threshold": "registration before the scoring look; 0 post-registration looks",
         "measured": f"P3.changhua_transfer.v1 sha256 {sc['prereg']['sha256'][:16]}...; "
                     f"post-registration looks {len(sc['prereg']['post_registration_looks'])}; "
                     f"legacy looks of unknown membership disclosed "
                     f"{sc['prereg']['unverifiable_prior_looks']}",
         "passed": len(sc["prereg"]["post_registration_looks"]) == 0},
        {"criterion": "Ledger (U6): every reported number has rows, one role per source",
         "threshold": "0 unledgered, 0 conflicts",
         "measured": f"{len(p3)} rows; unledgered {len(unl)}; conflicts {len(conf)}",
         "passed": not unl and not conf},
        {"criterion": "[registered test P3t.shape_5km] primary map (plan rule: LUIMAP random "
                      "forest, since the crop-dasymetric map fails the Yunlin battery gate) "
                      "transfers the within-Changhua shape",
         "threshold": "dAA_5km >= +0.03 vs uniform farmland and corr_5km >= 0.5",
         "measured": N["P3.T3.primary_shape"][0] + f"; maps passing {N['P3.T3.n_maps_pass'][0]}",
         "passed": bool(pm["shape_pass"])},
        {"criterion": "[registered test P3t.township (a)] primary map's in-fan township shares",
         "threshold": "R2 of township shares >= 0.5",
         "measured": N["P3.T4.primary_town_r2_share"][0] + f"; maps passing "
                     f"{N['P3.T4.n_maps_town_pass'][0]}",
         "passed": bool(pm["town_pass"])},
        {"criterion": "[registered test P3t.township (b)] crop-statistics township model "
                      "(Yunlin-fitted, shrunk per-year NNLS) predicts Changhua whole-township "
                      "irrigation electricity",
         "threshold": "R2 of levels (2012-22 mean) >= 0.5",
         "measured": N["P3.T4.crop_model_r2"][0], "passed": crop_mean["R2_level"] >= 0.5},
        {"criterion": "[registered test P3t.class_split] class-resolved maps reproduce the "
                      "Changhua class split (domestic in 'other')",
         "threshold": "TV distance <= 0.10 per map", "measured": N["P3.T5.class_tv"][0],
         "passed": len(cs_ok) == len(cls)},
        {"criterion": "[registered test P3t.battery_predicts] the Yunlin transfer battery ranks "
                      "the maps as the real Changhua transfer does",
         "threshold": "Spearman >= 0.5", "measured": N["P3.T8.battery_spearman"][0],
         "passed": bool(t8["pass"])},
        {"criterion": "[registered test P3t.heads_blocks] heads-only inversion I2 block pattern",
         "threshold": "Spearman >= 0.5 over its 10 blocks", "measured":
             N["P3.T6.heads_blocks"][0], "passed": heads["spearman_q_vs_truth"] >= 0.5},
        {"criterion": "[declared comparison, truth known] published prior-width range (LUIMAP "
                      "+ crop-dasymetric) covers the in-grid census total",
         "threshold": "truth inside the range", "measured": N["P3.L.published_range"][0],
         "passed": bool(t1["truth_in_range"])},
        {"criterion": "[declared comparison, truth known] level of each published scout "
                      "estimate",
         "threshold": "|ln(estimate/truth)| <= 0.30 (plan P4.3 tolerance)",
         "measured": N["P3.L.n_published_pass"][0] + "; LUIMAP " +
                     N["P3.L.luimap_ratios"][0] + "; crop-dasymetric " +
                     N["P3.L.cropdas_ratios"][0],
         "passed": t1["n_pass_level"] == t1["n_estimates"]},
        {"criterion": "[declared comparison, truth known] heads-only inversion level (I2)",
         "threshold": "|ln| <= 0.30", "measured": N["P3.T6.heads_level"][0],
         "passed": bool(heads["pass_level"])},
        {"criterion": "[declared comparison, truth known] phi_in estimators (plan P3 step 7) "
                      "bracket the census in-grid energy fraction",
         "threshold": "census fraction inside [min, max] of area, LUIMAP-farmland and "
                      "MOA-planted estimates", "measured": N["P3.L.phi_in"][0],
         "passed": bool(phi["pass"])},
    ]
    key_numbers = [f"{k}: {v[0]}" for k, v in N.items()]
    exec_ok = all(a["passed"] for a in acc[:7])
    tests = acc[7:13]
    n_tp = sum(a["passed"] for a in tests)
    summary = (
        "P3 was run, as repurposed, as an out-of-sample test of the plan's land-use and "
        "crop-statistics transfer methods against the Changhua census (C0 preorigin/v3, "
        "2012-2022). Every map was fitted on Yunlin only (pre-origin census years, crop surveys "
        "2012-2022 fetched for this WP), frozen, pre-registered (P3.changhua_transfer.v1) and "
        "only then scored. The rebuild reproduces the pilots: features identical, the plan's "
        "gate table reproduced (LUIMAP random forest passes; compA, compB, HGB, NNLS, MLP and "
        "now the crop-dasymetric map fail), crop-model Yunlin LOO R2 "
        f"{min(loo_pil):.2f}-{max(loo_pil):.2f}. Shape transfers: the primary map (LUIMAP RF, "
        f"chosen by the plan's rule) gives dAA_5km {pm['dAA_5km|D']:+.3f} and 5 km correlation "
        f"{pm['corr_5km_density|D']:.2f} in Changhua and township-share R2 "
        f"{pm['town_R2_share']:.2f}; {int(maps['shape_pass'].sum())} of {len(maps)} maps pass "
        "the shape rule, and the Yunlin battery ranks maps as the real transfer does (Spearman "
        f"{t8['spearman']:.2f}). Levels and crop statistics do not transfer: the crop-statistics "
        f"township model gives R2 {crop_mean['R2_level']:.2f} at {crop_mean['ratio']:.2f} x "
        "the census irrigation (Changhua pumps "
        f"{kpha['ratio_changhua_to_yunlin'].min():.2f}-{kpha['ratio_changhua_to_yunlin'].max():.2f}"
        " x Yunlin's kWh per planted hectare), the crop-dasymetric map is worse than uniform "
        f"(dAA {maps.loc['cropdas_plan', 'dAA_5km|D']:+.3f}), no class-resolved map gets the "
        "class split within 0.10 (Changhua's 23 % domestic class is not in Yunlin's land use), "
        f"and only {t1['n_pass_level']} of {t1['n_estimates']} published scout totals are "
        "within the P4.3 log tolerance of 0.30. The published prior-width range "
        f"{t1['published_range_GWh'][0]:.0f}-{t1['published_range_GWh'][1]:.0f} GWh/yr just "
        f"covers the truth {T:.1f}; the scouts' LUIMAP totals are {min(est[k]['ratio_all'] for k in lu_keys):.2f}-"
        f"{max(est[k]['ratio_all'] for k in lu_keys):.2f} x truth and the crop-dasymetric ones "
        f"{min(est[k]['ratio_all'] for k in cd_keys):.2f}-{max(est[k]['ratio_all'] for k in cd_keys):.2f}. "
        f"The heads-only inversion's {est['heads_only|I2_smooth_noprior']['GWh']:.0f} GWh/yr is "
        f"{heads['ratio_all']:.2f} x truth and its block pattern anti-correlates (Spearman "
        f"{heads['spearman_q_vs_truth']:.2f}). The plan's phi_in estimators "
        f"({phi['phi_area']:.2f}-{phi['phi_farm_luimap']:.2f}) understate the census in-grid "
        f"energy fraction {phi['census_energy_phi']:.3f}. Of the six registered method tests "
        f"{n_tp} pass and {6 - n_tp} fail; these are findings, the WP's execution criteria all "
        "pass. Not a product: nothing here feeds the model. No GPU, 0 Claude API tokens, "
        "nothing under hydrophysics/ or tests/ written, nothing staged in git.")
    deviations = [
        "Repurposed (UPDATE 2026-10-12): no changhua_prior product, no pi_maps.npz bootstrap or "
        "phi_in.json product; the plan's methods are run as an out-of-sample test and reported "
        "under reports/P3.*. Evidence files sit in reports/P3_evidence/.",
        "MOA 038/J51 fetched for 2012-2022 only (the plan says 2012-2025); only pre-origin years "
        "are used by the test.",
        "Crop-statistics shrinkage: the plan names no strength; lambda (penalty lam*||X_k||^2 "
        "on beta - beta_pooled) was chosen from a fixed grid by Yunlin leave-one-township-out "
        f"R2 (lambda {cm['lambda']}), before any Changhua look.",
        "The pilot joined census and MOA township names unnormalised and so dropped 臺西鄉 "
        "(census spelling 台西鄉) from its Yunlin crop fit; the plan variant normalises names "
        "(20 townships, also 員林鎮 -> 員林市 for Changhua); the pilot variant keeps the drop "
        "for reproduction.",
        "Crop-dasymetric map: aquaculture and livestock per-hectare rates are the own-class NNLS "
        f"coefficients (pond {pr['cropdas_info']['plan']['b_aq'] / 1e3:.2f}, livestock "
        f"{pr['cropdas_info']['plan']['b_ls'] / 1e3:.2f} MWh/ha/yr, the plan's 6.1 / 5.9); the "
        "other class uses the NNLS other-class model; the in-fan farmland fraction of a "
        "township uses cell farmland x MOI township share. In the Yunlin battery, townships "
        "touching test cells train with crop areas scaled by their farmland outside the test "
        "cells and with the energy of their pumps outside them (dropped below 20 % outside).",
        "Changhua part of a cell = map x MOI Changhua share (the landuse pilot's convention); the "
        "design_ml pilots' published totals used cells with share > 0.5; the census energy "
        f"differs by {T - truth['in_grid_GWh_cells_chg_gt05']:.3f} GWh/yr between the two "
        "domains.",
        "phi_in: three non-census estimators (MOI area, LUIMAP farmland, MOA planted area x area "
        "fraction) instead of the plan's two figures 0.75 / 0.82.",
        "Level and phi_in comparisons are declared comparisons on already-known county figures "
        "(look [P3-seed-known-totals]), not registered tests.",
        "predictions.json carries an ad-hoc 'shift' block (canal area per farm ha, dry share); "
        "the plan's step-6 diagnostics are in shift.json (port of the a1 pilot).",
        "No InSAR rank validator (S4 dependency; not needed now that the census exists).",
        "Pilot numbers reproduce to within 0.033 (R2, ratio) rather than exactly, because the "
        "target is C0 v3 (December-2022 fill, kept-set flips, class harmonisation), not the "
        "pilot's AMP_V2 cache.",
    ]
    issues = [
        "Do not use phi_in = 0.75-0.82 (plan P1/P4 scaling of Changhua county series to the "
        f"grid): the census in-grid energy fraction is {phi['census_energy_phi']:.3f} "
        f"(in grid {truth['county_from_pumps']['in_grid_GWh']:.1f} of "
        f"{truth['county_from_pumps']['all_GWh']:.1f} GWh/yr kept-pump energy 2012-22, pumps "
        "without coordinates counted outside). Area-based scaling would understate in-grid "
        "Changhua by about 22 %; use the in-grid census directly (P1 validator, P4m).",
        "Land-use transfer, for any area without a census: the LUIMAP random forest (and the "
        "other tree maps) transfers the within-county SHAPE (5 km and township) but not the "
        "level or the class split; NNLS-on-area maps barely beat uniform farmland; the "
        "crop-statistics NNLS township model and the crop-dasymetric map do not transfer "
        "(pooled coefficients zero on second-crop rice and fruit, and Changhua uses "
        f"{kpha['ratio_changhua_to_yunlin'].min():.2f}-{kpha['ratio_changhua_to_yunlin'].max():.2f}"
        " x Yunlin's kWh per planted hectare). After the fact, a single kWh per planted ha-season did better "
        f"(R2 {one['R2_level'].min():.2f}-{one['R2_level'].max():.2f}, ratio "
        f"{one['ratio'].min():.2f}-{one['ratio'].max():.2f}) but still low.",
        "Changhua's domestic class (23 % of in-grid energy, 43.1 GWh/yr) is invisible to "
        "Yunlin-trained land use (Yunlin 5.5 %): every class-resolved map fails the class split "
        f"with it and {len(cs_nd)} of {len(cls)} pass without it. This supports C0's flag that it may be "
        "household load; P4m's domestic-excluded member matters.",
        "Heads-only inversions should not be read as abstraction magnitudes or patterns: I2 is "
        f"{heads['ratio_all']:.2f} x the in-grid census and its 10-block pattern anti-correlates "
        f"with the census (Spearman {heads['spearman_q_vs_truth']:.2f}; two zero blocks hold "
        f"{heads['truth_share_in_zero_blocks'] * 100:.0f} % of the energy).",
        "The Yunlin transfer battery is a useful selector (rank Spearman "
        f"{t8['spearman']:.2f} with the real transfer) and its mean dAA for the primary map "
        f"(+{g[prim]['aa_mean_delta']:.3f}) matched the real one (+{pm['dAA_5km|D']:.3f}); its "
        f"split-mean 5 km correlation ({pbc['battery_8_mean_corr_5km']:.2f}) was pessimistic and "
        f"LOTO ({pbc['loto_corr_5km']:.2f}) optimistic against the real "
        f"{pbc['changhua_corr_5km']:.2f}.",
        "Plan figures checked: the plan's LUIMAP range 116-194 GWh/yr includes the a2 "
        "crop-feature HGB map (115.8); the landuse pilot's own models span 120.5-193.7. Shift "
        "diagnostics reproduce (AUC 0.95, Kish 5.5 %, canal water per "
        "farm ha outside Yunlin's range in 86 % of Changhua cells, upland share 0.013 vs 0.188).",
        "The LUIMAP 2024 snapshot was used as input (post-origin snapshot, the plan's U3 "
        "exemption); harmless here because nothing feeds a model, but any reuse of these maps as "
        "a pre-origin prior needs the exemption and its sensitivity runs.",
        "reports/P3_evidence/work/classified_w7.npy (62 MB) is a classification cache, "
        "byte-identical to the pilot's; removable.",
        f"The prereg3 registration disclosed {sc['prereg']['unverifiable_prior_looks']} legacy "
        "looks on units of unknown membership "
        "whose windows overlap 2012-2022 (other WPs' census and head looks); none names the "
        "P3 units.",
    ]
    code = [f"{REL}/{f}" for f in ("lib.py", "fetch_moa.py", "write_sources.py", "features.py",
                                   "verify_rebuild.py", "frame.py", "models_port.py",
                                   "crop_model.py", "cells_models.py", "shift.py", "score.py",
                                   "prereg_p3.py", "after_fact.py", "report_p3.py")]
    out = {"wp": "P3", "status": "passed" if exec_ok else "partial", "summary": summary,
           "acceptance": acc, "prereg": {"id": "P3.changhua_transfer.v1",
                                         "sha256": sc["prereg"]["sha256"]},
           "products": [], "evidence": EVR, "deviations_from_plan": deviations,
           "issues_for_downstream": issues, "key_numbers": key_numbers,
           "ledger": {"fragment": "results/dataprep/ledger/rows/P3.csv", "rows": len(p3),
                      "unledgered": unl, "conflicts": conf},
           "code_paths": code, "claude_api_tokens": 0, "gpu": "none"}
    lib.save_json(REP / "P3.json", out)
    write_md(out, maps, cls, sc, bat, cm, kpha, one, pbc, phi, heads, est, t1, prim, truth)
    print(json.dumps({"status": out["status"], "rows": len(p3), "unledgered": unl,
                      "conflicts": conf, "ledger": str(path)}, ensure_ascii=False))


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_md(out, maps, cls, sc, bat, cm, kpha, one, pbc, phi, heads, est, t1, prim, truth):
    L = []
    a = L.append
    a("# WP P3 report: out-of-sample test of land-use and crop-statistics transfer against the "
      "Changhua census")
    a("")
    a("Run 2026-10-14 (UTC; 2026-10-15 local). Repurposed by the 2026-10-12 update: the Changhua "
      "census (WP C0) replaces any land-use prior, so P3's methods are tested against it. This "
      "is a methods-validation report, not a product; nothing here feeds the model.")
    a("")
    a(f"**Status: {out['status']}.** All execution criteria pass. Of the six registered method "
      "tests, the shape, township-share and battery-ranking tests pass; the crop-statistics "
      "township model, the class split and the heads-only block pattern fail. The declared "
      "comparisons on known totals show that levels do not transfer and that the plan's "
      "phi_in estimators are too low.")
    a("")
    a("## Summary")
    a("")
    a(out["summary"])
    a("")
    a("## What was done")
    a("")
    a("1. **L0.** MOA 農情調查 (038, TownCropData) for Changhua and Yunlin and 糧情調查 (J51, "
      "TownRiceData, all counties, filtered locally) for ROC 101-111 (2012-2022), one raw page per "
      "request, `data_fetch_api/new_measurements/moa_crop_038_p3/` and `moa_rice_J51_p3/` with "
      "SOURCE.json (fetched 2026-10-14 by an earlier attempt of this WP; re-hashed now, 0 "
      "problems). Overlap years equal the pilot's rows.")
    a("2. **Features** rebuilt from L0 (LUIMAP 2024 classification, cell class areas, TW_COUNTY "
      "and MOI township shares, covariates): byte-identical to the landuse pilot.")
    a("3. **Targets.** Yunlin county component of C0 `pump_electricity/preorigin/v3` (guard use "
      "`prior`), 2012-01..2022-12 means; townships from census_L1/v3 (names normalised). The "
      "Changhua component was opened (use `validator`) only by `score.py` after "
      "`prereg3.assert_can_score`.")
    a("4. **Maps fitted on Yunlin and frozen** (`work/predictions.npz`): the landuse pilot's five "
      "per-class models; the design pilots' a2/a4/a5 total and composite models; the "
      "crop-dasymetric map (plan steps 2-4) and its crop-only sensitivity; the pilot's exact "
      "crop-dasymetric reproduction (2014-16, 2021). Crop-statistics township model: per-year "
      f"NNLS shrunk to the pooled fit, lambda {cm['lambda']} by Yunlin LOTO.")
    a("5. **Yunlin transfer battery** (eight plan splits + LOTO) and the plan's gate, before "
      f"registration: the crop-dasymetric map fails, so the primary map is `{prim}` (LUIMAP "
      "random forest), by the plan's rule.")
    a("6. **Pre-registration** `P3.changhua_transfer.v1` (sha256 "
      f"`{out['prereg']['sha256']}`), then the scoring run (0 post-registration looks).")
    a("")
    a("## Acceptance")
    a("")
    a("| criterion | threshold | measured | passed |")
    a("|---|---|---|---|")
    for r in out["acceptance"]:
        a(f"| {r['criterion']} | {r['threshold']} | {r['measured']} | "
          f"{'yes' if r['passed'] else '**no**'} |")
    a("")
    a("Rows 1-7 are the WP's execution criteria. Rows marked [registered test] were fixed in "
      "P3.changhua_transfer.v1 before the census was read at cell, block or township scale; a "
      "'no' there is a finding about the method, not a defect of the WP. Rows marked [declared "
      "comparison] use county figures this session already knew (in-grid total, domestic share, "
      "97 % of pumps in grid), so they are not blind.")
    a("")
    a("## Results")
    a("")
    a(f"Truth (C0 preorigin/v3, Changhua, in grid, 2012-22 mean): **{truth['in_grid_GWh_all_classes']:.2f} "
      f"GWh/yr** (without domestic {truth['in_grid_GWh_no_domestic']:.2f}); classes " +
      ", ".join(f"{k} {v:.1f}" for k, v in truth["class_GWh"].items()) + ".")
    a("")
    a("### Maps (frozen, Yunlin-trained), scored in Changhua (domain D)")
    a("")
    a("| map | GWh/yr | x truth | x truth (no dom.) | AA_5km | dAA_5km | corr_5km | town R2 share | "
      "shape | township |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for m, r in maps.iterrows():
        a(f"| {m}{' (primary)' if m == prim else ''} | {r['pred_GWh']:.1f} | {r['ratio_all']:.2f} | "
          f"{r['ratio_nodom']:.2f} | {r['AA_5km|D']:.3f} | {r['dAA_5km|D']:+.3f} | "
          f"{r['corr_5km_density|D']:.2f} | {r['town_R2_share']:.2f} | "
          f"{'pass' if r['shape_pass'] else 'fail'} | {'pass' if r['town_pass'] else 'fail'} |")
    a("")
    a("Uniform farmland (`farm_uniform`) is the reference of dAA. Sensitivities (domain share > "
      "0.5; truth without domestic) are in `P3_evidence/scores_maps.csv`; they do not change a "
      "verdict of the primary map.")
    a("")
    a("### Class split (class-resolved maps)")
    a("")
    a("| map | TV | TV without domestic | irrigation pred/true share | aquaculture | livestock | other |")
    a("|---|---|---|---|---|---|---|")
    for m, r in cls.iterrows():
        a(f"| {m} | {r['tv']:.3f} | {r['tv_nodom']:.3f} | "
          f"{r['pred_share|E_irrigation']:.2f} / {r['true_share|E_irrigation']:.2f} | "
          f"{r['pred_share|E_aquaculture']:.2f} / {r['true_share|E_aquaculture']:.2f} | "
          f"{r['pred_share|E_livestock']:.2f} / {r['true_share|E_livestock']:.2f} | "
          f"{r['pred_share|E_other']:.2f} / {r['true_share|E_other']:.2f} |")
    a("")
    a("### Crop-statistics township model (whole townships, irrigation)")
    a("")
    a("| year | R2 level | R2 share | Spearman | pred/true | true GWh | pred GWh |")
    a("|---|---|---|---|---|---|---|")
    for r in sc["T4_township"]["crop_model"]:
        a(f"| {r['year']} | {r['R2_level']:.2f} | {r['R2_share']:.2f} | {r['spearman']:.2f} | "
          f"{r['ratio']:.2f} | {r['true_GWh']:.1f} | {r['pred_GWh']:.1f} |")
    a("")
    a("Yunlin leave-one-township-out R2 of the same model: " +
      ", ".join(f"{y} {v['loo_r2_shrunk']:.2f}" for y, v in cm["per_year"].items()) +
      ". Pilot variant (2014/15/16/21, no shrinkage): Changhua R2 " +
      ", ".join(f"{r['year']} {r['R2_level']:.2f}" for r in
                sc["T4_township"]["crop_model_pilot_variant"]) +
      ". One rate per planted ha-season (after the fact): R2 " +
      f"{one['R2_level'].min():.2f}-{one['R2_level'].max():.2f}, ratio "
      f"{one['ratio'].min():.2f}-{one['ratio'].max():.2f}.")
    a("")
    a("After the fact (look [P3-after-fact]): Changhua uses "
      f"{kpha['changhua_kwh_per_ha'].min():.0f}-{kpha['changhua_kwh_per_ha'].max():.0f} kWh of "
      "irrigation electricity per planted ha-season against Yunlin's "
      f"{kpha['yunlin_kwh_per_ha'].min():.0f}-{kpha['yunlin_kwh_per_ha'].max():.0f} "
      f"({kpha['ratio_changhua_to_yunlin'].min():.2f}-{kpha['ratio_changhua_to_yunlin'].max():.2f} x), "
      "although it receives more canal water per farm hectare. Its crop mix is rice-heavy "
      "(second-crop rice and fruit carry zero coefficients in the pooled Yunlin fit) with far less "
      "upland crop and vegetables, so the Yunlin coefficients cannot carry over.")
    a("")
    a("### Published scout estimates (truth known; declared comparison)")
    a("")
    a("| estimate | GWh/yr | domain | x truth | x truth (no dom.) | within 0.30 log |")
    a("|---|---|---|---|---|---|")
    for k, v in est.items():
        a(f"| {k} | {v['GWh']:.1f} | {v['domain']} | {v['ratio_all']:.2f} | "
          f"{v['ratio_nodom']:.2f} | {'yes' if v['pass_level'] else 'no'} |")
    a("")
    a(f"Published LUIMAP + crop-dasymetric range {t1['published_range_GWh'][0]:.1f}-"
      f"{t1['published_range_GWh'][1]:.1f} GWh/yr: covers the truth "
      f"({truth['in_grid_GWh_all_classes']:.1f}) only at its top. The estimates sit closer to "
      "the domestic-excluded truth: the Yunlin-trained models carry Yunlin's 5.5 % domestic share.")
    a("")
    a("### Heads-only inversion I2 (physics_inversion_hooks, frozen theta)")
    a("")
    a(f"Level {est['heads_only|I2_smooth_noprior']['GWh']:.1f} GWh/yr equivalent = "
      f"{heads['ratio_all']:.2f} x truth. Blocks rebuilt and checked against the pilot csv "
      f"({heads['blocks_reproduced']}). Block Spearman {heads['spearman_q_vs_truth']:.2f}; block "
      f"share accuracy {heads['AA_blocks']:.2f} against {heads['AA_blocks_uniform_by_cells']:.2f} "
      f"for shares proportional to block size; its {heads['n_blocks_zero_estimate']} zero blocks "
      f"hold {heads['truth_share_in_zero_blocks'] * 100:.0f} % of the census energy.")
    a("")
    a("### phi_in (truth known; declared comparison)")
    a("")
    a(f"MOI area {phi['phi_area']:.3f}, LUIMAP farmland {phi['phi_farm_luimap']:.3f}, MOA planted "
      f"area {phi['phi_planted_moa']:.3f}; census in-grid energy fraction "
      f"**{phi['census_energy_phi']:.3f}**: the parts of the county outside the grid hold a "
      "quarter of its area but only about 4 % of its kept-pump energy.")
    a("")
    a("### Battery against the real transfer")
    a("")
    a(f"Rank correlation across {len(sc['T8_battery_predicts']['maps'])} maps between the "
      f"Yunlin battery's mean dAA_5km and the Changhua dAA_5km: "
      f"{sc['T8_battery_predicts']['spearman']:.2f}. For the primary map: battery mean dAA "
      f"+{bat['gate'][prim]['aa_mean_delta']:.3f} vs Changhua +{maps.loc[prim, 'dAA_5km|D']:.3f}; "
      f"5 km correlation 8-split mean {pbc['battery_8_mean_corr_5km']:.2f}, LOTO "
      f"{pbc['loto_corr_5km']:.2f}, Changhua {pbc['changhua_corr_5km']:.2f}; battery ratio range "
      f"{pbc['battery_8_ratio_range'][0]:.2f}-{pbc['battery_8_ratio_range'][1]:.2f}, Changhua "
      f"{pbc['changhua_ratio']:.2f}.")
    a("")
    a("## Deviations from plan")
    a("")
    for d in out["deviations_from_plan"]:
        a(f"- {d}")
    a("")
    a("## Issues for downstream")
    a("")
    for d in out["issues_for_downstream"]:
        a(f"- {d}")
    a("")
    a("## Provenance")
    a("")
    a("- Pre-registration: `results/dataprep/preregistration/P3.changhua_transfer.v1.json`; "
      "looks [P3-seed-known-totals] (disclosure, back-dated to the session start), [P3-battery] "
      "(Yunlin design-time), the scoring looks, [P3-after-fact].")
    a(f"- Ledger: `results/dataprep/ledger/rows/P3.csv`, {out['ledger']['rows']} rows, "
      f"unledgered {len(out['ledger']['unledgered'])}, conflicts {len(out['ledger']['conflicts'])}.")
    a("- Evidence (`results/dataprep/reports/P3_evidence/`): verify_rebuild.json, "
      "target_vs_pilot.json, repro_landuse_cv.{csv,json}, crop_model.json, battery.json, "
      "battery_scores.csv, battery_vs_pilot.csv, predictions.json, shift.json, scores.json, "
      "scores_maps.csv, scores_class.csv, scores_dry.json (synthetic dry run), after_fact.json, "
      "after_fact_townships.csv, logs/, work/ (features, predictions, crop-model tables).")
    a("- Code: `dataprep_staging/p3_changhua_prior/` (" +
      ", ".join(c.split("/")[-1] for c in out["code_paths"]) + ").")
    a("- CPU: every job longer than a minute ran in a cpu_slot (one slot each); one short run "
      "of shift.py was started outside a slot by mistake and killed after about two minutes. "
      "No GPU, 0 Claude API tokens. Nothing under hydrophysics/ or tests/ was written; nothing "
      "staged in git. Personal data: none in the report (pump coordinates and well numbers "
      "stay in C0's products; the evidence holds cell and township aggregates only).")
    (REP / "P3.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
