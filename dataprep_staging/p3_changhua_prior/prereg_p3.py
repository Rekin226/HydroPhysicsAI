"""WP P3 pre-registration (A1 tool harness.prereg3) of the out-of-sample test of the land-use /
crop-statistics transfer methods against the Changhua census, written AFTER every Yunlin-side
fit and every Changhua prediction was frozen (work/predictions.npz, crop model, battery) and
BEFORE any Changhua census value at cell, block or township scale was read.

Steps:
  seed      log what this session knew of the Changhua census before registering (C0 report
            and the task text: in-grid total, domestic share, pump fraction in grid) and the
            Yunlin-side battery look (design-time units).
  register  write P3.changhua_transfer.v1 (tests on unseen Changhua truth; declared
            comparisons on the already-known county total).
Usage: python -I prereg_p3.py {seed|register} [--dry-run]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import score  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg import not_applicable  # noqa: E402

PREREG_ID = score.PREREG_ID
SCOPE = "c0_pump_electricity_v3_changhua:2012-01..2022-12"
NONE_REASON = ("TPC census electricity of Changhua pumps (WP C0 pump_electricity v3) "
               "aggregated to model cells, 5 km blocks or townships; an electricity object, "
               "not head or leveling data: it contains no registered well or leveling "
               "benchmark, and the registries' cell membership (head data of active cells) is "
               "not read by this test")
HERE = Path(__file__).resolve().parent
REL = "dataprep_staging/p3_changhua_prior"
EV = "results/dataprep/reports/P3_evidence"


def ext(desc: str) -> dict:
    return {"description": desc, "none": NONE_REASON}


UNITS = {
    "P3:changhua_cells_2012_2022": ext(
        "Changhua-county component of C0 pump_electricity/preorigin/v3 cells_by_class "
        "(value_by_county, kWh/yr mean of 2012-01..2022-12) on the active cells with a MOI "
        "Changhua share > 0 or census energy > 0 (domain D), by class, and its 5 km blocks"),
    "P3:changhua_townships_2012_2022": ext(
        "Changhua kept census pumps (C0 preorigin/v3 pumps_changhua.npz, township from "
        "census_L1/v3 normalised to the MOA/MOI spelling): in-grid energy per township "
        "(2012-22 mean) and whole-township irrigation energy per calendar year 2012-2022"),
    "P3:changhua_i2_blocks": ext(
        "the 10 k-means blocks of the 954 Changhua cells of the heads-only inversion I2 "
        "(physics_inversion_hooks), census energy per block 2012-2022"),
}


def seed() -> None:
    known = {"P3:changhua_census_county_total": {
        "description": "county-level figures of the Changhua census known to this session "
                       "before registration: in-grid cleaned electricity 187.36 GWh/yr "
                       "2012-2022 (C0 report and key numbers), domestic share 23.0 % (43.1 "
                       "GWh/yr), about 188 GWh/yr after clean_census and 97 % of pumps in "
                       "active cells (task text / plan revision note), irrigation seasonal "
                       "CV 0.14",
        "none": NONE_REASON}}
    rec = prereg3.record_look(
        list(known), SCOPE, "P3", prereg3.SEED_PURPOSE, members=known,
        seeded_time="2026-10-14T15:47:00Z",
        note="[P3-seed-known-totals] read in the C0 report key numbers and the workflow task "
             "text at the start of this P3 session (2026-10-14 ~15:47Z); no cell, block, "
             "township or class value (other than domestic) of the Changhua census was read "
             "before the registration P3.changhua_transfer.v1")
    print(rec)
    bat = {"P3:yunlin_battery_2012_2022": {
        "description": "Yunlin-county component of C0 preorigin/v3 on the a2 training cells "
                       "(1091), scored on the eight plan-P3 transfer splits and LOTO "
                       "(battery_scores.csv; design-time units of the pilots)",
        "none": "TPC census electricity of Yunlin pumps aggregated to cells; no registered "
                "well or leveling benchmark"}}
    rec = prereg3.record_look(
        list(bat), "c0_pump_electricity_v3_yunlin:2012-01..2022-12", "P3", "design_time",
        members=bat,
        note="[P3-battery] cells_models.py battery (and repro): Yunlin-only transfer battery "
             "and the plan's gate, run before the Changhua registration to fix the primary "
             "map by the plan's rule")
    print(rec)


def inputs() -> list[str]:
    code = [f"{REL}/{f}" for f in ("lib.py", "frame.py", "features.py", "models_port.py",
                                   "crop_model.py", "cells_models.py", "score.py",
                                   "prereg_p3.py")]
    work = [f"{EV}/work/{f}" for f in ("features.npz", "township.npz", "predictions.npz",
                                       "crop_model_changhua_pred.csv", "yunlin_town_kwh.csv")]
    ev = [f"{EV}/{f}" for f in ("crop_model.json", "battery.json", "battery_scores.csv",
                                "predictions.json")]
    c0 = ["results/dataprep/pump_electricity/preorigin/v3/MANIFEST.json",
          "results/dataprep/pump_electricity/census_L1/v3/MANIFEST.json"]
    pil = ["results/dataprep/_pilots/landuse_crop_priors/prior_summary.json",
           "results/dataprep/_pilots/landuse_crop_priors/combine.json",
           "results/dataprep/_pilots/design_ml_transfer/a2_results.json",
           "results/dataprep/_pilots/design_ml_transfer/a4_results.json",
           "results/dataprep/_pilots/design_ml_transfer/a5_scores.csv",
           "results/dataprep/_pilots/physics_inversion_hooks/e_changhua_inversion.json",
           "results/dataprep/_pilots/physics_inversion_hooks/e_changhua_blocks_I2.csv",
           "results/dataprep/_pilots/physics_inversion_hooks/inputs_cwa.npz"]
    l0 = ["data_fetch_api/new_measurements/moa_crop_038_p3/SOURCE.json",
          "data_fetch_api/new_measurements/moa_rice_J51_p3/SOURCE.json"]
    return code + work + ev + c0 + pil + l0


def doc() -> dict:
    with open(lib.EVID / "battery.json", encoding="utf-8") as f:
        bat = json.load(f)
    with open(lib.EVID / "predictions.json", encoding="utf-8") as f:
        pred = json.load(f)
    import frame
    _, aux = frame.load("prior")
    phi = score.phi_estimates(aux)
    primary = bat["primary_by_plan_rule"]
    pub = score.published_estimates()
    lu = [v["GWh"] for k, v in pub.items() if k.startswith("landuse_pilot|") and
          "bootstrap" not in k]
    maps = pred["maps"]
    cls_maps = pred["class_maps"]
    tests = [
        {"test_id": "P3t.shape_5km", "data_scope": SCOPE,
         "fresh_units": ["P3:changhua_cells_2012_2022"],
         "external_units": {"P3:changhua_cells_2012_2022": UNITS["P3:changhua_cells_2012_2022"]},
         "what": "plan P3 acceptance rule applied to the real Yunlin -> Changhua transfer: each "
                 "frozen map (work/predictions.npz, Changhua part = map x MOI Changhua share) "
                 "scored with a2_robust_transfer.metrics on domain D against the census, all "
                 "classes; AA_5km = 1 - 0.5 sum |share_pred - share_true| over 5 km blocks "
                 "(row//5, col//5); corr_5km = Pearson correlation of block densities (block sum "
                 "/ domain cells in the block)",
         "metric": "dAA_5km = AA_5km(map) - AA_5km(farm_uniform); corr_5km",
         "threshold": f"per map: dAA_5km >= +{score.SHAPE['aa_delta_min']} and corr_5km >= "
                      f"{score.SHAPE['corr_min']}; headline = the primary map {primary!r} "
                      "(plan rule on the Yunlin battery)",
         "maps": maps,
         "secondary": ["domain D05 (MOI Changhua share > 0.5, the pilot's Changhua cells)",
                       "truth without the domestic class", "AA_1km, cell R2 and Spearman",
                       "primary-map ratio by propensity (> 0.9 vs <= 0.9)"],
         "gating": "none (methods-validation finding)"},
        {"test_id": "P3t.township", "data_scope": SCOPE,
         "fresh_units": ["P3:changhua_townships_2012_2022"],
         "external_units": {"P3:changhua_townships_2012_2022":
                            UNITS["P3:changhua_townships_2012_2022"]},
         "what": "(a) each frozen map's in-fan township totals (sum over cells of map x MOI "
                 "township share; Changhua townships with >= 1 km2 in the grid) against the "
                 "census in-grid energy by census township; (b) the crop-statistics model "
                 "(work/crop_model_changhua_pred.csv, per-year NNLS shrunk to the pooled fit, "
                 "lambda by Yunlin LOTO) whole-township irrigation against the census "
                 "whole-township irrigation energy (all kept pumps, in grid or not)",
         "metric": "(a) R2 of township shares; (b) R2 of levels of the 2012-22 mean",
         "threshold": f"(a) R2_share >= {score.R2_MIN} per map; (b) R2_level >= {score.R2_MIN}",
         "secondary": ["(a) R2 of levels, Spearman, truth without domestic",
                       "(b) per year 2012-2022, R2 of shares, total ratio; the pilot variant "
                       "(2014/15/16/21 betas, no shrinkage); the one-rate baseline"],
         "gating": "none (methods-validation finding)"},
        {"test_id": "P3t.class_split", "data_scope": SCOPE,
         "fresh_units": ["P3:changhua_cells_2012_2022"],
         "external_units": {"P3:changhua_cells_2012_2022": UNITS["P3:changhua_cells_2012_2022"]},
         "what": "class-resolved frozen maps: predicted Changhua shares of irrigation, "
                 "aquaculture, livestock and other (domestic + industry + other) against the "
                 "census class shares in grid",
         "metric": "total-variation distance TV = 0.5 sum |share_pred - share_true|",
         "threshold": f"TV <= {score.TV_MAX} per class-resolved map",
         "maps": cls_maps,
         "secondary": ["truth without the domestic class (other = industry + other)",
                       "per-class ratios"],
         "gating": "none (methods-validation finding)"},
        {"test_id": "P3t.battery_predicts", "data_scope": SCOPE,
         "fresh_units": ["P3:changhua_cells_2012_2022"],
         "external_units": {"P3:changhua_cells_2012_2022": UNITS["P3:changhua_cells_2012_2022"]},
         "what": "does the Yunlin transfer battery (battery.json, mean dAA_5km over the eight "
                 "splits) rank the maps as the real Changhua transfer does (dAA_5km on domain D)",
         "metric": "Spearman rank correlation across the maps scored in both",
         "threshold": f"Spearman >= {score.BATTERY_SPEARMAN_MIN}",
         "secondary": ["the primary map's battery means against its Changhua scores"],
         "gating": "none (methods-validation finding)"},
        {"test_id": "P3t.heads_blocks", "data_scope": SCOPE,
         "fresh_units": ["P3:changhua_i2_blocks"],
         "external_units": {"P3:changhua_i2_blocks": UNITS["P3:changhua_i2_blocks"]},
         "what": "spatial pattern of the heads-only inversion I2 (block abstraction Q_irr + "
                 "Q_rest, e_changhua_blocks_I2.csv; blocks rebuilt by inv_lib.make_blocks and "
                 "checked against the csv) against census energy per block",
         "metric": "Spearman across the 10 blocks; AA over blocks against the cell-count "
                   "uniform baseline",
         "threshold": "Spearman >= 0.5",
         "gating": "none (methods-validation finding)"},
    ]
    return {
        "id": PREREG_ID, "wp": "P3", "kind": "test",
        "title": "Out-of-sample test of the plan-P3 land-use / crop-statistics transfer methods "
                 "(fitted on Yunlin, pre-origin years) against the Changhua pump census (C0 "
                 "preorigin/v3, 2012-2022)",
        "tests": tests,
        "zones": not_applicable("no model zones; Changhua cells, 5 km blocks and townships"),
        "priors": not_applicable("no prior is fitted; the maps are frozen predictions"),
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("single real transfer Yunlin -> Changhua; the Yunlin battery "
                                "(eight splits + LOTO) was run before registration and fixed "
                                "the primary map"),
        "fresh_units": sorted(UNITS),
        "thresholds": {t["test_id"]: t["threshold"] for t in tests},
        "gate_scores": not_applicable("methods-validation report: each test passes or fails on "
                                      "its own threshold; no weighted gate; nothing is "
                                      "selected or published as a model input"),
        "primary_model": f"primary pi = {primary!r} by the plan's rule (crop-dasymetric if it "
                         f"passes the battery gate, else the LUIMAP random forest rf_LU+GEO); "
                         f"battery gate: {json.dumps(bat['gate'].get(primary))}; "
                         f"crop-dasymetric gate: {json.dumps(bat['gate'].get('cropdas_plan'))}",
        "inputs": inputs(),
        "interpretation": {
            "truth": "C0 pump_electricity/preorigin/v3 (gate passed), Changhua county "
                     "component, opened through the loader guard for use 'validator'; annual "
                     "mean 2012-01..2022-12; all classes primary (the model's forcing "
                     "definition), domestic-excluded as the sensitivity (C0 flags Changhua's "
                     "domestic class as possibly non-pumping load)",
            "domains": "pred_c = map_c x MOI Changhua share; D = cells with share > 0 or "
                       "census energy > 0; D05 = share > 0.5",
            "known_before_registration": "in-grid Changhua total 187.36 GWh/yr, domestic 23.0 "
                                         "% (43.1 GWh/yr), 97 % of pumps in active cells, "
                                         "irrigation seasonal CV 0.14 (look "
                                         "[P3-seed-known-totals]); the level comparisons "
                                         "below are therefore not blind and are not tests",
            "declared_comparisons_on_known_truth": {
                "level_scouts_and_rebuilt": f"|ln(estimate / truth)| <= {score.TOL_LOG} (plan "
                                            "P4.3 level tolerance) per estimate; the "
                                            "published LUIMAP + crop-dasymetric range "
                                            f"[{min(lu):.2f}, {max(lu):.2f}] GWh/yr covers "
                                            "the truth or not",
                "heads_only_level": "I2 43.43 GWh/yr equivalent against the truth, same "
                                    "tolerance",
                "phi_in": {"estimates_fixed_here": phi,
                           "rule": "census energy fraction in grid (kept Changhua pumps, "
                                   "2012-22) inside [min, max] of the three estimates"},
            },
            "meaning": "Shape tests passing with level failing would confirm the plan's "
                       "'shape only, prior-width only' use; shape failing would mean land use "
                       "does not transfer where no census exists. The battery test says "
                       "whether the in-Yunlin battery can choose maps for such a transfer. "
                       "Nothing here is fed back into any product or model (the census "
                       "itself replaces these priors).",
            "maps_frozen": maps,
            "predictions_sha256_note": "work/predictions.npz and the crop-model files are "
                                       "hashed in inputs; score.py refuses to run if they "
                                       "change",
        },
    }


def register(dry: bool) -> None:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("\n".join(errs))
    if dry:
        print(json.dumps({k: v for k, v in d.items() if k not in ("inputs",)},
                         ensure_ascii=False, indent=1, default=lib._jdefault)[:6000])
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    {"seed": lambda: seed(), "register": lambda: register("--dry-run" in sys.argv)}[
        sys.argv[1]]()
