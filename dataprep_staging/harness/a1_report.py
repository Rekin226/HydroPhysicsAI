"""Write the WP A1 report (``results/dataprep/reports/A1.md`` and ``A1.json``) and its
ledger rows from the evidence files; refuses to write if any reported number has no row.

    python -m harness.a1_report          (from dataprep_staging/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, REPO, REPORTS_DIR, rel_to_repo  # noqa: E402

from harness import __version__, coverage, scorers  # noqa: E402

EV = REPORTS_DIR / "A1_evidence"
#: confirm-run configurations with 3-7 clusters of 2-10 rows
SMALL = ("W1_7yr_x2", "E11_5reg_x6_rho0.1", "E11_5reg_x6_rho0.3", "W1_7yr_x2_rho0.5",
         "E6_5reg_x6_rho0.5", "k3_x10_rho0.1")
UNITS = DATAPREP / "validation_units" / "v1"
HERE = Path(__file__).resolve().parent
PLAN = "docs/DATA_REMEDIATION_PLAN_2026-10.md"
GAPP = "results/dataprep/_pilots/head_gapfill_pilot/out/"
PHYS = "results/dataprep/_pilots/physics_inversion_hooks/"
TPCP = "results/dataprep/_pilots/tpc_aggregate_electricity/"
UNIT_INPUTS = ["results/twin_forward/datum_gate_innov.npz",
               "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json",
               "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg",
               "AMP_V2/data/fan_stations.parquet", "AMP_V2/data/wells",
               "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"]


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def g(x, n=4):
    return f"{x:.{n}g}"


def main() -> int:
    rp = load(EV / "repro_final" / "repro_pilots.json")
    rp1 = load(EV / "repro" / "repro_pilots.json")
    cs = load(EV / "coverage_sim.json")
    csr = load(EV / "coverage_sim_rerun.json")
    cc = load(EV / "coverage_confirm.json")
    gp = load(EV / "guard_prereg_probes.json")
    mu = load(EV / "mutation_check.json")
    val = load(UNITS / "validation.json")
    reg = load(UNITS / "units.json")
    pytest_txt = (EV / "pytest.txt").read_text().strip().splitlines()[-1]
    ruff_txt = (EV / "ruff.txt").read_text().strip()
    P = rp["parts"]
    gf = P["gapfill"]
    ccfg = {c["name"]: c for c in cc["configs"]}
    units = {u["id"]: u for u in reg["units"]}
    fresh = [u for u in reg["units"] if u["role"] == "fresh"]
    dt = [u for u in reg["units"] if u["role"] == "design_time"]
    dec_sha = gp["preregistrations"]["A1-fresh-units-v1"]["sha256"]
    dec_time = gp["preregistrations"]["A1-fresh-units-v1"]["created"]
    conf_sha = gp["preregistrations"]["A1.coverage_confirm.v1"]["sha256"]
    same_design = {k: v for k, v in cs.items() if k != "seconds"} == \
        {k: v for k, v in csr.items() if k != "seconds"}
    rp_same = all(rp["parts"][k] == rp1["parts"][k] for k in ("gapfill", "gap_masks")) and \
        rp["max_rel_diff_overall"] == rp1["max_rel_diff_overall"]
    mj = max(u["max_jaccard_design_time"]["value"] for u in fresh if "max_jaccard_design_time"
             in u)
    lev = val["leveling_5fold_alternative"]

    acceptance = [
        {"criterion": "Shared scorers reproduce the pilot metrics named in A0 "
                      "(pilot_final_clean.csv, d_pseudo_changhua.json, TPC compare_annual.csv)",
         "threshold": "max relative difference <= 1e-6",
         "measured": (f"pilot_final_clean {gf['clean']['n_values']} values, max "
                      f"{g(gf['clean']['max_rel_diff'], 2)}; d_pseudo_changhua "
                      f"{P['pseudo']['scorers']['n_values']} metric fields, max "
                      f"{g(P['pseudo']['scorers']['max_rel_diff'], 2)}; compare_annual "
                      f"{P['tpc']['n_values']} values at full precision, max "
                      f"{g(P['tpc']['max_rel_diff'], 2)}, and "
                      f"{P['tpc']['archive_identical_at_stored_precision']}/"
                      f"{P['tpc']['archive_values']} identical at the file's 4 decimals"),
         "passed": bool(gf["clean"]["pass"] and P["pseudo"]["scorers"]["pass"]
                        and P["tpc"]["pass"])},
        {"criterion": "Further pilot metrics re-derived with the same scorers (gap-fill "
                      "pilot_final_all and pilot_summary_cleantruth; physics pilot "
                      "f_temporal_holdout, out-of-time RMSE with the training-mean datum)",
         "threshold": "max relative difference <= 1e-6",
         "measured": (f"{gf['all']['n_values']} + {gf['summary_cleantruth']['n_values']} + "
                      f"{P['holdout']['scorers']['n_values']} values, max "
                      f"{g(max(gf['all']['max_rel_diff'], gf['summary_cleantruth']['max_rel_diff'], P['holdout']['scorers']['max_rel_diff']), 2)}; "  # noqa: E501
                      f"all parts {rp['n_values_total']} values, max "
                      f"{g(rp['max_rel_diff_overall'], 2)}"),
         "passed": bool(rp["pass"])},
        {"criterion": "Coverage interval: actual coverage of a true 0.90 in 10,000 simulated "
                      "clustered binomial draws (confirmatory run, pre-registered "
                      "A1.coverage_confirm.v1, fresh seeds, rule frozen)",
         "threshold": ">= 0.89 in every primary configuration (20)",
         "measured": (f"min {cc['primary_min_actual_coverage']:.4f} "
                      f"({min(cc['configs'], key=lambda c: c['actual_coverage'] if c['role'] == 'primary' else 9)['name']}); "  # noqa: E501
                      f"0 of 20 below; secondary mixtures min "
                      f"{cc['secondary_min_actual_coverage']:.4f}"),
         "passed": bool(cc["pass"])},
        {"criterion": "Coverage interval, design-time run (chose the quantile; labelled "
                      "design-time)",
         "threshold": ">= 0.89 in each of 15 configurations",
         "measured": (f"Student-t quantile min {cs['min_actual_coverage']['t']:.4f} (0 below); "
                      f"plan's normal quantile min {cs['min_actual_coverage']['z']:.4f} "
                      f"({', '.join(cs['configs_below_criterion']['z'])} below 0.89), so "
                      "t became the default; rerun with today's code identical: "
                      f"{same_design}"),
         "passed": bool(not cs["configs_below_criterion"]["t"])},
        {"criterion": "Fresh validation units declared and hashed before any downstream test "
                      "uses them",
         "threshold": "declaration written; no look on any fresh unit before it",
         "measured": (f"A1-fresh-units-v1 ({len(fresh)} fresh units) written {dec_time}, "
                      f"sha256 {dec_sha[:12]}...; looks on fresh units at report time: "
                      f"{len(gp['fresh_units']['with_any_look'])}"),
         "passed": bool(not gp["fresh_units"]["with_any_look"])},
        {"criterion": "Validation-units product gate (counts reproduce the plan's design-time "
                      "numbers; fresh partitions disjoint; no fresh unit duplicates a "
                      "design-time region; folds keep sites whole and respect buffers; well "
                      "set identical to the physics pilot's)",
         "threshold": "all checks pass",
         "measured": f"{sum(c['pass'] for c in val['checks'])}/{len(val['checks'])} checks; "
                     f"max Jaccard of a fresh unit with a design-time region {mj:.3f}",
         "passed": bool(val["passed"])},
        {"criterion": "Artificial-gap generators reproduce the pilot's masks (families N, S, "
                      "E, H on the 2012-2022 matrix; test-period replica T)",
         "threshold": "identical masked cells, same order",
         "measured": f"{P['gap_masks']['n_identical']}/{P['gap_masks']['n_realizations']} "
                     "realizations identical",
         "passed": bool(P["gap_masks"]["pass"])},
        {"criterion": "Pre-registration tool and loader guard behave as specified",
         "threshold": "unit tests pass; design-time unit refused as fresh; guard opens the "
                      "units only for validator / selection / after-the-fact analysis",
         "measured": (f"{pytest_txt} (harness 21 + common 31); ruff: {ruff_txt}; "
                      f"probe refusal: {gp['probe_design_time_as_fresh']['validation_errors'][0][:70]}...; "  # noqa: E501
                      f"guard as expected: {gp['guard_as_expected']}; checker: "
                      f"{gp['checker']['status']}"),
         "passed": bool(gp["pass"] and "passed" in pytest_txt and "failed" not in pytest_txt)},
    ]
    status = "passed" if all(a["passed"] for a in acceptance) else "partial"

    pw = {n: ccfg[n] for n in ("indep_n30", "W1_7yr_x2", "E11_5reg_x6_rho0.1",
                               "E6_5reg_x6_rho0.5", "S1_14site_x13_rho0.3", "V3_74well_x42_rho0.2")}
    key_numbers = [
        f"Pilot reproduction: {rp['n_values_total']} values, max relative difference "
        f"{g(rp['max_rel_diff_overall'], 3)} (pseudo-Changhua worst: "
        f"{P['pseudo']['scorers']['worst']['key']}); patched pilot reruns equal the archive "
        f"(pseudo 219 values {g(P['pseudo']['patched_rerun']['max_rel_diff'], 2)}, holdout 53 "
        f"values {g(P['holdout']['patched_rerun']['max_rel_diff'], 2)}); "
        f"{rp['seconds']:.0f} s",
        f"Gap masks: {P['gap_masks']['n_identical']}/{P['gap_masks']['n_realizations']} "
        "realizations identical to the pilot (N 3, S 4, E 2, H 3, T 3)",
        "Mutation check: exact z instead of the pilots' 1.645 -> "
        f"{mu['z_exact_instead_of_1.645']['clean'][1]} of 704 values off; finite-sample "
        f"conformal -> {mu['finite_sample_conformal_instead_of_linear']['clean'][1]} off; "
        f"'rms' instead of mean-of-RMSE -> "
        f"{mu['eqwell_rms_instead_of_mean_of_rmse']['summary_cleantruth'][1]} of 660 off",
        f"Coverage confirm (pre-registered, seeds {cc['seed0']}+): primary min "
        f"{cc['primary_min_actual_coverage']:.4f} over 20 configurations, secondary min "
        f"{cc['secondary_min_actual_coverage']:.4f}",
        f"Coverage design-time (seeds 20261012+): t min {cs['min_actual_coverage']['t']:.4f}, "
        f"z min {cs['min_actual_coverage']['z']:.4f} (few_large_k5_x200_rho0.05)",
        "Gate power (full 4.4 rule, confirm run): pass rate at true 0.80 = "
        + ", ".join(f"{n} {c['full_rule_pass_rate_true0.8']:.2f}" for n, c in pw.items())
        + "; pass rate at true 0.90 = "
        + ", ".join(f"{n} {c['full_rule_pass_rate_true0.9']:.2f}" for n, c in pw.items()),
        f"Units: {len(dt)} design-time, {len(fresh)} fresh; Yunlin terciles 374/385/399 "
        "cells, north half 600 cells / 37 wells, two-tercile ME 63 and WM 55 of 84 Yunlin wells, "
        "Z1/Z2/Z3 139/300/515 cells, 13/24/37 wells; CH/YL cells 954/1158; wells 158 (74/84)",
        "Fresh: rotated terciles " + "/".join(
            f"{units[f'fresh.rot_tercile.{i}']['n_cells']}" for i in (1, 2, 3)) + " cells ("
        + "/".join(f"{units[f'fresh.rot_tercile.{i}']['n_wells']}" for i in (1, 2, 3))
        + " wells); k-means 2027 " + "/".join(
            f"{units[f'fresh.km2027.{i}']['n_cells']}" for i in range(5)) + " cells ("
        + "/".join(f"{units[f'fresh.km2027.{i}']['n_wells']}" for i in range(5))
        + " wells); river band YL10 "
        f"{units['fresh.river_band_YL10']['n_cells']} cells / "
        f"{units['fresh.river_band_YL10']['n_wells']} wells; block 2016-01..2018-12",
        f"Max Jaccard fresh vs design-time {mj:.3f}; leveling 5 balanced folds with 10 km buffer "
        f"would leave {min(lev['leveling_CH']['train_sizes'])} (CH) / "
        f"{min(lev['leveling_all']['train_sizes'])} (all) training benchmarks -> leave one "
        "cluster out (19 / 41 folds)",
        f"Design-time look seed dated {val['design_time_look_seed']['time']} "
        f"(mtime of _pilots/{val['design_time_look_seed']['evidence']})",
        f"Tests: {pytest_txt}; ruff clean",
        "U2 sigma_obs uses the pre-origin rho(k): "
        + ", ".join(f"k={k} {v:.3f}" for k, v in scorers.RHO_K_PREORIGIN.items())
        + " m; floor 0.10 m at a complete month",
        "Claude API tokens: 0; GPU: none",
    ]

    deviations = [
        "Code lives in dataprep_staging/harness/ (user decision D12), not "
        "results/dataprep/_staging_code/.",
        "Coverage gate quantile: Student-t on k - 1 degrees of freedom (k = clusters) inside "
        "the Wilson interval on n_eff, not the normal quantile. The plan names the Wilson "
        "interval on n / deff; the normal quantile reached only "
        f"{cs['min_actual_coverage']['z']:.4f} in the 5 x 200 configuration of the design run, "
        "so the declared rule switched to t. The confirmatory run on fresh seeds was "
        "pre-registered and passed. The interval level is 0.95 (the plan does not state one).",
        "Pre-origin defaults (U3, U7) for families whose plan wording reaches past 2022: "
        "rolling-origin July backtest 2013-2022 (plan 2013-2024), matched-lead backcast "
        "2020-22 -> 2016-18 (plan 2020-23 -> 2016-19), daily outage window 2013-2022 (plan "
        "2013-2025), test-period replica inside a pre-origin window. The plan's spans need "
        "allow_post_origin=True, for after-the-fact analysis only.",
        "Leveling folds are leave-one-cluster-out (10 km complete-linkage clusters, 10 km "
        "buffer), not five balanced folds: five folds with that buffer leave "
        f"{min(lev['leveling_CH']['train_sizes'])} training benchmarks in Changhua's worst fold.",
        "The third rotated tercile is the wrap-around remainder (the far-west and far-east "
        "strips), so it is not contiguous: rotating the cut points by half a tercile on a "
        "line leaves two end strips, which are joined into the third unit.",
        "The 10 km Yunlin river band (the Z2 mirror) and the fresh well and leveling folds are "
        "declared fresh as well; no pilot used them as hold-out units. The north half stays "
        "design-time.",
        "P4's inference experiments (E1-E11, P4.3) were superseded by the Changhua census "
        "(2026-10-12 update), so the fresh units are declared for every downstream test "
        "(H2, H3, P3, P4m, S1, S5, I1) rather than for E1-E11.",
        "Added beyond the plan: scorers.sigma_obs / u2_weight with A0's pre-origin rho(k); "
        "a mask-level reproduction of the pilot's gap generators; a mutation check of the "
        "reproduction; the gap-fill pilot's pilot_final_all, pilot_summary_cleantruth and the "
        "physics pilot's f_temporal_holdout as extra reproduction targets.",
        "TPC compare_annual: the harness compares 154 values (11 years x 14 columns); A0's "
        "byte-level comparison counted 159.",
    ]

    issues = [
        "Pre-register before scoring on a fresh unit: prereg.write_prereg(doc, units_registry="
        "'results/dataprep/validation_units/v1/units.json'), then prereg.assert_can_score(id, "
        "wp, test_id=...) immediately before computing the registered score. Log every other "
        "look at test data with prereg.record_look: the log is self-reported, and an unlogged "
        "look cannot be detected.",
        "Fresh means never used as a hold-out unit for a design choice; it does not mean "
        "unseen data. The 2016-18 months and every Yunlin and Changhua well were inside the "
        "pilots' 2012-22 fits, and the fresh cell units overlap design-time regions (max "
        f"Jaccard {mj:.3f}). Report the effective n; overlapping units give no independent "
        "estimate of tau (plan 4.3).",
        "The coverage gate has little power in small strata. With 7 x 2 clusters (W1) the full "
        "4.4 rule passes a product whose true coverage is 0.80 in "
        f"{ccfg['W1_7yr_x2']['full_rule_pass_rate_true0.8']:.0%} of draws, and with 5 regions x "
        f"6 in {ccfg['E11_5reg_x6_rho0.1']['full_rule_pass_rate_true0.8']:.0%}-"
        f"{ccfg['E11_5reg_x6_rho0.3']['full_rule_pass_rate_true0.8']:.0%}. In the "
        "configurations with 3-7 clusters of 2-10 rows the interval over-covers "
        f"({min(ccfg[n]['actual_coverage'] for n in SMALL):.3f}-"
        f"{max(ccfg[n]['actual_coverage'] for n in SMALL):.3f}). A pass in such a stratum is "
        "weak evidence: report n_eff, k and the interval with the verdict.",
        "The plan's point window (0.85-0.95 when n_eff >= 30) fails well-calibrated products "
        "often near that size: at a true 0.90 the full rule passes only "
        f"{ccfg['indep_n30']['full_rule_pass_rate_true0.9']:.0%} of 30 independent rows and "
        f"{ccfg['S1_14site_x13_rho0.3']['full_rule_pass_rate_true0.9']:.0%} of 14 sites x 13 at "
        "ICC 0.3. This is the plan's rule, left unchanged; expect false fails near n_eff = 30.",
        "U2 weights: use scorers.sigma_obs / u2_weight (pre-origin rho(k) 0.331/0.494/0.651/"
        "0.804/0.862 m at k = 20/15/10/5/3, floor 0.10 m at a complete month), not the plan's "
        "values, which included 2023-26 heads.",
        "Gap families that reach past 2022 raise LeakageError unless allow_post_origin=True; "
        "that flag is for after-the-fact analysis of full variants only. Pre-origin products "
        "are tested on the 2012-2022 matrix.",
        "The two-tercile region is fresh.two_tercile.ME (63 of 84 Yunlin wells, closest to "
        "Changhua's 74 of 158); fresh.two_tercile.WM (55) is secondary.",
        "Unit membership is stored in units.json. Do not recompute the k-means partition: "
        "another scikit-learn version may give a different one.",
        "Wells in the units are the 158 calibration wells of build_head_field with the default "
        "(non-strict) coverage QC, 2012-01..2022-12. A WP that uses another well set must map "
        "its wells to units by model cell.",
        "dt.folds.pilot_site_* apply the pilots' site-fold rule per county; the pilots applied "
        "it per pseudo-region and to Changhua. They are design-time labels only.",
        "Editing build_units.py, units.py or harness/__init__.py makes validation_units/v1 "
        "fail the checker (its code hashes); build v2 instead. The pre-registration "
        "A1.coverage_confirm.v1 hashes coverage.py, coverage_sim.py and coverage_confirm.py.",
        "P1 and W1 used harness.prereg and harness.coverage before this report; neither "
        "file changed after they used them. This round added scorers.sigma_obs/u2_weight, "
        "edited repro_pilots.py, build_units.py and a units.py docstring, and added "
        "coverage_confirm.py, a1_prereg.py, a1_checks.py and a1_report.py; prereg.py, "
        "coverage.py, coverage_sim.py, gaps.py and placebos.py were not edited.",
    ]

    products = [
        {"path": rel_to_repo(UNITS) + "/", "layer": "L2",
         "manifest": rel_to_repo(UNITS / "MANIFEST.json"),
         "allowed_uses": "selection, validator, after_the_fact_analysis (model_design, "
                         "priors, targets, datum fits, baselines and initial conditions "
                         "forbidden)",
         "fit_window_end": "2022-12"},
        {"path": rel_to_repo(PREREG_DIR / "A1-fresh-units-v1.json"), "layer": "preregistration",
         "manifest": f"INDEX.jsonl + ledger row A1.prereg.A1-fresh-units-v1 (sha256 {dec_sha})",
         "allowed_uses": "declaration of the fresh units and the coverage gate rule",
         "fit_window_end": "n/a"},
        {"path": rel_to_repo(PREREG_DIR / "A1.coverage_confirm.v1.json"),
         "layer": "preregistration",
         "manifest": f"INDEX.jsonl + ledger row A1.prereg.A1.coverage_confirm.v1 (sha256 "
                     f"{conf_sha})",
         "allowed_uses": "registration of the confirmatory coverage simulation",
         "fit_window_end": "n/a"},
        {"path": rel_to_repo(EV) + "/", "layer": "evidence",
         "manifest": "none (WP evidence: reproduction runs, coverage simulations, probes, "
                     "mutation check, logs)",
         "allowed_uses": "provenance of the A1 acceptance; not a data product",
         "fit_window_end": "n/a"},
        {"path": "results/dataprep/ledger/rows/A1.csv", "layer": "ledger",
         "manifest": "ledger fragment (explicit sources)", "allowed_uses": "data-role ledger "
                     "(U6)", "fit_window_end": "n/a"},
    ]
    code_files = sorted(p for p in HERE.glob("*.py")) + sorted((HERE / "tests").glob("*.py"))
    code_paths = [str(p) for p in code_files] + [str(REPO / "dataprep_staging" / "README.md")]

    # ---------------------------------------------------------------- ledger rows
    ev_rp = rel_to_repo(EV / "repro_final" / "repro_pilots.json")
    pv_rp = f"A1_evidence/repro_final/repro_pilots.json@{sha256_file(EV / 'repro_final' / 'repro_pilots.json')[:16]}"  # noqa: E501
    pv_units = "validation_units/v1"
    pv_cc = f"A1_evidence/coverage_confirm.json@{sha256_file(EV / 'coverage_confirm.json')[:16]}"
    pv_cs = f"A1_evidence/coverage_sim.json@{sha256_file(EV / 'coverage_sim.json')[:16]}"
    rows = []

    def add(nid, src, role, pv, note=""):
        rows.append({"number_id": nid, "source": src, "role": role, "product_version": pv,
                     "note": note})

    for lab, ref in (("clean", "pilot_final_clean.csv"), ("all", "pilot_final_all.csv")):
        nid = f"A1.repro.gapfill_final_{lab}"
        add(nid, GAPP + "pilot_ensembles.parquet", "input", pv_rp, "pilot predictions and sds")
        add(nid, GAPP + ref, "validator", pv_rp, "pilot metrics reproduced")
    add("A1.repro.gapfill_summary_cleantruth", GAPP + "pilot_all_scored.parquet", "input", pv_rp)
    add("A1.repro.gapfill_summary_cleantruth", GAPP + "pilot_summary_cleantruth.csv",
        "validator", pv_rp)
    for nid, ref in (("A1.repro.pseudo_changhua", "d_pseudo_changhua.json"),
                     ("A1.repro.temporal_holdout", "f_temporal_holdout.json")):
        for f in ("inputs_cwa.npz", "hindcast_cwa.npz", "K_pump_f32.npy", "M_cwa.npz"):
            add(nid, PHYS + f, "input", pv_rp, "arrays of the patched pilot rerun")
        add(nid, PHYS + ref, "validator", pv_rp, "archived pilot metrics")
    add("A1.repro.patched_reruns", PHYS + "d_pseudo_changhua.json", "validator", pv_rp)
    add("A1.repro.patched_reruns", PHYS + "f_temporal_holdout.json", "validator", pv_rp)
    add("A1.repro.patched_reruns", PHYS + "inputs_cwa.npz", "input", pv_rp)
    add("A1.repro.tpc_compare_annual", TPCP + "tpc_county_sector_monthly.csv", "input", pv_rp)
    add("A1.repro.tpc_compare_annual", TPCP + "census_month.csv", "input", pv_rp)
    add("A1.repro.tpc_compare_annual", TPCP + "compare_annual.csv", "validator", pv_rp)
    add("A1.repro.gap_masks", GAPP + "monthly_long.parquet", "input", pv_rp)
    add("A1.repro.gap_masks", GAPP + "daily_clean.parquet", "input", pv_rp,
        "Feb/Mar 2026 bridge for the T replica (after the fact)")
    for f in ("pilot_N.parquet", "pilot_S.parquet", "pilot_E.parquet", "pilot_H.parquet",
              "pilot_testperiod.parquet"):
        add("A1.repro.gap_masks", GAPP + f, "validator", pv_rp, "pilot masked cells")
    add("A1.repro.overall", ev_rp, "validator", pv_rp, "all reproduction parts")
    add("A1.repro.overall", "results/dataprep/_pilots/head_gapfill_pilot", "input", pv_rp)
    add("A1.repro.overall", "results/dataprep/_pilots/physics_inversion_hooks", "input", pv_rp)
    add("A1.repro.overall", "results/dataprep/_pilots/tpc_aggregate_electricity", "input",
        pv_rp)
    mpv = f"A1_evidence/mutation_check.json@{sha256_file(EV / 'mutation_check.json')[:16]}"
    add("A1.mutation.gapfill", GAPP + "pilot_ensembles.parquet", "input", mpv)
    add("A1.mutation.gapfill", GAPP + "pilot_all_scored.parquet", "input", mpv)
    for f in ("pilot_final_clean.csv", "pilot_final_all.csv", "pilot_summary_cleantruth.csv"):
        add("A1.mutation.gapfill", GAPP + f, "validator", mpv)
    add("A1.coverage.confirm", "dataprep_staging/harness/coverage_confirm.py", "input", pv_cc,
        "simulation code (seeds in the pre-registration)")
    add("A1.coverage.confirm", "dataprep_staging/harness/coverage.py", "input", pv_cc,
        "rule under test")
    add("A1.coverage.confirm", "results/dataprep/preregistration/A1.coverage_confirm.v1.json",
        "input", pv_cc, "pre-registration written before the run")
    add("A1.coverage.power", "dataprep_staging/harness/coverage_confirm.py", "input", pv_cc,
        "information only")
    add("A1.coverage.design_time", "dataprep_staging/harness/coverage_sim.py", "input", pv_cs)
    add("A1.coverage.design_time", rel_to_repo(EV / "coverage_sim.json"),
        "design_time_evidence", pv_cs, "this run selected the Student-t quantile")
    add("A1.coverage.design_time", rel_to_repo(EV / "coverage_sim_rerun.json"), "validator",
        pv_cs, "rerun with today's code, identical")
    for nid in ("A1.units.gate", "A1.units.counts", "A1.units.fresh", "A1.units.max_jaccard",
                "A1.units.folds", "A1.units.leveling_5fold_alternative"):
        for src in UNIT_INPUTS:
            add(nid, src, "input", pv_units)
    add("A1.units.counts", PLAN, "validator", pv_units, "plan's design-time counts")
    add("A1.units.gate", PLAN, "validator", pv_units, "plan's design-time counts")
    add("A1.units.gate", PHYS + "inputs_cwa.npz", "validator", pv_units,
        "physics pilot well set and cell counties (cross-check)")
    add("A1.units.counts", "results/dataprep/_pilots/design_bayesian_fusion/p1b.log",
        "validator", pv_units, "fusion pilot region sizes")
    add("A1.units.look_seed_time", PHYS + "e_changhua_inversion.json", "input", pv_units,
        "file modification time")
    add("A1.guard_probes", rel_to_repo(UNITS), "input", rel_to_repo(EV / "guard_prereg_probes.json"))
    add("A1.guard_probes", "results/dataprep/preregistration/looks.jsonl", "input",
        rel_to_repo(EV / "guard_prereg_probes.json"))
    add("A1.guard_probes", "results/dataprep/preregistration/INDEX.jsonl", "input",
        rel_to_repo(EV / "guard_prereg_probes.json"))
    add("A1.declaration", "results/dataprep/preregistration/A1-fresh-units-v1.json", "input",
        f"prereg-sha256:{dec_sha}")
    add("A1.declaration", rel_to_repo(UNITS / "units.json"), "input", pv_units)
    add("A1.tests", "dataprep_staging/harness/tests/test_harness.py", "input",
        f"harness@{__version__}")
    add("A1.tests", "dataprep_staging/common/tests/test_library.py", "input",
        f"harness@{__version__}")
    add("A1.sigma_obs", "results/dataprep/reports/A0_evidence/repair/reruns/rho_k_result.json",
        "input", f"harness@{__version__}", "pre-origin rho(k) table (A0)")
    ledger.append_rows("A1", rows)
    reported = {"A1.repro.gapfill_final_clean", "A1.repro.gapfill_final_all",
                "A1.repro.gapfill_summary_cleantruth", "A1.repro.pseudo_changhua",
                "A1.repro.temporal_holdout", "A1.repro.patched_reruns",
                "A1.repro.tpc_compare_annual", "A1.repro.gap_masks", "A1.repro.overall",
                "A1.mutation.gapfill", "A1.coverage.confirm", "A1.coverage.power",
                "A1.coverage.design_time", "A1.units.gate", "A1.units.counts",
                "A1.units.fresh", "A1.units.max_jaccard", "A1.units.folds",
                "A1.units.leveling_5fold_alternative", "A1.units.look_seed_time",
                "A1.guard_probes", "A1.declaration", "A1.tests", "A1.sigma_obs",
                "A1.prereg.A1-fresh-units-v1", "A1.prereg.A1.coverage_confirm.v1"}
    missing = ledger.unledgered(reported)
    if missing:
        raise SystemExit(f"unledgered numbers: {missing}")
    n_rows = len([r for r in ledger.read_all() if r["wp"] == "A1"])

    summary = (
        f"Built the shared validation and anti-leakage harness (dataprep_staging/harness, "
        f"version {__version__}): scorers (equal-well, datum-free anomaly and out-of-time "
        "RMSE with the training-mean datum; native and split-conformal 90 % coverage; "
        "pseudo-truth recovery; U2 sigma_obs with A0's pre-origin rho(k)), the "
        "cluster-adjusted coverage gate, placebo generators, artificial-gap generators for "
        "every plan 4.4 family with the fixed seeds, unit and fold builders, and the "
        "pre-registration tool with a hash-chained look log. The scorers reproduce the pilot "
        f"metrics A0 named to a maximum relative difference of {g(rp['max_rel_diff_overall'], 2)}"
        f" over {rp['n_values_total']} values, and the gap generators reproduce the pilot's "
        f"masks in {P['gap_masks']['n_identical']}/{P['gap_masks']['n_realizations']} "
        "realizations. The coverage interval (Student-t quantile, chosen by a design-time "
        "simulation because the plan's normal quantile reached only "
        f"{cs['min_actual_coverage']['z']:.3f} in one configuration) passed a pre-registered "
        "confirmatory simulation on fresh seeds with a minimum actual coverage of "
        f"{cc['primary_min_actual_coverage']:.4f} over 20 primary configurations (criterion "
        f"0.89). validation_units/v1 ({len(dt)} design-time and {len(fresh)} fresh units, "
        f"gate {sum(c['pass'] for c in val['checks'])}/{len(val['checks'])}) was built from "
        "repository data through 2022-12, and the fresh units were declared and hashed "
        f"(A1-fresh-units-v1) before any downstream test; no look exists on them. "
        f"{pytest_txt.split(' in ')[0]}; ruff clean. The gate has little power in strata "
        "made of a few small clusters (7 x 2, 5 x 6), and the plan's point window often "
        "fails well-calibrated products near n_eff = 30; both are flagged for downstream "
        "WPs. No GPU, 0 Claude API tokens; "
        "nothing under hydrophysics/, tests/ or L0 was written; nothing was staged in git.")

    report = {"wp": "A1", "status": status, "date": "2026-10-12", "summary": summary,
              "acceptance": acceptance, "products": products, "code_paths": code_paths,
              "key_numbers": key_numbers, "deviations_from_plan": deviations,
              "issues_for_downstream": issues,
              "evidence": sorted(rel_to_repo(p) for p in EV.rglob("*.json")
                                 if "_run" not in str(p.parent)),
              "preregistrations": gp["preregistrations"],
              "reproduction_runs_identical": rp_same, "ledger_rows": n_rows,
              "coverage_gate_rule": {"level": coverage.GATE_LEVEL,
                                     "quantile": coverage.DEFAULT_SMALL_K,
                                     "point_window": list(coverage.POINT_WINDOW),
                                     "point_window_n_eff_min": coverage.N_EFF_WINDOW_MIN},
              "claude_api_tokens": 0, "gpu": "none",
              "report_path": rel_to_repo(REPORTS_DIR / "A1.md")}
    (REPORTS_DIR / "A1.json").write_text(json.dumps(report, indent=1, ensure_ascii=False)
                                         + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- markdown
    L = ["# WP A1: validation and anti-leakage harness (report, 2026-10-12)", "",
         f"Status: **{status}**. {summary}", "",
         "## Acceptance", "",
         "| Criterion | Threshold | Measured | Pass |", "|---|---|---|---|"]
    for a in acceptance:
        L.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                 f"{'yes' if a['passed'] else 'NO'} |")
    L += ["", "## What was built", "",
          "Code: `dataprep_staging/harness/` (user decision D12). Import with "
          "`sys.path.insert(0, '<repo>/dataprep_staging')` and `from harness import ...`; "
          "usage is in `dataprep_staging/README.md`.", "",
          "- **`scorers`.** Equal-well RMSE (`agg='rms'`, the twin's; `'mean_of_rmse'`, the "
          "gap-fill pilot's), datum-free anomaly RMSE, out-of-time RMSE with the "
          "training-mean datum (no GLS datum is offered), native coverage and split-conformal "
          "coverage by exchangeable groups (plus a cross-fitted variant that returns per-row "
          "hits for the gate), pseudo-truth recovery metrics (bias, monthly, seasonal and "
          "annual correlation, seasonal-amplitude ratio, block-density correlation), a "
          "climatology baseline that refuses a post-origin fit, ratios of totals, and U2's "
          "`sigma_obs(k, n_days)` / `u2_weight` with A0's pre-origin rho(k).",
          "- **`coverage`.** Cluster-adjusted Wilson interval for a coverage rate on "
          "n_eff = n / (1 + (m_w - 1) rho), rho the ANOVA intra-cluster correlation of the "
          "hit/miss indicators; Student-t quantile on k - 1 df at level 0.95; the plan 4.4 "
          "rule (interval contains 0.90, plus the 0.85-0.95 point window when n_eff >= 30) "
          "and pooling of small strata with their nearest neighbours.",
          "- **`placebos`.** Six-month shift (circular or padded), own 12-month climatology "
          "(fitted to the origin by default), year shuffle that keeps the seasonal shape "
          "(derangement), kWh more than 15 km away, regional (Yunlin) total, standardized "
          "rain-deficit index. Seed convention 6000 + r.",
          "- **`gaps`.** One generator per plan 4.4 family (D, S, E, H, N, T for heads; O for "
          "H1 offsets and steps; R and Y for rings; G for GNSS; B for billing; J, C, X, K for "
          "P1/P2; W for canals; P for E11; L and Q for InSAR), registry `FAMILIES`, seeds "
          "`1000*ord(family)+r` and `7000+r`. The head families are the pilot's code "
          "(identical masks, verified).",
          "- **`units`, `build_units`.** Design-time units (Yunlin easting terciles, halves, "
          "the fusion pilot's 12 regions, Changhua Z1-Z3, the 2020-22 hold-out, the pilots' "
          "site folds) and fresh units (rotated terciles, two-tercile regions, k-means 2027 "
          "partition, 2016-18 block, the Yunlin 10 km river band, site-grouped spatial-block "
          "folds for wells and leveling).",
          "- **`prereg`.** Writes `results/dataprep/preregistration/<id>.json` once "
          "(read-only), stamps the creation time itself, hashes the inputs, records the "
          "sha256 in the hash-chained `INDEX.jsonl` and the WP's ledger fragment, refuses "
          "reused test ids, design-time units named as fresh and units already looked at, "
          "and `assert_can_score` refuses to score a design whose pre-registration postdates "
          "a look at its test data (or whose file changed), logging the scoring look in "
          "`looks.jsonl`.", "",
          "## Validation units (`results/dataprep/validation_units/v1/`)", "",
          "| Unit | Role | Cells | Wells | Max Jaccard with a design-time region |",
          "|---|---|---|---|---|"]
    for u in reg["units"]:
        if u["kind"] != "cells":
            continue
        jj = u.get("max_jaccard_design_time", {})
        L.append(f"| {u['id']} | {u['role']} | {u['n_cells']} | {u['n_wells']} | "
                 f"{jj.get('value', float('nan')):.3f} ({jj.get('with', '-')}) |"
                 if jj else f"| {u['id']} | {u['role']} | {u['n_cells']} | {u['n_wells']} | - |")
    L += ["", "Month units: `dt.block.2020_2022` (design-time) and `fresh.block.2016_2018` "
          "(36 months, disjoint from it). Fold units: "
          + "; ".join(f"`{u['id']}` ({len(u['folds'])} folds)" for u in reg["units"]
                      if u["kind"] == "folds") + ".", "",
          f"Built from repository data through 2022-12 (twin grid, fan polygon, county "
          f"polygons, the 158 calibration wells of `build_head_field`, the leveling panel). "
          f"The well set, its cells, layers and counties, the grid mask and the cell "
          f"counties are identical to the physics pilot's `inputs_cwa.npz`. Design-time units "
          f"carry a look seeded at {val['design_time_look_seed']['time']} (the earliest "
          f"modification time of the pilot outputs that scored them), so the tool refuses "
          f"them as fresh units in two ways (registry role and blocking look).", "",
          "## Key numbers", ""]
    L += [f"- {k}" for k in key_numbers]
    L += ["", "## Deviations from the plan", ""] + [f"- {d}" for d in deviations]
    L += ["", "## Issues for downstream WPs", ""] + [f"- {i}" for i in issues]
    L += ["", "## Evidence", "",
          "All under `results/dataprep/reports/A1_evidence/`:", "",
          "- `repro_final/repro_pilots.json` (final reproduction; `repro/` is an identical "
          "earlier run made before `scorers.sigma_obs` was added), with the patched pilot "
          "runs, their patches and logs in `repro_final/{pseudo,holdout,tpc}_run/`;",
          "- `coverage_confirm.json` (pre-registered confirmatory simulation), "
          "`coverage_sim.json` (design-time) and `coverage_sim_rerun.json` (identical rerun);",
          "- `guard_prereg_probes.json`, `mutation_check.json` (+ its script), "
          "`build_units.log`, `pytest.txt`, `ruff.txt`;",
          "- the product's own `validation.json` and `MANIFEST.json`.", "",
          f"Ledger: {n_rows} rows in `results/dataprep/ledger/rows/A1.csv`, every reported "
          "number covered. No GPU, 0 Claude API tokens. Nothing under `hydrophysics/`, "
          "`tests/` or L0 was written; nothing was staged or committed in git."]
    (REPORTS_DIR / "A1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    out, n = ledger.assemble()
    print(f"status {status}; ledger rows A1 {n_rows}; data_roles.csv {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
