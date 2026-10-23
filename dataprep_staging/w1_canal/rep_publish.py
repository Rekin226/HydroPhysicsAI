"""Publish the staged repair round as surface_water_v4/full/v<N+1> (W1 repair round).

    cpu_slot.sh python -I dataprep_staging/w1_canal/rep_publish.py STAGE_DIR

Copies the rep_build.py staging tree into a new version directory (never an existing one),
sets the product version in every long table, writes validation.json, QC_REPORT.md and the
manifest, and appends the W1 ledger rows of the repair round. The manifest lists full/v2's
inputs and code, the full/v2 files read, the A1 declaration, the repair code, and (verifier
minor issue: incomplete provenance) the inputs of the reporting checks that full/v1-v2
carried in validation.json without listing them: the head check (AMP_V2 wells tree and
station metadata, the CODIS fan rain table) and the comparison with the scout's parse
(_pilots/surface_water_monthly files).

``main(stage, dest)`` with ``dest`` set writes into ``dest`` without ledger rows (synthetic
dry runs only).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
from common import ledger, manifest  # noqa: E402
from common.flags import validate_long_table  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

from w1_canal import checks  # noqa: E402
from w1_canal import config as C  # noqa: E402
from w1_canal import rep_config as RC  # noqa: E402
from w1_canal.rep_build import DECL_ID, FULL_V2  # noqa: E402

LONG_TABLES = ["district_monthly.csv", "percolation_basis.csv", "basin_rain_index_monthly.csv",
               "inflow_predicted_monthly.csv", "gauges_daily_L1.parquet",
               "inflow_composite_daily.parquet"]
OTHER = ["janmay_validation.csv", "statespace_loo_coverage.csv", "printed_monthly.csv",
         "printed_annual.csv", "district_draws.npz", "grid_sw_v4.npz",
         "station_rain_monthly.csv", "ext2026_loo_annual.csv", "ext2026_loo_monthly.csv",
         "rep_coverage_annual.csv", "rep_coverage_janmay.csv", "rep_coverage_inflow.csv",
         "rep_double_loo_monthly.csv"]
REP_CODE = ["rep_config.py", "rep_uncert.py", "rep_build.py", "rep_prereg.py",
            "rep_publish.py"]
REP_COMMON = ["dataprep_staging/harness/prereg2.py", "dataprep_staging/harness/coverage2.py",
              "dataprep_staging/harness/coverage.py"]
V2_READ = ["MANIFEST.json", "district_monthly.csv", "validation.json", "grid_sw_v4.npz",
           "inflow_predicted_monthly.csv", "basin_rain_index_monthly.csv",
           "ext2026_loo_monthly.csv", "QC_REPORT.md", "gauges_daily_L1.parquet",
           "inflow_composite_daily.parquet", "janmay_validation.csv",
           "statespace_loo_coverage.csv", "printed_monthly.csv", "printed_annual.csv",
           "station_rain_monthly.csv", "ext2026_loo_annual.csv"]
PILOT = DATAPREP / "_pilots" / "surface_water_monthly"
REPORTING_INPUTS = [C.WELL_META, checks.CODIS_RAIN, PILOT / "MANIFEST.json",
                    PILOT / "choushui_daily_q.csv", PILOT / "jiji_janmay_supply_2015_2021.csv"]


def f3(x):
    return f"{x:.3f}"


def qc_text(version: str, rep: dict) -> str:
    v2 = (FULL_V2 / "QC_REPORT.md").read_text()
    v2 = v2.replace("# surface_water_v4/full/v2 - QC report",
                    f"# surface_water_v4/full/{version} - QC report (values as full/v2)")
    v2 = v2.replace("## 2026 extension (plan W1 step 6; flag 6; U4 recorded arm only)",
                    "## 2026 extension (plan W1 step 6; flag 6; U4 recorded arm only; the 2026 "
                    "sd listed in this section is full/v2's, superseded by the repair round "
                    "below)")
    ga, gj, gi = rep["coverage_annual"], rep["coverage_janmay"], rep["coverage_inflow"]
    lines = [
        "", "## Repair round (W1, 2026-10-12): repaired 2026 uncertainty and interval labels",
        "", f"- declaration {rep['declaration']['id']} ({rep['declaration']['created']}), "
            "written before any coverage was computed; design-time units only",
        "- values unchanged (2012-2025 = full/v1, 2026 = registered chain); 2026 sd and the "
        "200 members per district come from the repaired error model (rep_config.py)",
        f"- annual coverage of the 90 % intervals (double leave-one-year-out, "
        f"{ga['n_years']} years x 2 districts): {ga['hits']}/{ga['n']} = "
        f"{ga['coverage']:.3f}; A1 gate-2 interval [{ga['a1_gate2']['lo']:.3f}, "
        f"{ga['a1_gate2']['hi']:.3f}] -> {'pass' if ga['a1_gate2']['pass'] else 'FAIL'}",
        f"- Jan-May 2017-2021 (reported): {gj['hits']}/{gj['n']}; inflow monthly 90 % "
        f"(reported, observed months only): {gi['hits']}/{gi['n']} = {gi['coverage']:.3f}, "
        f"A1 gate-2 {'pass' if gi['a1_gate2']['pass'] else 'FAIL'}",
        f"- 2026 interval label: {rep['label_2026']}",
        f"- 2012-2025 rows (full/v1 state-space draws): {rep['label_hist']}",
        "- structural-zero months (winter weight 0): sd 0 and every member 0; the winter-0.5 "
        "alternative is in structural_alt_value (table) and value_structural_alt (grid)",
        "- every row of district_monthly.csv carries sd_basis and interval_calibration "
        "(plan 4.7)", ""]
    for nm, s in rep["series_2026"].items():
        lines.append(f"- {nm} 2026: value {[round(x, 3) for x in s['value']]}; sd "
                     f"{[round(x, 3) for x in s['sd']]}; Jan-Sep {s['jan_sep_total']:.3f}")
    for nm, mc in rep["member_checks"].items():
        lines.append(f"- {nm} members: distinct trajectories {mc['distinct_trajectories']}, "
                     f"Jan-Sep p05/p50/p95 {[round(x, 3) for x in mc['jan_sep_p05_p50_p95']]}"
                     f", Jan-Sep member mean - value {mc['jan_sep_mean_minus_value']:+.3f} "
                     f"(truncation at 0), negative members {mc['negative_members']}")
    return v2 + "\n".join(lines) + "\n"


def main(stage: str, dest: str | None = None):
    stage = Path(stage)
    val = json.loads((stage / "validation_rep.json").read_text())
    rep = val["repair_r1"]
    m2 = json.loads((FULL_V2 / "MANIFEST.json").read_text())
    if dest is None:
        d = manifest.new_version_dir(f"{C.PRODUCT_BASE}/full")
    else:
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=False)
    version = d.name
    pv = f"{C.PRODUCT_BASE}/full/{version}"
    for f in LONG_TABLES:
        t = pd.read_csv(stage / f) if f.endswith(".csv") else pd.read_parquet(stage / f)
        if f == "percolation_basis.csv":
            t["flag"] = t.pop("flag_value")
            t["method"] = "delivery_by_crop_calendar"
            t["source_ids"] = "surface_water_v4/full district_monthly"
        t["product_version"] = pv
        errs = validate_long_table(t, require_all_columns=True)
        if errs:
            raise ValueError(f"{f}: {errs}")
        (t.to_csv(d / f, index=False) if f.endswith(".csv") else
         t.to_parquet(d / f, index=False))
    for f in OTHER:
        shutil.copy2(stage / f, d / f)
    full_val = {**val, "product": f"{C.PRODUCT_BASE}/full", "version": version}
    with open(d / "validation.json", "w") as fh:
        json.dump(full_val, fh, ensure_ascii=False, indent=1)
    (d / "QC_REPORT.md").write_text(qc_text(version, rep))
    gate = dict(m2["gate"])
    gate["interval_calibration_2026"] = {
        "label": rep["label_2026"], "annual_check_passed": rep["annual_check_passed"],
        "annual_hits": f"{rep['coverage_annual']['hits']}/{rep['coverage_annual']['n']}",
        "rule": f"declared in {DECL_ID}: the 22 annual hits (11 evaluation years x 2 "
                "districts, double leave-one-year-out) through the A1 gate-2 interval "
                "(clusters = years); monthly and Jan-Sep intervals are labelled uncalibrated "
                "at those scales whatever the result (plan 4.7)",
        "inflow_monthly": {"hits": f"{rep['coverage_inflow']['hits']}/"
                                   f"{rep['coverage_inflow']['n']}",
                           "a1_gate2_pass": rep["coverage_inflow"]["a1_gate2"]["pass"]}}
    inputs = ([r["path"] for r in m2["inputs"]]
              + [rel_to_repo(FULL_V2 / f) for f in V2_READ]
              + [rel_to_repo(DATAPREP / "preregistration" / f"{DECL_ID}.json")]
              + [rel_to_repo(p) for p in REPORTING_INPUTS])
    inputs = list(dict.fromkeys(inputs))
    trees = [r["path"] for r in m2.get("input_trees") or []] + [rel_to_repo(C.WELL_DIR)]
    code = list(dict.fromkeys([r["path"] for r in m2["code"]]
                              + [rel_to_repo(HERE / f) for f in REP_CODE] + REP_COMMON))
    params = dict(m2["parameters"])
    params["repair_r1"] = {
        "declaration": f"results/dataprep/preregistration/{DECL_ID}.json",
        "settings": "dataprep_staging/w1_canal/rep_config.py",
        "n_draws": RC.N_DRAWS, "draw_seed": RC.DRAW_SEED, "coverage_seed": RC.COVERAGE_SEED,
        "level": RC.LEVEL, "pred_quantile": RC.PRED_QUANTILE, "silverman": RC.SILVERMAN,
        "winter_alt": RC.WINTER_ALT, "coverage_target": RC.COVERAGE_TARGET,
        "janmay_years": list(RC.JANMAY_YEARS), "inflow_max_flag": RC.INFLOW_MAX_FLAG,
        "pool_years": rep["error_model"]["pool_years"]}
    params["staged_by"] = "dataprep_staging/w1_canal/rep_build.py"
    outputs = LONG_TABLES + OTHER + ["validation.json", "QC_REPORT.md"]
    m = manifest.build_manifest(
        d, product=f"{C.PRODUCT_BASE}/full", layer="L2", inputs=inputs, input_trees=trees,
        code=code, parameters=params, fit_window_end=m2["fit_window_end"], outputs=outputs,
        validation_report="validation.json", gate=gate,
        allowed_uses=m2["allowed_uses"], forbidden_uses=m2["forbidden_uses"],
        personal_data="none",
        extra={"layer_note": m2["layer_note"] + "; repair round: repaired 2026 sd and members "
                             "with a coverage-checked or 'uncalibrated' label on every row "
                             "(plan 4.7)",
               "supersedes": "surface_water_v4/full/v2 (identical values; 2026 sd and members "
                             "repaired, interval labels added, structural-zero sd set to 0 "
                             "with the winter-0.5 alternative moved to its own column)"})
    manifest.write_manifest(d, m)
    if dest is None:
        rows = ledger_rows(pv, rep)
        ledger.append_rows(C.WP, rows)
        print("published", d, "ledger rows", len(rows))
    else:
        print("dry-run manifest written to", d)
    return d


def ledger_rows(pv: str, rep: dict) -> list[dict]:
    jj = rel_to_repo(C.JIJI_CSV)
    v1d = rel_to_repo(DATAPREP / C.PRODUCT_BASE / "full" / "v1" / "district_draws.npz")
    comp = rel_to_repo(DATAPREP / C.PRODUCT_BASE / "full" / "v1"
                       / "inflow_composite_daily.parquet")
    rain = rel_to_repo(DATAPREP / C.PRODUCT_BASE / "full" / "v2"
                       / "basin_rain_index_monthly.csv")
    brief = rel_to_repo(C.BRIEFING_110_PDF)
    rows = []

    def add(n, src, role, note=""):
        rows.append({"number_id": n, "source": src, "role": role, "product_version": pv,
                     "note": note})

    for scale in ("annual", "janmay", "inflow"):
        n = f"W1.full_v3.rep.coverage_{scale}"
        add(n, rain, "input", "basin rain index (chain regressor)")
        add(n, comp, "input" if scale != "inflow" else "validator",
            "composite: regression target of the chain folds" if scale != "inflow" else
            "observed composite months (every day flag <= 1) scored; all months fit the folds")
        if scale == "annual":
            add(n, jj + "#held_out_year_district_total", "design_time_evidence",
                "design-time units already scored by W1.ext2026_rain_runoff.v1")
            add(n, jj + "#training_year_totals", "fit_target",
                "beta and district scales of the double leave-one-year-out folds")
            add(n, v1d, "input", "residual trajectories D_k(y, m) - chain_y(m) of the pool "
                                 "years (annual sums = observed totals)")
        elif scale == "janmay":
            add(n, brief, "design_time_evidence", "Jan-May 2017-2021, design-time units")
            add(n, jj + "#training_year_totals", "fit_target")
            add(n, v1d, "input")
        else:
            add(n, jj + "#training_year_totals", "fit_target")
    for d in ("chc", "yl"):
        n = f"W1.full_v3.rep.sd_2026.{d}"
        add(n, v1d, "input", "residual trajectories of the 11 evaluation years")
        add(n, jj + "#training_year_totals", "fit_target", "chain folds")
        add(n, rain, "input")
        add(n, comp, "input", "regression target of the chain folds")
        n = f"W1.full_v3.rep.members_2026.{d}"
        add(n, v1d, "input")
        add(n, rain, "input")
        n = f"W1.full_v3.rep.structural_alt_2026.{d}"
        add(n, jj + "#training_year_totals", "fit_target", "winter-0.5 refit")
        add(n, rain, "input")
    n = "W1.full_v3.rep.inflow_sd_2026"
    add(n, comp, "input", "leave-one-year-out monthly residuals, all months of the 11 years")
    add(n, rain, "input")
    n = "W1.full_v3.rep.label_2026"
    add(n, rel_to_repo(DATAPREP / "preregistration" / f"{DECL_ID}.json"), "input",
        "declared coverage rule")
    return rows


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
