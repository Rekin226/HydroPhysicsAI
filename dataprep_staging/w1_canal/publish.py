"""Publish a checked W1 staging tree as new product versions with manifests and ledger rows.

    python -m w1_canal.publish STAGE_DIR          (from dataprep_staging/, in a CPU slot)

Creates results/dataprep/surface_water_v4/{preorigin,full}/v<N+1>/ (never reuses a version),
copies the staged outputs, writes validation.json, QC_REPORT.md and MANIFEST.json, and
appends the W1 ledger rows.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from common import ledger, manifest  # noqa: E402
from common.flags import validate_long_table  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal.prereg import PREREG_ID  # noqa: E402

PRE_OUTPUTS = ["gauges_daily_L1.parquet", "inflow_composite_daily.parquet",
               "janmay_validation.csv", "statespace_loo_coverage.csv", "district_monthly.csv",
               "district_draws.npz", "percolation_basis.csv", "printed_monthly.csv",
               "printed_annual.csv"]
GRID_OUTPUT = "grid_sw_v4.npz"
CODE_FILES = ["__init__.py", "config.py", "parse_yearbook.py", "sources.py", "composite.py",
              "diversion.py", "validate.py", "statespace.py", "grid.py", "checks.py",
              "build.py", "publish.py", "prereg.py"]
COMMON_CODE = ["dataprep_staging/common/hashing.py", "dataprep_staging/common/flags.py",
               "dataprep_staging/common/paths.py", "dataprep_staging/harness/coverage.py"]
LUIMAP_END = "2024-12"


def code_list() -> list[str]:
    return [rel_to_repo(C.HERE / f) for f in CODE_FILES] + COMMON_CODE


def inputs_for(years) -> list[str]:
    pdfs = [p for y in years for p in sorted((C.YEARBOOK_DIR / f"hyb{y}").glob("*.pdf"))]
    return [rel_to_repo(p) for p in pdfs] + [rel_to_repo(p) for p in (
        C.JIJI_CSV, C.JIJI_HTML, C.BRIEFING_110_PDF, C.MOA_691_CSV, C.V3_NPZ, C.TWIN_GRID_NPZ)]


def parameters(variant: str, val: dict) -> dict:
    v = C.VARIANTS[variant]
    return {
        "variant": variant, "years": [v["years"][0], v["years"][-1]],
        "fit_window": [f"{v['years'][0]}-01", v["fit_window_end"]],
        "inflow_gauges": list(C.INFLOW_GAUGES), "gap_fill_days": C.GAP_FILL_DAYS,
        "donor_ratio_tol": C.DONOR_RATIO_TOL,
        "qc_exclude": {k: [[y, list(m) if m else None] for y, m in it]
                       for k, it in C.QC_EXCLUDE.items()},
        "calendar": {str(k): list(x) for k, x in C.CALENDAR.items()},
        "share_grid": list(C.SHARE_GRID), "winter_grid": list(C.WINTER_GRID),
        "bench_years": list(C.BENCH_YEARS),
        "final_fit": val["final_fit"],
        "bootstrap": {"B": C.BOOT_B, "seed": C.BOOT_SEED, "level": C.BOOT_LEVEL},
        "state_space": {"sigma_janmay_rel": C.SS_SIGMA_JANMAY_REL,
                        "sigma_janmay_min": C.SS_SIGMA_JANMAY_MIN,
                        "sigma_cropshare": C.SS_SIGMA_CROPSHARE,
                        "annual_totals": "exact (normalised within each year)",
                        "rho_grid": list(C.SS_RHO_GRID), "sigma_grid": list(C.SS_SIGMA_GRID),
                        "n_draws": C.SS_N_DRAWS, "seed": C.SS_SEED,
                        "sampler": "elliptical slice sampling, Laplace reference",
                        "thin": C.SS_ESS_THIN, "burn": C.SS_ESS_BURN},
        "preregistration": f"results/dataprep/preregistration/{PREREG_ID}.json",
        "licence_note": "derived from WRA hydrological yearbooks (copying disallowed, plan "
                        "2.6): keep under results/ (gitignored); cite WRA",
        "staged_by": "dataprep_staging/w1_canal/build.py",
    }


def qc_report(variant: str, val: dict, common: dict, version: str) -> str:
    b = val["benchmark"]
    s = b["scores"]
    lines = [f"# surface_water_v4/{variant}/{version} - QC report", "",
             f"Years {val['years'][0]}-{val['years'][1]}. WP W1 (plan section 3). Values in "
             "10^8 m3 unless noted.", "", "## Yearbook parse", ""]
    pq = common["parse_qc"]
    lines += [f"- station-years parsed: {pq['station_years']}; missing days: "
              f"{pq['missing_days']}",
              f"- monthly means from daily values differing from the printed monthly mean by "
              f"more than 5 %: {pq['printed_monthly_off_gt_tol']} of "
              f"{pq['printed_monthly_compared']} station-months (all years 2012-2025)",
              "", "## Inflow composite", ""]
    lines += [f"- {x}" for x in val["composite"]["log"]]
    lines += [f"- annual scale to the weir inflow: {val['composite']['scale_to_weir_inflow']}",
              "", "## Fit", "", f"- {json.dumps(val['final_fit'], ensure_ascii=False)}", "",
              "## Jan-May benchmark (leave-one-year-out, 2015-2021)", "",
              "| method | RMSE Changhua | RMSE Yunlin |", "|---|---|---|"]
    for m, sc in s.items():
        lines.append(f"| {m} | {sc['chc']['rmse']:.3f} | {sc['yl']['rmse']:.3f} |")
    lines += ["", "| comparison | district | RMSE diff | 90 % paired bootstrap | sign test |",
              "|---|---|---|---|---|"]
    for base, by in b["compare"].items():
        for d, r in by.items():
            lines.append(f"| supply_limited_denton - {base} | {d} | {r['rmse_diff']:+.3f} | "
                         f"[{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}] | {r['sign_wins']}/"
                         f"{r['sign_n']}, p = {r['sign_p_two_sided']:.3f} |")
    a = b["acceptance"]
    lines += ["", f"Acceptance (pre-registered): passed = {a['passed']}; label: {a['label']}",
              "", "## State space", ""]
    cv = val["state_space_loo_coverage"]
    lines += [f"- leave-one-year-out 90 % coverage: {cv['hits']}/{cv['n']} "
              f"({cv['coverage']:.2f}); Clopper-Pearson 95 % at n_eff = 7: "
              f"{[round(x, 3) for x in cv['clopper_pearson_95_neff']]}; pre-registered rule "
              f"passed = {cv['passed_prereg_rule']}",
              f"- A1 harness gate: {cv.get('a1_harness_gate')}", "",
              "## Allocation (v3, LUIMAP 2024)", "",
              f"- {json.dumps(val['allocation'], ensure_ascii=False)}", ""]
    return "\n".join(lines) + "\n"


def publish_variant(stage: Path, variant: str, common: dict, root=None) -> tuple[Path, dict]:
    src = stage / variant
    val = json.loads((src / "validation_variant.json").read_text())
    years = C.VARIANTS[variant]["years"]
    d = (manifest.new_version_dir(f"{C.PRODUCT_BASE}/{variant}") if root is None else
         manifest.new_version_dir(f"{C.PRODUCT_BASE}/{variant}", root=root))
    version = d.name
    pv = f"{C.PRODUCT_BASE}/{variant}/{version}"
    for f in PRE_OUTPUTS + [GRID_OUTPUT]:
        if f.startswith("printed_"):
            t = pd.read_csv(stage / f)
            t[t.year.isin(years)].to_csv(d / f, index=False)
        elif f.endswith((".csv", ".parquet")):
            t = pd.read_csv(src / f) if f.endswith(".csv") else pd.read_parquet(src / f)
            if f == "percolation_basis.csv":
                t["flag"] = 5
                t["method"] = "supply_limited_denton_pfd_by_crop_calendar"
                t["source_ids"] = "surface_water_v4 district_monthly"
            if "product_version" in t.columns or f == "percolation_basis.csv":
                t["product_version"] = pv
            if "flag" in t.columns:
                errs = validate_long_table(t, require_all_columns=True)
                if errs:
                    raise ValueError(f"{variant}/{f}: {errs}")
            if f.endswith(".csv"):
                t.to_csv(d / f, index=False)
            else:
                t.to_parquet(d / f, index=False)
        else:
            shutil.copy2(src / f, d / f)
    full_val = {"product": f"{C.PRODUCT_BASE}/{variant}", "version": version,
                "variant_validation": val, "common_checks": common}
    with open(d / "validation.json", "w") as fh:
        json.dump(full_val, fh, ensure_ascii=False, indent=1)
    (d / "QC_REPORT.md").write_text(qc_report(variant, val, common, version))
    acc = val["benchmark"]["acceptance"]
    ships = all(all(v.values()) for v in acc["beats_point"].values())
    gate = {"passed": bool(ships),
            "criteria": "plan W1: v4 ships if its leave-one-year-out Jan-May RMSE is below "
                        "v3 and the climatological fraction in both districts; acceptance "
                        "(90 % paired-bootstrap interval of the RMSE difference below zero "
                        "for all four comparisons) decides the label",
            "acceptance_passed": bool(acc["passed"]), "label": acc["label"]}
    outputs = []
    if variant == "preorigin":
        for f in PRE_OUTPUTS:
            outputs.append({"path": f, "fit_window_end": "2022-12"})
        outputs.append({"path": GRID_OUTPUT, "fit_window_end": LUIMAP_END,
                        "note": "allocation from LUIMAP 2024 (exemption)"})
        outputs += [{"path": "validation.json"}, {"path": "QC_REPORT.md"}]
        sens = val["allocation"]["luimap_sensitivity"]
        exemptions = [{
            "input": rel_to_repo(C.V3_NPZ),
            "data_end": LUIMAP_END,
            "reason": "the cell allocation (paddy_m2, dry_m2) comes from the NLSC LUIMAP 2024 "
                      "land-use snapshot; no pre-origin snapshot is in the repository (plan "
                      "U3 names LUIMAP 2024 as the standard exemption)",
            "ledger_number_id": "W1.preorigin.luimap_sensitivity",
            "sensitivity": (f"same in-fan district volumes spread by county area instead of "
                            f"farmland: correlation of mean cell depth "
                            f"{sens['corr_mean_depth']:.3f}, median |relative change| "
                            f"{sens['median_abs_rel_diff']:.3f}, cells changing by > 50 %: "
                            f"{sens['frac_cells_rel_diff_gt_0.5']:.3f}; district series "
                            f"unaffected"),
            "uses": ["forcing_frozen", "model_design", "design_time_evidence"]}]
        m = manifest.build_manifest(
            d, product=f"{C.PRODUCT_BASE}/preorigin", layer="L2",
            inputs=inputs_for(years) + [rel_to_repo(C.WELL_META), rel_to_repo(
                REPO / "data_fetch_api/new_measurements/cwa_codis_rain/rf_timeseries_codis.csv")],
            input_trees=[rel_to_repo(C.WELL_DIR)],
            code=code_list(), parameters=parameters(variant, val),
            fit_window_end=LUIMAP_END, non_exempt_fit_window_end="2022-12",
            exemptions=exemptions, outputs=outputs, validation_report="validation.json",
            gate=gate,
            allowed_uses=["forcing_frozen", "model_design", "selection", "prior",
                          "prior_width_input", "design_time_evidence",
                          "after_the_fact_analysis"],
            forbidden_uses=["scoring_target", "baseline", "datum_fit", "calibration_target",
                            "prospective_scorer", "validator", "initial_conditions",
                            "idw_field", "forcing_recorded_arm",
                            "statistical_layer_history"],
            personal_data="none",
            extra={"layer_note": "district series and draws are L2 (flag 5); grid_sw_v4.npz "
                                 "is the L3 grid array (value, sd, flag, dates, mask_hash) "
                                 "on the twin grid"})
    else:
        outputs = PRE_OUTPUTS + [GRID_OUTPUT, "validation.json", "QC_REPORT.md"]
        m = manifest.build_manifest(
            d, product=f"{C.PRODUCT_BASE}/full", layer="L2", inputs=inputs_for(years),
            code=code_list(), parameters=parameters(variant, val),
            fit_window_end="2025-12", outputs=outputs, validation_report="validation.json",
            gate=gate,
            allowed_uses=["forcing_recorded_arm", "statistical_layer_history",
                          "after_the_fact_analysis"],
            forbidden_uses=["scoring_target", "baseline", "datum_fit", "calibration_target",
                            "prospective_scorer", "validator", "initial_conditions",
                            "idw_field", "prior", "prior_width_input", "model_design",
                            "selection", "forcing_frozen", "design_time_evidence"],
            personal_data="none",
            extra={"layer_note": "district series and draws are L2 (flag 5); grid_sw_v4.npz "
                                 "is the L3 grid array; U4: post-origin forcing, recorded "
                                 "arm only"})
    manifest.write_manifest(d, m)
    return d, val


def ledger_rows(pv: dict[str, str], common: dict) -> list[dict]:
    yb = "data_fetch_api/new_measurements/wra_hydro_yearbook"
    jj = rel_to_repo(C.JIJI_CSV)
    br = rel_to_repo(C.BRIEFING_110_PDF)
    moa = rel_to_repo(C.MOA_691_CSV)
    v3 = rel_to_repo(C.V3_NPZ)
    rows = []

    def add(num, src, role, ver, note=""):
        rows.append({"number_id": num, "source": src, "role": role, "product_version": ver,
                     "note": note})

    for v, ver in pv.items():
        loo = "leave-one-year-out: the other six years choose the shares or the fraction"
        for d in ("chc", "yl"):
            for m in ("supply_limited_denton", "calendar_denton", "supply_limited_raw_indicator",
                      "clim_fraction_loo", "v3"):
                n = f"W1.{v}.janmay_rmse.{m}.{d}"
                add(n, br, "validator", ver, loo)
                add(n, jj, "input", ver)
                if m == "v3":
                    add(n, v3, "input", ver)
                elif m != "clim_fraction_loo":
                    add(n, yb, "input", ver)
            for b in ("clim_fraction_loo", "v3"):
                n = f"W1.{v}.janmay_boot.vs_{b}.{d}"
                add(n, br, "validator", ver, loo)
                add(n, jj, "input", ver)
                add(n, yb, "input", ver)
                if b == "v3":
                    add(n, v3, "input", ver)
        n = f"W1.{v}.beta_a"
        add(n, jj, "fit_target", ver, "annual district totals of the variant's years")
        add(n, yb, "input", ver)
        n = f"W1.{v}.shares_final"
        add(n, br, "fit_target", ver, "crop-1 shares and winter weight chosen on all 7 years")
        add(n, jj, "input", ver)
        add(n, yb, "input", ver)
        n = f"W1.{v}.statespace_coverage"
        add(n, br, "validator", ver, "held-out year's Jan-May; the other six enter the "
                                     "likelihood")
        add(n, moa, "input", ver)
        add(n, jj, "input", ver)
        add(n, yb, "input", ver)
        n = f"W1.{v}.cropshare_vs_moa691"
        add(n, moa, "validator", ver)
        add(n, jj, "input", ver)
        add(n, yb, "input", ver)
        n = f"W1.{v}.composite_scale"
        add(n, yb, "input", ver)
        add(n, jj, "input", ver, "weir annual inflow")
        n = f"W1.{v}.annual_infan_volume"
        add(n, jj, "input", ver)
        add(n, v3, "input", ver, "in-fan farmland fraction (LUIMAP 2024)")
    pre = pv["preorigin"]
    n = "W1.preorigin.head_pcorr"
    add(n, "AMP_V2/data/wells", "validator", pre, "Changhua/Yunlin layer 1-2 wells 2012-2022")
    add(n, "AMP_V2/data/fan_stations.parquet", "input", pre, "columns sid, CountyName, "
        "GroundwaterLayerCode only")
    add(n, "data_fetch_api/new_measurements/cwa_codis_rain/rf_timeseries_codis.csv", "input",
        pre)
    add(n, v3, "input", pre)
    add(n, yb, "input", pre)
    add(n, jj, "input", pre)
    add("W1.preorigin.luimap_sensitivity", v3, "input", pre, "exemption sensitivity (U3)")
    full = pv["full"]
    add("W1.parse.printed_monthly_off", yb, "validator", full, "printed 月平均 vs daily means")
    add("W1.parse.days_vs_pilot", yb, "input", full)
    add("W1.parse.days_vs_pilot", "results/dataprep/_pilots/surface_water_monthly/"
        "choushui_daily_q.csv", "validator", full)
    add("W1.check.jiji_csv_vs_html", jj, "input", full)
    add("W1.check.jiji_csv_vs_html", rel_to_repo(C.JIJI_HTML), "validator", full)
    add("W1.check.janmay_vs_pilot", br, "input", full)
    add("W1.check.janmay_vs_pilot", "results/dataprep/_pilots/surface_water_monthly/"
        "jiji_janmay_supply_2015_2021.csv", "validator", full)
    add("W1.preregistration", f"results/dataprep/preregistration/{PREREG_ID}.json", "input",
        pre, "written 2026-10-12T02:40Z, before the benchmark ran")
    return rows


def main(stage: str, dry_root: str | None = None):
    stage = Path(stage)
    common = json.loads((stage / "common.json").read_text())
    pv = {}
    for variant in ("preorigin", "full"):
        d, _ = publish_variant(stage, variant, common, root=dry_root)
        pv[variant] = f"{C.PRODUCT_BASE}/{variant}/{d.name}"
        print("published" if dry_root is None else "dry run", d)
    rows = ledger_rows(pv, common)
    if dry_root is None:
        ledger.append_rows(C.WP, rows)
        print("ledger rows appended:", pv, len(rows))
    else:
        print("ledger rows (not written):", len(rows), ledger.conflicts(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
