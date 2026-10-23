"""Publish the staged 2026 extension as surface_water_v4/full/v<N+1> (W1 step 6).

    cpu_slot.sh python -I dataprep_staging/w1_canal/ext_publish.py STAGE_DIR

Copies the ext_build.py staging tree into a new version directory (never an existing one),
sets the product version in every long table, writes validation.json, QC_REPORT.md and the
manifest (full/v1's inputs and code plus the CODIS L0 records, the full/v1 outputs read, and
the extension code), and appends the W1 ledger rows of the extension.
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
from common.paths import REPO, rel_to_repo  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal import ext_config as E  # noqa: E402
from w1_canal.ext_prereg import PREREG_ID  # noqa: E402

FULL_V1 = C.product_dir_base("full") / "v1"
LONG_TABLES = ["district_monthly.csv", "percolation_basis.csv", "basin_rain_index_monthly.csv",
               "inflow_predicted_monthly.csv", "gauges_daily_L1.parquet",
               "inflow_composite_daily.parquet"]
OTHER = ["janmay_validation.csv", "statespace_loo_coverage.csv", "printed_monthly.csv",
         "printed_annual.csv", "district_draws.npz", "grid_sw_v4.npz",
         "station_rain_monthly.csv", "ext2026_loo_annual.csv", "ext2026_loo_monthly.csv"]
EXT_CODE = ["ext_config.py", "ext_fetch.py", "ext_model.py", "ext_prereg.py", "ext_build.py",
            "ext_publish.py"]
EXT_COMMON = ["hydrophysics/twin/cwa_rain.py", "hydrophysics/twin/field_observations.py",
              "dataprep_staging/harness/prereg.py", "dataprep_staging/common/source.py"]


def qc_text(version: str, ext: dict) -> str:
    v1 = (FULL_V1 / "QC_REPORT.md").read_text()
    v1 = v1.replace("# surface_water_v4/full/v1 - QC report",
                    f"# surface_water_v4/full/{version} - QC report (2012-2025 as full/v1)")
    ct = ext["chain_test"]
    lines = ["", "## 2026 extension (plan W1 step 6; flag 6; U4 recorded arm only)", "",
             f"- pre-registration {ext['preregistration']['id']} "
             f"({ext['preregistration']['created']})",
             f"- months {ext['extension_months'][0]}..{ext['extension_months'][1]}; rain index "
             f"from {len(ext['rain_index']['members']['mainstem']) + len(ext['rain_index']['members']['chenyoulan']) + len(ext['rain_index']['members']['shuili'])} "  # noqa: E501
             f"CODIS records; months without an index: "
             f"{ext['months_without_index_2011_11_on']}",
             f"- evaluation years of the leave-one-year-out chain test: "
             f"{ext['evaluation_years']}",
             f"- regression spec for 2026: {ext['fit_2026']['spec']} (inner LOO RMSE "
             f"{ext['fit_2026']['inner_loo_rmse']}); R from {ext['fit_2026']['R_from_year']}",
             "", "| method | RMSE Changhua | bias Changhua | RMSE Yunlin | bias Yunlin |",
             "|---|---|---|---|---|"]
    for m in ("rain_chain", "actual_inflow", "climatology", "persistence"):
        a, b = ct["scores"]["chc"][m], ct["scores"]["yl"][m]
        lines.append(f"| {m} | {a['rmse']:.3f} | {a['bias']:+.3f} | {b['rmse']:.3f} | "
                     f"{b['bias']:+.3f} |")
    lines += ["", "| comparison | district | RMSE diff | 90 % paired bootstrap | sign test |",
              "|---|---|---|---|---|"]
    for d in ("chc", "yl"):
        for k, r in ct["compare"][d].items():
            lines.append(f"| {k} | {d} | {r['rmse_diff']:+.3f} | [{r['boot_lo']:+.3f}, "
                         f"{r['boot_hi']:+.3f}] | {r['sign_wins']}/{r['sign_n']}, p = "
                         f"{r['sign_p_two_sided']:.3f} |")
    rule = ext["shipping_rule"]
    lines += ["", f"- shipping rule: ships {rule['ships']}; label: {rule['label']}",
              f"- inflow monthly LOO RMSE {ct['inflow_monthly_rmse_loo']:.3f} vs in-sample "
              f"monthly climatology {ct['inflow_monthly_rmse_climatology_insample']:.3f} "
              f"(10^8 m3)"]
    for d in ("chc", "yl"):
        s = ext["series_2026"][d]
        lines.append(f"- {d} 2026 monthly: {[round(x, 3) for x in s['value']]}; sd "
                     f"{[round(x, 3) for x in s['sd']]}; Jan-Sep {s['jan_sep_total']:.3f}")
    return v1 + "\n".join(lines) + "\n"


def main(stage: str):
    stage = Path(stage)
    val = json.loads((stage / "validation_ext.json").read_text())
    ext = val["ext2026"]
    m1 = json.loads((FULL_V1 / "MANIFEST.json").read_text())
    d = manifest.new_version_dir(f"{C.PRODUCT_BASE}/full")
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
    full_val = {"product": f"{C.PRODUCT_BASE}/full", "version": version, **val}
    with open(d / "validation.json", "w") as fh:
        json.dump(full_val, fh, ensure_ascii=False, indent=1)
    (d / "QC_REPORT.md").write_text(qc_text(version, ext))
    rule = ext["shipping_rule"]
    gate = dict(m1["gate"])
    gate["extension_2026"] = {
        "ships": rule["ships"], "label": rule["label"],
        "beats_point": rule["beats_point"], "interval_below_zero": rule["interval_below_zero"],
        "rule": "pre-registered (W1.ext2026_rain_runoff.v1): rain-driven chain ships if its "
                "leave-one-year-out annual RMSE is below annual climatology in both "
                "districts; label from the 90 % paired bootstrap"}
    inputs = [r["path"] for r in m1["inputs"]] + [rel_to_repo(p) for p in (
        FULL_V1 / "MANIFEST.json", FULL_V1 / "district_monthly.csv",
        FULL_V1 / "district_draws.npz", FULL_V1 / "grid_sw_v4.npz",
        FULL_V1 / "inflow_composite_daily.parquet", FULL_V1 / "validation.json",
        FULL_V1 / "QC_REPORT.md", E.L0_DIR / "SOURCE.json",
        C.L0_NEW / "cwa_station_inventory" / "codis_station_list_2026-10-11.json",
        REPO / "results" / "dataprep" / "preregistration" / f"{PREREG_ID}.json")]
    code = ([r["path"] for r in m1["code"]] + [rel_to_repo(HERE / f) for f in EXT_CODE]
            + EXT_COMMON)
    params = dict(m1["parameters"])
    params["fit_window"] = ["2012-01", "2026-09"]
    params["extension_2026"] = {
        "plan_step": "W1 step 6", "preregistration": f"results/dataprep/preregistration/"
                                                     f"{PREREG_ID}.json",
        "stations": {k: {"subbasin": sb, "name": nm, "chain": [list(c) for c in ch]}
                     for k, (sb, nm, ch) in E.STATIONS.items()},
        "subbasins": {k: list(v) for k, v in E.SUBBASINS.items()},
        "min_day_fraction": E.MIN_DAY_FRACTION, "normal_window": list(E.NORMAL_WINDOW),
        "min_stations_month": E.MIN_STATIONS_MONTH, "specs": list(E.SPECS), "lags": E.LAGS,
        "fit_years": list(E.FIT_YEARS), "evaluation_years": ext["evaluation_years"],
        "R_from_year": E.R_EXT_FROM_YEAR,
        "bootstrap": {"B": E.BOOT_B, "seed": E.BOOT_SEED, "level": E.BOOT_LEVEL},
        "draws": {"n": E.N_DRAWS, "seed": E.DRAW_SEED},
        "months": ext["extension_months"], "spec_2026": ext["fit_2026"]["spec"]}
    params["staged_by"] = "dataprep_staging/w1_canal/ext_build.py"
    outputs = LONG_TABLES + OTHER + ["validation.json", "QC_REPORT.md"]
    m = manifest.build_manifest(
        d, product=f"{C.PRODUCT_BASE}/full", layer="L2", inputs=inputs,
        input_trees=[rel_to_repo(E.RAW_DIR)], code=code, parameters=params,
        fit_window_end=ext["extension_months"][1], outputs=outputs,
        validation_report="validation.json", gate=gate,
        allowed_uses=m1["allowed_uses"], forbidden_uses=m1["forbidden_uses"],
        personal_data="none",
        extra={"layer_note": "full/v1 (2012-2025, flag 5) plus 2026 months (flag 6, rain-"
                             "driven extension, plan W1 step 6); district series and draws "
                             "are L2, grid_sw_v4.npz the L3 grid array; U4: post-origin "
                             "forcing, recorded arm only",
               "supersedes": "surface_water_v4/full/v1 (identical 2012-2025 values)"})
    manifest.write_manifest(d, m)
    rows = ledger_rows(pv, ext)
    ledger.append_rows(C.WP, rows)
    print("published", d, "ledger rows", len(rows), "conflicts", ledger.conflicts(rows))


def ledger_rows(pv: str, ext: dict) -> list[dict]:
    rain = f"data_fetch_api/new_measurements/{E.L0_SOURCE_ID}"
    jj = rel_to_repo(C.JIJI_CSV)
    comp = rel_to_repo(FULL_V1 / "inflow_composite_daily.parquet")
    den = rel_to_repo(FULL_V1 / "district_monthly.csv")
    rows = []

    def add(n, src, role, note=""):
        rows.append({"number_id": n, "source": src, "role": role, "product_version": pv,
                     "note": note})

    for d in ("chc", "yl"):
        for m in ("rain_chain", "actual_inflow", "climatology", "persistence"):
            n = f"W1.full_v2.ext2026.loo_annual_rmse.{m}.{d}"
            add(n, jj, "validator", "held-out year's annual district total; the training "
                                    "years' totals refit beta and the district scales")
            if m in ("rain_chain", "actual_inflow"):
                add(n, comp, "input", "regression target (rain chain) or inflow (reference)")
            if m == "rain_chain":
                add(n, rain, "input")
        n = f"W1.full_v2.ext2026.loo_boot_vs_climatology.{d}"
        add(n, jj, "validator")
        add(n, comp, "input")
        add(n, rain, "input")
        n = f"W1.full_v2.ext2026.series_2026.{d}"
        add(n, rain, "input", "basin rain index 2025-11..2026-09")
        add(n, comp, "fit_target", "rainfall-runoff regression 2012-2025")
        add(n, jj, "input", "beta and district scales (annual totals 2012-2025), R of 2025")
        n = f"W1.full_v2.ext2026.sd_2026.{d}"
        add(n, den, "validator", "full/v1 Denton series of the scored folds")
        add(n, rain, "input")
        add(n, jj, "input")
    n = "W1.full_v2.ext2026.inflow_loo_rmse"
    add(n, comp, "validator")
    add(n, rain, "input")
    n = "W1.full_v2.ext2026.rain_index_availability"
    add(n, rain, "input", "CODIS records valid per month (>= 80 % numeric days)")
    return rows


if __name__ == "__main__":
    main(sys.argv[1])
