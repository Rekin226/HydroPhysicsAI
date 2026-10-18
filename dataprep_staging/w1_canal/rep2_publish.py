"""Publish the staged repair round 2 as surface_water_v4/preorigin/v<N+1> and full/v<N+1>.

    cpu_slot.sh python -I dataprep_staging/w1_canal/rep2_publish.py STAGE_DIR

Creates new version directories (never an existing one), sets product_version in every long
table, writes validation.json, QC_REPORT.md and MANIFEST.json, and APPENDS the W1 ledger rows
of this round (the fragment is not rewritten; verifier minor issue). Provenance: each new
version lists the previous version's MANIFEST.json and every file read from it as inputs, so
its own inputs and code stay pinned through that manifest; the new code is listed directly.
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

from w1_canal import config as C  # noqa: E402
from w1_canal import rep2_config as R2  # noqa: E402
from w1_canal.rep2_build import (  # noqa: E402
    DECL_ID,
    FULL_COPY,
    FULL_V3,
    PRE_COPY,
    PRE_V1,
)

LONG = ["district_monthly.csv", "percolation_basis.csv", "basin_rain_index_monthly.csv",
        "inflow_predicted_monthly.csv", "gauges_daily_L1.parquet",
        "inflow_composite_daily.parquet"]
NPZ = ["district_draws.npz", "grid_sw_v4.npz"]
GRID = "grid_sw_v4.npz"
REP2_CODE = ["rep2_config.py", "rep2_build.py", "rep2_publish.py"]
SHARED_CODE = ["__init__.py", "config.py", "rep_config.py", "rep_uncert.py", "build.py"]
PRE_ONLY_CODE = ["parse_yearbook.py", "grid.py"]
COMMON_CODE = ["dataprep_staging/common/hashing.py", "dataprep_staging/common/flags.py",
               "dataprep_staging/common/paths.py", "dataprep_staging/harness/coverage2.py",
               "dataprep_staging/harness/coverage.py"]


def code_list(pre: bool) -> list[str]:
    files = REP2_CODE + SHARED_CODE + (PRE_ONLY_CODE if pre else [])
    return [rel_to_repo(HERE / f) for f in files] + COMMON_CODE


def write_tables(stage: Path, d: Path, pv: str, files: list[str]):
    for f in files:
        src = stage / f
        if f in LONG:
            t = pd.read_csv(src) if f.endswith(".csv") else pd.read_parquet(src)
            t["product_version"] = pv
            errs = validate_long_table(t, require_all_columns=True)
            if errs:
                raise ValueError(f"{f}: {errs}")
            (t.to_csv(d / f, index=False) if f.endswith(".csv") else
             t.to_parquet(d / f, index=False))
        else:
            shutil.copy2(src, d / f)


def qc_section(variant: str, st: dict) -> str:
    r = st["rep2"]
    lines = ["", f"## {R2.ROUND_TAG}: interval labels on every output (plan 4.7)", "",
             "- values, flags and members unchanged; every sd, percentile and member array "
             "carries a label (row column interval_calibration, per-month arrays in the npz "
             "files, manifest interval_labels for the coverage-check tables)",
             f"- 2012-2022/2025 district rows: {r['label_hist']}",
             f"- inflow composite sd: {r['label_composite']}",
             f"- structural-zero rows: {r['label_structural']}",
             "- member_mean and value_minus_member_mean (district_monthly.csv): the Denton "
             "value and the state-space members have different centres; max |value - member "
             "mean| " + ", ".join(f"{k} {v.get('max_abs', v.get('2012_2025_max_abs')):.3f}"
                                   for k, v in r["value_minus_member_mean"].items())]
    if variant == "full":
        lines += [f"- 2026 rows: {r['label_2026']}",
                  f"- basin rain index sd: {r['label_rain']}",
                  "- post hoc residual diagnostic (Yunlin 2020-2025 all negative): see "
                  "validation.json repair_r2.residual_diagnostics_post_hoc"]
    else:
        lines += ["- structural-zero months: sd 0 (the members are 0); the winter-0.5 "
                  "alternative moved from sd to structural_alt_value / value_structural_alt",
                  "- prior_width_input is forbidden: no sd of this product is "
                  "coverage-checked at a scale a prior width would use",
                  f"- parse QC recomputed from the 2012-2022 yearbooks only: "
                  f"{st['parse_qc_2012_2022']}",
                  "- gate: the pre-registered Jan-May acceptance passed on 7 design-time years "
                  "per district (weak evidence; see MANIFEST gate.caveat)"]
    return "\n".join(lines) + "\n"


def publish_variant(stage: Path, variant: str, state: dict) -> tuple[Path, dict]:
    pre = variant == "preorigin"
    prev = PRE_V1 if pre else FULL_V3
    m_prev = json.loads((prev / "MANIFEST.json").read_text())
    st = state[variant]
    d = manifest.new_version_dir(f"{C.PRODUCT_BASE}/{variant}")
    version = d.name
    pv = f"{C.PRODUCT_BASE}/{variant}/{version}"
    sdir = stage / variant
    data_files = (["district_monthly.csv", "percolation_basis.csv", "gauges_daily_L1.parquet",
                   "inflow_composite_daily.parquet"] + NPZ + PRE_COPY) if pre else (
        ["district_monthly.csv", "percolation_basis.csv", "basin_rain_index_monthly.csv",
         "inflow_predicted_monthly.csv", "gauges_daily_L1.parquet",
         "inflow_composite_daily.parquet"] + NPZ + FULL_COPY)
    write_tables(sdir, d, pv, data_files)
    val = json.loads((sdir / "validation_stage.json").read_text())
    val["version"] = version
    with open(d / "validation.json", "w") as fh:
        json.dump(val, fh, ensure_ascii=False, indent=1)
    qc_prev = (prev / "QC_REPORT.md").read_text()
    (d / "QC_REPORT.md").write_text(
        qc_prev.replace(f"{variant}/{m_prev['version']} - QC report",
                        f"{variant}/{version} - QC report (values as {variant}/"
                        f"{m_prev['version']})", 1) + qc_section(variant, st))
    # ---------------------------------------------------------------- manifest
    read_prev = ["MANIFEST.json", "QC_REPORT.md", "validation.json"] + data_files
    inputs = [rel_to_repo(prev / f) for f in read_prev]
    if pre:
        inputs += st["pdfs"] + [rel_to_repo(C.V3_NPZ)]
    else:
        inputs += [rel_to_repo(DATAPREP / "preregistration" / f"{DECL_ID}.json")]
    inputs = list(dict.fromkeys(inputs))
    params = dict(m_prev["parameters"])
    params[R2.ROUND] = {"derived_from": f"{C.PRODUCT_BASE}/{variant}/{m_prev['version']}",
                        "settings": "dataprep_staging/w1_canal/rep2_config.py",
                        "fitted": "nothing (labels, member statistics, structural-zero sd, "
                                  "pre-origin parse QC)"}
    params["staged_by"] = "dataprep_staging/w1_canal/rep2_build.py"
    gate = dict(m_prev["gate"])
    labels = st["labels"]
    if pre:
        gate["label"] = ("pre-registered Jan-May acceptance passed on 7 design-time years per "
                         "district: weak evidence")
        gate["caveat"] = ("the scout chose the method on the same 7 years (beta fitted on "
                          "2012-2025 at design time); percentile paired bootstrap at n = 7 "
                          "(anti-conservative); Yunlin-vs-climatology upper bound -0.002 "
                          "(-0.005 with another seed); sign tests 5/7, 5/7, 4/7, 6/7 (none "
                          "significant); the supporting head check failed in Yunlin layer 1 "
                          "(-0.0087)")
        gate["evidence_class"] = "design_time"
        gate["interval_calibration"] = {"district_rows": st["rep2"]["label_hist"],
                                        "structural_rows": st["rep2"]["label_structural"],
                                        "composite": st["rep2"]["label_composite"]}
        outputs = [{"path": f, "fit_window_end": "2022-12"} for f in data_files if f != GRID]
        outputs.append({"path": GRID, "fit_window_end": m_prev["fit_window_end"],
                        "note": "allocation from LUIMAP 2024 (exemption)"})
        allowed = [u for u in m_prev["allowed_uses"] if u not in R2.PRE_REMOVED_USES]
        forbidden = list(dict.fromkeys(m_prev["forbidden_uses"] + list(R2.PRE_REMOVED_USES)))
        kw = {"exemptions": m_prev["exemptions"],
              "non_exempt_fit_window_end": m_prev["non_exempt_fit_window_end"]}
    else:
        ic = dict(gate.get("interval_calibration_2026") or {})
        ic["label"] = st["rep2"]["label_2026"]
        ic["note"] = ("repair round 2 wording: every published 2026 interval (monthly, "
                      "Jan-Sep) is uncalibrated; the annual check covers 12-month sums the "
                      "product does not publish")
        gate["interval_calibration_2026"] = ic
        gate["interval_calibration"] = {"district_rows_2012_2025": st["rep2"]["label_hist"],
                                        "structural_rows": st["rep2"]["label_structural"],
                                        "composite": st["rep2"]["label_composite"],
                                        "rain_index": st["rep2"]["label_rain"]}
        outputs = list(data_files)
        allowed, forbidden = m_prev["allowed_uses"], m_prev["forbidden_uses"]
        kw = {}
    outputs += ["validation.json", "QC_REPORT.md"]
    m = manifest.build_manifest(
        d, product=f"{C.PRODUCT_BASE}/{variant}", layer="L2", inputs=inputs,
        code=code_list(pre), parameters=params, fit_window_end=m_prev["fit_window_end"],
        outputs=outputs, validation_report="validation.json", gate=gate,
        allowed_uses=allowed, forbidden_uses=forbidden, personal_data="none",
        extra={"layer_note": m_prev["layer_note"] + "; repair round 2: interval labels on "
                                                    "every output (plan 4.7)",
               "supersedes": f"{C.PRODUCT_BASE}/{variant}/{m_prev['version']} (identical "
                             "values, flags and members; interval labels, member means"
                             + ("; structural-zero sd 0; prior_width_input forbidden; "
                                "pre-origin-only parse QC" if pre else "") + ")",
               "interval_labels": labels["outputs"],
               "provenance_note": f"inputs and code of {variant}/{m_prev['version']} are "
                                  "pinned by its MANIFEST.json, listed here as an input"},
        **kw)
    errs = manifest.validate_manifest(m, d)
    if errs:
        raise manifest.ManifestError(errs)
    manifest.write_manifest(d, m)
    return d, m


# ----------------------------------------------------------------------------- ledger
def ledger_rows(pre_pv: str, full_pv: str) -> list[dict]:
    rows = []

    def add(n, src, role, pv, note=""):
        rows.append({"number_id": n, "source": src, "role": role, "product_version": pv,
                     "note": note})

    jj = rel_to_repo(C.JIJI_CSV)
    brief = rel_to_repo(C.BRIEFING_110_PDF)
    moa = rel_to_repo(C.MOA_691_CSV)
    hyb = rel_to_repo(C.YEARBOOK_DIR)
    ssnote = ("state-space likelihood: annual totals imposed exactly, Jan-May totals and crop-1 "
              "shares as observed aggregates (hyperparameter weights by marginal likelihood); "
              "corrects the 'input' role of W1.preorigin/full.statespace_hyper (v1 rows kept: "
              "append-only fragment)")
    for tag, pv in (("preorigin_v2", pre_pv), ("full_v4", full_pv)):
        n = f"W1.{tag}.statespace_hyper"
        add(n, jj, "fit_target", pv, ssnote)
        add(n, brief, "fit_target", pv, ssnote + "; the slide-19 table parsed by pdftotext "
            "(not a digitised chart)")
        add(n, moa, "fit_target", pv, ssnote)
        add(n, hyb, "input", pv, "indicator (supply-limited diversion of the composite)")
        n = f"W1.{tag}.interval_labels"
        prev = PRE_V1 if tag.startswith("pre") else FULL_V3
        add(n, rel_to_repo(prev / "statespace_loo_coverage.csv"), "input", pv,
            "Jan-May leave-one-year-out hits behind the 2012-2022/2025 row label (A1 gate-2)")
        add(n, rel_to_repo(prev / "inflow_composite_daily.parquet"), "input", pv,
            "composite sd (heuristic; labelled uncalibrated)")
        n = f"W1.{tag}.member_mean_offset"
        add(n, rel_to_repo(prev / "district_draws.npz"), "input", pv, "member means")
        add(n, rel_to_repo(prev / "district_monthly.csv"), "input", pv, "Denton values")
    add("W1.full_v4.interval_labels", rel_to_repo(FULL_V3 / "rep_coverage_annual.csv"), "input",
        full_pv, "annual 19/22 behind the 2026 label (declared rule)")
    add("W1.full_v4.interval_labels", rel_to_repo(FULL_V3 / "basin_rain_index_monthly.csv"),
        "input", full_pv, "rain index sd (labelled uncalibrated)")
    n = "W1.full_v4.residual_diagnostics_post_hoc"
    add(n, rel_to_repo(FULL_V3 / "ext2026_loo_annual.csv"), "design_time_evidence", full_pv,
        "observed annual totals and single-fold chain predictions of the 11 evaluation years "
        "(design-time units; post hoc; nothing chosen from it)")
    n = "W1.preorigin_v2.parse_qc_2012_2022"
    add(n, hyb, "input", pre_pv, "daily tables vs the printed monthly and annual means of the "
        "same 2012-2022 yearbooks (internal consistency)")
    add(n, rel_to_repo(PRE_V1 / "gauges_daily_L1.parquet"), "validator", pre_pv,
        "identity of the re-parse with preorigin/v1")
    n = "W1.rep2.forward_archive_probe"
    add(n, "civil_iot_sta_colife_STA_WaterResource_v2", "input", full_pv,
        "reachable mirror; no discharge or stage datastream for H075/H049/H064 or the weir")
    add(n, "wra_fhy_WraApi_v1", "input", full_pv, "503")
    add(n, "wra_opendata_api_v2", "input", full_pv, "bot-challenge HTML page")
    return rows


def main(stage: str):
    stage = Path(stage)
    state = json.loads((stage / "rep2_state.json").read_text())
    d_pre, _ = publish_variant(stage, "preorigin", state)
    d_full, _ = publish_variant(stage, "full", state)
    pre_pv = f"{C.PRODUCT_BASE}/preorigin/{d_pre.name}"
    full_pv = f"{C.PRODUCT_BASE}/full/{d_full.name}"
    rows = ledger_rows(pre_pv, full_pv)
    ledger.append_rows(C.WP, rows)
    print("published", d_pre, d_full, "ledger rows appended", len(rows))


if __name__ == "__main__":
    main(sys.argv[1])
