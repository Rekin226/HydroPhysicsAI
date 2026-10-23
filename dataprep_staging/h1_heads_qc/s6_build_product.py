"""H1a stage 6: write the L1 product ``results/dataprep/heads/L1_clean/v<N>/`` with its
MANIFEST.json (plan 2.3), validation report, product card and ledger rows.

Usage: s6_build_product.py WORKDIR BUILD_DIR VALIDATION_JSON INJECTION_JSON LEAKPROBE_JSON
       [--dry-run] [--root DIR]   (--root: write under DIR instead of results/dataprep; tests)

Outputs copied from BUILD_DIR: heads_daily.parquet, heads_monthly.parquet, events.csv,
barometric_efficiency.csv. Written here: pressure_hourly.parquet (the ERA5 cell series, long
form), station_meta_clean.parquet (all non-personal columns, repaired),
station_meta_preorigin.parquet (static columns only, for pre-origin uses), validation.json,
PRODUCT_CARD.md. The gate passes only if every acceptance criterion passes.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import guard, manifest  # noqa: E402
from common.paths import PILOTS, REPO, rel_to_repo  # noqa: E402

from h1_heads_qc import baro, meta, rawio  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402

PRODUCT = "heads/L1_clean"
CODE_FILES = ["__init__.py", "config.py", "rawio.py", "screens.py", "baro.py", "tides.py",
              "meta.py", "pipeline.py", "backing.py", "assemble.py", "p0_profile.py",
              "s0_meta.py", "s1_screen.py", "s2_backing.py", "s3_assemble.py", "s4_validate.py",
              "s5_inject.py", "s6_build_product.py", "s7_leakprobe.py", "dev_baro_methods.py",
              "fetch_openmeteo.py", "fetch_codis_pressure.py", "prereg_h1a.py", "run_all.sh"]
STATIC_COLS = ["sid", "site_id", "WellIdentifier", "WellName", "WellNumber", "CountyCode",
               "CountyName", "TownName", "CountyName_raw", "TownName_raw", "label_flag",
               "label_method", "x_twd97", "y_twd97", "lon", "lat", "xy_flag", "xy_method",
               "xy_source", "xy_class_raw", "xy_evidence", "LocationByTWD97", "poly_county",
               "poly_town", "dist_to_named_town_m", "dist_to_valid_sibling_m", "prefix_town",
               "town_match_after_label_repair", "layer_code_raw", "layer_code", "layer_flag",
               "layer_method", "LayerAttribute", "WellDepth", "FinishDepth", "WellElevation",
               "BaseElevation", "GroundHeight", "ground_height_missing_code", "WellType",
               "WellRadius", "PipeMaterial", "GroundwaterZone", "WaterResourcesDistrictName",
               "EstablishDate", "FinishDate", "era5_cell", "era5_cell_from"]


def criteria(val: dict, inj: dict, leak: dict) -> list[dict]:
    out = []
    for kind, label, thr in (("paired", "injected paired offsets >= 1 m recovered (2012-22)",
                              C.RECALL_MIN),
                             ("step", "injected steps >= 3 m recovered (2012-22)",
                              C.RECALL_MIN)):
        s = inj["summary"].get(kind, {})
        r = s.get("recall")
        out.append({"criterion": label, "threshold": f"recall >= {thr}",
                    "measured": f"{r:.3f} ({s.get('n_recovered')}/{s.get('n_injected')})"
                    if r is not None else "n/a", "passed": bool(r is not None and r >= thr),
                    "data": "pre-origin"})
    a = val["audit_agreement"]["hist"]
    out.append({"criterion": "agreement with the audit's 2012-22 raw-record events (same well,"
                             " +-1 day)", "threshold": f">= {C.AUDIT_AGREE_MIN}",
                "measured": f"{a['agreement']:.4f} ({a['matched']}/{a['n_audit_events']})",
                "passed": bool(a["agreement"] >= C.AUDIT_AGREE_MIN), "data": "pre-origin"})
    re = val["real_events"]
    n_untestable = sum(1 for e in re["events"] if e["n_days_in_record"] == 0)
    out.append({"criterion": "no flag on the known real events (09200321, 09190132 recovery "
                             "2024-05-28..06-04; 09070311 typhoon recharge end of July 2024)",
                "threshold": "0 of 3 flagged (each event must have days in the record)",
                "measured": f"{re['n_flagged']} of {re['n_events']} flagged; "
                            f"{n_untestable} without record days",
                "passed": re["n_flagged"] == 0 and n_untestable == 0,
                "data": "post-origin (plan-mandated; changes no parameter)"})
    o = val["o1_coherence"]["pre"]["corrected_wells"]
    md = o["median_paired_diff"]
    out.append({"criterion": "barometric step: median O1 phase coherence improves (wells with a"
                             " correction, 2012-22, paired)", "threshold": "median paired "
                "difference > 0",
                "measured": (f"{md:+.4f} (median {o['median_raw']:.3f} -> {o['median_corr']:.3f};"
                             f" {o['n_improved']} better / {o['n_worse']} worse of "
                             f"{o['n_wells']}; sign p {o['sign_test_p']:.3g})")
                if md is not None else "n/a",
                "passed": bool(md is not None and md > 0), "data": "pre-origin"})
    pdp = val["pressure_drops"]["pre"]
    out.append({"criterion": "no flagged offset episode coincides with a pressure drop >= 20 "
                             "hPa (2012-22)", "threshold": "0 episodes",
                "measured": f"{pdp['n_offset_episodes_on_fall_ge_20hpa']} of "
                            f"{pdp['n_offset_episodes']} (largest fall at an episode "
                            f"{pdp['max_fall_at_offset_episodes_hpa']:.1f} hPa)",
                "passed": pdp["n_offset_episodes_on_fall_ge_20hpa"] == 0, "data": "pre-origin"})
    out.append({"criterion": "leakage probe: pre-origin rows rebuilt without any post-origin "
                             "file are identical to the product's pre-origin rows",
                "threshold": "0 differing rows",
                "measured": f"{leak['n_diff_total']} differing rows "
                            f"({leak['n_rows_compared']} compared)",
                "passed": leak["n_diff_total"] == 0, "data": "pre-origin"})
    return out


def main():
    work, build = Path(sys.argv[1]), Path(sys.argv[2])
    val = json.loads(Path(sys.argv[3]).read_text())
    inj = json.loads(Path(sys.argv[4]).read_text())
    leak = json.loads(Path(sys.argv[5]).read_text())
    dry = "--dry-run" in sys.argv
    crit = criteria(val, inj, leak)
    passed = all(c["passed"] for c in crit)
    for c in crit:
        print(("PASS " if c["passed"] else "FAIL ") + c["criterion"] + ": " + c["measured"])
    if dry:
        return
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else None
    d = manifest.new_version_dir(PRODUCT, root=root) if root else \
        manifest.new_version_dir(PRODUCT)
    for f in ("heads_daily.parquet", "heads_monthly.parquet", "events.csv",
              "barometric_efficiency.csv"):
        shutil.copy2(build / f, d / f)
    # pressure (long form)
    P = baro.load_pressure()
    P = P[(P.index >= "2011-12-25") & (P.index < "2026-10-01")]
    PL = P.rename_axis("date").reset_index().melt(id_vars="date", var_name="era5_cell",
                                                  value_name="surface_pressure_hpa")
    PL = PL.dropna(subset=["surface_pressure_hpa"]).sort_values(["era5_cell", "date"])
    PL.to_parquet(d / "pressure_hourly.parquet", index=False)
    # station metadata
    sm = pd.read_parquet(work / "station_meta_clean.parquet")
    sm.to_parquet(d / "station_meta_clean.parquet", index=False)
    sm[[c for c in STATIC_COLS if c in sm.columns]].to_parquet(
        d / "station_meta_preorigin.parquet", index=False)
    # validation report
    report = {"wp": "H1a", "criteria": crit, "gate_passed": passed,
              "validation": val, "injection": {k: v for k, v in inj.items()
                                               if k not in ("rows", "spurious")},
              "leakage_probe": leak,
              "evidence_files": ["results/dataprep/reports/H1a_evidence/validation_full.json",
                                 "results/dataprep/reports/H1a_evidence/injection_O.json",
                                 "results/dataprep/reports/H1a_evidence/leakprobe.json"]}
    Path(d / "validation.json").write_text(json.dumps(report, indent=1, default=str))
    (d / "PRODUCT_CARD.md").write_text(card(val, inj, crit, passed))

    here = Path(__file__).resolve().parent
    code = [here / f for f in CODE_FILES if (here / f).exists()]
    code += [REPO / "dataprep_staging" / "harness" / "gaps.py",
             REPO / "dataprep_staging" / "common" / "paths.py"]
    inputs = [rawio.STATIONS, meta.WRA_LIST, meta.TOWNS,
              PILOTS / "audit_2026-10-11" / "artefact_events.csv",
              REPO / "results" / "dataprep" / "reports" / "A0_evidence" / "repair" / "reruns"
              / "rho_k_result.json",
              REPO / "results" / "twin" / "hybrid_datum_gate_innov" / "evaluation.csv"]
    inputs += baro.pressure_files()
    codis = sorted((REPO / "data_fetch_api" / "new_measurements" / "pressure_hourly")
                   .glob("cwa_codis_*"))[-1]
    inputs += [codis / "SOURCE.json"]
    inputs = [rel_to_repo(p) for p in inputs if Path(p).exists()]
    outputs = [{"path": "heads_daily.parquet"}, {"path": "heads_monthly.parquet"},
               {"path": "events.csv"}, {"path": "pressure_hourly.parquet"},
               {"path": "barometric_efficiency.csv"},
               {"path": "station_meta_clean.parquet"},
               {"path": "station_meta_preorigin.parquet", "fit_window_end": "2022-12"},
               {"path": "validation.json"}, {"path": "PRODUCT_CARD.md"}]
    m = manifest.build_manifest(
        d, product=PRODUCT, layer="L1", inputs=inputs, code=[str(c) for c in code],
        input_trees=[rel_to_repo(rawio.HIST_DIR), rel_to_repo(rawio.NEW_DIR), rel_to_repo(codis)],
        parameters={
            "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
            "merge": "historical cache 2012-01-01..2022-12-31 + new network 2023-01-01.."
                     "2026-09-30 (2022-10..12 overlap identical, p0_profile.json)",
            "pre_origin_rule": "rows dated <= 2022-12 are built from pre-origin data only "
                               "(separate pre-origin pass, pre-origin backing evidence, "
                               "pre-origin daily screen); verified by the leakage probe",
            "barometric_design": C.BARO_DESIGN, "seed": C.BARO_SEED,
            "pressure_source": "Open-Meteo ERA5 surface_pressure, nearest 0.25-degree cell",
            "flags": {"0": "observed, passed QC", "1": "observed, corrected (barometric loading "
                      "removed, or an offset/step corrected with metadata or co-located "
                      "backing)", "3": "corrected with neighbour-regression corroboration only",
                      "8": "missing / sentinel only", "9": "suspect (kept, excluded "
                      "downstream)"},
            "prereg": "results/dataprep/preregistration/H1a.qc_acceptance.v1.json"},
        fit_window_end="2026-09",
        time_slicing={"column": "date", "params_fit_window_end": "2022-12", "encoding": "iso"},
        validation_report="validation.json",
        gate={"passed": passed, "criteria": "plan WP H1 (H1a) acceptance, pre-registered in "
                                            "H1a.qc_acceptance.v1; see validation.json"},
        allowed_uses=["scoring_target", "calibration_target", "datum_fit", "baseline",
                      "initial_conditions", "idw_field", "model_design", "selection",
                      "validator", "statistical_layer_history", "after_the_fact_analysis",
                      "design_time_evidence"],
        forbidden_uses=["prospective_scorer", "prior", "prior_width_input", "forcing_frozen",
                        "forcing_recorded_arm"],
        personal_data="none (station metadata: non-personal columns only; LocationAddress and "
                      "contact fields never read)",
        outputs=outputs)
    errs = guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit("time slicing errors: " + "; ".join(errs))
    manifest.write_manifest(d, m)
    print("wrote", d, "gate passed" if passed else "GATE NOT PASSED")


def card(val, inj, crit, passed) -> str:
    s = val["summary"]
    cov = s["monthly_coverage"]
    lines = ["# heads/L1_clean (WP H1a)", "",
             "Cleaned groundwater heads of the Choushui fan monitoring network, 2012-01 to "
             "2026-09: the historical cache (2012-22) merged with the new network (2023-26), "
             "screened, barometric loading removed where significant, offset episodes and "
             "steps detected and flagged (or corrected with metadata/co-located backing), "
             "daily and C20 monthly aggregates with sigma_obs.", "",
             f"Gate: {'passed' if passed else 'NOT passed'}.", "", "## Coverage", ""]
    for per, c in cov.items():
        lines.append(f"- {per}: {c['wells_with_record']} wells with a record; "
                     f"{c['observed_flag01']} observed well-months (flag 0/1), "
                     f"{c['observed_flag01_c20']} of them C20; flag 3: {c['flag3']}, "
                     f"flag 9: {c['flag9']}, flag 8: {c['flag8']}.")
    lines += ["", "## Flags in use", "",
              "0 observed; 1 observed and corrected (barometric loading removed, or an offset "
              "or step corrected with physical metadata or a co-located nested well); 3 "
              "corrected with neighbour-regression corroboration only (not a calibration "
              "target, U2); 8 missing or sentinel only; 9 suspect (kept, excluded "
              "downstream).", "", "## Use", "",
              "Pre-origin uses (calibration targets, datum fits, baselines, initial "
              "conditions) must open the product with `window_end <= 2022-12` (time slice on "
              "`date`); `station_meta_preorigin.parquet` is the pre-origin station table. "
              "The registered prospective scorer keeps its registered QC (forbidden use).", "",
              "## Acceptance", ""]
    for c in crit:
        lines.append(f"- {'PASS' if c['passed'] else 'FAIL'}: {c['criterion']} "
                     f"({c['threshold']}): {c['measured']}")
    lines += ["", "## Known limits", "",
              "- Jumps across logger gaps longer than 70 minutes are not detected (the audit's "
              "rule); real recoveries across gaps stay unflagged by design.",
              "- Whether a flagged offset episode is an artefact is unverified: the default is "
              "to flag (9), not correct.",
              "- Barometric efficiency is estimated only where significant; elsewhere no "
              "correction is applied (flag 0).",
              "- GroundHeight 0.0 is a missing-value code (H1b replaces it after W3).",
              "- Layer codes proposed from WellDepth are flag 5, unverified.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
