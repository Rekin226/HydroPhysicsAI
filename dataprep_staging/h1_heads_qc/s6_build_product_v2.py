"""H1a v2 stage 6 (round 2): write ``results/dataprep/heads/L1_clean/v<N>/`` (a new version;
v1 stays published) with its MANIFEST.json (plan 2.3), validation report and product card.

Usage: s6_build_product_v2.py WORKDIR BUILD_DIR VALIDATION_JSON INJECTION_JSON LEAKPROBE_JSON
       [--dry-run]

Gate (H1a.qc_acceptance.v2, conjunctive): the registered tests on fresh units (T1 paired and
T2 step recall on family O r = 10..19, T7 leakage probe of the v2 pre-origin rows) and the
conformance checks with the plan's thresholds (C3 real events, C4 audit agreement, C5 O1
coherence, C6 pressure drops).

Outputs copied from BUILD_DIR: heads_daily.parquet, heads_monthly.parquet, events.csv,
barometric_efficiency.csv. Written here: pressure_hourly.parquet, station_meta_clean.parquet,
station_meta_preorigin.parquet, well_qc_summary.csv (per well and period: jump catalogue,
episodes, steps, flag-9 share; date = last day of the period's data), validation.json,
PRODUCT_CARD.md.
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
from h1_heads_qc.s6_build_product import STATIC_COLS  # noqa: E402

PRODUCT = "heads/L1_clean"
PREREG = "results/dataprep/preregistration/H1a.qc_acceptance.v2.json"
EVID = "results/dataprep/reports/H1a_evidence/v2"
CODE_FILES = ["__init__.py", "config.py", "rawio.py", "screens.py", "baro.py", "tides.py",
              "meta.py", "pipeline.py", "backing.py", "assemble.py", "p0_profile.py",
              "s0_meta.py", "s1_screen.py", "s4_validate.py", "s5_inject.py",
              "s6_build_product.py", "s7_leakprobe.py", "dev_baro_methods.py",
              "dev_baro_epochs.py", "fetch_openmeteo.py", "fetch_codis_pressure.py",
              "levelcheck.py", "decide_v2.py", "assemble_v2.py", "s2b_decide_v2.py",
              "s3_assemble_v2.py", "s4_validate_v2.py", "s5_inject_v2.py",
              "s6_build_product_v2.py", "prereg_h1a_v2.py", "looks_h1a_v2.py", "run_v2.sh"]


def criteria(val: dict, inj: dict, leak: dict) -> list[dict]:
    out = []
    for kind, tid, label in (("paired", "H1a.v2.inject_O.paired",
                              "injected paired offsets >= 1 m recovered (family O r=10..19, "
                              "fresh; 2012-22)"),
                             ("step", "H1a.v2.inject_O.step",
                              "injected steps >= 3 m recovered (family O r=10..19, fresh; "
                              "2012-22)")):
        s = inj["summary"].get(kind, {})
        r = s.get("recall")
        out.append({"id": tid, "type": "registered test (fresh units)", "criterion": label,
                    "threshold": f"recall >= {C.RECALL_MIN}",
                    "measured": f"{r:.3f} ({s.get('n_recovered')}/{s.get('n_injected')})"
                    if r is not None else "n/a",
                    "passed": bool(r is not None and r >= C.RECALL_MIN), "data": "pre-origin"})
    cf = val["conformance"]
    re = cf["real_events"]
    out.append({"id": "H1a.v2.C3.real_events", "type": "conformance (units seen in v1)",
                "criterion": "no flag on the known real events (09200321, 09190132 recovery "
                             "2024-05-28..06-04; 09070311 typhoon recharge end of July 2024)",
                "threshold": "0 of 3 flagged (each event must have days in the record)",
                "measured": f"{re['n_flagged']} of {re['n_events']} flagged; "
                            f"{re['n_untestable']} without record days",
                "passed": re["n_flagged"] == 0 and re["n_untestable"] == 0,
                "data": "post-origin windows (plan-mandated; changes no parameter)"})
    a = cf["audit_agreement"]["hist"]
    out.append({"id": "H1a.v2.C4.audit_agreement", "type": "conformance (units seen in v1)",
                "criterion": "agreement with the audit's 2012-22 raw-record events (same well,"
                             " +-1 day)", "threshold": f">= {C.AUDIT_AGREE_MIN}",
                "measured": f"{a['agreement']:.4f} ({a['matched']}/{a['n_audit_events']})",
                "passed": bool(a["agreement"] >= C.AUDIT_AGREE_MIN), "data": "pre-origin"})
    o = cf["o1_coherence"]["pre"]["corrected_wells"]
    md = o["median_paired_diff"]
    out.append({"id": "H1a.v2.C5.baro_o1", "type": "conformance (units seen in v1)",
                "criterion": "barometric step: median O1 phase coherence improves (wells with a"
                             " correction, 2012-22, paired)",
                "threshold": "median paired difference > 0",
                "measured": (f"{md:+.4f} (median {o['median_raw']:.3f} -> {o['median_corr']:.3f};"
                             f" {o['n_improved']} better / {o['n_worse']} worse of "
                             f"{o['n_wells']}; sign p {o['sign_test_p']:.3g})")
                if md is not None else "n/a",
                "passed": bool(md is not None and md > 0), "data": "pre-origin"})
    pdp = cf["pressure_drops"]["pre"]
    out.append({"id": "H1a.v2.C6.pressure_drop", "type": "conformance (units seen in v1; holds "
                "by construction in v2)",
                "criterion": "no flagged offset episode (nor any flagged v2 object with an edge: "
                             "dropout, step, excursion, jump edge) coincides with a pressure "
                             "drop >= 20 hPa (2012-22)",
                "threshold": "0", "measured": f"{pdp['n_on_fall_ge_20hpa']} of "
                                              f"{pdp['n_objects']} flagged objects",
                "passed": pdp["n_on_fall_ge_20hpa"] == 0, "data": "pre-origin"})
    out.append({"id": "H1a.v2.leakprobe", "type": "registered test (fresh units)",
                "criterion": "leakage probe: v2 pre-origin rows rebuilt without any "
                             "post-origin file are identical to the product's pre-origin rows",
                "threshold": "0 differing rows",
                "measured": f"{leak['n_diff_total']} differing rows "
                            f"({leak['n_rows_compared']} compared)",
                "passed": leak["n_diff_total"] == 0, "data": "pre-origin"})
    return out


def well_qc_summary(work: Path, build: Path) -> pd.DataFrame:
    d2 = pd.read_parquet(work / "decisions_v2_stage1.parquet")
    D = pd.read_parquet(build / "heads_daily.parquet", columns=["sid", "date", "period", "flag"])
    rows = []
    for (sid, per), g in D.groupby(["sid", "period"]):
        rec = g[g.flag != 8]
        if rec.empty:
            continue
        dd = d2[(d2.sid == sid) & (d2.period == per)]
        j = dd[dd.kind == "jump"]
        years = len(rec) / 365.25
        rows.append({"sid": sid, "period": per,
                     "date": "2022-12-31" if per == "pre" else "2026-09-30",
                     "n_days_with_record": int(len(rec)),
                     "share_days_flag9": float((rec.flag == 9).mean()),
                     "jumps_ge_1m": int(len(j)),
                     "jumps_persistent": int((j.v2_status == "persistent").sum()),
                     "jumps_typhoon_time": int((j.v2_status == "typhoon_time").sum()),
                     "persistent_jumps_per_year": float((j.v2_status == "persistent").sum()
                                                        / years) if years > 0 else None,
                     "offset_episodes": int((dd.kind == "offset_episode").sum()),
                     "dropouts": int((dd.kind == "dropout").sum()),
                     "steps_accepted": int((dd.kind == "step").sum()),
                     "steps_corrected": int(((dd.kind == "step")
                                             & dd.flag.isin([1, 3])).sum()),
                     "rejected_step_runs": int(dd.kind.isin(["excursion", "jump_edge"]).sum())})
    return pd.DataFrame(rows).sort_values(["sid", "period"]).reset_index(drop=True)


def main():
    work, build = Path(sys.argv[1]), Path(sys.argv[2])
    val = json.loads(Path(sys.argv[3]).read_text())
    inj = json.loads(Path(sys.argv[4]).read_text())
    leak = json.loads(Path(sys.argv[5]).read_text())
    dry = "--dry-run" in sys.argv
    crit = criteria(val, inj, leak)
    passed = all(c["passed"] for c in crit)
    for c in crit:
        print(("PASS " if c["passed"] else "FAIL ") + c["id"] + ": " + c["measured"])
    if dry:
        return
    d = manifest.new_version_dir(PRODUCT)
    for f in ("heads_daily.parquet", "heads_monthly.parquet", "events.csv",
              "barometric_efficiency.csv"):
        shutil.copy2(build / f, d / f)
    P = baro.load_pressure()
    P = P[(P.index >= "2011-12-25") & (P.index < "2026-10-01")]
    PL = P.rename_axis("date").reset_index().melt(id_vars="date", var_name="era5_cell",
                                                  value_name="surface_pressure_hpa")
    PL = PL.dropna(subset=["surface_pressure_hpa"]).sort_values(["era5_cell", "date"])
    PL.to_parquet(d / "pressure_hourly.parquet", index=False)
    sm = pd.read_parquet(work / "station_meta_clean.parquet")
    sm.to_parquet(d / "station_meta_clean.parquet", index=False)
    sm[[c for c in STATIC_COLS if c in sm.columns]].to_parquet(
        d / "station_meta_preorigin.parquet", index=False)
    well_qc_summary(work, build).to_csv(d / "well_qc_summary.csv", index=False)
    report = {"wp": "H1a", "version": d.name, "prereg": PREREG, "criteria": crit,
              "gate_passed": passed, "gate_rule": "conjunctive over the 7 criteria",
              "validation": val,
              "injection": {k: v for k, v in inj.items() if k not in ("rows", "spurious")},
              "leakage_probe": leak,
              "evidence_files": [f"{EVID}/validation_v2.json", f"{EVID}/injection_O_r10-19.json",
                                 f"{EVID}/leakprobe_v2.json"]}
    Path(d / "validation.json").write_text(json.dumps(report, indent=1, default=str))
    (d / "PRODUCT_CARD.md").write_text(card(val, crit, passed, d.name))

    here = Path(__file__).resolve().parent
    code = [here / f for f in CODE_FILES if (here / f).exists()]
    code += [REPO / "dataprep_staging" / "harness" / "gaps.py",
             REPO / "dataprep_staging" / "harness" / "gaps_guarded.py",
             REPO / "dataprep_staging" / "common" / "paths.py"]
    inputs = [rawio.STATIONS, meta.WRA_LIST, meta.TOWNS,
              PILOTS / "audit_2026-10-11" / "artefact_events.csv",
              REPO / "results" / "dataprep" / "reports" / "A0_evidence" / "repair" / "reruns"
              / "rho_k_result.json",
              REPO / "results" / "twin" / "hybrid_datum_gate_innov" / "evaluation.csv",
              REPO / "results" / "dataprep" / "heads" / "L1_clean" / "v1" / "events.csv",
              REPO / "results" / "dataprep" / "heads" / "L1_clean" / "v1"
              / "heads_monthly.parquet",
              REPO / PREREG]
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
               {"path": "well_qc_summary.csv"},
               {"path": "validation.json"}, {"path": "PRODUCT_CARD.md"}]
    m = manifest.build_manifest(
        d, product=PRODUCT, layer="L1", inputs=inputs, code=[str(c) for c in code],
        input_trees=[rel_to_repo(rawio.HIST_DIR), rel_to_repo(rawio.NEW_DIR), rel_to_repo(codis)],
        parameters={
            "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
            "decision_layer": "decide_v2 (round 2): 10-day LAD level persistence of every "
                              "stage-1 jump >= 1 m (not spikes), typhoon-time jumps (>= 20 hPa "
                              "fall within +-24 h) excluded from edges, greedy pairing of "
                              "persistent jumps into episodes (flag 9, never corrected), "
                              "segment-checked steps >= 3 m with the plan's flag rule",
            "levelcheck": {"window_days": 10, "min_hourly_medians": 12, "frac_of_jump": 0.5},
            "merge": "historical cache 2012-01-01..2022-12-31 + new network 2023-01-01.."
                     "2026-09-30 (2022-10..12 overlap identical, p0_profile.json)",
            "pre_origin_rule": "rows dated <= 2022-12 are built from pre-origin data only "
                               "(separate pre-origin pass, pre-origin backing evidence, "
                               "pre-origin daily screen); verified by the leakage probe",
            "barometric_design": C.BARO_DESIGN, "seed": C.BARO_SEED,
            "pressure_source": "Open-Meteo ERA5 surface_pressure, nearest 0.25-degree cell",
            "flags": {"0": "observed, passed QC", "1": "observed, corrected (barometric loading "
                      "removed, or an accepted step corrected with physical metadata or "
                      "co-located backing that the neighbours do not refute)",
                      "3": "step corrected with neighbour-regression corroboration only",
                      "8": "missing / sentinel only", "9": "suspect (kept, excluded "
                      "downstream): offset episodes, dropouts, uncorrected steps, rejected "
                      "step runs, sample screens"},
            "supersedes": "heads/L1_clean v1 (gate not passed; kept published)",
            "prereg": PREREG},
        fit_window_end="2026-09",
        time_slicing={"column": "date", "params_fit_window_end": "2022-12", "encoding": "iso"},
        validation_report="validation.json",
        gate={"passed": passed, "criteria": "plan WP H1 (H1a) acceptance, pre-registered in "
                                            "H1a.qc_acceptance.v2; see validation.json"},
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


def card(val, crit, passed, version) -> str:
    s = val["summary"]
    cov = s["monthly_coverage"]
    lines = [f"# heads/L1_clean {version} (WP H1a, round 2)", "",
             "Cleaned groundwater heads of the Choushui fan monitoring network, 2012-01 to "
             "2026-09: the historical cache (2012-22) merged with the new network (2023-26), "
             "screened, barometric loading removed where significant, datum-offset episodes "
             "and dropouts flagged (never corrected), steps of 3 m or more flagged or corrected "
             "with physical metadata or co-located backing, daily and C20 monthly aggregates "
             "with sigma_obs. Supersedes v1 (whose gate failed and whose episode/step "
             "corrections were mostly spurious).", "",
             f"Gate: {'passed' if passed else 'NOT passed'}.", "", "## Coverage", ""]
    for per, c in cov.items():
        lines.append(f"- {per}: {c['wells_with_record']} wells with a record; "
                     f"{c['observed_flag01']} observed well-months (flag 0/1), "
                     f"{c['observed_flag01_c20']} of them C20; flag 3: {c['flag3']}, "
                     f"flag 9: {c['flag9']}, flag 8: {c['flag8']}.")
    lines += ["", "## Flags in use", "",
              "0 observed; 1 observed and corrected (barometric loading removed, or an "
              "accepted step corrected with physical metadata or a co-located nested well); "
              "3 step corrected with neighbour-regression corroboration only (not a "
              "calibration target, U2); 8 missing or sentinel only; 9 suspect (kept, "
              "excluded downstream).", "", "## Use", "",
              "Pre-origin uses (calibration targets, datum fits, baselines, initial "
              "conditions) must open the product with `window_end <= 2022-12` (time slice on "
              "`date`); `station_meta_preorigin.parquet` is the pre-origin station table. "
              "The registered prospective scorer keeps its registered QC (forbidden use). "
              "`value_raw` and `baro_m` keep every correction reversible.", "",
              "## Acceptance (H1a.qc_acceptance.v2)", ""]
    for c in crit:
        lines.append(f"- {'PASS' if c['passed'] else 'FAIL'} [{c['type']}]: {c['criterion']} "
                     f"({c['threshold']}): {c['measured']}")
    lines += ["", "## Known limits", "",
              "- Jumps across logger gaps longer than 70 minutes are not detected (the audit's "
              "rule); episodes shorter than about 10 days are not paired (level persistence).",
              "- In pumping-cycle wells (pump switches of 1-4 m within an hour that last days "
              "to weeks) the decision layer cannot tell a datum episode from a pump regime; "
              "it flags them (9). See well_qc_summary.csv (share_days_flag9, "
              "persistent_jumps_per_year); value_raw keeps the readings.",
              "- Typhoon-time jumps (>= 20 hPa fall within +-24 h) are kept as real-event "
              "candidates, never used as datum edges.",
              "- The displaced side of an accepted step is the impossible side, else the "
              "shorter side; for absolute levels the side may be wrong by the step size.",
              "- Barometric efficiency is estimated only where significant; elsewhere no "
              "correction is applied (flag 0).",
              "- GroundHeight 0.0 is a missing-value code (H1b replaces it after W3).",
              "- Layer codes proposed from WellDepth are flag 5, unverified.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
