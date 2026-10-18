"""Write the H1a pre-registration with the A1 tool, before the acceptance tests run.

Usage: prereg_h1a.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg  # noqa: E402

HERE = Path(__file__).resolve().parent
NA = prereg.not_applicable

CODE = ["config.py", "rawio.py", "screens.py", "baro.py", "tides.py", "meta.py",
        "pipeline.py", "backing.py", "assemble.py", "s0_meta.py", "s1_screen.py",
        "s2_backing.py", "s3_assemble.py", "s4_validate.py", "s5_inject.py", "s7_leakprobe.py",
        "s6_build_product.py"]

DOC = {
    "id": "H1a.qc_acceptance.v1",
    "wp": "H1a",
    "title": "WP H1a head QC: acceptance of the frozen screens, barometric correction, "
             "offset/step detection and flag rule",
    "kind": "test",
    "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP H1 (validation and acceptance), "
                    "4.4 (family O), 2.4-2.5",
    "tests": [
        {"test_id": "H1a.inject_O.paired", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:inject_O_r0-9_paired"],
         "metric": "recall of injected paired offsets (1-3 m, 15-365 days; harness "
                   "inject_offsets_steps seeds 1000*ord('O')+r, r=0..9, 20 per realization): "
                   "an episode with both jumps within +-1 day", "threshold": ">= 0.9"},
        {"test_id": "H1a.inject_O.step", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:inject_O_r0-9_step"],
         "metric": "recall of injected steps (3-6 m, 20 per realization): a step or episode "
                   "starting within +-1 day", "threshold": ">= 0.9"},
        {"test_id": "H1a.audit_agreement.v1", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:audit_artefact_events_hist"],
         "metric": "share of the audit's 2012-22 raw-record events (artefact_events.csv, "
                   "spikes and offsets) matched by an H1a jump of the same well within +-1 "
                   "day", "threshold": ">= 0.95"},
        {"test_id": "H1a.real_events", "data_scope": "heads:2024-05..2024-08",
         "fresh_units": ["H1a:real_events_2024"],
         "metric": "number of the 3 known real events (config.REAL_EVENTS) carrying any "
                   "flag-9 day, artefact event or correction in their window",
         "threshold": "0"},
        {"test_id": "H1a.baro_o1", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:o1_corrected_wells_2012_2022"],
         "metric": "paired median over wells with an accepted barometric correction and >= 12 "
                   "monthly O1 phases of coherence(corrected) - coherence(raw)",
         "threshold": "> 0"},
        {"test_id": "H1a.baro_pressure_drop", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:offset_episodes_2012_2022"],
         "metric": "offset episodes (any flag) with a pressure fall >= 20 hPa within +-24 h "
                   "of either jump, at the well's ERA5 cell", "threshold": "0"},
        {"test_id": "H1a.leakprobe", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:preorigin_rows"],
         "metric": "pre-origin rows (daily, monthly, events dated <= 2022-12, barometric "
                   "table) rebuilt with --pre-only that differ from the product",
         "threshold": "0"},
    ],
    "zones": NA("QC of point records; no zones"),
    "priors": NA("no priors"),
    "burn_in": NA("no model run"),
    "folds": NA("no folds; synthetic injections, audit events and fixed real-event windows"),
    "fresh_units": ["H1a:inject_O_r0-9_paired", "H1a:inject_O_r0-9_step",
                    "H1a:audit_artefact_events_hist", "H1a:real_events_2024",
                    "H1a:o1_corrected_wells_2012_2022", "H1a:offset_episodes_2012_2022",
                    "H1a:preorigin_rows"],
    "thresholds": {"recall_paired": 0.9, "recall_step": 0.9, "audit_agreement": 0.95,
                   "real_events_flagged": 0, "o1_median_paired_diff": "> 0",
                   "episodes_on_20hpa_drop": 0, "leakprobe_differing_rows": 0},
    "gate_scores": [{"name": t, "weight": 1.0 / 7.0, "rule": "pass/fail"} for t in
                    ("inject_paired", "inject_step", "audit_agreement", "real_events",
                     "baro_o1", "baro_pressure_drop", "leakprobe")],
    "gate_rule": "conjunctive: the product gate passes only if all seven tests pass; the "
                 "equal weights only satisfy the schema",
    "primary_model": "the H1a QC configuration frozen in dataprep_staging/h1_heads_qc/config.py "
                     "and the code files listed in inputs (sha256 recorded here); barometric "
                     "design 'hourly_grouped' chosen beforehand on pre-origin split-half "
                     "reliability (design-time, results/dataprep/reports/H1a_evidence/"
                     "design_time/baro_methods.json)",
    "inputs": [rel_to_repo(HERE / f) for f in CODE] + [
        "dataprep_staging/harness/gaps.py",
        "results/dataprep/_pilots/audit_2026-10-11/artefact_events.csv",
        "results/dataprep/reports/A0_evidence/repair/reruns/rho_k_result.json",
        "results/dataprep/reports/H1a_evidence/design_time/baro_methods.json",
        "results/dataprep/reports/H1a_evidence/design_time/baro_epochs.json"],
    "reported_not_gated": [
        "post-origin audit agreement (2023-26 events)", "O1 coherence 2023-26",
        "pressure-drop check on 2023-26 episodes", "CWA station pressure vs ERA5",
        "frozen-challenge rescore on L1 monthly means (audit expects < 0.01 in the ratio)",
        "false-flag rate of the injection runs", "base-run share of days flagged 9"],
    "interpretation": (
        "Thresholds are the plan's (WP H1 acceptance) and are not changed after the results. "
        "All gated tests except the real-event check use pre-origin data only (U7); the "
        "real-event check is plan-mandated, reads 2024 records and changes no parameter or "
        "pre-origin row. A failed test is reported as failed with its number and the gate "
        "is not passed; nothing is re-tuned on these results. Design-time disclosure: the "
        "jump/offset thresholds come from the audit (2026-10-11), whose detector was "
        "designed while viewing both periods, and the three real events were identified by "
        "the audit, so the real-event check is in-sample for the audit's detector design; "
        "the floor-dropout and 12 m rules come from the head-signature pilot. The "
        "barometric design, the per-logger-epoch fit and the se inflation (1.23) were chosen "
        "on 2012-22 split-half reliability only (the design-time runs used the code before "
        "the inflation was added). Before this registration the code was smoke-tested on 8 "
        "wells none of which is a real-event well (07020111, 07020121, 07020131, 07010121, "
        "07040121, 07040141, 09180341, 09180331), including one injection realization r=0 "
        "on those 8 wells (8 paired events, 8 recovered) and the audit/O1/pressure checks on "
        "them; bugs found were code errors only and no threshold or parameter was changed "
        "after these smoke runs. A real event without record days counts as untestable and "
        "fails the real-event test."),
}


def main():
    errs = prereg.validate_prereg(DOC)
    if errs:
        raise SystemExit("; ".join(errs))
    if "--dry-run" in sys.argv:
        print(json.dumps(DOC, indent=1, ensure_ascii=False)[:2000])
        return
    path, sha = prereg.write_prereg(DOC)
    print(path, sha)


if __name__ == "__main__":
    main()
