"""Write the H1a round-2 pre-registration (``H1a.qc_acceptance.v2``) with the repaired A1 tool
(``harness.prereg2``), before any v2 registered test runs.

Usage: prereg_h1a_v2.py [--dry-run]

The design-time evidence files named in ``inputs`` must already sit in
results/dataprep/reports/H1a_evidence/v2/design_time/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg2  # noqa: E402

HERE = Path(__file__).resolve().parent
NA = prereg2.not_applicable
DT = "results/dataprep/reports/H1a_evidence/v2/design_time"

CODE = ["config.py", "rawio.py", "screens.py", "baro.py", "tides.py", "meta.py",
        "pipeline.py", "backing.py", "assemble.py", "s0_meta.py", "s1_screen.py",
        "s4_validate.py", "s5_inject.py", "s6_build_product.py", "s7_leakprobe.py",
        "levelcheck.py", "decide_v2.py", "assemble_v2.py", "s2b_decide_v2.py",
        "s3_assemble_v2.py", "s4_validate_v2.py", "s5_inject_v2.py", "s6_build_product_v2.py",
        "looks_h1a_v2.py", "run_v2.sh"]

EXT = {
    "H1a:inject_O_r10-19_paired": "family O realizations r = 10..19 of harness.gaps "
                                  "(seed 1000*ord('O') + r), paired offsets; never generated "
                                  "or scored before this registration",
    "H1a:inject_O_r10-19_step": "family O realizations r = 10..19, steps; never generated or "
                                "scored before this registration",
    "H1a:v2_preorigin_rows": "the pre-origin rows of the v2 build (daily, monthly, events "
                             "dated <= 2022-12, barometric table) against the v2 pre-only "
                             "rebuild; the v2 builds did not exist before this registration",
}

DOC = {
    "id": "H1a.qc_acceptance.v2",
    "wp": "H1a",
    "title": "WP H1a round 2: acceptance of the v2 decision layer (level-persistent jumps, "
             "typhoon-time rule, best-match pairing, trend-checked steps, plan flag rule)",
    "kind": "test",
    "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP H1 (validation and acceptance), "
                    "4.4 (family O), 4.5, 2.4-2.5",
    "supersedes": "H1a.qc_acceptance.v1 (stands as registered and scored; its gate failed: "
                  "paired recall 0.889, 1 of 3 real events flagged, 3 of 228 episodes on "
                  ">= 20 hPa falls)",
    "tests": [
        {"test_id": "H1a.v2.inject_O.paired", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:inject_O_r10-19_paired"],
         "external_units": {"H1a:inject_O_r10-19_paired": EXT["H1a:inject_O_r10-19_paired"]},
         "metric": "recall of injected paired offsets (1-3 m, 15-365 days; 20 per realization, "
                   "r = 10..19; s5_inject_v2.py): an offset episode or dropout of the v2 "
                   "decisions whose first jump is within +-1 day of the injected jump and "
                   "whose return is within +-1 day of the reversal",
         "threshold": ">= 0.9"},
        {"test_id": "H1a.v2.inject_O.step", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:inject_O_r10-19_step"],
         "external_units": {"H1a:inject_O_r10-19_step": EXT["H1a:inject_O_r10-19_step"]},
         "metric": "recall of injected steps (3-6 m, 20 per realization, r = 10..19): an "
                   "accepted step, offset episode or dropout of the v2 decisions whose first "
                   "jump is within +-1 day of the injected jump",
         "threshold": ">= 0.9"},
        {"test_id": "H1a.v2.leakprobe", "data_scope": "heads:2012-01..2022-12",
         "fresh_units": ["H1a:v2_preorigin_rows"],
         "external_units": {"H1a:v2_preorigin_rows": EXT["H1a:v2_preorigin_rows"]},
         "metric": "v2 pre-origin rows rebuilt with --pre-only (no post-origin file opened) "
                   "that differ from the v2 product (s7_leakprobe.py)",
         "threshold": "0"},
    ],
    "conformance_checks": [
        {"id": "H1a.v2.C3.real_events", "units_seen_in_v1": ["H1a:real_events_2024"],
         "data_scope": "heads:2024-05..2024-08",
         "metric": "number of the 3 known real events (config.REAL_EVENTS) carrying any flag-9 "
                   "day, artefact event (v1 kinds plus excursion and jump_edge) or correction "
                   "in their window; an event without record days fails",
         "threshold": "0"},
        {"id": "H1a.v2.C4.audit_agreement", "units_seen_in_v1": ["H1a:audit_artefact_events_hist"],
         "data_scope": "heads:2012-01..2022-12",
         "metric": "share of the audit's 2012-22 raw-record events matched by a stage-1 jump "
                   "of the same well within +-1 day (stage 1 unchanged from v1)",
         "threshold": ">= 0.95"},
        {"id": "H1a.v2.C5.baro_o1", "units_seen_in_v1": ["H1a:o1_corrected_wells_2012_2022"],
         "data_scope": "heads:2012-01..2022-12",
         "metric": "paired median over wells with an accepted barometric correction and >= 12 "
                   "monthly O1 phases of coherence(corrected) - coherence(raw) (barometric "
                   "fits unchanged from v1)",
         "threshold": "> 0"},
        {"id": "H1a.v2.C6.pressure_drop", "units_seen_in_v1": ["H1a:offset_episodes_2012_2022"],
         "data_scope": "heads:2012-01..2022-12",
         "metric": "flagged v2 objects with an edge (offset episodes, dropouts, accepted steps, "
                   "excursion and jump-edge runs) of 2012-22 with a pressure fall >= 20 hPa "
                   "within +-24 h of an edge jump at the well's ERA5 cell",
         "threshold": "0"},
    ],
    "zones": NA("QC of point records; no zones"),
    "priors": NA("no priors"),
    "burn_in": NA("no model run"),
    "folds": NA("no folds; synthetic injections, audit events and fixed real-event windows"),
    "fresh_units": ["H1a:inject_O_r10-19_paired", "H1a:inject_O_r10-19_step",
                    "H1a:v2_preorigin_rows"],
    "design_time_units": ["H1a:inject_O_r0-9_paired", "H1a:inject_O_r0-9_step",
                          "H1a:real_events_2024", "H1a:audit_artefact_events_hist",
                          "H1a:o1_corrected_wells_2012_2022", "H1a:offset_episodes_2012_2022",
                          "H1a:preorigin_rows"],
    "thresholds": {"recall_paired_r10_19": 0.9, "recall_step_r10_19": 0.9,
                   "leakprobe_differing_rows": 0, "real_events_flagged": 0,
                   "audit_agreement": 0.95, "o1_median_paired_diff": "> 0",
                   "flagged_objects_on_20hpa_drop": 0},
    "gate_scores": [{"name": t, "weight": 1.0 / 7.0, "rule": "pass/fail"} for t in
                    ("inject_paired_r10_19", "inject_step_r10_19", "leakprobe_v2",
                     "C3_real_events", "C4_audit_agreement", "C5_baro_o1",
                     "C6_pressure_drop")],
    "gate_rule": "conjunctive: the v2 product gate passes only if the three registered tests "
                 "and the four conformance checks all pass; the equal weights only satisfy "
                 "the schema",
    "primary_model": "the H1a v2 configuration: config.py (unchanged from v1), the stage-1 code "
                     "of v1 (unchanged) and the v2 decision layer decide_v2.py / levelcheck.py "
                     "with assemble_v2.py, all hashed in inputs",
    "inputs": [rel_to_repo(HERE / f) for f in CODE] + [
        "dataprep_staging/harness/gaps.py",
        "dataprep_staging/harness/gaps_guarded.py",
        "results/dataprep/_pilots/audit_2026-10-11/artefact_events.csv",
        "results/dataprep/reports/A0_evidence/repair/reruns/rho_k_result.json",
        "results/dataprep/heads/L1_clean/v1/MANIFEST.json",
        f"{DT}/design_inject_d1a_r0-9.json",
        f"{DT}/design_inject_d1b_r0-9.json",
        f"{DT}/design_inject_d2_r0-9.json",
        f"{DT}/design_misses_d2.csv",
        f"{DT}/design_inject_d3_r0-9.json",
        f"{DT}/design_step_misses_d3.csv",
        f"{DT}/design_inject_d4_r0-9.json"],
    "reported_not_gated": [
        "pooled recall over r = 0..19 (r = 0..9 design-time)",
        "recall lenient and strict, by size, side-correct share",
        "false flags created by the injections (per injected event)",
        "real-data false-flag proxies: v1 corrections withdrawn, jump catalogue, flag-9 share "
        "per well, pumping-cycle wells",
        "chance coincidences of flagged objects with >= 20 hPa falls (year-shifted times)",
        "post-origin analogues of C3-C6, CWA station pressure vs ERA5",
        "frozen-challenge rescore on the v2 monthly means (split dropped months / changed "
        "values)",
        "v1 -> v2 transitions of flags, C20 months and corrections"],
    "interpretation": (
        "Round 2 of WP H1a after the v1 gate failed (H1a.qc_acceptance.v1 stands; its results "
        "are not re-scored). Thresholds are the plan's and are not changed after the results. "
        "The v2 change is the decision layer only; stage 1 (screens, jump detector, barometric "
        "fits) is v1's code unchanged. "
        "Fresh units: the family O realizations r = 10..19 and the v2 pre-origin rows; no "
        "look names them before this registration. Design-time units: family O r = 0..9 "
        "(scored by v1, then scored five times to design v2: design_inject_d1a, d1b, d2, d3, d4), the three "
        "real events, the audit events, the O1 wells and the 2012-22 episodes (all looked at "
        "in v1, and the v1 failure diagnosis inspected the episodes at 07080132, 07240223 and "
        "the spurious 2014/2019 steps of 09190132). The conformance checks C3-C6 are "
        "requirements of the plan evaluated on those seen units: a pass there is necessary, "
        "not independent evidence. C6 holds by construction, because v2 excludes jumps "
        "within +-24 h of a >= 20 hPa pressure fall from episode and step edges (the plan's "
        "'real events are kept: typhoon recharge'); the barometric evidence is C5 and the "
        "physical bound (the fitted responses give a few centimetres for a 40 hPa fall). "
        "The v2 design was motivated by after-the-fact diagnoses that read post-origin "
        "records only through the v1 real-event check (09190132's 2024 flag traced to a "
        "2019 step); every v2 rule and constant was chosen and evaluated on 2012-22 data "
        "only (U7). The level-check constants (10 days, 12 hourly medians, half the jump) "
        "were fixed before any v2 score. Across the design rounds on r = 0..9 the rules "
        "changed (d1: stage-1 pairs re-checked; d2: level-persistent jumps paired; d3: "
        "undetermined fallback, best-match pairing, trend-adjusted one-year segment check; "
        "d4: one-year median segment check) but no threshold did; the final design-time recall on r = 0..9 is "
        "in design_inject_d4_r0-9.json. A failed test is reported as failed with its number "
        "and the gate is not passed."),
}


def main():
    errs = prereg2.validate_prereg(DOC)
    if errs:
        raise SystemExit("; ".join(errs))
    if "--dry-run" in sys.argv:
        print(json.dumps(DOC, indent=1, ensure_ascii=False)[:3000])
        print("blocking looks:", prereg2.blocking_looks(DOC["tests"], None))
        return
    path, sha = prereg2.write_prereg(DOC)
    print(path, sha)


if __name__ == "__main__":
    main()
