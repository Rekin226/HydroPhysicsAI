"""H1a repair-round ledger rows (plan 4.1, rule U6) for every number the repair-round section of
the WP report quotes (python -I). Number ids start with ``H1a.r2.``; each names its data
sources with one role each (input, validator, design_time_evidence).

Usage: s8b_ledger_r2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ledger  # noqa: E402

HIST = "AMP_V2/data/wells"
NEW = "data_fetch_api/new_measurements/heads_202210_202610"
OM = "data_fetch_api/new_measurements/pressure_hourly/openmeteo_era5_2026-10-12"
AUDIT = "results/dataprep/_pilots/audit_2026-10-11/artefact_events.csv"
EVAL = "results/twin/hybrid_datum_gate_innov/evaluation.csv"
RHO = "results/dataprep/reports/A0_evidence/repair/reruns/rho_k_result.json"
GAPS = "dataprep_staging/harness/gaps.py"
E = "results/dataprep/reports/H1a_evidence/v2"
P1 = "results/dataprep/heads/L1_clean/v1"
P2 = "results/dataprep/heads/L1_clean/v2"
P3 = "results/dataprep/heads/L1_clean/v3"
LOOKS = "results/dataprep/preregistration/looks.jsonl"
PREREG = "results/dataprep/preregistration/H1a.qc_acceptance.v2.json"


def rows() -> list[dict]:
    R = []

    def add(nid, src, role, pv, note=""):
        R.append({"number_id": f"H1a.r2.{nid}", "source": src, "role": role,
                  "product_version": pv, "note": note})

    # registered tests (H1a.qc_acceptance.v2, scored on the registered v2 build = product v2)
    for k, note in (("inject.recall_paired", "0.920 (184/200), family O r=10..19"),
                    ("inject.recall_step", "0.920 (184/200), family O r=10..19"),
                    ("inject.spurious", "1 flag created per 400 injected events"),
                    ("inject.harness_artefacts", "2 of 16 paired misses: effective "
                                                 "durations 406 d and 1 d")):
        add(k, HIST, "input", P2, "pre-origin raw records the injections were added to")
        add(k, GAPS, "input", P2, "family O generator (seed 1000*ord('O')+r)")
        add(k, OM, "input", P2, "ERA5 pressure for the typhoon-time rule")
        add(k, f"{E}/injection_O_r10-19.json", "validator", P2, f"injection record: {note}")
    add("leakprobe.v2", HIST, "input", P2, "pre-origin raw records")
    add("leakprobe.v2", NEW, "input", P2, "post-origin records (full build only)")
    add("leakprobe.v2", f"{E}/leakprobe_v2.json", "validator", P2,
        "0 of 907,684 rows differ")
    # conformance checks (units seen in v1), v2 and v3 builds
    for pv, tag in ((P2, "v2"), (P3, "v3")):
        add(f"C3.real_events.{tag}", NEW, "validator", pv, "2024 records of the 3 wells")
        add(f"C3.real_events.{tag}", f"{E}/validation_{tag}.json", "input", pv,
            "0 of 3 flagged")
        add(f"C4.audit.{tag}", HIST, "input", pv, "stage-1 jumps (unchanged from v1)")
        add(f"C4.audit.{tag}", AUDIT, "validator", pv, "1.000 (4736/4736)")
        add(f"C5.o1.{tag}", HIST, "input", pv, "hourly heads 2012-22")
        add(f"C5.o1.{tag}", OM, "input", pv, "+0.0023 over 68 corrected wells; all 258 "
                                             "wells 0.1576 -> 0.1576")
        add(f"C6.pressure.{tag}", HIST, "input", pv, "v2 flagged objects with an edge")
        add(f"C6.pressure.{tag}", OM, "validator", pv, "0 of 307 on >= 20 hPa falls")
    add("leakprobe.v3", HIST, "input", P3, "pre-origin raw records")
    add("leakprobe.v3", NEW, "input", P3, "post-origin records (full build only)")
    add("leakprobe.v3", f"{E}/leakprobe_v3.json", "validator", P3,
        "0 of 907,684 rows differ (post-registration recheck)")
    add("chance_coincidences", HIST, "input", P2, "v1 episodes 3 observed vs 1.2 expected; "
                                                   "v2 objects 0 vs 0.4")
    add("chance_coincidences", OM, "validator", P2, "year-shifted ERA5 falls")
    add("stage1_identity", HIST, "input", P2, "344 wells: jumps, episodes, steps, BE, O1 "
                                               "identical to the v1 run")
    # after-the-fact verification of the corrections
    add("corrections.v2.level_check", HIST, "input", P2, "8 of 9 step corrections pass")
    add("corrections.v2.level_check", NEW, "input", P2, "3 post-origin corrections")
    add("corrections.v2.level_check", f"{E}/verify_v2.json", "validator", P2,
        "independent 10-day medians")
    add("corrections.v2.sibling", HIST, "input", P2, "7 not confirmed, 2 without sibling "
                                                      "data (1 co-located, 1 neighbour-only)")
    add("corrections.v2.sibling", NEW, "input", P2, "post-origin siblings")
    add("corrections.v2.sibling", f"{E}/verify_v2.json", "validator", P2,
        "well minus nested sibling, 365 d")
    add("corrections.v2.months", HIST, "input", P2, "pre 106 well-months at 4 wells")
    add("corrections.v2.months", NEW, "input", P2, "post 4 well-months at 3 wells")
    add("corrections.v3.months", HIST, "input", P3, "pre 0")
    add("corrections.v3.months", NEW, "input", P3, "post 2 well-months at 1 well (flag 3)")
    add("corrections.v1.level_check", P1, "input", P3,
        "v1 pre-origin flag-1 steps 18 of 39 fail, flag-1 episodes 125 of 210 fail")
    add("corrections.v1.level_check", HIST, "validator", P3, "raw records, 10-day medians")
    add("examples.verifier", P1, "input", P3, "07230332 and 07260122: v1 corrected "
                                               "1796 / 2879 days, v3 0 / 0")
    add("examples.verifier", HIST, "validator", P3, "raw records")
    add("examples.verifier", NEW, "validator", P3, "09190132 2019-26: v1 2458 flag-9 days, "
                                                    "v3 0")
    # false-flag proxies and coverage (v3)
    add("false_flags.episodes", HIST, "input", P3, "78 of 271 flagged pre-origin episodes "
                                                    "fail the level check; 68 at 3 wells")
    add("false_flags.flag9_share", HIST, "input", P3, "pre 1.29 % of days")
    add("false_flags.flag9_share", NEW, "input", P3, "post 0.62 % of days")
    add("coverage.v3", HIST, "input", P3, "pre 27,694 flag 0/1 well-months (27,133 C20)")
    add("coverage.v3", NEW, "input", P3, "post 6,868 (6,285 C20)")
    add("coverage.v3", RHO, "input", P3, "pre-origin rho(k) for sigma_obs")
    add("events.v3", HIST, "input", P3, "272 episodes at 31 wells, 10 dropouts, 6 steps")
    add("events.v3", NEW, "input", P3, "6 episodes, 2 dropouts, 4 steps")
    add("pumping_cycle_wells", HIST, "input", P3, "6 wells with >= 2 persistent jumps/yr")
    # rescore (secondary, reported)
    add("rescore.v3", EVAL, "input", P3, "frozen challenge predictions")
    add("rescore.v3", NEW, "input", P3, "+0.0044 = +0.0032 dropped months + 0.0012 values")
    add("rescore.v1_split", EVAL, "input", P1, "frozen challenge predictions")
    add("rescore.v1_split", P1, "input", P1, "-0.0525 = -0.0071 dropped months - 0.0455 "
                                             "changed values")
    # barometric plausibility (fits unchanged from v1)
    add("baro.implausible", HIST, "input", P3, "2 accepted epochs A_K < -1, 24 A_K > 0 at "
                                                "22 wells; max |baro_m| 0.32 m")
    add("baro.implausible", OM, "input", P3, "ERA5 pressure")
    # provenance
    add("check.v2_v3", f"{E}/check_v2_v3.json", "validator", P3,
        "v3 100 files ok; v2 3 repair-round code files edited after packaging")
    add("guard_probe", f"{E}/guard_probe_v1_v2_v3.json", "validator", P3,
        "v3 grants pre-origin uses with window_end 2022-12; v2 provenance only; v1 gate")
    add("looks.repair_round", LOOKS, "validator", P3,
        "history disclosure 1, registered scores 3, conformance 5, diagnose 10 (19 in "
        "the repair round; 36 H1a looks in all)")
    add("prereg.inputs_intact", PREREG, "validator", P3, "37 registered inputs unchanged")
    return R


def main():
    path = ledger.append_rows("H1a", rows())
    print("ledger rows ->", path, len(rows()))


if __name__ == "__main__":
    main()
